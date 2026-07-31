#!/usr/bin/env python3
"""多视图几何一致性过滤，思路照搬 COLMAP PatchMatch Stereo 的 filter 阶段。

**做什么**

FoundationStereo 对每个像素都给稠密深度，不管那个像素是不是真的可信。
这里用相机位姿做多视图交叉验证：一个像素的深度如果是对的，那么把它投到
别的视角、再投回来，应该能回到原处。回不来的就是错的。

对每个参考视角的每个像素，逐个源视角检查四项：

1. **前后向重投影误差**（几何）按深度反投到 3D，投进源视角取该处深度，
   再用源视角的深度反投回 3D、投回参考视角。偏移超过阈值就不算一致。
2. **相对深度误差**（几何）正向投影算出的深度 与 源视角深度图里读到的深度，
   两者相对差超过阈值就不算一致。
3. **三角化角**（几何）两个相机对该 3D 点的张角太小时深度本来就不可解，
   直接不让它参与投票。
4. **NCC**（光度）把源图按当前深度 warp 回参考视角，做滑动窗口 NCC。
   前三项都是纯几何的，管不住"纹理很差、深度靠猜但恰好几何自洽"的区域；
   这一项补上。需要 --image-dir 才启用。

通过的源视角计入票数，票数少于阈值的像素判为不可信。

**和 COLMAP 的区别**

COLMAP 的 filter 也是三项（光度 NCC + 几何一致性 + 三角化角），但它的 NCC
来自 PatchMatch 匹配过程中算出的选择概率。这里 FoundationStereo 不输出置信度，
所以 NCC 是自己重新算的，并且额外加了一项"相对深度误差"。

COLMAP 是把不可信像素直接置 0（宁可给洞也不给错）。这里默认也置 0，
但同时输出一张票数图，需要软置信度的话可以直接用。

**为什么这一步能轻松打满 GPU**

PatchMatch 的深度估计有传播依赖（第 r 行要等第 r-1 行），COLMAP 因此只能
每列开一个线程串行扫，占用率不到 1%。而过滤没有这个依赖——每个
(像素, 源视角) 对都是独立的，可以整块张量一次算完。
一张 480x640 配 8 个源视角就是 246 万个并行单元。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- 数据加载 ----
def load_scene(transforms_json: Path, depth_dir: Path, device):
    """读 transforms.json 和深度图，返回位姿、内参、深度张量。

    transforms.json 用的是 OpenGL/Blender 约定（x 右, y 上, z 后，相机看向 -z）。
    深度图存的是沿光轴的 z 深度（FoundationStereo 的 fx*baseline/disp）。
    """
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = data["frames"]

    def _intr(frame):
        g = lambda k: float(frame.get(k, data[k]))          # noqa: E731
        return g("fl_x"), g("fl_y"), g("cx"), g("cy")

    poses, intrinsics, depths, names = [], [], [], []
    missing = []
    for frame in frames:
        stem = Path(frame["file_path"]).stem
        # 深度图可能按顺序命名（00000.npy），也可能与 file_path 同名
        cand = depth_dir / f"{stem}.npy"
        if not cand.exists() and frame.get("depth_file_path"):
            cand = (transforms_json.parent / frame["depth_file_path"]).with_suffix(".npy")
        if not cand.exists():
            missing.append(stem)
            continue
        poses.append(np.asarray(frame["transform_matrix"], dtype=np.float64))
        intrinsics.append(_intr(frame))
        depths.append(np.load(cand).astype(np.float32))
        names.append(cand.stem)

    if not depths:
        raise FileNotFoundError(
            f"{depth_dir} 里没有能和 {transforms_json} 对上的深度图")
    if missing:
        print(f"[filter_depth] 警告: {len(missing)} 个 frame 没有对应深度图，已跳过")

    shapes = {d.shape for d in depths}
    if len(shapes) != 1:
        raise ValueError(f"深度图尺寸不一致: {shapes}")

    c2w = torch.tensor(np.stack(poses), dtype=torch.float32, device=device)   # (N,4,4)
    if c2w.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1.0], device=device).expand(len(c2w), 1, 4)
        c2w = torch.cat([c2w, bottom], dim=1)
    K = torch.tensor(intrinsics, dtype=torch.float32, device=device)          # (N,4) fx,fy,cx,cy
    D = torch.tensor(np.stack(depths), dtype=torch.float32, device=device)    # (N,H,W)
    return c2w, K, D, names


def load_images(image_dir: Path, names, shape, device):
    """按深度图的名字读对应的渲染图，转灰度。返回 (N,H,W) 或 None。

    NCC 检验要用渲染出来的图，所以名字必须和深度图对得上——
    两者都来自同一次 ns-render，命名是一致的。
    """
    import cv2
    H, W = shape
    imgs, missing = [], []
    for name in names:
        hit = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = image_dir / f"{name}{ext}"
            if cand.exists():
                hit = cand
                break
        if hit is None:
            missing.append(name)
            imgs.append(np.zeros((H, W), np.float32))
            continue
        g = cv2.imread(str(hit), cv2.IMREAD_GRAYSCALE)
        if g is None:
            missing.append(name)
            g = np.zeros((H, W), np.uint8)
        if g.shape != (H, W):
            raise ValueError(f"{hit} 尺寸 {g.shape} 与深度图 {(H, W)} 不符")
        imgs.append(g.astype(np.float32) / 255.0)

    if len(missing) == len(names):
        raise FileNotFoundError(
            f"{image_dir} 里一张都对不上深度图的名字（例如 {names[0]}.png）")
    if missing:
        print(f"[filter_depth] 警告: {len(missing)} 张渲染图缺失，这些视角不参与 NCC")
    return torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)


# ------------------------------------------------------------ 投影/反投影 ----
def unproject(depth, K, c2w):
    """像素 + z 深度 -> 世界坐标。depth (B,H,W) -> (B,H,W,3)。"""
    B, H, W = depth.shape
    fx, fy, cx, cy = K.unbind(-1)                                   # 各 (B,)
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij")
    u = u.expand(B, H, W)
    v = v.expand(B, H, W)
    s = lambda t: t.view(B, 1, 1)                                   # noqa: E731

    # OpenGL 相机系：x 右, y 上, 看向 -z
    x = (u - s(cx)) / s(fx) * depth
    y = -(v - s(cy)) / s(fy) * depth
    z = -depth
    pts_cam = torch.stack([x, y, z], dim=-1)                        # (B,H,W,3)

    R = c2w[:, :3, :3].view(B, 1, 1, 3, 3)
    t = c2w[:, :3, 3].view(B, 1, 1, 3)
    return (R @ pts_cam.unsqueeze(-1)).squeeze(-1) + t


def project(points_world, K, c2w):
    """世界坐标 -> 像素 + z 深度。points (B,H,W,3) -> uv (B,H,W,2), depth (B,H,W)。"""
    B = points_world.shape[0]
    R = c2w[:, :3, :3].view(B, 1, 1, 3, 3)
    t = c2w[:, :3, 3].view(B, 1, 1, 3)
    pts_cam = (R.transpose(-1, -2) @ (points_world - t).unsqueeze(-1)).squeeze(-1)

    fx, fy, cx, cy = K.unbind(-1)
    s = lambda x: x.view(B, 1, 1)                                   # noqa: E731

    depth = -pts_cam[..., 2]                                        # 看向 -z
    safe = depth.clamp(min=1e-6)
    u = s(fx) * pts_cam[..., 0] / safe + s(cx)
    v = -s(fy) * pts_cam[..., 1] / safe + s(cy)
    return torch.stack([u, v], dim=-1), depth


def _uv_to_grid(uv, H, W):
    """像素坐标 -> grid_sample 的归一化坐标（align_corners=True）。"""
    gx = 2.0 * uv[..., 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * uv[..., 1] / max(H - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_depth(depth_maps, uv):
    """在深度图上按像素坐标双线性采样。depth (B,H,W), uv (B,H,W,2) -> (B,H,W)。

    用最近邻会在深度不连续处引入伪一致，但双线性会跨边界插值出中间值。
    这里用双线性 + 后面的相对深度检查来兜底。
    """
    B, H, W = depth_maps.shape
    out = torch.nn.functional.grid_sample(
        depth_maps.unsqueeze(1), _uv_to_grid(uv, H, W), mode="bilinear",
        padding_mode="zeros", align_corners=True)
    return out.squeeze(1)


def sample_image(images, uv):
    """在灰度图上按像素坐标双线性采样。images (B,H,W), uv (B,H,W,2) -> (B,H,W)。"""
    B, H, W = images.shape
    out = torch.nn.functional.grid_sample(
        images.unsqueeze(1), _uv_to_grid(uv, H, W), mode="bilinear",
        padding_mode="zeros", align_corners=True)
    return out.squeeze(1)


def _box(x, radius):
    """半径 radius 的滑动窗口均值。边界处按实际有效像素数归一化。"""
    k = 2 * radius + 1
    return torch.nn.functional.avg_pool2d(
        x.unsqueeze(1), kernel_size=k, stride=1, padding=radius,
        count_include_pad=False).squeeze(1)


def compute_ncc(ref, warped, valid, radius, min_texture_std=0.0, eps=1e-6):
    """参考图与 warp 过来的源图之间的滑动窗口 NCC。

    ref/warped/valid 形状 (B,H,W)，返回 (B,H,W) 的 NCC，取值 [-1,1]。

    实现上不逐窗口循环——NCC 的每一项都是窗口内的一阶/二阶矩，
    全部用盒式滤波（avg_pool2d, stride=1）一次算出：

        NCC = (E[rw] - E[r]E[w]) / sqrt(Var[r] * Var[w])

    warp 用的是每个像素**自己的深度**，不是拟合一个平面，所以不需要法向；
    斜面只要深度图局部准确就能对上。深度不连续处窗口会混进前后景，
    NCC 自然掉下来——那恰好就是不该信的地方。
    """
    r = ref * valid
    w = warped * valid
    n = _box(valid.float(), radius).clamp(min=eps)      # 窗口内有效像素占比

    mu_r = _box(r, radius) / n
    mu_w = _box(w, radius) / n
    var_r = (_box(r * r, radius) / n - mu_r * mu_r).clamp(min=0)
    var_w = (_box(w * w, radius) / n - mu_w * mu_w).clamp(min=0)
    cov = _box(r * w, radius) / n - mu_r * mu_w

    ncc = cov / torch.sqrt(var_r * var_w + eps)
    ncc = ncc.clamp(-1, 1)

    # 参考窗口没纹理时 NCC 的分母趋零，结果被噪声主导，此时它**不该表态**：
    # 没纹理不代表深度错。所以这里返回 NaN 表示弃权，由调用方当作"通过"处理，
    # 而不是判成 -1 去否决——那会把大片平坦区域无差别砍掉。
    abstain = (n < 0.5) | (torch.sqrt(var_r) < min_texture_std) | (var_r < eps)
    return torch.where(abstain, torch.full_like(ncc, float("nan")), ncc)


# ---------------------------------------------------------------- 选源视角 ----
def select_sources(c2w, num_src):
    """给每个参考视角挑 num_src 个源视角。

    按相机中心距离取最近的若干个——太近则基线不足（三角化角小），
    太远则视野重叠少。距离排序是个简单有效的折中，实际是否可用还有
    三角化角那一关兜底。
    """
    centers = c2w[:, :3, 3]                                          # (N,3)
    dist = torch.cdist(centers, centers)                             # (N,N)
    dist.fill_diagonal_(float("inf"))                                # 排除自己
    k = min(num_src, len(c2w) - 1)
    return dist.topk(k, largest=False).indices                       # (N,k)


# ---------------------------------------------------------------- 核心过滤 ----
@torch.no_grad()
def filter_depths(c2w, K, D, src_idx, args, images=None, diag=None):
    """返回 (过滤后的深度, 每像素一致票数)。

    对每个参考视角，把它的全部源视角打包成一个 batch 一次算完——
    没有任何跨像素依赖，纯张量运算。
    """
    N, H, W = D.shape
    out_depth = torch.zeros_like(D)
    out_votes = torch.zeros((N, H, W), dtype=torch.int16, device=D.device)
    cos_min_tri = float(np.cos(np.deg2rad(args.min_triangulation_angle)))

    for i in range(N):
        js = src_idx[i]                                              # (k,)
        k = len(js)
        d_ref = D[i:i + 1]                                           # (1,H,W)
        valid_ref = torch.isfinite(d_ref) & (d_ref > 0)

        # 参考像素 -> 世界（一次，随后广播到 k 个源视角）
        pts_w = unproject(d_ref.nan_to_num(), K[i:i + 1], c2w[i:i + 1])
        pts_w_k = pts_w.expand(k, H, W, 3)

        # 正向：投进各源视角
        uv_src, d_proj = project(pts_w_k, K[js], c2w[js])
        in_bounds = ((uv_src[..., 0] >= 0) & (uv_src[..., 0] <= W - 1) &
                     (uv_src[..., 1] >= 0) & (uv_src[..., 1] <= H - 1) &
                     (d_proj > 0))

        # 源视角深度图上采样
        d_src = sample_depth(D[js], uv_src)
        has_src = torch.isfinite(d_src) & (d_src > 0)

        # 判据 2：相对深度误差
        depth_err = (d_proj - d_src).abs() / d_src.clamp(min=1e-6)

        # 反向：在 uv_src 处（非整数像素）按源视角深度反投，再投回参考视角。
        # 注意这里必须用 uv_src 构造射线，而不是源视角的整数像素网格。
        pts_back = _unproject_at(uv_src, d_src.nan_to_num(), K[js], c2w[js])
        uv_back, _ = project(pts_back, K[i:i + 1].expand(k, 4), c2w[i:i + 1].expand(k, 4, 4))

        v, u = torch.meshgrid(
            torch.arange(H, device=D.device, dtype=torch.float32),
            torch.arange(W, device=D.device, dtype=torch.float32),
            indexing="ij")
        uv_ref = torch.stack([u, v], dim=-1).expand(k, H, W, 2)
        reproj_err = (uv_back - uv_ref).norm(dim=-1)

        # 判据 3：三角化角（参考相机中心、源相机中心 对 3D 点的张角）
        c_ref = c2w[i, :3, 3].view(1, 1, 1, 3)
        c_src = c2w[js, :3, 3].view(k, 1, 1, 3)
        r1 = torch.nn.functional.normalize(c_ref - pts_w_k, dim=-1)
        r2 = torch.nn.functional.normalize(c_src - pts_w_k, dim=-1)
        cos_tri = (r1 * r2).sum(-1)

        ok_bounds = in_bounds & has_src
        ok_reproj = reproj_err < args.max_reproj_error
        ok_depth = depth_err < args.max_depth_error
        ok_tri = cos_tri < cos_min_tri                               # 角越大 cos 越小

        # 判据 4：光度一致性（NCC）。把源图按当前深度 warp 回参考视角，
        # 与参考图做滑动窗口 NCC。纯几何的三项管不住"纹理很差、深度靠猜
        # 但恰好几何自洽"的区域（大片白墙那种），这一项就是补这个。
        if images is not None:
            warped = sample_image(images[js], uv_src)                # (k,H,W)
            ncc = compute_ncc(images[i:i + 1].expand(k, H, W), warped, ok_bounds,
                              args.ncc_window, args.min_texture_std)
            # NaN = 弃权（窗口没纹理），当作通过；只有真正算得出且偏低才否决
            ok_ncc = torch.nan_to_num(ncc, nan=1.0) >= args.min_ncc
            if diag is not None:
                diag["_ncc_abstain"] += torch.isnan(ncc).sum().item()
                m = ~torch.isnan(ncc)
                if m.any():
                    diag["_ncc_sum"] += ncc[m].sum().item()
                    diag["_ncc_n"] += m.sum().item()
        else:
            ncc = None
            ok_ncc = torch.ones_like(ok_bounds)

        consistent = ok_bounds & ok_reproj & ok_depth & ok_tri & ok_ncc

        votes = consistent.sum(0).to(torch.int16)                    # (H,W)
        keep = valid_ref.squeeze(0) & (votes >= args.min_num_consistent)

        out_votes[i] = votes
        out_depth[i] = torch.where(keep, d_ref.squeeze(0), torch.zeros(()).to(D))

        # 诊断：统计每一项各自否掉了多少 (像素,源视角) 对。
        # 调参时最想知道的就是"到底是哪一关卡得太死"。
        if diag is not None:
            n = ok_bounds.numel()
            diag["总对数"] += n
            diag["出界/源无深度"] += (~ok_bounds).sum().item()
            diag["重投影误差超限"] += (ok_bounds & ~ok_reproj).sum().item()
            diag["相对深度误差超限"] += (ok_bounds & ok_reproj & ~ok_depth).sum().item()
            diag["三角化角过小"] += (ok_bounds & ok_reproj & ok_depth & ~ok_tri).sum().item()
            diag["NCC 过低"] += (ok_bounds & ok_reproj & ok_depth & ok_tri & ~ok_ncc).sum().item()
            diag["通过"] += consistent.sum().item()

    return out_depth, out_votes


def _unproject_at(uv, depth, K, c2w):
    """在给定的（非整数）像素坐标处按深度反投到世界。uv (B,H,W,2)。"""
    B = uv.shape[0]
    fx, fy, cx, cy = K.unbind(-1)
    s = lambda t: t.view(B, 1, 1)                                    # noqa: E731
    x = (uv[..., 0] - s(cx)) / s(fx) * depth
    y = -(uv[..., 1] - s(cy)) / s(fy) * depth
    z = -depth
    pts_cam = torch.stack([x, y, z], dim=-1)
    R = c2w[:, :3, :3].view(B, 1, 1, 3, 3)
    t = c2w[:, :3, 3].view(B, 1, 1, 3)
    return (R @ pts_cam.unsqueeze(-1)).squeeze(-1) + t


# ---------------------------------------------------------------- 可视化 ----
def save_visualization(vis_dir, name, depth_before, depth_after, votes, num_src):
    """三联图：过滤前深度 / 一致票数 / 过滤后深度。"""
    import cv2
    vis_dir.mkdir(parents=True, exist_ok=True)

    def colorize(d):
        v = d.copy()
        m = np.isfinite(v) & (v > 0)
        if not m.any():
            return np.zeros((*d.shape, 3), np.uint8)
        lo, hi = np.percentile(v[m], [2, 98])
        n = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        img = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~m] = 0
        return img

    vote_img = cv2.applyColorMap(
        (255 * votes.astype(np.float32) / max(num_src, 1)).clip(0, 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS)

    panel = np.hstack([colorize(depth_before), vote_img, colorize(depth_after)])
    label = np.zeros((28, panel.shape[1], 3), np.uint8)
    w = panel.shape[1] // 3
    for i, txt in enumerate(["before", f"votes /{num_src}", "after"]):
        cv2.putText(label, txt, (i * w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(vis_dir / f"{name}.png"), np.vstack([label, panel]))


# ---------------------------------------------------------------------- CLI ----
def main():
    ap = argparse.ArgumentParser(
        description="多视图几何一致性过滤（COLMAP PatchMatch filter 的 GPU 张量版）")
    ap.add_argument("--transforms-json", required=True, type=Path)
    ap.add_argument("--depth-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--num-src", type=int, default=8,
                    help="每个参考视角用几个源视角交叉验证（默认 8）")
    ap.add_argument("--max-reproj-error", type=float, default=2.0,
                    help="前后向重投影误差上限，像素（COLMAP 默认 1.0）")
    ap.add_argument("--max-depth-error", type=float, default=0.01,
                    help="相对深度误差上限（默认 1%%）")
    ap.add_argument("--min-triangulation-angle", type=float, default=3.0,
                    help="最小三角化角，度（COLMAP 默认 3）")
    ap.add_argument("--min-num-consistent", type=int, default=2,
                    help="至少几个源视角一致才保留（COLMAP 默认 2）")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="渲染图目录（通常是 render/left）。给了才做 NCC 光度检验")
    ap.add_argument("--ncc-window", type=int, default=4,
                    help="NCC 滑动窗口半径，实际窗口 (2r+1)^2（默认 4 -> 9x9）")
    ap.add_argument("--min-ncc", type=float, default=0.3,
                    help="NCC 下限，取值 [-1,1]（默认 0.3）")
    ap.add_argument("--min-texture-std", type=float, default=0.02,
                    help="参考窗口局部标准差低于此值时 NCC 弃权而非否决（默认 0.02）")
    ap.add_argument("--save-confidence", action="store_true",
                    help="额外输出逐像素置信度（票数/源视角数）到 <output>/confidence/")
    ap.add_argument("--vis-dir", type=Path, default=None,
                    help="给了就输出可视化三联图；不给就不生成")
    ap.add_argument("--no-diagnose", action="store_true",
                    help="不打印逐判据的拒绝统计")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    c2w, K, D, names = load_scene(args.transforms_json, args.depth_dir, device)
    N, H, W = D.shape
    k = min(args.num_src, N - 1)
    print(f"[filter_depth] {N} 个视角 {W}x{H}，每个配 {k} 个源视角")
    print(f"[filter_depth] 并行单元 = {H*W*k:,} (像素 x 源视角)，逐参考视角批处理")

    images = None
    if args.image_dir:
        images = load_images(args.image_dir, names, (H, W), device)
        print(f"[filter_depth] NCC 已启用: 窗口 {2*args.ncc_window+1}x{2*args.ncc_window+1}, "
              f"阈值 {args.min_ncc}")
    else:
        print("[filter_depth] 未给 --image-dir，跳过 NCC 光度检验（只做几何三项）")

    src_idx = select_sources(c2w, args.num_src)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.time()
    diag = None if args.no_diagnose else dict.fromkeys(
        ["总对数", "出界/源无深度", "重投影误差超限", "相对深度误差超限",
         "三角化角过小", "NCC 过低", "通过",
         "_ncc_abstain", "_ncc_sum", "_ncc_n"], 0)
    filtered, votes = filter_depths(c2w, K, D, src_idx, args, images, diag)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = (D > 0) & torch.isfinite(D)
    after = filtered > 0
    kept = after.sum().item()
    total = before.sum().item()

    D_cpu, F_cpu, V_cpu = D.cpu().numpy(), filtered.cpu().numpy(), votes.cpu().numpy()
    conf_dir = args.output_dir / "confidence"
    if args.save_confidence:
        conf_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        np.save(args.output_dir / f"{name}.npy", F_cpu[i])
        if args.save_confidence:
            # 票数 / 源视角数 -> [0,1] 的逐像素置信度。
            # 比二值的"信/不信"多保留了信息：下游可以按置信度加权，
            # 而不是在覆盖率和精度之间二选一。放子目录是为了不被
            # sorted(glob("*.npy")) 这类调用误读成深度图。
            np.save(conf_dir / f"{name}.npy", (V_cpu[i] / max(k, 1)).astype(np.float32))
        if args.vis_dir:
            save_visualization(args.vis_dir, name, D_cpu[i], F_cpu[i], V_cpu[i], k)

    print(f"[filter_depth] 耗时 {dt:.2f}s ({dt/N*1000:.1f} ms/视角)")
    print(f"[filter_depth] 保留 {kept:,}/{total:,} 像素 ({kept/max(total,1)*100:.1f}%)")
    print(f"[filter_depth] 平均票数 {V_cpu.mean():.2f}/{k}")
    if diag:
        tot = max(diag["总对数"], 1)
        print("[filter_depth] 逐判据拒绝统计 (按 像素x源视角 对计):")
        for k in ["出界/源无深度", "重投影误差超限", "相对深度误差超限",
                  "三角化角过小", "NCC 过低", "通过"]:
            print(f"    {k:<18} {diag[k]:>12,}  {diag[k]/tot*100:5.1f}%")
        if diag["_ncc_n"] or diag["_ncc_abstain"]:
            ab = diag["_ncc_abstain"]
            print(f"    {'(NCC 弃权/无纹理)':<18} {ab:>12,}  {ab/tot*100:5.1f}%"
                  f"   有效 NCC 均值 {diag['_ncc_sum']/max(diag['_ncc_n'],1):.3f}")
    print(f"[filter_depth] 输出 -> {args.output_dir}")
    if args.save_confidence:
        print(f"[filter_depth] 置信度 -> {args.output_dir/'confidence'}")
    if args.vis_dir:
        print(f"[filter_depth] 可视化 -> {args.vis_dir}")


if __name__ == "__main__":
    main()
