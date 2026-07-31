#!/usr/bin/env python3
"""把过滤后的深度当作可信深度喂给 DROID-SLAM，跑 RGBD SLAM 出位姿。

**思路**

前面这条链路已经产出了渲染的 RGB（render/left）和经多视图一致性过滤的深度
（depth_filtered）。把这两样交给 DROID-SLAM 的 RGBD 模式，深度作为强先验，
解出相机轨迹。

**为什么过滤后的深度特别适合当先验**

DROID-SLAM 内部是 `disps_sens = where(depth > 0, 1/depth, depth)`——
深度为 0 的像素被当作"没有深度先验"，只用光度项去解。
而我们的过滤器正是把不可信的像素置 0，语义天然吻合：
留下的都是多视图交叉验证过的，置 0 的地方 SLAM 自己去猜。不需要额外处理。

**两个要对齐的地方**

1. 内参。slam.py 要 `np.loadtxt(f)[:4]` 即 4 个数 fx fy cx cy，
   而立体匹配用的 K.txt 是 9 个数 + 基线、且是**旋转过**的。
   SLAM 用的图和深度都是原始朝向，所以这里从 transforms.json 取未旋转的内参。
2. 图像与深度靠**排序后的下标**配对（`sorted(glob)`），不是靠文件名匹配。
   render/left/00000.png 和 depth_filtered/00000.npy 排序一致，没问题。

**评估**

transforms.json 里的位姿是已知的（来自 COLMAP / nerfstudio），
所以可以直接拿来量 SLAM 恢复得准不准——这等于反过来量深度的质量。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_intrinsic(transforms_json: Path, out: Path):
    """写 slam.py 要的 4 元素内参文件（原始朝向，未旋转）。"""
    d = json.loads(transforms_json.read_text(encoding="utf-8"))
    missing = [k for k in ("fl_x", "fl_y", "cx", "cy") if d.get(k) is None]
    if missing:
        raise SystemExit(f"[run_rgbd_slam] {transforms_json} 缺内参: {missing}")
    vals = [float(d["fl_x"]), float(d["fl_y"]), float(d["cx"]), float(d["cy"])]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(str(v) for v in vals) + "\n")
    return vals


def umeyama(X, Y, with_scale=True):
    """求 s,R,t 使 s·R·X + t ≈ Y。X,Y 形状 (3,N)。"""
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc @ Xc.T / X.shape[1])
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = (np.trace(np.diag(D) @ W) / (Xc ** 2).sum(0).mean()) if with_scale else 1.0
    return s, R, my - s * R @ mx


def evaluate(poses_dir: Path, transforms_json: Path):
    """把 SLAM 出的位姿和 transforms.json 的已知位姿对齐后算 ATE。"""
    files = sorted(poses_dir.glob("*.txt"))
    if not files:
        return None
    est = np.stack([np.loadtxt(f) for f in files])

    d = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = sorted(d["frames"], key=lambda f: str(f["file_path"]))
    gt = np.stack([np.asarray(f["transform_matrix"], dtype=float) for f in frames])

    n = min(len(est), len(gt))
    if n < 3:
        return None
    est, gt = est[:n], gt[:n]
    gt_t = gt[:, :3, 3].T

    # SLAM 位姿的约定未知，cam->world 和 world->cam 都试，取拟合更好的
    best = None
    for name, M in (("cam->world", est), ("world->cam", np.linalg.inv(est))):
        t = M[:, :3, 3].T
        s, R, tr = umeyama(t, gt_t, with_scale=False)
        rmse = np.sqrt((np.linalg.norm(s * R @ t + tr - gt_t, axis=0) ** 2).mean())
        if best is None or rmse < best[0]:
            best = (rmse, name, t)
    _, conv, est_t = best

    s, R, tr = umeyama(est_t, gt_t, with_scale=True)
    err = np.linalg.norm(s * R @ est_t + tr - gt_t, axis=0)
    length = np.linalg.norm(np.diff(gt_t, axis=1), axis=0).sum()
    return {
        "n": n, "convention": conv, "scale": float(s),
        "ate_rmse": float(np.sqrt((err ** 2).mean())), "ate_max": float(err.max()),
        "traj_length": float(length),
    }


def main():
    ap = argparse.ArgumentParser(description="用过滤后的深度跑 DROID RGBD SLAM")
    ap.add_argument("--droid-metric-dir", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path, help="渲染的 RGB（原始朝向）")
    ap.add_argument("--depth-dir", required=True, type=Path, help="深度 .npy（原始朝向）")
    ap.add_argument("--transforms-json", required=True, type=Path, help="提供内参和真值位姿")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="droid.pth，默认取 <droid-metric-dir>/weights/droid.pth")
    ap.add_argument("--global-ba-frontend", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="不和 transforms.json 对比")
    args = ap.parse_args()

    slam_py = args.droid_metric_dir / "slam.py"
    if not slam_py.is_file():
        raise SystemExit(f"[run_rgbd_slam] 找不到 {slam_py}")
    ckpt = args.checkpoint or (args.droid_metric_dir / "weights/droid.pth")
    if not ckpt.is_file():
        raise SystemExit(
            f"[run_rgbd_slam] 找不到权重 {ckpt}\n"
            f"  在 {args.droid_metric_dir} 下跑 python download_models.py")

    n_img = len(sorted(args.image_dir.glob("*.[pj][np]g")))
    n_dep = len(sorted(args.depth_dir.glob("*.npy")))
    if n_img == 0 or n_dep == 0:
        raise SystemExit(f"[run_rgbd_slam] 图像 {n_img} 张、深度 {n_dep} 张，有一边是空的")
    if n_img != n_dep:
        # 靠排序下标配对，数量不等一定错位
        raise SystemExit(
            f"[run_rgbd_slam] 图像 {n_img} 张与深度 {n_dep} 张数量不等。"
            f"两者靠排序后的下标配对，数量必须一致")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intr_file = args.output_dir / "intrinsic_slam.txt"
    fx, fy, cx, cy = write_intrinsic(args.transforms_json, intr_file)

    # 统计深度有效率——过滤后 0 的比例就是 SLAM 拿不到先验的比例
    sample = np.load(sorted(args.depth_dir.glob("*.npy"))[0])
    valid = float(np.mean(np.isfinite(sample) & (sample > 0)))

    print(f"[run_rgbd_slam] {n_img} 帧, 内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"[run_rgbd_slam] 深度有效像素约 {valid:.1%}"
          f"（置 0 的部分 DROID 当作无先验，只用光度项）")

    traj = args.output_dir / "trajectory.txt"
    poses = args.output_dir / "poses"
    cmd = [sys.executable, str(slam_py),
           "--images", str(args.image_dir),
           "--depth", str(args.depth_dir),
           "--intr", str(intr_file),
           "--out-traj", str(traj),
           "--out-poses", str(poses),
           "--checkpoint", str(ckpt),
           "--global-ba-frontend", str(args.global_ba_frontend)]
    print(f"[run_rgbd_slam] 启动 DROID-SLAM ...")
    # slam.py 用相对导入找 modules/，必须在 droid_metric 根目录跑
    r = subprocess.run(cmd, cwd=str(args.droid_metric_dir))
    if r.returncode != 0:
        raise SystemExit(f"[run_rgbd_slam] DROID-SLAM 失败 (exit {r.returncode})")

    n_out = len(sorted(poses.glob("*.txt"))) if poses.is_dir() else 0
    print(f"[run_rgbd_slam] 输出 {n_out} 帧位姿 -> {poses}")
    print(f"[run_rgbd_slam] 轨迹 (TUM 格式) -> {traj}")

    if not args.no_eval:
        st = evaluate(poses, args.transforms_json)
        if st is None:
            print("[run_rgbd_slam] 位姿太少或缺失，跳过评估")
        else:
            print(f"[run_rgbd_slam] 与 transforms.json 已知位姿对比 "
                  f"（{st['n']} 帧, 判定为 {st['convention']}）:")
            print(f"    真值轨迹总长  {st['traj_length']:.4f}")
            print(f"    ATE RMSE      {st['ate_rmse']:.4f}  "
                  f"({st['ate_rmse']/max(st['traj_length'],1e-9):.2%} 轨迹长度)")
            print(f"    ATE max       {st['ate_max']:.4f}")
            print(f"    恢复尺度      {st['scale']:.4f}   "
                  f"（越接近 1 说明深度的绝对尺度越准）")


if __name__ == "__main__":
    main()
