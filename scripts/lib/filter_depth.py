#!/usr/bin/env python3
"""Multi-view geometric consistency filtering based on COLMAP PatchMatch Stereo.

FoundationStereo produces dense depth for every pixel regardless of reliability.
This module cross-validates depth using camera poses. A correct depth should
return to the original pixel after projection into another view and back.

Four checks are applied for every reference pixel and source view:

1. **Round-trip reprojection error**: unproject to 3D, project into the source,
   sample source depth, unproject again, and project back to the reference.
2. **Relative depth error** between projected and sampled source depth.
3. **Triangulation angle**: views with insufficient angular separation abstain.
4. **NCC**: warp the source image using current depth and compute windowed NCC.
   This catches low-texture regions whose guessed depth happens to be
   geometrically self-consistent. It is enabled with --image-dir.

Source views passing all checks cast votes; pixels below the vote threshold are
untrusted. Unlike COLMAP, which obtains NCC confidence during PatchMatch, this
implementation computes NCC independently and adds relative-depth checking.
Untrusted pixels are set to zero, and a vote map is also produced as soft confidence.

Filtering has no propagation dependency between pixels or views, so an entire
tensor can be evaluated in parallel. A 480x640 image with eight source views
provides about 2.46 million parallel units.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- Data loading ----
def load_scene(transforms_json: Path, depth_dir: Path, device):
    """Load transforms.json and depth maps; return poses, intrinsics, and depth.

    transforms.json uses the OpenGL/Blender convention (x-right, y-up, z-back,
    camera facing -z). Depth maps store z-depth along the optical axis.
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
        # Depth maps may be sequentially named or share the file_path stem.
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
            f"No depth maps in {depth_dir} match {transforms_json}")
    if missing:
        print(f"[filter_depth] Warning: skipped {len(missing)} frames without depth maps")

    shapes = {d.shape for d in depths}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent depth-map dimensions: {shapes}")

    c2w = torch.tensor(np.stack(poses), dtype=torch.float32, device=device)   # (N,4,4)
    if c2w.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1.0], device=device).expand(len(c2w), 1, 4)
        c2w = torch.cat([c2w, bottom], dim=1)
    K = torch.tensor(intrinsics, dtype=torch.float32, device=device)          # (N,4) fx,fy,cx,cy
    D = torch.tensor(np.stack(depths), dtype=torch.float32, device=device)    # (N,H,W)
    return c2w, K, D, names


def load_images(image_dir: Path, names, shape, device):
    """Load rendered images by depth-map name and convert them to grayscale.

    NCC requires rendered images whose names match the depth maps. Both are
    produced by the same ns-render invocation and therefore share names.
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
            raise ValueError(f"{hit} dimensions {g.shape} do not match depth map {(H, W)}")
        imgs.append(g.astype(np.float32) / 255.0)

    if len(missing) == len(names):
        raise FileNotFoundError(
            f"No image in {image_dir} matches a depth-map name such as {names[0]}.png")
    if missing:
        print(f"[filter_depth] Warning: {len(missing)} rendered images are missing; excluded from NCC")
    return torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)


# ------------------------------------------------------ Projection/unprojection ----
def unproject(depth, K, c2w):
    """Convert pixels plus z-depth to world coordinates."""
    B, H, W = depth.shape
    fx, fy, cx, cy = K.unbind(-1)                                   # Each is (B,)
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij")
    u = u.expand(B, H, W)
    v = v.expand(B, H, W)
    s = lambda t: t.view(B, 1, 1)                                   # noqa: E731

    # OpenGL camera coordinates: x-right, y-up, facing -z.
    x = (u - s(cx)) / s(fx) * depth
    y = -(v - s(cy)) / s(fy) * depth
    z = -depth
    pts_cam = torch.stack([x, y, z], dim=-1)                        # (B,H,W,3)

    R = c2w[:, :3, :3].view(B, 1, 1, 3, 3)
    t = c2w[:, :3, 3].view(B, 1, 1, 3)
    return (R @ pts_cam.unsqueeze(-1)).squeeze(-1) + t


def project(points_world, K, c2w):
    """Convert world coordinates to pixels plus z-depth."""
    B = points_world.shape[0]
    R = c2w[:, :3, :3].view(B, 1, 1, 3, 3)
    t = c2w[:, :3, 3].view(B, 1, 1, 3)
    pts_cam = (R.transpose(-1, -2) @ (points_world - t).unsqueeze(-1)).squeeze(-1)

    fx, fy, cx, cy = K.unbind(-1)
    s = lambda x: x.view(B, 1, 1)                                   # noqa: E731

    depth = -pts_cam[..., 2]                                        # Camera faces -z.
    safe = depth.clamp(min=1e-6)
    u = s(fx) * pts_cam[..., 0] / safe + s(cx)
    v = -s(fy) * pts_cam[..., 1] / safe + s(cy)
    return torch.stack([u, v], dim=-1), depth


def _uv_to_grid(uv, H, W):
    """Convert pixel coordinates to normalized grid_sample coordinates."""
    gx = 2.0 * uv[..., 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * uv[..., 1] / max(H - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_depth(depth_maps, uv):
    """Bilinearly sample depth maps at pixel coordinates.

    Nearest-neighbor sampling creates false consistency at depth discontinuities,
    while bilinear sampling interpolates across boundaries. The subsequent
    relative-depth check guards against the latter.
    """
    B, H, W = depth_maps.shape
    out = torch.nn.functional.grid_sample(
        depth_maps.unsqueeze(1), _uv_to_grid(uv, H, W), mode="bilinear",
        padding_mode="zeros", align_corners=True)
    return out.squeeze(1)


def sample_image(images, uv):
    """Bilinearly sample grayscale images at pixel coordinates."""
    B, H, W = images.shape
    out = torch.nn.functional.grid_sample(
        images.unsqueeze(1), _uv_to_grid(uv, H, W), mode="bilinear",
        padding_mode="zeros", align_corners=True)
    return out.squeeze(1)


def _box(x, radius):
    """Sliding-window mean with the given radius, normalized at boundaries."""
    k = 2 * radius + 1
    return torch.nn.functional.avg_pool2d(
        x.unsqueeze(1), kernel_size=k, stride=1, padding=radius,
        count_include_pad=False).squeeze(1)


def compute_ncc(ref, warped, valid, radius, min_texture_std=0.0, eps=1e-6):
    """Windowed NCC between a reference image and a warped source image.

    Inputs and output have shape (B,H,W); NCC values are in [-1,1].

    First- and second-order window moments are computed with box filters:

        NCC = (E[rw] - E[r]E[w]) / sqrt(Var[r] * Var[w])

    The warp uses each pixel's own depth rather than a fitted plane, so no normals
    are required. Windows spanning depth discontinuities mix foreground and
    background, naturally lowering NCC in precisely the unreliable regions.
    """
    r = ref * valid
    w = warped * valid
    n = _box(valid.float(), radius).clamp(min=eps)      # Valid-pixel ratio in the window

    mu_r = _box(r, radius) / n
    mu_w = _box(w, radius) / n
    var_r = (_box(r * r, radius) / n - mu_r * mu_r).clamp(min=0)
    var_w = (_box(w * w, radius) / n - mu_w * mu_w).clamp(min=0)
    cov = _box(r * w, radius) / n - mu_r * mu_w

    ncc = cov / torch.sqrt(var_r * var_w + eps)
    ncc = ncc.clamp(-1, 1)

    # In textureless windows the denominator approaches zero and noise dominates.
    # Texturelessness does not imply bad depth, so NaN means abstain and callers
    # treat it as passing rather than rejecting large uniform areas.
    abstain = (n < 0.5) | (torch.sqrt(var_r) < min_texture_std) | (var_r < eps)
    return torch.where(abstain, torch.full_like(ncc, float("nan")), ncc)


# ---------------------------------------------------------- Source-view selection ----
def select_sources(c2w, num_src):
    """Select num_src source views for every reference view.

    Choose nearest camera centers. Views that are too close have inadequate
    baseline, while distant views have little overlap. Distance is a practical
    compromise, with triangulation-angle checking as a final guard.
    """
    centers = c2w[:, :3, 3]                                          # (N,3)
    dist = torch.cdist(centers, centers)                             # (N,N)
    dist.fill_diagonal_(float("inf"))                                # Exclude self.
    k = min(num_src, len(c2w) - 1)
    return dist.topk(k, largest=False).indices                       # (N,k)


# ---------------------------------------------------------------- Core filter ----
@torch.no_grad()
def filter_depths(c2w, K, D, src_idx, args, images=None, diag=None):
    """Return filtered depth and per-pixel consistency votes.

    All source views for a reference view are evaluated in one batch using only
    tensor operations and no cross-pixel dependencies.
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

        # Reference pixels -> world coordinates, then broadcast to k source views.
        pts_w = unproject(d_ref.nan_to_num(), K[i:i + 1], c2w[i:i + 1])
        pts_w_k = pts_w.expand(k, H, W, 3)

        # Forward projection into each source view.
        uv_src, d_proj = project(pts_w_k, K[js], c2w[js])
        in_bounds = ((uv_src[..., 0] >= 0) & (uv_src[..., 0] <= W - 1) &
                     (uv_src[..., 1] >= 0) & (uv_src[..., 1] <= H - 1) &
                     (d_proj > 0))

        # Sample the source-view depth maps.
        d_src = sample_depth(D[js], uv_src)
        has_src = torch.isfinite(d_src) & (d_src > 0)

        # Criterion 2: relative depth error.
        depth_err = (d_proj - d_src).abs() / d_src.clamp(min=1e-6)

        # Reverse: unproject source depth at noninteger uv_src, then project back.
        # Rays must use uv_src rather than the source view's integer pixel grid.
        pts_back = _unproject_at(uv_src, d_src.nan_to_num(), K[js], c2w[js])
        uv_back, _ = project(pts_back, K[i:i + 1].expand(k, 4), c2w[i:i + 1].expand(k, 4, 4))

        v, u = torch.meshgrid(
            torch.arange(H, device=D.device, dtype=torch.float32),
            torch.arange(W, device=D.device, dtype=torch.float32),
            indexing="ij")
        uv_ref = torch.stack([u, v], dim=-1).expand(k, H, W, 2)
        reproj_err = (uv_back - uv_ref).norm(dim=-1)

        # Criterion 3: angle subtended by reference/source centers at the 3D point.
        c_ref = c2w[i, :3, 3].view(1, 1, 1, 3)
        c_src = c2w[js, :3, 3].view(k, 1, 1, 3)
        r1 = torch.nn.functional.normalize(c_ref - pts_w_k, dim=-1)
        r2 = torch.nn.functional.normalize(c_src - pts_w_k, dim=-1)
        cos_tri = (r1 * r2).sum(-1)

        ok_bounds = in_bounds & has_src
        ok_reproj = reproj_err < args.max_reproj_error
        ok_depth = depth_err < args.max_depth_error
        ok_tri = cos_tri < cos_min_tri                               # Larger angles have smaller cosine.

        # Criterion 4: photometric consistency (NCC). Warp source images into
        # the reference view and compare windows. This catches low-texture areas
        # whose guessed depths happen to be geometrically self-consistent.
        if images is not None:
            warped = sample_image(images[js], uv_src)                # (k,H,W)
            ncc = compute_ncc(images[i:i + 1].expand(k, H, W), warped, ok_bounds,
                              args.ncc_window, args.min_texture_std)
            # NaN means abstain for textureless windows; only a valid low NCC rejects.
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

        # Count how many pixel/source-view pairs each criterion rejects.
        if diag is not None:
            n = ok_bounds.numel()
            diag["total_pairs"] += n
            diag["out_of_bounds_or_no_source_depth"] += (~ok_bounds).sum().item()
            diag["reprojection_error"] += (ok_bounds & ~ok_reproj).sum().item()
            diag["relative_depth_error"] += (ok_bounds & ok_reproj & ~ok_depth).sum().item()
            diag["small_triangulation_angle"] += (ok_bounds & ok_reproj & ok_depth & ~ok_tri).sum().item()
            diag["low_ncc"] += (ok_bounds & ok_reproj & ok_depth & ok_tri & ~ok_ncc).sum().item()
            diag["passed"] += consistent.sum().item()

    return out_depth, out_votes


def _unproject_at(uv, depth, K, c2w):
    """Unproject depth at given noninteger pixel coordinates into world space."""
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


# --------------------------------------------------------------- Visualization ----
def save_visualization(vis_dir, name, depth_before, depth_after, votes, num_src):
    """Three-panel view: original depth / consistency votes / filtered depth."""
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
        description="GPU tensor implementation of COLMAP-style multi-view consistency filtering")
    ap.add_argument("--transforms-json", required=True, type=Path)
    ap.add_argument("--depth-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--num-src", type=int, default=8,
                    help="Number of source views per reference view (default: 8)")
    ap.add_argument("--max-reproj-error", type=float, default=2.0,
                    help="Maximum round-trip reprojection error in pixels (COLMAP default: 1.0)")
    ap.add_argument("--max-depth-error", type=float, default=0.01,
                    help="Maximum relative depth error (default: 1%%)")
    ap.add_argument("--min-triangulation-angle", type=float, default=3.0,
                    help="Minimum triangulation angle in degrees (COLMAP default: 3)")
    ap.add_argument("--min-num-consistent", type=int, default=2,
                    help="Minimum consistent source views required (COLMAP default: 2)")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Rendered-image directory, usually render/left; enables NCC")
    ap.add_argument("--ncc-window", type=int, default=4,
                    help="NCC window radius; actual size is (2r+1)^2 (default: 4 -> 9x9)")
    ap.add_argument("--min-ncc", type=float, default=0.3,
                    help="Minimum NCC in [-1,1] (default: 0.3)")
    ap.add_argument("--min-texture-std", type=float, default=0.02,
                    help="NCC abstains below this reference-window standard deviation (default: 0.02)")
    ap.add_argument("--save-confidence", action="store_true",
                    help="Save per-pixel confidence (votes/source views) under <output>/confidence/")
    ap.add_argument("--vis-dir", type=Path, default=None,
                    help="Optional directory for three-panel visualizations")
    ap.add_argument("--no-diagnose", action="store_true",
                    help="Do not print rejection statistics by criterion")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    c2w, K, D, names = load_scene(args.transforms_json, args.depth_dir, device)
    N, H, W = D.shape
    k = min(args.num_src, N - 1)
    print(f"[filter_depth] {N} views at {W}x{H}, with {k} source views each")
    print(f"[filter_depth] Parallel units = {H*W*k:,} (pixels x source views), batched by reference")

    images = None
    if args.image_dir:
        images = load_images(args.image_dir, names, (H, W), device)
        print(f"[filter_depth] NCC enabled: window {2*args.ncc_window+1}x{2*args.ncc_window+1}, "
              f"threshold {args.min_ncc}")
    else:
        print("[filter_depth] No --image-dir; skipping NCC and using geometry only")

    src_idx = select_sources(c2w, args.num_src)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.time()
    diag = None if args.no_diagnose else dict.fromkeys(
        ["total_pairs", "out_of_bounds_or_no_source_depth", "reprojection_error",
         "relative_depth_error", "small_triangulation_angle", "low_ncc", "passed",
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
            # Votes/source-view count gives [0,1] confidence. This preserves more
            # information than a binary mask for downstream weighting. A subdirectory
            # prevents sorted(glob("*.npy")) calls from treating it as depth.
            np.save(conf_dir / f"{name}.npy", (V_cpu[i] / max(k, 1)).astype(np.float32))
        if args.vis_dir:
            save_visualization(args.vis_dir, name, D_cpu[i], F_cpu[i], V_cpu[i], k)

    print(f"[filter_depth] Elapsed {dt:.2f}s ({dt/N*1000:.1f} ms/view)")
    print(f"[filter_depth] Kept {kept:,}/{total:,} pixels ({kept/max(total,1)*100:.1f}%)")
    print(f"[filter_depth] Mean votes {V_cpu.mean():.2f}/{k}")
    if diag:
        tot = max(diag["total_pairs"], 1)
        print("[filter_depth] Rejections by criterion (pixel x source-view pairs):")
        for k in ["out_of_bounds_or_no_source_depth", "reprojection_error",
                  "relative_depth_error", "small_triangulation_angle", "low_ncc", "passed"]:
            print(f"    {k:<18} {diag[k]:>12,}  {diag[k]/tot*100:5.1f}%")
        if diag["_ncc_n"] or diag["_ncc_abstain"]:
            ab = diag["_ncc_abstain"]
            print(f"    {'(NCC abstained/no texture)':<18} {ab:>12,}  {ab/tot*100:5.1f}%"
                  f"   mean valid NCC {diag['_ncc_sum']/max(diag['_ncc_n'],1):.3f}")
    print(f"[filter_depth] Output -> {args.output_dir}")
    if args.save_confidence:
        print(f"[filter_depth] Confidence -> {args.output_dir/'confidence'}")
    if args.vis_dir:
        print(f"[filter_depth] Visualizations -> {args.vis_dir}")


if __name__ == "__main__":
    main()
