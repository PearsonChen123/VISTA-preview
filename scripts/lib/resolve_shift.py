#!/usr/bin/env python3
"""Convert target disparity as a fraction of image width into baseline length.

Disparity and depth are inversely related:

    disparity_px = fx * baseline / Z

For one baseline, near objects have greater disparity. A target such as 10% of
image width therefore needs an anchor depth, defined here as scene reference depth.

Reference depth is estimated quickly from the COLMAP sparse cloud, or supplied
directly as stereo.reference_depth.

Pose translation occurs in normalized nerfstudio space while depth has real scale:

    baseline_metric = frac * W * Z_ref / fx
    shift_norm      = baseline_metric * scale

make_intrinsics.py later converts back with baseline = shift_norm / scale.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_model import find_sparse_dir, read_images_binary, read_points3D_binary


def scene_depth_from_colmap(colmap_dir: Path, percentile: float):
    """Estimate scene depth from the sparse point cloud.

    Project each registered image's visible 3D points into camera coordinates and
    use z-depth, measuring the distance to content the camera actually observed.
    """
    sparse = find_sparse_dir(colmap_dir)
    if not (sparse / "points3D.bin").is_file():
        raise FileNotFoundError(f"No points3D.bin under {sparse}")

    images = read_images_binary(sparse / "images.bin")
    points = read_points3D_binary(sparse / "points3D.bin")
    if not points:
        raise ValueError(f"{sparse}/points3D.bin is empty")

    xyz = {pid: p.xyz for pid, p in points.items()}
    depths = []
    for im in images.values():
        ids = im.point3D_ids
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue
        pts = np.array([xyz[i] for i in ids if i in xyz])
        if pts.size == 0:
            continue
        w2c = im.world_to_camera()
        z = (pts @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]      # OpenCV z is depth.
        depths.append(z[z > 0])

    if not depths:
        raise ValueError("No image observes a valid 3D point")
    allz = np.concatenate(depths)
    return float(np.percentile(allz, percentile)), allz


def main():
    ap = argparse.ArgumentParser(
        description="Target disparity fraction -> baseline in normalized coordinates")
    ap.add_argument("--transforms-json", required=True, type=Path)
    ap.add_argument("--dataparser-transforms", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["pixels", "baseline"])
    ap.add_argument("--shift", type=float, default=0.2,
                    help="Direct normalized value for mode=baseline")
    ap.add_argument("--shift-pixels", type=float, default=0.1,
                    help="Target disparity fraction for mode=pixels (0.1 = 10%%)")
    ap.add_argument("--colmap-dir", type=Path, default=None,
                    help="COLMAP directory used to estimate scene depth")
    ap.add_argument("--reference-depth", type=float, default=None,
                    help="Explicit reference depth in meters; skips point-cloud estimation")
    ap.add_argument("--percentile", type=float, default=25.0,
                    help="Reference percentile of point-cloud depth (default: 25); "
                         "a near anchor keeps foreground disparity manageable")
    ap.add_argument("--quiet", action="store_true", help="Print only the final value")
    args = ap.parse_args()

    log = (lambda *a: None) if args.quiet else (
        lambda *a: print("[resolve_shift]", *a, file=sys.stderr))

    if args.mode == "baseline":
        log(f"mode=baseline, using shift={args.shift} directly")
        print(args.shift)
        return

    tf = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    fx, W = float(tf["fl_x"]), int(tf["w"])
    scale = float(json.loads(
        args.dataparser_transforms.read_text(encoding="utf-8"))["scale"])

    if args.reference_depth is not None:
        Z = args.reference_depth
        log(f"Using configured reference depth {Z:.4f} m")
    else:
        if args.colmap_dir is None:
            raise SystemExit(
                "[resolve_shift] mode=pixels requires a reference depth:\n"
                "  run the COLMAP stage for automatic sparse-cloud estimation,\n"
                "  or set stereo.reference_depth in config.json")
        try:
            Z, allz = scene_depth_from_colmap(args.colmap_dir, args.percentile)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(
                f"[resolve_shift] Cannot estimate scene depth from {args.colmap_dir}: {e}\n"
                f"  Set stereo.reference_depth in config.json")
        log(f"Sparse cloud has {len(allz):,} observations; depth distribution "
            f"5%={np.percentile(allz,5):.3f}  50%={np.percentile(allz,50):.3f}  "
            f"95%={np.percentile(allz,95):.3f} m")
        log(f"Reference depth uses percentile {args.percentile:g}% = {Z:.4f} m")

    target_px = args.shift_pixels * W
    baseline = target_px * Z / fx
    shift_norm = baseline * scale

    log(f"Target disparity {args.shift_pixels:.1%} x {W}px = {target_px:.1f} px @ {Z:.3f} m")
    log(f"Baseline = {target_px:.1f} * {Z:.3f} / {fx:.1f} = {baseline:.6f} m")
    log(f"Normalized shift = {baseline:.6f} * {scale:.6f} = {shift_norm:.6f}")
    if args.reference_depth is None:
        log("Actual disparity by depth for this baseline:")
        for p in (5, 25, 50, 95):
            zp = float(np.percentile(allz, p))
            d = fx * baseline / zp
            mark = "  <- anchor" if abs(p - args.percentile) < 1e-6 else ""
            log(f"    {p:>2}% percentile Z={zp:7.3f} m -> {d:6.1f} px ({d/W:5.1%} width){mark}")
    else:
        log("(Nearer depths have greater disparity, as required by stereo geometry)")

    print(f"{shift_norm:.9f}")


if __name__ == "__main__":
    main()
