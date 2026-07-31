#!/usr/bin/env python3
"""Run DROID RGB-D SLAM using filtered depth as a trusted prior.

The pipeline provides rendered RGB and multi-view-filtered depth. DROID treats
zero depth as no prior and uses only photometric terms there, exactly matching
the filter's convention of zeroing unreliable pixels.

SLAM needs unrotated fx/fy/cx/cy from transforms.json rather than the rotated
stereo K.txt. Images and depth are paired by sorted index. Known transforms.json
poses provide direct ATE evaluation of the recovered trajectory.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_intrinsic(transforms_json: Path, out: Path):
    """Write the four-element unrotated intrinsic file required by slam.py."""
    d = json.loads(transforms_json.read_text(encoding="utf-8"))
    missing = [k for k in ("fl_x", "fl_y", "cx", "cy") if d.get(k) is None]
    if missing:
        raise SystemExit(f"[run_rgbd_slam] {transforms_json} lacks intrinsics: {missing}")
    vals = [float(d["fl_x"]), float(d["fl_y"]), float(d["cx"]), float(d["cy"])]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(str(v) for v in vals) + "\n")
    return vals


def umeyama(X, Y, with_scale=True):
    """Find s,R,t such that s*R*X + t ~= Y for X,Y shaped (3,N)."""
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
    """Align SLAM poses with known transforms.json poses and compute ATE."""
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

    # Try both camera-to-world and world-to-camera conventions; retain the better fit.
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
    ap = argparse.ArgumentParser(description="Run DROID RGB-D SLAM with filtered depth")
    ap.add_argument("--droid-metric-dir", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path, help="Rendered RGB in original orientation")
    ap.add_argument("--depth-dir", required=True, type=Path, help="Depth .npy in original orientation")
    ap.add_argument("--transforms-json", required=True, type=Path, help="Intrinsics and reference poses")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="droid.pth; defaults to <droid-metric-dir>/weights/droid.pth")
    ap.add_argument("--global-ba-frontend", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="Do not compare against transforms.json")
    args = ap.parse_args()

    slam_py = args.droid_metric_dir / "slam.py"
    if not slam_py.is_file():
        raise SystemExit(f"[run_rgbd_slam] {slam_py} not found")
    ckpt = args.checkpoint or (args.droid_metric_dir / "weights/droid.pth")
    if not ckpt.is_file():
        raise SystemExit(
            f"[run_rgbd_slam] Weights not found: {ckpt}\n"
            f"  Run python download_models.py under {args.droid_metric_dir}")

    n_img = len(sorted(args.image_dir.glob("*.[pj][np]g")))
    n_dep = len(sorted(args.depth_dir.glob("*.npy")))
    if n_img == 0 or n_dep == 0:
        raise SystemExit(f"[run_rgbd_slam] Images: {n_img}, depth maps: {n_dep}; one side is empty")
    if n_img != n_dep:
        # Sorted-index pairing requires equal counts.
        raise SystemExit(
            f"[run_rgbd_slam] Image count {n_img} differs from depth count {n_dep}; "
            f"sorted-index pairing requires equal counts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intr_file = args.output_dir / "intrinsic_slam.txt"
    fx, fy, cx, cy = write_intrinsic(args.transforms_json, intr_file)

    # Zero-valued filtered depth is the fraction without a SLAM depth prior.
    sample = np.load(sorted(args.depth_dir.glob("*.npy"))[0])
    valid = float(np.mean(np.isfinite(sample) & (sample > 0)))

    print(f"[run_rgbd_slam] {n_img} frames, intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"[run_rgbd_slam] Approximately {valid:.1%} valid depth pixels; zeroes have no prior")

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
    print("[run_rgbd_slam] Starting DROID-SLAM ...")
    # slam.py uses relative imports; run from the droid_metric root.
    r = subprocess.run(cmd, cwd=str(args.droid_metric_dir))
    if r.returncode != 0:
        raise SystemExit(f"[run_rgbd_slam] DROID-SLAM failed (exit {r.returncode})")

    n_out = len(sorted(poses.glob("*.txt"))) if poses.is_dir() else 0
    print(f"[run_rgbd_slam] Wrote {n_out} frame poses -> {poses}")
    print(f"[run_rgbd_slam] Trajectory (TUM format) -> {traj}")

    if not args.no_eval:
        st = evaluate(poses, args.transforms_json)
        if st is None:
            print("[run_rgbd_slam] Too few or missing poses; skipping evaluation")
        else:
            print(f"[run_rgbd_slam] Comparison with known transforms.json poses "
                  f"({st['n']} frames, interpreted as {st['convention']}):")
            print(f"    Reference trajectory length {st['traj_length']:.4f}")
            print(f"    ATE RMSE      {st['ate_rmse']:.4f}  "
                  f"({st['ate_rmse']/max(st['traj_length'],1e-9):.2%} of trajectory length)")
            print(f"    ATE max       {st['ate_max']:.4f}")
            print(f"    Recovered scale {st['scale']:.4f} "
                  f"(closer to 1 means more accurate absolute depth scale)")


if __name__ == "__main__":
    main()
