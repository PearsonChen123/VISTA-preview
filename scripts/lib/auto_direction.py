#!/usr/bin/env python3
"""Select the stereo translation direction from the camera trajectory.

NeRF is best trained near observed viewpoints. Translating the virtual camera
along the trajectory stays on that observed-view manifold, while perpendicular
translation extrapolates and produces blur and artifacts.

For each frame, the trajectory tangent is projected into camera coordinates.
The dominant x/y component and majority sign determine one global direction.
FoundationStereo shares one K.txt across a batch, and different directions
require different image and intrinsic rotations, so direction cannot vary by
frame. Sequences with rapidly changing direction should be processed in segments.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Camera motion in local coordinates -> project direction name.
#
# Names describe right-image content motion relative to the left image. Camera
# motion and image-content motion are opposite; see the verified map in common.py.
CAMERA_MOTION_TO_DIRECTION = {
    ("y", +1): "down",     # Camera moves up.
    ("y", -1): "up",       # Camera moves down.
    ("x", +1): "left",     # Camera moves right.
    ("x", -1): "right",    # Camera moves left.
}


def load_poses(path: Path):
    """Load c2w poses from transforms.json or an exported frame list, sorted by name."""
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["frames"] if isinstance(data, dict) else data
    key = lambda f: str(f.get("file_path", ""))            # noqa: E731
    mats = []
    for f in sorted(frames, key=key):
        m = np.asarray(f.get("transform_matrix", f.get("transform")), dtype=float)
        if m.shape == (3, 4):
            m = np.vstack([m, [0, 0, 0, 1.0]])
        mats.append(m)
    return np.stack(mats)


def analyze(c2w: np.ndarray):
    """Return the direction and statistics."""
    if len(c2w) < 2:
        raise SystemExit("[auto_direction] At least two frames are required")

    centers = c2w[:, :3, 3]
    tangent = np.gradient(centers, axis=0)                  # Centered difference
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    if (norms < 1e-9).any():
        raise SystemExit("[auto_direction] Coincident camera positions prevent tangent estimation")
    tangent /= norms

    # World -> each frame's camera coordinates.
    local = np.einsum("nij,nj->ni", c2w[:, :3, :3].transpose(0, 2, 1), tangent)
    tx, ty = local[:, 0], local[:, 1]

    sum_x, sum_y = np.abs(tx).sum(), np.abs(ty).sum()
    axis = "x" if sum_x >= sum_y else "y"
    dominance = max(sum_x, sum_y) / (sum_x + sum_y)

    comp = tx if axis == "x" else ty
    sign = 1 if (comp > 0).sum() >= (comp <= 0).sum() else -1
    consistency = max((comp > 0).mean(), (comp <= 0).mean())

    direction = CAMERA_MOTION_TO_DIRECTION[(axis, sign)]
    return direction, {
        "axis": axis,
        "sign": sign,
        "dominance": float(dominance),
        "sign_consistency": float(consistency),
        "mean_abs_x": float(np.abs(tx).mean()),
        "mean_abs_y": float(np.abs(ty).mean()),
        "n_frames": len(c2w),
    }


def main():
    ap = argparse.ArgumentParser(description="Select stereo translation from the camera trajectory")
    ap.add_argument("--poses", required=True, type=Path,
                    help="transforms.json or a frame list exported by export_poses.py")
    ap.add_argument("--fallback", default="up",
                    help="Fallback direction when dominance is insufficient (default: up)")
    ap.add_argument("--min-dominance", type=float, default=0.6,
                    help="Fall back below this trajectory dominance (default: 0.6)")
    ap.add_argument("--quiet", action="store_true", help="Print only the final direction")
    args = ap.parse_args()

    log = (lambda *a: None) if args.quiet else (
        lambda *a: print("[auto_direction]", *a, file=sys.stderr))

    direction, st = analyze(load_poses(args.poses))

    log(f"{st['n_frames']} frames; trajectory tangent in camera coordinates: "
        f"mean |x| {st['mean_abs_x']:.3f}, mean |y| {st['mean_abs_y']:.3f}")
    log(f"Dominant axis {st['axis']} ({st['dominance']:.1%} dominance), "
        f"sign consistency {st['sign_consistency']:.0%}")

    if st["dominance"] < args.min_dominance:
        log(f"Dominance is below {args.min_dominance:.0%}; trajectory direction is ambiguous")
        log(f"Using configured fallback {args.fallback}; process such sequences in segments")
        print(args.fallback)
        return

    log(f"Camera moves along local {'+' if st['sign'] > 0 else '-'}{st['axis']}"
        f"  ->  direction {direction}")
    if st["sign_consistency"] < 0.8:
        log(f"Warning: sign consistency is only {st['sign_consistency']:.0%}; "
            f"the trajectory reverses direction, so the majority sign is used")

    print(direction)


if __name__ == "__main__":
    main()
