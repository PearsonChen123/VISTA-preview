#!/usr/bin/env python3
"""Generate the K.txt required by FoundationStereo (intrinsics + baseline).

The first line contains nine flattened 3x3 intrinsic values; the second is the baseline in meters.

This performs two operations:

1. **Baseline conversion.** Render poses use normalized nerfstudio coordinates,
   while FoundationStereo needs a real-scale baseline. The dataparser scale maps
   real to normalized scale, so baseline = shift / scale.

2. **Intrinsic rotation.** Intrinsics must follow the image rotation used for
   stereo matching, or fx and the principal point will be wrong.

The old version used two inline heredocs and duplicated the rotation map. This
script combines both operations and uses the shared definitions in common.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIRECTION_CHOICES, rotate_intrinsics, rotated_size,
                    rotation_for, write_intrinsic_file)


def main():
    ap = argparse.ArgumentParser(description="Generate FoundationStereo K.txt")
    ap.add_argument("--transforms-json", required=True, type=Path,
                    help="Dataset transforms.json providing intrinsics and image dimensions")
    ap.add_argument("--dataparser-transforms", required=True, type=Path,
                    help="Training dataparser_transforms.json providing scale")
    ap.add_argument("--shift", required=True, type=float, help="Baseline in normalized coordinates")
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right (x/-x/y/-y also accepted)")
    ap.add_argument("--output", required=True, type=Path, help="Final rotated K.txt")
    ap.add_argument("--output-original", type=Path, help="Optional original K for debugging")
    args = ap.parse_args()

    data = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    missing = [k for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if data.get(k) is None]
    if missing:
        raise ValueError(f"{args.transforms_json} is missing intrinsic fields: {missing}")

    fl_x, fl_y = float(data["fl_x"]), float(data["fl_y"])
    cx, cy = float(data["cx"]), float(data["cy"])
    width, height = int(data["w"]), int(data["h"])

    K = np.array([[fl_x, 0.0, cx],
                  [0.0, fl_y, cy],
                  [0.0, 0.0, 1.0]])

    dp = json.loads(args.dataparser_transforms.read_text(encoding="utf-8"))
    scale = dp.get("scale")
    if scale is None:
        raise ValueError(f"{args.dataparser_transforms} has no scale for baseline conversion")
    scale = float(scale)

    # Baseline is a positive distance; direction is encoded by translation and rotation.
    baseline = abs(args.shift / scale)

    rotation = rotation_for(args.shift_direction)
    K_rot = rotate_intrinsics(K, rotation, width, height)
    new_w, new_h = rotated_size(width, height, rotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_intrinsic_file(args.output, K_rot, baseline)
    if args.output_original:
        write_intrinsic_file(args.output_original, K, baseline)

    print(f"[make_intrinsics] Original {width}x{height}  "
          f"fx={fl_x:.3f} fy={fl_y:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"[make_intrinsics] Rotation {rotation} -> {new_w}x{new_h}  "
          f"fx={K_rot[0,0]:.3f} fy={K_rot[1,1]:.3f} "
          f"cx={K_rot[0,2]:.3f} cy={K_rot[1,2]:.3f}")
    print(f"[make_intrinsics] Baseline {baseline:.6f} m  (shift {args.shift} / scale {scale})")
    print(f"[make_intrinsics] Wrote {args.output}")


if __name__ == "__main__":
    main()
