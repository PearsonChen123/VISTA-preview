#!/usr/bin/env python3
"""Derive right-eye poses by translating the left-eye poses.

Input is the transforms_train.json written by export_poses.py (a list of frames,
each with file_path and a 3x4 camera-to-world matrix); output is the right-eye
version in the same format.

The translation happens in the **camera's own frame**:
    new_c2w = c2w @ inv(T_shift)
so the baseline follows each camera's orientation instead of being a fixed
displacement in world space.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIRECTION_CHOICES, normalize_direction


def _axis_and_sign(shift_direction: str):
    key = normalize_direction(shift_direction)
    sign = -1.0 if key.startswith("-") else 1.0
    axis = {"x": 0, "y": 1}[key.lstrip("-")]
    return axis, sign


def shift_pose(c2w_3x4, shift: float, shift_direction: str) -> list:
    axis, sign = _axis_and_sign(shift_direction)

    c2w = np.asarray(c2w_3x4, dtype=float)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])

    t_shift = np.eye(4)
    t_shift[axis, 3] = sign * shift

    # inv(T_shift @ inv(c2w)) == c2w @ inv(T_shift)
    shifted = c2w @ np.linalg.inv(t_shift)
    return shifted[:3, :4].tolist()


def main():
    ap = argparse.ArgumentParser(description="Derive right-eye poses from left-eye poses")
    ap.add_argument("--input", required=True, type=Path, help="transforms_train.json")
    ap.add_argument("--output", required=True, type=Path, help="right-eye output json")
    ap.add_argument("--shift", required=True, type=float, help="baseline length (in the normalised frame)")
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right (x/-x/y/-y also accepted)")
    args = ap.parse_args()

    frames = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        raise ValueError(f"{args.input} should contain a list of frames")

    for frame in frames:
        frame["transform"] = shift_pose(frame["transform"], args.shift, args.shift_direction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    print(f"[stereo_shift] {len(frames)} right-eye poses -> {args.output}"
          f"  (shift={args.shift} direction={args.shift_direction})")


if __name__ == "__main__":
    main()
