#!/usr/bin/env python3
"""Rotate the rendered stereo pair into the orientation the matcher expects.

The rotation is fully determined by the baseline direction; see
common.rotation_for().

The old version inferred the rotation by grepping `DEFAULT_SHIFT=` out of
process_stereo.sh (positive shift meant counter-clockwise), which silently broke
whenever that script changed. This one takes the baseline direction directly.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIRECTION_CHOICES, rotate_image, rotation_for

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _rotate_one(args):
    src, dst, rotation = args
    image = cv2.imread(str(src))
    if image is None:
        return f"cannot read image: {src}"
    cv2.imwrite(str(dst), rotate_image(image, rotation))
    return None


def rotate_dir(input_dir: Path, output_dir: Path, rotation: str) -> int:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, output_dir / p.name, rotation) for p in files]

    errors = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        for err in tqdm(pool.map(_rotate_one, jobs), total=len(jobs),
                        desc=f"rotating {input_dir.name} ({rotation})"):
            if err:
                errors.append(err)

    if errors:
        raise RuntimeError(f"{len(errors)} images failed to rotate, e.g. {errors[0]}")
    return len(files)


def main():
    ap = argparse.ArgumentParser(description="Rotate the stereo pair according to the baseline direction")
    ap.add_argument("--left-dir", required=True, type=Path)
    ap.add_argument("--right-dir", required=True, type=Path)
    ap.add_argument("--output-left-dir", required=True, type=Path)
    ap.add_argument("--output-right-dir", required=True, type=Path)
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right (x/-x/y/-y also accepted)")
    args = ap.parse_args()

    rotation = rotation_for(args.shift_direction)
    if rotation == "none":
        print(f"[rotate_images] direction {args.shift_direction} needs no rotation; copying as-is")

    n_left = rotate_dir(args.left_dir, args.output_left_dir, rotation)
    n_right = rotate_dir(args.right_dir, args.output_right_dir, rotation)

    if n_left != n_right:
        raise RuntimeError(f"stereo pair count mismatch: left {n_left}, right {n_right}")
    print(f"[rotate_images] rotated {n_left} image pairs ({rotation})")


if __name__ == "__main__":
    main()
