#!/usr/bin/env python3
"""Restore FoundationStereo depth maps and save nerfstudio-compatible 16-bit PNGs.

batch_process.py .npy files already contain depth in meters, not disparity; it
computes depth = fx * baseline / disp internally. The old --disp-dir option was
misleading and is now named --depth-in.

Images are rotated before inference, so depth maps need the inverse rotation.
depth_file_path can optionally be written to transforms.json.
"""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIRECTION_CHOICES, inverse_of, rotate_array, rotation_for

# 16-bit PNG range: DEPTH_MAX meters maps to 65535.
DEPTH_MAX_METERS = 400.0


def _process_one(job):
    src, out_dir, rotation, save_png = job
    src = Path(src)
    try:
        depth = np.load(src).astype(np.float32)
        depth = rotate_array(depth, rotation)

        npy_path = Path(out_dir) / src.name
        np.save(npy_path, depth)

        if save_png:
            scaled = np.clip(depth * (65535.0 / DEPTH_MAX_METERS), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(npy_path.with_suffix(".png")), scaled)

        valid = depth[np.isfinite(depth) & (depth > 0)]
        stats = (float(np.min(valid)), float(np.median(valid)), float(np.max(valid))) if valid.size else None
        return {"stem": src.stem, "png": npy_path.with_suffix(".png").name, "stats": stats}
    except Exception as exc:                                   # noqa: BLE001
        return {"stem": src.stem, "error": str(exc)}


def update_transforms(transforms_json: Path, depth_dir: Path, results):
    """Write depth_file_path into corresponding transforms.json frames.

    Match by file_path stem when possible. Otherwise, align by order only when
    counts match exactly. This handles ns-render's five-digit names versus common
    four-digit dataset names without silently misaligning frames.
    """
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not frames:
        raise ValueError(f"{transforms_json} contains no frames")

    rel_dir = os.path.relpath(depth_dir.resolve(), transforms_json.resolve().parent)
    results = sorted(results, key=lambda r: r["stem"])

    def _assign(frame, item):
        frame["depth_file_path"] = os.path.join(rel_dir, item["png"]).replace("\\", "/")

    by_stem = {Path(f.get("file_path", "")).stem: f for f in frames}
    matched = [(by_stem[r["stem"]], r) for r in results if r["stem"] in by_stem]

    if matched:
        mode = "filename"
    elif len(results) == len(frames):
        mode = "sequence"
        matched = list(zip(frames, results))
    else:
        print(f"[depth_postprocess] Warning: {len(results)} depth maps and {len(frames)} frames "
              f"do not match by count or name; not updating transforms.json")
        return

    for frame, item in matched:
        _assign(frame, item)

    transforms_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[depth_postprocess] Updated depth_file_path for {len(matched)}/{len(frames)} "
          f"transforms.json frames using {mode} alignment")


def main():
    ap = argparse.ArgumentParser(description="Restore depth orientation and save 16-bit PNGs")
    ap.add_argument("--depth-in", required=True, type=Path,
                    help="FoundationStereo output directory (.npy depth files)")
    ap.add_argument("--depth-out", required=True, type=Path, help="Output directory")
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right (x/-x/y/-y also accepted)")
    ap.add_argument("--transforms-json", type=Path,
                    help="Optionally write depth_file_path into this transforms.json")
    ap.add_argument("--no-png", action="store_true", help="Save .npy only, without 16-bit PNG")
    args = ap.parse_args()

    files = sorted(args.depth_in.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {args.depth_in}")

    # Restore depth after the forward image rotation.
    rotation = inverse_of(rotation_for(args.shift_direction))
    args.depth_out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(p), str(args.depth_out), rotation, not args.no_png) for p in files]
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        for res in tqdm(pool.map(_process_one, jobs), total=len(jobs),
                        desc=f"Depth postprocessing ({rotation})"):
            results.append(res)

    failed = [r for r in results if "error" in r]
    if failed:
        raise RuntimeError(f"{len(failed)} depth maps failed; example {failed[0]['stem']}: {failed[0]['error']}")

    stats = [r["stats"] for r in results if r.get("stats")]
    if stats:
        arr = np.array(stats)
        print(f"[depth_postprocess] Completed {len(results)} frames | depth (m): "
              f"min {arr[:,0].min():.3f}, median {np.median(arr[:,1]):.3f}, max {arr[:,2].max():.3f}")
    else:
        print(f"[depth_postprocess] Warning: no valid depth in {len(results)} frames")

    if args.transforms_json:
        update_transforms(args.transforms_json, args.depth_out, results)


if __name__ == "__main__":
    main()
