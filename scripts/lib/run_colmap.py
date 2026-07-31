#!/usr/bin/env python3
"""Run COLMAP feature extraction, matching, mapping, and undistortion.

Existing products skip their stage automatically; --overwrite forces reruns.

Output layout:

    <work>/colmap/
    ├── database.db
    ├── sparse/0/          mapper output (original distorted model)
    └── undistorted/
        ├── images/        undistorted images used for training and rendering
        └── sparse/        undistorted PINHOLE model

--camera-model controls lens fitting during SfM. PINHOLE asserts an undistorted
lens; OPENCV estimates distortion and is appropriate for real wide-angle lenses.
Downstream models are always undistorted PINHOLE models. With PINHOLE SfM input,
image_undistorter only copies images and may be disabled with --undistort 0.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MATCHERS = {
    "exhaustive": "exhaustive_matcher",       # Unordered images; robust but O(n^2)
    "sequential": "sequential_matcher",       # Video frames; much faster temporal matching
    "spatial": "spatial_matcher",             # GPS/position priors
    "vocab_tree": "vocab_tree_matcher",       # Large image collections
}


def run(cmd, label):
    print(f"\n\033[1;34m[colmap]\033[0m {label}")
    print("  " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"[run_colmap] {label} failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description="Run COLMAP SfM and undistortion")
    ap.add_argument("--image-dir", required=True, type=Path, help="Input image directory")
    ap.add_argument("--work-dir", required=True, type=Path, help="COLMAP working directory")
    ap.add_argument("--matcher", default="exhaustive", choices=list(MATCHERS),
                    help="Matching strategy; sequential is faster for video (default: exhaustive)")
    ap.add_argument("--camera-model", default="PINHOLE",
                    help="SfM camera model; PINHOLE assumes no distortion (default); "
                         "use OPENCV/RADIAL for distorted real lenses")
    ap.add_argument("--undistort", type=int, default=1,
                    help="Run image_undistorter; use 0 with PINHOLE to avoid duplicate images")
    ap.add_argument("--single-camera", type=int, default=1,
                    help="Share one camera intrinsic model across images (default: 1)")
    ap.add_argument("--use-gpu", type=int, default=1)
    ap.add_argument("--colmap-bin", default="colmap")
    ap.add_argument("--overwrite", action="store_true", help="Rerun stages with existing products")
    args = ap.parse_args()

    if shutil.which(args.colmap_bin) is None:
        raise SystemExit(f"[run_colmap] COLMAP executable not found: {args.colmap_bin}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"[run_colmap] Image directory does not exist: {args.image_dir}")

    n_img = sum(1 for p in args.image_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if n_img == 0:
        raise SystemExit(f"[run_colmap] No images in {args.image_dir}")

    work = args.work_dir
    db = work / "database.db"
    sparse = work / "sparse"
    undist = work / "undistorted"
    work.mkdir(parents=True, exist_ok=True)

    print(f"[run_colmap] {n_img} images -> {work}")
    print(f"[run_colmap] Matcher {args.matcher}, camera model {args.camera_model}")
    if args.camera_model in ("PINHOLE", "SIMPLE_PINHOLE"):
        print(f"[run_colmap] Note: {args.camera_model} assumes an undistorted lens.")
        print("[run_colmap] Use OPENCV for real wide-angle/fisheye lenses.")

    # 1. Feature extraction
    if args.overwrite or not db.exists():
        run([args.colmap_bin, "feature_extractor",
             "--database_path", db,
             "--image_path", args.image_dir,
             "--ImageReader.camera_model", args.camera_model,
             "--ImageReader.single_camera", args.single_camera,
             "--SiftExtraction.use_gpu", args.use_gpu], "feature extraction")
    else:
        print(f"\n[colmap] Feature extraction - {db} exists, skipping")

    # 2. Matching
    stamp = work / ".matched"
    if args.overwrite or not stamp.exists():
        run([args.colmap_bin, MATCHERS[args.matcher],
             "--database_path", db,
             "--SiftMatching.use_gpu", args.use_gpu], f"matching ({args.matcher})")
        stamp.touch()
    else:
        print("\n[colmap] Matching - already complete, skipping")

    # 3. Sparse reconstruction
    if args.overwrite or not (sparse / "0" / "cameras.bin").exists():
        sparse.mkdir(parents=True, exist_ok=True)
        run([args.colmap_bin, "mapper",
             "--database_path", db,
             "--image_path", args.image_dir,
             "--output_path", sparse], "sparse reconstruction")
    else:
        print(f"\n[colmap] Sparse reconstruction - {sparse}/0 exists, skipping")

    models = sorted(p for p in sparse.glob("*") if (p / "cameras.bin").exists())
    if not models:
        raise SystemExit("[run_colmap] Sparse reconstruction produced no model")
    if len(models) > 1:
        print(f"[run_colmap] Warning: produced {len(models)} disconnected models; "
              f"using largest {models[0].name}")

    # 4. Undistortion
    if not args.undistort:
        print(f"\n[colmap] Undistortion disabled; downstream uses {models[0]}")
        print(f"[run_colmap] Complete: sparse model {models[0]}")
        return
    if args.overwrite or not (undist / "sparse" / "cameras.bin").exists():
        run([args.colmap_bin, "image_undistorter",
             "--image_path", args.image_dir,
             "--input_path", models[0],
             "--output_path", undist,
             "--output_type", "COLMAP"], "undistortion")
    else:
        print(f"\n[colmap] Undistortion - {undist} exists, skipping")

    n_out = sum(1 for p in (undist / "images").iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"\n[run_colmap] Complete: registered {n_out}/{n_img} images")
    print(f"[run_colmap] Undistorted images {undist/'images'}")
    print(f"[run_colmap] Undistorted model {undist/'sparse'}")
    if n_out < n_img:
        print(f"[run_colmap] Note: {n_img-n_out} images were not registered. "
              "Try --matcher sequential for video or check overlap/texture.")


if __name__ == "__main__":
    main()
