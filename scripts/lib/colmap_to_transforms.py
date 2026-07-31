#!/usr/bin/env python3
"""Convert a COLMAP sparse model to nerfstudio transforms.json.

Undistorted PINHOLE input is preferred because it reduces downstream intrinsics
to fx/fy/cx/cy. Distorted mapper output requires every rendering and stereo stage
to propagate distortion correctly. Use `colmap image_undistorter` first unless
--allow-distorted is explicitly requested.

COLMAP stores world-to-camera in OpenCV coordinates (x-right, y-down, z-forward).
nerfstudio needs camera-to-world in OpenGL coordinates (x-right, y-up, z-back):

    transform_matrix = inv(w2c) @ diag(1, -1, -1, 1)
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_model import find_sparse_dir, read_cameras_binary, read_images_binary

# OpenCV camera coordinates -> OpenGL camera coordinates.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def build_transforms(cameras, images, image_dir: Path, out_path: Path,
                     allow_distorted: bool = False, downscale: float = 1.0):
    distorted = [c for c in cameras.values() if not c.is_undistorted]
    if distorted and not allow_distorted:
        models = sorted({c.model for c in distorted})
        raise SystemExit(
            f"[colmap_to_transforms] Model has distortion ({', '.join(models)}).\n"
            f"  First run: colmap image_undistorter --image_path <images> \\\n"
            f"             --input_path <sparse/0> --output_path <undistorted>\n"
            f"  Then use <undistorted> as input.\n"
            f"  Add --allow-distorted only if distortion is intentional.")

    # Stable filename order aligns pose export, rendering, and depth write-back.
    ordered = sorted(images.values(), key=lambda im: im.name)

    frames = []
    for im in ordered:
        cam = cameras[im.camera_id]
        c2w_gl = im.camera_to_world() @ CV_TO_GL
        frame = {
            "file_path": str((image_dir / im.name).as_posix()),
            "transform_matrix": c2w_gl.tolist(),
            "colmap_im_id": int(im.id),
        }
        # Store intrinsics per frame for multi-camera models.
        if len(cameras) > 1:
            fx, fy, cx, cy = (v / downscale for v in cam.pinhole_intrinsics())
            frame.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                          "w": round(cam.width / downscale),
                          "h": round(cam.height / downscale)})
        frames.append(frame)

    # Store intrinsics at top level for a single-camera model.
    ref = cameras[ordered[0].camera_id]
    # Scale intrinsics with downsampled images such as images_4.
    fx, fy, cx, cy = (v / downscale for v in ref.pinhole_intrinsics())
    k1, k2, p1, p2 = ref.distortion()

    out = {
        "camera_model": "OPENCV",
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": round(ref.width / downscale), "h": round(ref.height / downscale),
        "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out, ref


def main():
    ap = argparse.ArgumentParser(description="COLMAP sparse model -> transforms.json")
    ap.add_argument("--colmap-dir", required=True, type=Path,
                    help="Directory containing cameras.bin/images.bin, or a parent "
                         "(automatically searches sparse/ and sparse/0/)")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Image directory; <undistorted>/images for the undistorted workflow")
    ap.add_argument("--output", required=True, type=Path, help="transforms.json path")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="Image downscale relative to the COLMAP model; use 4 for images_4")
    ap.add_argument("--allow-distorted", action="store_true",
                    help="Allow distorted models (not recommended)")
    args = ap.parse_args()

    sparse = find_sparse_dir(args.colmap_dir)
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    if not images:
        raise SystemExit(f"[colmap_to_transforms] No registered images in {sparse}")

    # Relative file_path values keep the directory portable.
    try:
        rel = args.image_dir.resolve().relative_to(args.output.resolve().parent)
    except ValueError:
        rel = args.image_dir.resolve()

    out, ref = build_transforms(cameras, images, Path(rel), args.output,
                                args.allow_distorted, args.downscale)

    missing = [f["file_path"] for f in out["frames"]
               if not (args.output.parent / f["file_path"]).exists()][:3]

    print(f"[colmap_to_transforms] Sparse model {sparse}")
    print(f"[colmap_to_transforms] Camera model {ref.model} ({len(cameras)} cameras)")
    print(f"[colmap_to_transforms] Downscale {args.downscale}x")
    print(f"[colmap_to_transforms] Intrinsics {out['w']}x{out['h']}  "
          f"fx={out['fl_x']:.3f} fy={out['fl_y']:.3f} "
          f"cx={out['cx']:.3f} cy={out['cy']:.3f}")
    print(f"[colmap_to_transforms] Registered images {len(out['frames'])}")
    print(f"[colmap_to_transforms] Wrote {args.output}")
    if missing:
        print(f"[colmap_to_transforms] Warning: image files are missing, e.g. {missing}")


if __name__ == "__main__":
    main()
