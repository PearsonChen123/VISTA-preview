#!/usr/bin/env python3
"""Project COLMAP sparse points into each image to produce sparse depth maps."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:  # pragma: no cover - fallback if cv2 missing
    _HAVE_CV2 = False
    from imageio import imwrite as imwrite_png

    def cv2_imwrite(path: str, array: np.ndarray) -> None:
        imwrite_png(path, array)
else:
    def cv2_imwrite(path: str, array: np.ndarray) -> None:
        cv2.imwrite(path, array)


@dataclass
class CameraEntry:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImageEntry:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class PointEntry:
    id: int
    xyz: np.ndarray


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next(fid, num_bytes: int, fmt: str):
    return struct.unpack(fmt, fid.read(num_bytes))


def read_cameras_binary(path: Path) -> Dict[int, CameraEntry]:
    cameras: Dict[int, CameraEntry] = {}
    with path.open("rb") as fid:
        num_cameras, = _read_next(fid, 8, "<Q")
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _read_next(fid, 24, "<iiQQ")
            model_info = _CAMERA_MODELS.get(model_id)
            if model_info is None:
                raise ValueError(f"Unsupported camera model id {model_id} in {path}")
            model_name, num_params = model_info
            params = np.array(_read_next(fid, 8 * num_params, "<" + "d" * num_params), dtype=np.float64)
            cameras[cam_id] = CameraEntry(cam_id, model_name, int(width), int(height), params)
    return cameras


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def read_images_binary(path: Path) -> Dict[int, ImageEntry]:
    images: Dict[int, ImageEntry] = {}
    with path.open("rb") as fid:
        num_images, = _read_next(fid, 8, "<Q")
        for _ in range(num_images):
            image_id, = _read_next(fid, 4, "<I")
            qvec = np.array(_read_next(fid, 32, "<dddd"), dtype=np.float64)
            tvec = np.array(_read_next(fid, 24, "<ddd"), dtype=np.float64)
            camera_id, = _read_next(fid, 4, "<I")
            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")
            num_points2D, = _read_next(fid, 8, "<Q")
            xys = np.zeros((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.full((num_points2D,), -1, dtype=np.int64)
            for idx in range(num_points2D):
                x, y, point3D_id = _read_next(fid, 24, "<ddq")
                xys[idx] = [x, y]
                point3D_ids[idx] = point3D_id
            images[image_id] = ImageEntry(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3d_binary(path: Path) -> Dict[int, PointEntry]:
    points: Dict[int, PointEntry] = {}
    with path.open("rb") as fid:
        num_points, = _read_next(fid, 8, "<Q")
        for _ in range(num_points):
            point_id, = _read_next(fid, 8, "<Q")
            xyz = np.array(_read_next(fid, 24, "<ddd"), dtype=np.float64)
            fid.read(3)  # skip RGB
            fid.read(1)  # alignment padding
            fid.read(8)  # reprojection error
            track_len, = _read_next(fid, 8, "<Q")
            # Skip track entries (image_id, point2D_idx) pairs
            for _ in range(track_len):
                fid.read(8)
            points[point_id] = PointEntry(point_id, xyz)
    return points


def create_sparse_depth_maps(
    cameras: Dict[int, CameraEntry],
    images: Iterable[ImageEntry],
    points: Dict[int, PointEntry],
    output_dir: Path,
    scale_factor: float,
    write_mask: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask"
    if write_mask:
        mask_dir.mkdir(exist_ok=True)

    total_pixels = 0
    total_valid = 0

    image_list = sorted(images, key=lambda entry: entry.name)

    for image in image_list:
        camera = cameras[image.camera_id]
        height, width = camera.height, camera.width
        depth_map = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8) if write_mask else None

        R = qvec_to_rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)

        for xy, point_id in zip(image.xys, image.point3D_ids):
            if point_id == -1:
                continue
            point = points.get(point_id)
            if point is None:
                continue
            xyz = point.xyz.reshape(3, 1)
            cam_coord = R @ xyz + t
            depth = cam_coord[2, 0]
            if depth <= 0:
                continue
            x_pix = int(round(xy[0]))
            y_pix = int(round(xy[1]))
            if not (0 <= x_pix < width and 0 <= y_pix < height):
                continue
            current = depth_map[y_pix, x_pix]
            if current == 0 or depth < current:
                depth_map[y_pix, x_pix] = depth
                if mask is not None:
                    mask[y_pix, x_pix] = 255

        stem = Path(image.name).stem
        np.save(output_dir / f"{stem}.npy", depth_map)
        depth_png = np.clip(depth_map * scale_factor, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2_imwrite(str(output_dir / f"{stem}.png"), depth_png)
        if mask is not None:
            cv2_imwrite(str(mask_dir / f"{stem}.png"), mask)

        total_pixels += depth_map.size
        total_valid += int(np.count_nonzero(depth_map))

    coverage = (total_valid / total_pixels) * 100 if total_pixels else 0.0
    print(f"Processed {len(image_list)} images. Valid depth coverage: {coverage:.2f}%")


def main() -> None:
    default_base = Path("/mnt/h/keyframe_3/test_images_100")

    parser = argparse.ArgumentParser(description="Generate sparse depth maps from COLMAP reconstruction.")
    parser.add_argument("--base-dir", type=Path, default=default_base,
                        help="Base dataset directory (default: %(default)s)")
    parser.add_argument("--cameras-bin", type=Path, default=None,
                        help="Path to cameras.bin (default: <base>/undistorted/sparse/cameras.bin)")
    parser.add_argument("--images-bin", type=Path, default=None,
                        help="Path to images.bin (default: <base>/undistorted/sparse/images.bin)")
    parser.add_argument("--points3d-bin", type=Path, default=None,
                        help="Path to points3D.bin (default: <base>/undistorted/sparse/points3D.bin)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for sparse depth maps (default: <base>/stereo/sparse_depth)")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Scale factor for PNG depth export (meters->scaled units, default: 1000)")
    parser.add_argument("--no-mask", action="store_true", help="Do not write binary confidence masks")
    args = parser.parse_args()

    base_dir = args.base_dir
    sparse_dir = base_dir / "undistorted" / "sparse"

    cameras_path = args.cameras_bin or sparse_dir / "cameras.bin"
    images_path = args.images_bin or sparse_dir / "images.bin"
    points_path = args.points3d_bin or sparse_dir / "points3D.bin"
    output_dir = args.output_dir or base_dir / "stereo" / "sparse_depth"

    for path in (cameras_path, images_path, points_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required COLMAP file: {path}")

    print("Reading COLMAP models...")
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    points = read_points3d_binary(points_path)

    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points)} points.")

    create_sparse_depth_maps(cameras, images.values(), points, output_dir, args.scale, not args.no_mask)
    print(f"Sparse depth maps written to {output_dir}")


if __name__ == "__main__":
    main()
