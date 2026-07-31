#!/usr/bin/env python3
"""Binary reader for COLMAP sparse models (cameras.bin / images.bin / points3D.bin).

The old version converted cameras.bin to cameras.txt and invoked instant-ngp's
colmap2nerf.py. Reading binary directly avoids that intermediate dependency.

The format follows COLMAP src/colmap/scene/reconstruction.cc.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# model_id -> (name, parameter count)
CAMERA_MODELS = {
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


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole_intrinsics(self):
        """Return (fx, fy, cx, cy); meaningful only for undistorted models."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        if self.model == "PINHOLE":
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if self.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE",
                          "RADIAL_FISHEYE"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        if self.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                          "THIN_PRISM_FISHEYE"):
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        raise ValueError(f"Unsupported camera model: {self.model}")

    def distortion(self):
        """Return (k1, k2, p1, p2), using zero for absent distortion terms."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return 0.0, 0.0, 0.0, 0.0
        if self.model == "SIMPLE_RADIAL":
            return float(p[3]), 0.0, 0.0, 0.0
        if self.model == "RADIAL":
            return float(p[3]), float(p[4]), 0.0, 0.0
        if self.model in ("OPENCV", "FULL_OPENCV"):
            return float(p[4]), float(p[5]), float(p[6]), float(p[7])
        return 0.0, 0.0, 0.0, 0.0

    @property
    def is_undistorted(self) -> bool:
        return self.model in ("PINHOLE", "SIMPLE_PINHOLE")


@dataclass
class Image:
    id: int
    qvec: np.ndarray                 # (4,) w,x,y,z world-to-camera rotation
    tvec: np.ndarray                 # (3,) world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray                  # (M,2) feature pixel coordinates
    point3D_ids: np.ndarray          # (M,) -1 means no corresponding 3D point

    def world_to_camera(self) -> np.ndarray:
        """Return 4x4 world-to-camera using OpenCV x-right, y-down, z-forward."""
        T = np.eye(4)
        T[:3, :3] = qvec_to_rotmat(self.qvec)
        T[:3, 3] = self.tvec
        return T

    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera())


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def qvec_to_rotmat(q) -> np.ndarray:
    """COLMAP quaternion order is (w, x, y, z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to (w, x, y, z)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _read(fid, num_bytes: int, fmt: str):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_cameras_binary(path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            cid, model_id, w, h = _read(f, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id: {model_id}")
            name, n_params = CAMERA_MODELS[model_id]
            params = _read(f, 8 * n_params, "d" * n_params)
            cameras[cid] = Camera(cid, name, int(w), int(h), np.array(params))
    return cameras


def read_images_binary(path) -> Dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            props = _read(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            (n_pts,) = _read(f, 8, "Q")
            raw = _read(f, 24 * n_pts, "ddq" * n_pts)
            xys = np.array(raw).reshape(-1, 3)[:, :2] if n_pts else np.zeros((0, 2))
            ids = np.array(raw).reshape(-1, 3)[:, 2].astype(np.int64) if n_pts else np.zeros((0,), np.int64)

            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, ids)
    return images


def read_points3D_binary(path) -> Dict[int, Point3D]:
    points = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            props = _read(f, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            err = props[7]
            (track_len,) = _read(f, 8, "Q")
            track = _read(f, 8 * track_len, "ii" * track_len)
            track = np.array(track).reshape(-1, 2) if track_len else np.zeros((0, 2), int)
            points[pid] = Point3D(pid, xyz, rgb, err, track[:, 0], track[:, 1])
    return points


def find_sparse_dir(root: Path) -> Path:
    """Find the sparse-model directory containing cameras.bin.

    Supported COLMAP layouts:
        <root>/cameras.bin                       image_undistorter output
        <root>/sparse/cameras.bin
        <root>/sparse/0/cameras.bin              mapper output (possibly multiple models)
    """
    root = Path(root)
    candidates = [root, root / "sparse", *sorted(root.glob("sparse/*"))]
    for c in candidates:
        if (c / "cameras.bin").is_file() and (c / "images.bin").is_file():
            return c
    raise FileNotFoundError(
        f"No COLMAP sparse model under {root} (requires cameras.bin + images.bin)")


def read_model(sparse_dir):
    """Read a sparse model and return (cameras, images)."""
    d = find_sparse_dir(sparse_dir)
    return read_cameras_binary(d / "cameras.bin"), read_images_binary(d / "images.bin")
