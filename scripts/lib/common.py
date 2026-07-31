"""Shared stereo geometry definitions.

**Why rotation is necessary**

nerfstudio's camera coordinates are x-right / y-up / z-back. We create stereo
pairs by translating the camera, but stereo matching networks such as
FoundationStereo require horizontal disparity with right-image content shifted
left relative to the left image. When the baseline is not along -x, rendered
images must be rotated into that convention and depth maps rotated back afterward.

The old implementation copied this mapping into three places: the image-rotation
script inferred it from the shift sign, an inline K.txt script had another copy,
and depth postprocessing had a third labeled "opposite of the former." Keeping
them consistent manually was error-prone, so this module defines one shared map.
"""

import cv2
import numpy as np

# User-facing direction names describe where right-image content moves relative
# to the left image after the result is restored to its original orientation.
#
# This is clearer than x / -x / y / -y, which describe camera translation in
# camera coordinates. Camera motion and image-content motion have opposite signs.
#
# The mapping follows projection geometry and has been verified with renders:
#   right -> +x axis -> content right      up   -> +y axis -> content up
#   left  -> -x axis -> content left       down -> -y axis -> content down
DIRECTION_ALIASES = {
    "right": "x",  "x": "x",
    "left": "-x",  "-x": "-x",
    "up": "y",     "y": "y",
    "down": "-y",  "-y": "-y",
}

# Recommended CLI spellings; x/-x/y/-y remain accepted for compatibility.
DIRECTION_CHOICES = ("up", "down", "left", "right", "x", "-x", "y", "-y")

# Axis direction -> rotation applied to rendered images and intrinsics.
ROTATION_FOR_SHIFT = {
    "x": "180",
    "-x": "none",
    "y": "90cc",
    "-y": "90c",
}


def normalize_direction(direction: str) -> str:
    """Normalize up/down/left/right or x/-x/y/-y to axis notation."""
    key = str(direction).strip().lower()
    if key not in DIRECTION_ALIASES:
        raise ValueError(
            f"Unknown direction {direction!r}; choices: {', '.join(DIRECTION_CHOICES)}")
    return DIRECTION_ALIASES[key]

# Inverse rotations used to restore depth maps to their original orientation.
INVERSE_ROTATION = {
    "none": "none",
    "180": "180",
    "90cc": "90c",
    "90c": "90cc",
}

def rotation_for(shift_direction: str) -> str:
    """Map a direction to the image/intrinsics rotation type."""
    return ROTATION_FOR_SHIFT[normalize_direction(shift_direction)]


def inverse_of(rotation: str) -> str:
    """Return the inverse rotation."""
    if rotation not in INVERSE_ROTATION:
        raise ValueError(f"Unknown rotation type {rotation!r}")
    return INVERSE_ROTATION[rotation]


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate an HxW or HxWxC image with cv2 (for uint8 images)."""
    if rotation == "none":
        return image
    codes = {
        "90cc": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "90c": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    return cv2.rotate(image, codes[rotation])


def rotate_array(array: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate a 2D array with NumPy (for float depth/disparity maps).

    np.rot90 with k>0 is counterclockwise, matching cv2 COUNTERCLOCKWISE.
    """
    if rotation == "none":
        return array
    k = {"90cc": 1, "90c": -1, "180": 2}[rotation]
    return np.ascontiguousarray(np.rot90(array, k=k))


def rotate_intrinsics(K: np.ndarray, rotation: str, width: int, height: int) -> np.ndarray:
    """Rotate a 3x3 intrinsic matrix to match the rotated image.

    width/height are the image dimensions before rotation. A 90-degree rotation
    swaps fx/fy and recalculates the principal point against the new bounds.
    """
    if rotation == "none":
        return K.copy()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if rotation == "90cc":
        return np.array([[fy, 0, cy],
                         [0, fx, width - 1.0 - cx],
                         [0, 0, 1]])
    if rotation == "90c":
        return np.array([[fy, 0, height - 1.0 - cy],
                         [0, fx, cx],
                         [0, 0, 1]])
    # 180
    return np.array([[fx, 0, width - 1.0 - cx],
                     [0, fy, height - 1.0 - cy],
                     [0, 0, 1]])


def rotated_size(width: int, height: int, rotation: str):
    """Return (width, height) after rotation."""
    if rotation in ("90cc", "90c"):
        return height, width
    return width, height


def read_intrinsic_file(path):
    """Read a FoundationStereo K file: nine intrinsic values, then baseline."""
    with open(path, "r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} has invalid format: expected K's 9 values and a baseline")
    K = np.array([float(v) for v in lines[0].split()], dtype=float).reshape(3, 3)
    return K, float(lines[1])


def write_intrinsic_file(path, K: np.ndarray, baseline: float):
    """Write a FoundationStereo-format K file."""
    with open(path, "w") as f:
        f.write(" ".join(str(v) for v in K.flatten()) + "\n")
        f.write(f"{baseline}\n")
