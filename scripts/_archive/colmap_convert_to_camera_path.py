import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _matrix_from_data(data: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        if arr.size == 16:
            arr = arr.reshape(4, 4)
        elif arr.size == 12:
            arr = arr.reshape(3, 4)
    if arr.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        arr = np.vstack([arr, bottom])
    if arr.shape != (4, 4):
        raise ValueError(f"Unsupported transform shape {arr.shape}; expected 3x4 or 4x4.")
    return arr.astype(float)


def _extract_matrix(frame: Dict) -> np.ndarray:
    keys = (
        "transform",
        "transform_matrix",
        "camera_to_world",
        "camera_to_world_matrix",
        "pose",
    )
    for key in keys:
        if key in frame:
            return _matrix_from_data(frame[key])
    raise KeyError("Frame missing transform matrix; expected one of: " + ", ".join(keys))


def _frame_file_path(frame: Dict, fallback_index: int) -> str:
    keys = ("file_path", "image_path", "path", "filename")
    for key in keys:
        if key in frame and frame[key]:
            return str(frame[key])
    return f"frame_{fallback_index:06d}"


def _load_camera_frames(path: Path) -> List[Dict]:
    data = _read_json(path)
    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            frames = data["frames"]
        elif "camera_path" in data and isinstance(data["camera_path"], list):
            frames = data["camera_path"]
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    result = []
    for idx, frame in enumerate(frames):
        matrix = _extract_matrix(frame)
        file_path = _frame_file_path(frame, idx)
        result.append({
            "index": idx,
            "file_path": file_path,
            "matrix": matrix,
        })
    return result


def _canonicalize_path(value: str) -> str:
    value = value.replace("\\", "/")
    value = value.lstrip("./")
    return value


def _build_original_lookup(original_frames: List[Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    path_lookup: Dict[str, np.ndarray] = {}
    name_lookup: Dict[str, np.ndarray] = {}
    for idx, frame in enumerate(original_frames):
        try:
            matrix = _extract_matrix(frame)
        except KeyError:
            continue
        file_path = _frame_file_path(frame, idx)
        canonical = _canonicalize_path(file_path)
        if canonical:
            path_lookup[canonical] = matrix
        name_lookup[Path(file_path).name] = matrix
    return path_lookup, name_lookup


def _match_original_matrix(file_path: str,
                            path_lookup: Dict[str, np.ndarray],
                            name_lookup: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    canonical = _canonicalize_path(file_path)
    if canonical in path_lookup:
        return path_lookup[canonical]
    name = Path(file_path).name
    return name_lookup.get(name)


def _frames_match_original(frames: List[Dict],
                           path_lookup: Dict[str, np.ndarray],
                           name_lookup: Dict[str, np.ndarray],
                           tolerance: float = 1e-4) -> Tuple[bool, Optional[float]]:
    diffs = []
    for frame in frames:
        original = _match_original_matrix(frame["file_path"], path_lookup, name_lookup)
        if original is None:
            continue
        diff = np.max(np.abs(original - frame["matrix"]))
        diffs.append(diff)
    if not diffs:
        return False, None
    median_diff = float(np.median(diffs))
    return median_diff <= tolerance, median_diff


def _median_translation_norm(frames: List[Dict]) -> float:
    norms = [float(np.linalg.norm(frame["matrix"][:3, 3])) for frame in frames]
    if not norms:
        return 0.0
    return float(np.median(norms))


def _load_original_transforms(path: Path) -> Dict:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Original transforms file must contain an object: {path}")
    return data


def _to_fov(fl: float, size: float) -> float:
    return 2.0 * math.atan(size / (2.0 * fl)) * 180.0 / math.pi


def _apply_dataparser_normalization(matrix: np.ndarray,
                                    dp_transform: np.ndarray,
                                    dp_scale: float) -> np.ndarray:
    normalized = dp_transform @ matrix
    normalized = normalized.copy()
    normalized[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    normalized[:3, 3] *= dp_scale
    return normalized


def _load_dataparser_info(path: Path) -> Tuple[np.ndarray, float]:
    data = _read_json(path)
    if "transform" not in data:
        raise ValueError(f"dataparser_transforms.json missing 'transform': {path}")
    transform = _matrix_from_data(data["transform"])
    scale = float(data.get("scale", 1.0))
    return transform, scale


def _resolve_dataparser_info(explicit_path: Optional[str],
                              config_path: Optional[str],
                              original_path: Path) -> Tuple[Optional[np.ndarray], Optional[float], Optional[Path]]:
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    if config_path:
        cfg = Path(config_path).expanduser().resolve()
        maybe = cfg.parent / "dataparser_transforms.json"
        candidates.append(maybe)
    env_path = os.environ.get("DATAPARSER_TRANSFORMS_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    # Auto-discover under outputs directories near the original path
    root_candidates: List[Path] = []
    for parent in original_path.resolve().parents:
        outputs_dir = parent / "outputs"
        if outputs_dir.is_dir():
            root_candidates.append(outputs_dir)
    # include sibling 'outputs' next to dataparser file as fallback
    outputs_dir = original_path.parent / "outputs"
    if outputs_dir.is_dir():
        root_candidates.append(outputs_dir)

    visited = set()
    for root in root_candidates:
        if root in visited:
            continue
        visited.add(root)
        matches = sorted(root.rglob("dataparser_transforms.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(matches[:5])  # take up to five most recent per root

    for candidate in candidates:
        if candidate.is_file():
            try:
                transform, scale = _load_dataparser_info(candidate)
                return transform, scale, candidate
            except Exception:
                continue
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Convert dataparser transforms to Nerfstudio camera_path format with COLMAP support.")
    parser.add_argument("dataparser_file", help="Path to transformed camera JSON (e.g., transforms_train.json)")
    parser.add_argument("original_transform_file", help="Path to original transforms.json with intrinsics")
    parser.add_argument("output_file", help="Path to output camera_path.json")
    parser.add_argument("--dataparser-transforms", dest="dataparser_transforms", help="Optional path to dataparser_transforms.json (for normalization)")
    parser.add_argument("--config", dest="config_path", help="Optional path to training config.yml (to locate dataparser info)")
    parser.add_argument("--fps", type=float, default=1.0, help="Desired FPS metadata for camera path (default: 1)")
    parser.add_argument("--seconds", type=float, default=None, help="Override seconds metadata; defaults to number of frames / fps")
    parser.add_argument("--translation-threshold", type=float, default=1.0,
                        help="Translation norm threshold to decide if poses are still in original scale (default: 1.0)")
    parser.add_argument("--force-original", action="store_true",
                        help="Force treating input transforms as original (apply normalization if dataparser info is available)")
    parser.add_argument("--force-normalized", action="store_true",
                        help="Force treating input transforms as already normalized")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics")

    args = parser.parse_args()

    dataparser_path = Path(args.dataparser_file).expanduser().resolve()
    original_path = Path(args.original_transform_file).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()

    frames = _load_camera_frames(dataparser_path)
    if not frames:
        raise RuntimeError(f"No frames found in {dataparser_path}")

    original_data = _load_original_transforms(original_path)
    original_frames = original_data.get("frames", [])
    path_lookup, name_lookup = _build_original_lookup(original_frames)

    frames_equal_original, median_diff = _frames_match_original(frames, path_lookup, name_lookup)
    if args.verbose:
        if median_diff is None:
            print("[INFO] Unable to compare frames with original transforms (no matches).")
        else:
            print(f"[INFO] Median difference between inputs and original: {median_diff:.6f}")

    median_translation = _median_translation_norm(frames)
    if args.verbose:
        print(f"[INFO] Median translation norm of input poses: {median_translation:.6f}")

    if args.force_original and args.force_normalized:
        raise ValueError("Cannot set both --force-original and --force-normalized")

    if args.force_original:
        needs_normalization = True
    elif args.force_normalized:
        needs_normalization = False
    else:
        needs_normalization = frames_equal_original or median_translation > args.translation_threshold

    if args.verbose:
        if needs_normalization:
            print("[INFO] Input poses appear to be in original scale; normalization will be applied if possible.")
        else:
            print("[INFO] Input poses appear to be normalized; conversion will keep them as-is.")

    dp_transform = None
    dp_scale = None
    dp_source = None
    if needs_normalization:
        dp_transform, dp_scale, dp_source = _resolve_dataparser_info(
            args.dataparser_transforms,
            args.config_path,
            original_path,
        )
        if dp_transform is None and args.verbose:
            print("[INFO] dataparser transforms not found; poses will remain in original scale.")
        elif dp_transform is not None and args.verbose:
            print(f"[INFO] Using dataparser transforms from {dp_source}")
            print(f"[INFO] dataparser scale: {dp_scale}")
    elif args.verbose and args.dataparser_transforms:
        print("[INFO] --dataparser-transforms provided but normalization not required; flag ignored.")

    processed_matrices: List[np.ndarray] = []
    for frame in frames:
        matrix = frame["matrix"]
        if needs_normalization and dp_transform is not None:
            matrix = _apply_dataparser_normalization(matrix, dp_transform, dp_scale)
        processed_matrices.append(matrix)

    if needs_normalization and dp_transform is None:
        print("Warning: dataparser transforms unavailable; output is still in original coordinate system.")

    # Compute metadata
    w = float(original_data.get("w"))
    h = float(original_data.get("h"))
    fl_x = float(original_data.get("fl_x"))
    fl_y = float(original_data.get("fl_y", fl_x))

    fov = _to_fov(fl_y, h)
    aspect = w / h

    camera_path_entries = []
    for matrix in processed_matrices:
        camera_path_entries.append({
            "camera_to_world": matrix.tolist(),
            "fov": fov,
            "aspect": aspect,
        })

    fps = float(args.fps)
    seconds = float(args.seconds) if args.seconds is not None else float(len(camera_path_entries)) / fps

    result = OrderedDict([
        ("camera_type", "perspective"),
        ("render_height", h),
        ("render_width", w),
        ("fps", fps),
        ("seconds", seconds),
        ("is_cycle", False),
        ("smoothness_value", 0.0),
        ("camera_path", camera_path_entries),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.verbose:
        print(f"[INFO] Wrote camera path with {len(camera_path_entries)} frames to {output_path}")


if __name__ == "__main__":
    main()
