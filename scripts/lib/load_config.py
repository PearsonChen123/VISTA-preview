#!/usr/bin/env python3
"""Load config.json, apply defaults, resolve paths, and emit shell assignments.

config.json is the single configuration source. run_pipeline.sh evaluates this
output, after which command-line arguments may still override values.

Usage:
    eval "$(python3 lib/load_config.py config.json)"
    python3 lib/load_config.py config.json --dump     # Human-readable format
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

# Defaults let config.json specify only values that differ.
DEFAULTS = {
    "project": {
        "root": None,                      # Required scene root
        "work_dir": None,                  # Empty -> <root>/stereo_depth
    },
    "colmap": {
        "enabled": True,
        "image_dir": "images",             # Relative to root or absolute
        "work_dir": None,                  # Empty -> <root>/colmap
        "matcher": "exhaustive",           # exhaustive/sequential/spatial/vocab_tree
        "camera_model": "PINHOLE",
        "undistort": True,
        "single_camera": 1,
        "use_gpu": 1,
        "colmap_bin": "colmap",
    },
    "nerfstudio": {
        "method": "nerfacto",
        "config_path": None,               # Empty -> latest under <root>/outputs
        "max_num_iterations": 30000,
        "output_dir": None,                # Empty -> <root>/outputs
        "extra_args": [],
    },
    "stereo": {
        # pixels: derive baseline from target disparity as a fraction of image width.
        # baseline: supply baseline directly in normalized coordinates (legacy).
        "shift_mode": "pixels",
        "shift_pixels": 0.1,               # 0.1 = 10% of image width
        "reference_depth": None,           # Empty -> estimate from COLMAP sparse points
        "reference_depth_percentile": 25,  # Reference percentile of depth distribution
                                           # Near-depth anchoring limits foreground disparity.
        "shift": 0.2,                      # Used when shift_mode="baseline"
        # Automatically choose translation along the observed camera trajectory.
        "auto_direction": True,
        "auto_direction_min_dominance": 0.6,   # Fall back to direction below this.
        "direction": "up",                 # Used when automatic selection is off/falls back.
        "valid_iters": 32,
        "foundation_dir": None,            # Empty -> project third_party/FoundationStereo
        "foundation_model": None,          # Empty -> project models/foundation_stereo/...
    },
    "filter": {
        "enabled": True,
        "num_src": 8,
        "max_reproj_error": 2.0,
        "max_depth_error": 0.01,
        "min_triangulation_angle": 3.0,
        "min_num_consistent": 2,
        "use_ncc": True,
        "ncc_window": 4,
        "min_ncc": 0.3,
        "min_texture_std": 0.02,
        # Save per-pixel confidence for confidence-weighted depth-guided sampling.
        "save_confidence": True,
    },
    "slam": {
        # Feed filtered depth to DROID-SLAM as trusted RGB-D depth. DROID treats
        # filtered zeroes as "no depth prior", which matches their meaning.
        "enabled": False,
        "droid_metric_dir": None,          # Empty -> <nevstereo_root>/droid_metric
        "use_filtered_depth": True,        # False uses unfiltered depth/
        "global_ba_frontend": 0,
        "evaluate": True,                  # Compare with known transforms.json poses
    },
    "output": {
        "depth_png": True,
        "clean_intermediate": False,
        "visualize": False,
    },
    "env": {
        "nevstereo_root": "/mnt/g/algorithm_backup/Nevstereo",
        "conda_root": "/home/pengcc/miniconda3",
        "nerfstudio_env": "nerfstudio_sm120",
        "stereo_env": "nevstereo",
        "cuda_home": "/usr/local/cuda-12.8",
        "max_jobs": 4,
    },
}

# JSON path -> shell variable name
SHELL_VARS = {
    "project.root": "DATA_DIR",
    "project.work_dir": "WORK_DIR",
    "colmap.enabled": "COLMAP_ENABLED",
    "colmap.image_dir": "COLMAP_IMAGE_DIR",
    "colmap.work_dir": "COLMAP_WORK_DIR",
    "colmap.matcher": "COLMAP_MATCHER",
    "colmap.camera_model": "COLMAP_CAMERA_MODEL",
    "colmap.undistort": "COLMAP_UNDISTORT",
    "colmap.single_camera": "COLMAP_SINGLE_CAMERA",
    "colmap.use_gpu": "COLMAP_USE_GPU",
    "colmap.colmap_bin": "COLMAP_BIN",
    "nerfstudio.method": "NS_METHOD",
    "nerfstudio.config_path": "CONFIG_PATH",
    "nerfstudio.max_num_iterations": "NS_ITERS",
    "nerfstudio.output_dir": "NS_OUTPUT_DIR",
    "stereo.shift_mode": "SHIFT_MODE",
    "stereo.shift_pixels": "SHIFT_PIXELS",
    "stereo.reference_depth": "REFERENCE_DEPTH",
    "stereo.reference_depth_percentile": "REFERENCE_DEPTH_PCT",
    "stereo.shift": "SHIFT",
    "stereo.auto_direction": "AUTO_DIRECTION",
    "stereo.auto_direction_min_dominance": "AUTO_DIR_MIN_DOM",
    "stereo.direction": "SHIFT_DIRECTION",
    "stereo.valid_iters": "VALID_ITERS",
    "stereo.foundation_dir": "FOUNDATION_DIR",
    "stereo.foundation_model": "FOUNDATION_MODEL",
    "filter.enabled": "FILTER_ENABLED",
    "filter.num_src": "FILTER_NUM_SRC",
    "filter.max_reproj_error": "FILTER_MAX_REPROJ",
    "filter.max_depth_error": "FILTER_MAX_DEPTH_ERR",
    "filter.min_triangulation_angle": "FILTER_MIN_TRI_ANGLE",
    "filter.min_num_consistent": "FILTER_MIN_CONSISTENT",
    "filter.use_ncc": "FILTER_USE_NCC",
    "filter.ncc_window": "FILTER_NCC_WINDOW",
    "filter.min_ncc": "FILTER_MIN_NCC",
    "filter.min_texture_std": "FILTER_MIN_TEXTURE_STD",
    "filter.save_confidence": "FILTER_SAVE_CONF",
    "slam.enabled": "SLAM_ENABLED",
    "slam.droid_metric_dir": "DROID_METRIC_DIR",
    "slam.use_filtered_depth": "SLAM_USE_FILTERED",
    "slam.global_ba_frontend": "SLAM_GLOBAL_BA",
    "slam.evaluate": "SLAM_EVALUATE",
    "output.depth_png": "DEPTH_PNG",
    "output.clean_intermediate": "CLEAN_INTERMEDIATE",
    "output.visualize": "VIS",
    "env.nevstereo_root": "NEVSTEREO_ROOT",
    "env.conda_root": "CONDA_ROOT",
    "env.cuda_home": "CUDA_HOME",
    "env.max_jobs": "MAX_JOBS",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if k not in out:
            raise SystemExit(f"[load_config] Unknown configuration key: {k}"
                             f" (choices: {', '.join(base)})")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg: dict, config_path: Path) -> dict:
    """Resolve relative paths and populate empty derived values."""
    root = cfg["project"]["root"]
    if not root:
        raise SystemExit("[load_config] project.root is required in config.json")
    # Resolve relative paths against config.json so the configuration remains portable.
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    cfg["project"]["root"] = str(root)

    def under_root(value, default_rel):
        p = Path(value).expanduser() if value else Path(default_rel)
        return str(p if p.is_absolute() else (root / p))

    cfg["project"]["work_dir"] = under_root(cfg["project"]["work_dir"], "stereo_depth")
    cfg["colmap"]["image_dir"] = under_root(cfg["colmap"]["image_dir"], "images")
    cfg["colmap"]["work_dir"] = under_root(cfg["colmap"]["work_dir"], "colmap")
    cfg["nerfstudio"]["output_dir"] = under_root(cfg["nerfstudio"]["output_dir"], "outputs")

    nev = Path(cfg["env"]["nevstereo_root"])
    if cfg["stereo"]["foundation_dir"] is None:
        cfg["stereo"]["foundation_dir"] = str(nev / "third_party/FoundationStereo")
    if cfg["stereo"]["foundation_model"] is None:
        cfg["stereo"]["foundation_model"] = str(
            nev / "models/foundation_stereo/model_best_bp2.pth")
    if cfg["slam"]["droid_metric_dir"] is None:
        cfg["slam"]["droid_metric_dir"] = str(nev / "droid_metric")
    if cfg["nerfstudio"]["config_path"]:
        cfg["nerfstudio"]["config_path"] = str(
            Path(cfg["nerfstudio"]["config_path"]).expanduser())
    return cfg


def get(cfg, dotted):
    section, key = dotted.split(".")
    return cfg[section][key]


def to_shell(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def main():
    ap = argparse.ArgumentParser(description="config.json -> shell variables")
    ap.add_argument("config", type=Path)
    ap.add_argument("--dump", action="store_true", help="Print the complete resolved configuration")
    args = ap.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"[load_config] Configuration file not found: {args.config}")
    try:
        user = json.loads(args.config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[load_config] {args.config} is not valid JSON: {e}")

    cfg = resolve(deep_merge(DEFAULTS, user), args.config.resolve())

    if args.dump:
        json.dump(cfg, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    for dotted, var in SHELL_VARS.items():
        print(f"{var}={shlex.quote(to_shell(get(cfg, dotted)))}")
    # Handle arrays separately.
    extra = cfg["nerfstudio"]["extra_args"]
    print(f"NS_EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")
    # Construct interpreter paths from conda_root and environment names.
    cr, ne, se = (cfg["env"]["conda_root"], cfg["env"]["nerfstudio_env"],
                  cfg["env"]["stereo_env"])
    print(f"NERF_PY={shlex.quote(f'{cr}/envs/{ne}/bin/python')}")
    print(f"NS_RENDER={shlex.quote(f'{cr}/envs/{ne}/bin/ns-render')}")
    print(f"NS_TRAIN={shlex.quote(f'{cr}/envs/{ne}/bin/ns-train')}")
    print(f"STEREO_PY={shlex.quote(f'{cr}/envs/{se}/bin/python')}")


if __name__ == "__main__":
    main()
