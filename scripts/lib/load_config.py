#!/usr/bin/env python3
"""读 config.json，填默认值、解析路径，输出 shell 变量赋值供 eval。

所有配置的唯一来源就是 config.json。run_pipeline.sh 只是把它 eval 进来，
命令行参数仍可覆盖（覆盖发生在 eval 之后）。

用法:
    eval "$(python3 lib/load_config.py config.json)"
    python3 lib/load_config.py config.json --dump     # 人看的格式
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

# 默认值。config.json 里没写的项走这里，所以 config.json 可以只写要改的部分。
DEFAULTS = {
    "project": {
        "root": None,                      # 必填：场景根目录
        "work_dir": None,                  # 留空 -> <root>/stereo_depth
    },
    "colmap": {
        "enabled": True,
        "image_dir": "images",             # 相对 root 或绝对路径
        "work_dir": None,                  # 留空 -> <root>/colmap
        "matcher": "exhaustive",           # exhaustive/sequential/spatial/vocab_tree
        "camera_model": "PINHOLE",
        "undistort": True,
        "single_camera": 1,
        "use_gpu": 1,
        "colmap_bin": "colmap",
    },
    "nerfstudio": {
        "method": "nerfacto",
        "config_path": None,               # 留空 -> 自动找 <root>/outputs 下最新的
        "max_num_iterations": 30000,
        "output_dir": None,                # 留空 -> <root>/outputs
        "extra_args": [],
    },
    "stereo": {
        # shift_mode = "pixels": 用目标视差占图像宽度的百分比来定基线（推荐，直观）
        #              "baseline": 直接给归一化坐标系下的基线长度（旧行为）
        "shift_mode": "pixels",
        "shift_pixels": 0.1,               # 0.1 = 视差达到图像宽度的 10%
        "reference_depth": None,           # 留空 -> 从 COLMAP 稀疏点云统计
        "reference_depth_percentile": 25,  # 取深度分布的哪个分位作参考
                                           # 25 而非中位数：锚近处能把近平面视差压住
        "shift": 0.2,                      # shift_mode="baseline" 时用
        # auto_direction: 按相机轨迹自动选平移方向。NeRF 只在相机去过的视角附近
        # 训练充分，沿轨迹平移留在流形内，垂直于轨迹就是外推、渲染会糊。
        "auto_direction": True,
        "auto_direction_min_dominance": 0.6,   # 主导性不足则退回 direction
        "direction": "up",                 # up/down/left/right（auto 关掉或退回时用）
        "valid_iters": 32,
        "foundation_dir": None,            # 留空 -> 项目内 third_party/FoundationStereo
        "foundation_model": None,          # 留空 -> 项目内 models/foundation_stereo/...
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
        # 额外输出逐像素置信度（票数/源视角数），供深度引导采样按置信度加权
        "save_confidence": True,
    },
    "slam": {
        # 把过滤后的深度当可信深度，喂给 DROID-SLAM 跑 RGBD 出位姿。
        # 深度里被过滤掉的 0 值，DROID 正好当作"无深度先验"，语义吻合。
        "enabled": False,
        "droid_metric_dir": None,          # 留空 -> <nevstereo_root>/droid_metric
        "use_filtered_depth": True,        # False 则用未过滤的 depth/
        "global_ba_frontend": 0,
        "evaluate": True,                  # 与 transforms.json 的已知位姿对比
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

# JSON 路径 -> shell 变量名
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
            raise SystemExit(f"[load_config] 未知的配置项: {k}"
                             f"（可选: {', '.join(base)}）")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg: dict, config_path: Path) -> dict:
    """把相对路径解析成绝对路径，填好留空的派生项。"""
    root = cfg["project"]["root"]
    if not root:
        raise SystemExit("[load_config] config.json 里必须指定 project.root")
    # 相对路径按 config.json 所在目录解析，这样配置文件可以跟着场景走
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
    ap = argparse.ArgumentParser(description="config.json -> shell 变量")
    ap.add_argument("config", type=Path)
    ap.add_argument("--dump", action="store_true", help="打印解析后的完整配置")
    args = ap.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"[load_config] 找不到配置文件: {args.config}")
    try:
        user = json.loads(args.config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[load_config] {args.config} 不是合法 JSON: {e}")

    cfg = resolve(deep_merge(DEFAULTS, user), args.config.resolve())

    if args.dump:
        json.dump(cfg, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    for dotted, var in SHELL_VARS.items():
        print(f"{var}={shlex.quote(to_shell(get(cfg, dotted)))}")
    # 数组单独处理
    extra = cfg["nerfstudio"]["extra_args"]
    print(f"NS_EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")
    # 解释器路径由 conda_root + 环境名拼出来
    cr, ne, se = (cfg["env"]["conda_root"], cfg["env"]["nerfstudio_env"],
                  cfg["env"]["stereo_env"])
    print(f"NERF_PY={shlex.quote(f'{cr}/envs/{ne}/bin/python')}")
    print(f"NS_RENDER={shlex.quote(f'{cr}/envs/{ne}/bin/ns-render')}")
    print(f"NS_TRAIN={shlex.quote(f'{cr}/envs/{ne}/bin/ns-train')}")
    print(f"STEREO_PY={shlex.quote(f'{cr}/envs/{se}/bin/python')}")


if __name__ == "__main__":
    main()
