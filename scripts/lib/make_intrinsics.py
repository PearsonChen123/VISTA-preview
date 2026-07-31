#!/usr/bin/env python3
"""生成 FoundationStereo 需要的 K.txt（内参 + 基线）。

格式：第一行是 3x3 内参展平成 9 个数，第二行是基线（米）。

两件事：

1. **基线换算**。渲染用的位姿在 nerfstudio 归一化坐标系里，而 FoundationStereo
   要用真实尺度的基线来算 depth = fx * baseline / disp。dataparser_transforms.json
   里的 scale 就是"真实尺度 -> 归一化尺度"的系数，所以 baseline = shift / scale。

2. **内参旋转**。图像会被旋转到立体匹配要求的朝向（见 common.py），内参必须
   跟着转，否则 fx 用错、主点偏移，深度整体错。

旧版把这两步写成 process_stereo_foundation.sh 里两段内联 heredoc，
旋转对应表还和别处重复。现在合并成一个脚本，旋转口径统一走 common.py。
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIRECTION_CHOICES, rotate_intrinsics, rotated_size,
                    rotation_for, write_intrinsic_file)


def main():
    ap = argparse.ArgumentParser(description="生成 FoundationStereo 的 K.txt")
    ap.add_argument("--transforms-json", required=True, type=Path,
                    help="数据集的 transforms.json（提供内参和图像尺寸）")
    ap.add_argument("--dataparser-transforms", required=True, type=Path,
                    help="训练输出目录里的 dataparser_transforms.json（提供 scale）")
    ap.add_argument("--shift", required=True, type=float, help="归一化坐标系下的基线")
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right（也接受 x/-x/y/-y）")
    ap.add_argument("--output", required=True, type=Path, help="最终 K.txt（旋转后）")
    ap.add_argument("--output-original", type=Path, help="可选：旋转前的 K，便于排查")
    args = ap.parse_args()

    data = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    missing = [k for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if data.get(k) is None]
    if missing:
        raise ValueError(f"{args.transforms_json} 缺少内参字段: {missing}")

    fl_x, fl_y = float(data["fl_x"]), float(data["fl_y"])
    cx, cy = float(data["cx"]), float(data["cy"])
    width, height = int(data["w"]), int(data["h"])

    K = np.array([[fl_x, 0.0, cx],
                  [0.0, fl_y, cy],
                  [0.0, 0.0, 1.0]])

    dp = json.loads(args.dataparser_transforms.read_text(encoding="utf-8"))
    scale = dp.get("scale")
    if scale is None:
        raise ValueError(f"{args.dataparser_transforms} 里没有 scale，无法把基线换算成真实尺度")
    scale = float(scale)

    # 基线是距离，恒为正；方向已经体现在位姿平移和旋转里了
    baseline = abs(args.shift / scale)

    rotation = rotation_for(args.shift_direction)
    K_rot = rotate_intrinsics(K, rotation, width, height)
    new_w, new_h = rotated_size(width, height, rotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_intrinsic_file(args.output, K_rot, baseline)
    if args.output_original:
        write_intrinsic_file(args.output_original, K, baseline)

    print(f"[make_intrinsics] 原始  {width}x{height}  "
          f"fx={fl_x:.3f} fy={fl_y:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"[make_intrinsics] 旋转  {rotation} -> {new_w}x{new_h}  "
          f"fx={K_rot[0,0]:.3f} fy={K_rot[1,1]:.3f} "
          f"cx={K_rot[0,2]:.3f} cy={K_rot[1,2]:.3f}")
    print(f"[make_intrinsics] 基线  {baseline:.6f} m  (shift {args.shift} / scale {scale})")
    print(f"[make_intrinsics] 写出  {args.output}")


if __name__ == "__main__":
    main()
