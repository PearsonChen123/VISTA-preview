#!/usr/bin/env python3
"""
将 depth_500 的GT深度图转换为TUM格式
Redwood格式: 16位PNG，像素值直接以毫米为单位
TUM格式: 16位PNG，像素值 / 5000 = 深度（米）
转换: TUM像素值 = Redwood像素值 * 5
"""
import os
import cv2
import numpy as np
from pathlib import Path

# 源目录和目标目录
source_dir = Path("/mnt/h/RGBD-500/depth_500")
target_dir = Path("/mnt/h/RGBD-500/BAD_SLAM/depth")
depth_txt_file = Path("/mnt/h/RGBD-500/BAD_SLAM/depth.txt")

# 创建目标目录
target_dir.mkdir(parents=True, exist_ok=True)

# 获取所有深度图文件并排序
depth_files = sorted(source_dir.glob("*.png"))

print(f"找到 {len(depth_files)} 个深度图文件")
print(f"从 {source_dir} 转换并保存到 {target_dir}")
print(f"深度格式转换: Redwood (mm) -> TUM (pixel_value / 5000 = meters)")

# 转换所有深度图文件，重新编号从000000开始
for i, depth_file in enumerate(depth_files):
    # 读取Redwood深度图 (16位PNG，像素值为毫米)
    redwood_depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    
    if redwood_depth is None:
        print(f"警告: 无法读取 {depth_file.name}")
        continue
    
    # 转换为TUM格式
    # Redwood: 像素值(mm) -> TUM: 像素值 / 5000 = 米
    # 所以: TUM像素值 = Redwood像素值(mm) / 1000 * 5000 = Redwood像素值 * 5
    tum_depth = redwood_depth.astype(np.float32) * 5.0
    
    # 保持无效深度值（0）不变
    tum_depth[redwood_depth == 0] = 0
    
    # 转换为16位整数
    tum_depth = np.clip(tum_depth, 0, 65535).astype(np.uint16)
    
    # 使用新的文件名（从000000开始）
    new_filename = f"{i:06d}.png"
    target_file = target_dir / new_filename
    cv2.imwrite(str(target_file), tum_depth)
    
    if i % 50 == 0:
        print(f"已处理 {i + 1}/{len(depth_files)} 个文件... (示例: {redwood_depth[redwood_depth>0].mean():.1f}mm -> {tum_depth[tum_depth>0].mean():.1f} TUM单位)")

print(f"\n所有深度图已转换并保存到 {target_dir}")

# 生成 depth.txt 文件
# 根据 rgb.txt 的格式,时间戳从 0 开始,增量为 0.033333 (30fps)
with open(depth_txt_file, 'w') as f:
    f.write("# Depth images\n")
    f.write("# timestamp filename\n")
    
    for i, depth_file in enumerate(depth_files):
        # 提取原始文件编号用于计算时间戳
        file_num = int(depth_file.stem)
        # 计算时间戳 (假设30fps)
        timestamp = file_num / 30.0
        # 使用新的文件名（从000000开始）
        new_filename = f"{i:06d}.png"
        f.write(f"{timestamp:.6f} depth/{new_filename}\n")

print(f"depth.txt 文件已生成: {depth_txt_file}")
print(f"共 {len(depth_files)} 条记录")
print("\n转换完成!")

