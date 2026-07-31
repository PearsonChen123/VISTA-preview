#!/usr/bin/env python3
"""
将深度图除以1.8进行scale校正
"""
import cv2
import numpy as np
from pathlib import Path
import shutil

# 深度目录
depth_dir = Path("/mnt/h/RGBD-500/BAD_SLAM/depth")
backup_dir = Path("/mnt/h/RGBD-500/BAD_SLAM/depth_backup")

# 备份原始数据
if not backup_dir.exists():
    print("备份原始深度图到 depth_backup...")
    shutil.copytree(depth_dir, backup_dir)
    print(f"备份完成: {backup_dir}")
else:
    print(f"备份目录已存在: {backup_dir}")

print()

# 获取所有深度图
depth_files = sorted(depth_dir.glob("*.png"))
print(f"找到 {len(depth_files)} 个深度图文件")
print("开始scale校正: 深度值 / 1.8")
print()

scale_factor = 1.8

# 处理所有深度图
for i, depth_file in enumerate(depth_files):
    # 读取深度图
    depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        print(f"警告: 无法读取 {depth_file.name}")
        continue
    
    # 记录原始统计
    valid_mask = depth > 0
    if i == 0:
        orig_mean = depth[valid_mask].mean() if valid_mask.sum() > 0 else 0
    
    # 应用scale校正
    scaled_depth = depth.astype(np.float32) / scale_factor
    
    # 保持无效值（0）不变
    scaled_depth[depth == 0] = 0
    
    # 转换回uint16
    scaled_depth = np.clip(scaled_depth, 0, 65535).astype(np.uint16)
    
    # 保存
    cv2.imwrite(str(depth_file), scaled_depth)
    
    # 显示进度
    if i % 100 == 0:
        if valid_mask.sum() > 0:
            new_mean = scaled_depth[valid_mask].mean()
            print(f"已处理 {i + 1}/{len(depth_files)} 个文件... (示例: {depth[valid_mask].mean():.1f} -> {new_mean:.1f})")
        else:
            print(f"已处理 {i + 1}/{len(depth_files)} 个文件...")

print()
print("=" * 60)
print("Scale校正完成!")
print("=" * 60)

# 验证结果
print("\n验证几个样本:")
for idx in [0, len(depth_files)//2, -1]:
    depth_file = depth_files[idx]
    depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    valid_mask = depth > 0
    if valid_mask.sum() > 0:
        avg_value = depth[valid_mask].mean()
        avg_meters = avg_value / 5000.0
        print(f"  {depth_file.name}: 平均值={avg_value:.1f}, 约{avg_meters:.3f}米")

print(f"\n原始数据已备份至: {backup_dir}")
print(f"校正后的数据位于: {depth_dir}")
