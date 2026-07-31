#!/usr/bin/env python3
"""
将RGB-D数据转换为TUM格式
- RGB: /mnt/h/RGBD-500/images
- Depth: /mnt/h/RGBD-500/stereo/depth (NPY格式，真值)
- 输出: /mnt/h/RGBD-500/BAD-SLAM_0.1
"""
import numpy as np
import cv2
import shutil
from pathlib import Path

# 路径配置
rgb_dir = Path("/mnt/h/RGBD-500/images")
depth_dir = Path("/mnt/h/RGBD-500/stereo/depth_monster")
output_dir = Path("/mnt/h/RGBD-500/BAD-SLAM_monster")

# 创建输出目录结构
rgb_output = output_dir / "rgb"
depth_output = output_dir / "depth"
rgb_output.mkdir(parents=True, exist_ok=True)
depth_output.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("转换RGB-D数据到TUM格式")
print("=" * 70)
print(f"RGB源目录: {rgb_dir}")
print(f"深度源目录: {depth_dir}")
print(f"输出目录: {output_dir}")
print()

# 1. 获取所有RGB文件
rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
if not rgb_files:
    print("错误: 找不到RGB图像！")
    exit(1)

print(f"找到 {len(rgb_files)} 张RGB图像")

# 2. 获取所有深度文件（NPY格式）
depth_files = sorted(depth_dir.glob("*.npy"))
if not depth_files:
    print("错误: 找不到深度文件（.npy）！")
    exit(1)

print(f"找到 {len(depth_files)} 个深度文件")
print()

# 检查数量是否匹配
if len(rgb_files) != len(depth_files):
    print(f"警告: RGB数量({len(rgb_files)})和深度数量({len(depth_files)})不匹配")
    # 取较小值
    num_frames = min(len(rgb_files), len(depth_files))
    print(f"将处理前 {num_frames} 帧")
else:
    num_frames = len(rgb_files)

print(f"注意: 每张深度图将只保留前50%最浅的部分作为有效值")
print()

print("=" * 70)
print("阶段1: 转换RGB图像")
print("=" * 70)
print()

rgb_entries = []
for i in range(num_frames):
    rgb_file = rgb_files[i]
    
    # 复制RGB图像（统一命名为6位数字）
    output_name = f"{i:06d}.png"
    output_path = rgb_output / output_name
    
    # 如果是jpg，转换为png
    if rgb_file.suffix.lower() == '.jpg':
        img = cv2.imread(str(rgb_file))
        cv2.imwrite(str(output_path), img)
    else:
        shutil.copy2(rgb_file, output_path)
    
    # 生成时间戳（假设30fps）
    timestamp = i / 30.0
    rgb_entries.append((timestamp, f"rgb/{output_name}"))
    
    if i % 50 == 0:
        print(f"  处理 {i+1}/{num_frames}: {rgb_file.name} -> {output_name}")

print(f"✓ 完成 {len(rgb_entries)} 张RGB图像")

print()
print("=" * 70)
print("阶段2: 转换深度图（NPY -> PNG TUM格式）")
print("=" * 70)
print()

depth_entries = []
depth_stats = []

for i in range(num_frames):
    depth_file = depth_files[i]
    
    # 读取NPY深度图
    try:
        depth = np.load(str(depth_file))
    except Exception as e:
        print(f"错误: 无法读取 {depth_file.name}: {e}")
        continue
    
    # 检查深度图格式并转换为浮点（米）
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        # 如果已经是整数格式，转换为浮点进行处理
        if depth.dtype == np.uint16:
            # 假设是TUM格式，转回米
            depth_float = depth.astype(np.float32) / 5000.0
        else:
            depth_float = depth.astype(np.float32)
    else:
        # 已经是浮点格式（米）
        depth_float = depth.astype(np.float32)

    # 只保留前50%最浅的部分
    # 计算50%分位数，只保留深度值小于等于该分位数的像素
    valid_mask_initial = depth_float > 0
    if valid_mask_initial.sum() > 0:
        valid_depths = depth_float[valid_mask_initial]
        depth_p50 = np.percentile(valid_depths, 100)

        # 创建过滤mask：只保留前50%最浅的部分
        percentile_mask = valid_mask_initial & (depth_float <= depth_p50)

        # 将深度值大于50%分位数的像素设为0（无效）
        depth_filtered = np.copy(depth_float)
        depth_filtered[~percentile_mask] = 0

        # 统计过滤效果
        filtered_count = percentile_mask.sum()
        original_count = valid_mask_initial.sum()
    else:
        depth_filtered = depth_float
        percentile_mask = valid_mask_initial
        filtered_count = 0
        original_count = 0

    # 转换为TUM格式
    # TUM格式: 像素值 / 5000 = 深度（米）
    # 所以: 像素值 = 深度（米）* 5000
    depth_tum = (depth_filtered * 5000.0).astype(np.uint16)

    # 统计信息（使用过滤后的mask）
    valid_mask = depth_tum > 0
    if valid_mask.sum() > 0:
        depth_min = depth_tum[valid_mask].min()
        depth_max = depth_tum[valid_mask].max()
        depth_mean = depth_tum[valid_mask].mean()
        depth_stats.append({
            'min': depth_min,
            'max': depth_max,
            'mean': depth_mean
        })
    
    # 保存为PNG
    output_name = f"{i:06d}.png"
    output_path = depth_output / output_name
    cv2.imwrite(str(output_path), depth_tum)
    
    # 生成时间戳
    timestamp = i / 30.0
    depth_entries.append((timestamp, f"depth/{output_name}"))
    
    if i % 50 == 0:
        if valid_mask.sum() > 0:
            print(f"  处理 {i+1}/{num_frames}: {depth_file.name}")
            print(f"    深度范围: {depth_min/5000:.3f}~{depth_max/5000:.3f}m, "
                  f"平均: {depth_mean/5000:.3f}m")
            if original_count > 0:
                filter_ratio = filtered_count / original_count * 100
                print(f"    保留前50%最浅: {original_count} -> {filtered_count} 像素 ({filter_ratio:.1f}% 保留)")
        else:
            print(f"  处理 {i+1}/{num_frames}: {depth_file.name} (无有效深度)")

print(f"✓ 完成 {len(depth_entries)} 张深度图")

# 深度统计
if depth_stats:
    all_mins = [s['min'] for s in depth_stats]
    all_maxs = [s['max'] for s in depth_stats]
    all_means = [s['mean'] for s in depth_stats]
    print()
    print("深度统计 (TUM单位):")
    print(f"  最小值范围: {min(all_mins):.1f} ~ {max(all_mins):.1f}")
    print(f"  最大值范围: {min(all_maxs):.1f} ~ {max(all_maxs):.1f}")
    print(f"  平均值范围: {min(all_means):.1f} ~ {max(all_means):.1f}")
    print()
    print("深度统计 (米):")
    print(f"  最小值范围: {min(all_mins)/5000:.3f} ~ {max(all_mins)/5000:.3f}m")
    print(f"  最大值范围: {min(all_maxs)/5000:.3f} ~ {max(all_maxs)/5000:.3f}m")
    print(f"  平均值范围: {min(all_means)/5000:.3f} ~ {max(all_means)/5000:.3f}m")

print()
print("=" * 70)
print("阶段3: 生成TUM格式配置文件")
print("=" * 70)
print()

# 3. 生成 rgb.txt
rgb_txt = output_dir / "rgb.txt"
with open(rgb_txt, 'w') as f:
    f.write("# RGB images\n")
    f.write("# timestamp filename\n")
    for timestamp, filename in rgb_entries:
        f.write(f"{timestamp:.6f} {filename}\n")
print(f"✓ 生成 rgb.txt ({len(rgb_entries)} 条记录)")

# 4. 生成 depth.txt
depth_txt = output_dir / "depth.txt"
with open(depth_txt, 'w') as f:
    f.write("# Depth images\n")
    f.write("# timestamp filename\n")
    for timestamp, filename in depth_entries:
        f.write(f"{timestamp:.6f} {filename}\n")
print(f"✓ 生成 depth.txt ({len(depth_entries)} 条记录)")

# 5. 生成 associated.txt
associated_txt = output_dir / "associated.txt"
with open(associated_txt, 'w') as f:
    f.write("# Associated RGB and Depth images\n")
    f.write("# timestamp_rgb rgb_file timestamp_depth depth_file\n")
    for i in range(min(len(rgb_entries), len(depth_entries))):
        ts_rgb, rgb_file = rgb_entries[i]
        ts_depth, depth_file = depth_entries[i]
        f.write(f"{ts_rgb:.6f} {rgb_file} {ts_depth:.6f} {depth_file}\n")
print(f"✓ 生成 associated.txt ({min(len(rgb_entries), len(depth_entries))} 条记录)")

# 6. 生成 calibration.txt（需要相机内参）
# 这里使用默认的Redwood参数，如果有实际参数请修改
calibration_txt = output_dir / "calibration.txt"
with open(calibration_txt, 'w') as f:
    # 格式: fx fy cx cy
    # 默认使用640x480分辨率的典型参数
    f.write("525.0 525.0 319.5 239.5\n")
    f.write("640 480\n")
    f.write("0.0002\n")  # depth scale factor (如果需要的话)
print(f"✓ 生成 calibration.txt (使用默认参数)")
print("  注意: 请根据实际相机参数修改calibration.txt")

print()
print("=" * 70)
print("转换完成!")
print("=" * 70)
print()
print(f"输出目录: {output_dir}")
print(f"  - rgb/        : {len(rgb_entries)} 张RGB图像")
print(f"  - depth/      : {len(depth_entries)} 张深度图")
print(f"  - rgb.txt     : RGB时间戳文件")
print(f"  - depth.txt   : 深度时间戳文件")
print(f"  - associated.txt : 关联文件")
print(f"  - calibration.txt : 相机参数（请检查并修改）")
print()
print("可以直接用于BAD SLAM!")
