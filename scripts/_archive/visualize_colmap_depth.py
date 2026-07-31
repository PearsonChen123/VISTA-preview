#!/usr/bin/env python3
"""
读取COLMAP Dense深度图并转换为TUM格式进行可视化
- 输入: /mnt/h/RGBD-500/undistorted/dense/stereo/depth_maps/*.geometric.bin (COLMAP Dense格式)
- 处理: 保留5%-95%百分位的深度值
- 输出: /mnt/h/RGBD-500/undistorted/dense/stereo/depth_maps/vis/*.png (TUM格式)
"""
import numpy as np
import cv2
from pathlib import Path
import struct


def read_colmap_array(path):
    """
    读取COLMAP的二进制深度图文件
    格式: ASCII头部 "width&height&channels&" + float32数据
    """
    with open(path, "rb") as f:
        # 读取header，找到"width&height&channels&"
        header_bytes = b''
        while True:
            byte = f.read(1)
            if not byte:
                break
            header_bytes += byte
            # 找到第三个&后停止
            if header_bytes.count(b'&') == 3:
                break
        
        # 解析头部
        header = header_bytes.decode('ascii')
        parts = header.split('&')
        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])
        
        # 读取剩余的所有float32数据
        data = np.fromfile(f, dtype=np.float32)
        
        # 重塑为图像形状
        expected_size = width * height * channels
        if len(data) == expected_size:
            array = data.reshape((height, width, channels))
        else:
            raise ValueError(f"数据大小不匹配: 预期{expected_size}个元素，实际{len(data)}个")
        
        # 如果只有一个通道，压缩维度
        if channels == 1:
            array = array[:, :, 0]
        
        return array, width, height


# 路径配置
depth_dir = Path("/mnt/h/RGBD-500/undistorted/dense/stereo/depth_maps")
output_dir = depth_dir / "vis"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("COLMAP Dense深度图转换为TUM格式（5%-95%百分位过滤）")
print("=" * 70)
print(f"输入目录: {depth_dir}")
print(f"输出目录: {output_dir}")
print()

# 获取所有geometric深度图文件
depth_files = sorted(depth_dir.glob("*.geometric.bin"))

if not depth_files:
    print("错误: 找不到COLMAP深度图文件！")
    exit(1)

print(f"找到 {len(depth_files)} 个COLMAP深度图")
print()

# 统计信息
depth_stats = []

print("开始转换...")
print()

for i, depth_file in enumerate(depth_files):
    try:
        # 读取COLMAP深度图
        depth_array, width, height = read_colmap_array(str(depth_file))
        
        # COLMAP深度图已经是米单位的浮点数
        depth_float = depth_array.astype(np.float32)
        
        # 只保留5%-95%百分位的深度
        valid_mask = depth_float > 0
        if valid_mask.sum() > 0:
            valid_depths = depth_float[valid_mask]
            depth_p5 = np.percentile(valid_depths, 5)
            depth_p95 = np.percentile(valid_depths, 95)
            
            # 创建过滤mask：只保留5%-95%百分位的深度
            percentile_mask = valid_mask & (depth_float >= depth_p5) & (depth_float <= depth_p95)
            
            # 将不在范围内的像素设为0（无效）
            depth_filtered = np.copy(depth_float)
            depth_filtered[~percentile_mask] = 0
        else:
            depth_filtered = depth_float
        
        # 转换为TUM格式
        # TUM格式: 像素值 / 5000 = 深度（米）
        # 所以: 像素值 = 深度（米）* 5000
        depth_tum = (depth_filtered * 5000.0).astype(np.uint16)
        
        # 统计信息
        valid_mask = depth_tum > 0
        if valid_mask.sum() > 0:
            depth_min = depth_tum[valid_mask].min()
            depth_max = depth_tum[valid_mask].max()
            depth_mean = depth_tum[valid_mask].mean()
            valid_ratio = valid_mask.sum() / (width * height) * 100
            
            depth_stats.append({
                'min': depth_min,
                'max': depth_max,
                'mean': depth_mean,
                'valid_ratio': valid_ratio
            })
        else:
            valid_ratio = 0.0
        
        # 提取文件名（去掉.geometric.bin）
        base_name = depth_file.name.replace('.geometric.bin', '')
        output_name = f"{base_name}.png"
        output_path = output_dir / output_name
        
        # 保存为PNG (TUM格式)
        cv2.imwrite(str(output_path), depth_tum)
        
        if i % 10 == 0 or i < 5:
            print(f"  [{i+1}/{len(depth_files)}] {depth_file.name}")
            print(f"    分辨率: {width}x{height}")
            if valid_mask.sum() > 0:
                print(f"    深度范围(过滤后5%-95%): {depth_min/5000:.3f}~{depth_max/5000:.3f}m, "
                      f"平均: {depth_mean/5000:.3f}m")
                print(f"    有效像素: {valid_ratio:.1f}%")
            else:
                print(f"    无有效深度")
            print()
    
    except Exception as e:
        print(f"  错误: 无法处理 {depth_file.name}: {e}")
        continue

print()
print("=" * 70)
print("转换完成!")
print("=" * 70)
print()

# 全局统计
if depth_stats:
    all_mins = [s['min'] for s in depth_stats]
    all_maxs = [s['max'] for s in depth_stats]
    all_means = [s['mean'] for s in depth_stats]
    all_valid_ratios = [s['valid_ratio'] for s in depth_stats]
    
    print(f"成功转换: {len(depth_stats)}/{len(depth_files)} 个深度图")
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
    print(f"有效像素比例: {min(all_valid_ratios):.1f}% ~ {max(all_valid_ratios):.1f}% "
          f"(平均: {np.mean(all_valid_ratios):.1f}%)")
else:
    print("警告: 没有成功转换任何深度图")

print()
print(f"输出目录: {output_dir}")
print("TUM格式深度图已保存（5%-95%百分位过滤），可用于可视化或SLAM系统！")
print()

