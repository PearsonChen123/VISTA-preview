#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import re
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import argparse

def rotate_depth(depth, rotation_type="90cc"):
    """Rotate a depth map"""
    if rotation_type == "90c":
        # Rotate 90 degrees clockwise
        return np.rot90(depth, k=-1)
    elif rotation_type == "90cc":
        # Rotate 90 degrees counter-clockwise
        return np.rot90(depth, k=1)
    elif rotation_type == "180":
        # Rotate 180 degrees
        return np.rot90(depth, k=2)
    else:
        # No rotation
        return depth

def save_depth_as_png(depth, output_path):
    """Save depth map as 16-bit PNG for nerfstudio compatibility"""
    # 确保深度值为浮点数
    depth = depth.astype(np.float32)
    # 计算每个单位深度对应的像素值
    scaling_factor = 65535.0 / 400.0  # 32.7675
    # Map depth values directly to 16-bit range where 1000 units = 65535
    depth_scaled = (depth * scaling_factor).astype(np.uint16)
    # Clip values to ensure they don't exceed 65535 (which corresponds to depth of 1000)
    depth_scaled = np.clip(depth_scaled, 0, 65535)
    # Save as 16-bit PNG
    cv2.imwrite(str(output_path), depth_scaled)
    return depth_scaled

def process_depth_file(disp_file, depth_dir, rotation_type, index):
    """Process a single disparity file: rotate, create PNG version and determine frame index"""
    try:
        # Original file path
        file_path = Path(disp_file)
        filename = file_path.stem
        try:
            frame_idx = int(filename)
        except ValueError:
            frame_idx = index

        print(f"\n===== 处理视差图文件: {file_path.name} (帧 {frame_idx}) =====")

        # Load depth map from disparity directory
        depth = np.load(file_path)

        # Create output path in depth directory
        depth_npy_path = Path(depth_dir) / file_path.name

        # Rotate depth map based on rotation type
        rotated_depth = rotate_depth(depth, rotation_type)

        # 打印旋转后深度图统计信息
        valid_depth = rotated_depth[(rotated_depth > 0) & (np.isfinite(rotated_depth))]
        if len(valid_depth) > 0:
            min_depth = np.min(valid_depth)
            max_depth = np.max(valid_depth)
            median_depth = np.median(valid_depth)
            print(f"【旋转后深度图统计】")
            print(f"  - 最小深度: {min_depth:.4f}")
            print(f"  - 最大深度: {max_depth:.4f}")
            print(f"  - 中值深度: {median_depth:.4f}")
        else:
            print(f"警告：在旋转后的深度图中没有找到有效的深度值！")

        # Save rotated depth to depth directory
        np.save(depth_npy_path, rotated_depth)

        # Create PNG filename
        png_path = depth_npy_path.with_suffix('.png')

        # Save as PNG for nerfstudio compatibility
        save_depth_as_png(rotated_depth, png_path)

        # Return information for transforms.json update
        return {
            'frame_idx': frame_idx,
            'depth_file_path': png_path.name
        }
    except Exception as e:
        print(f"处理视差图文件时出错 {disp_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_transforms_json(transforms_json_path, depth_dir_path, frames_data):
    """Update transforms.json to include depth file paths"""
    try:
        # Read existing transforms.json
        with open(transforms_json_path, 'r') as f:
            data = json.load(f)

        if 'frames' not in data:
            print(f"Error: 'frames' not found in {transforms_json_path}")
            return False

        # Get transforms.json directory and depth directory for relative path calculation
        transforms_dir = os.path.abspath(os.path.dirname(transforms_json_path))
        depth_dir = os.path.abspath(depth_dir_path)
        rel_depth_dir = os.path.relpath(depth_dir, transforms_dir)

        # Update frames with depth file paths
        for frame_info in frames_data:
            frame_idx = frame_info['frame_idx']
            depth_file_name = frame_info['depth_file_path']
            depth_file_path = os.path.join(rel_depth_dir, depth_file_name)
            depth_file_path = depth_file_path.replace('\\\\', '/')

            if frame_idx < len(data['frames']):
                data['frames'][frame_idx]['depth_file_path'] = depth_file_path
            else:
                print(f"Warning: Frame index {frame_idx} out of range (max: {len(data['frames'])-1})")

        # Write updated transforms.json
        with open(transforms_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Updated transforms.json with depth file paths for {len(frames_data)} frames")
        return True
    except Exception as e:
        print(f"Error updating transforms.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    # Determine rotation type based on shift direction
    if args.shift_direction.lower() == "x":
        rotation_type = "180"
    elif args.shift_direction.lower() == "-x":
        rotation_type = "none" # No rotation for -x direction
    elif args.shift_direction.lower() == "y":
        rotation_type = "90c" # Clockwise rotation for y-direction
    elif args.shift_direction.lower() == "-y":
        rotation_type = "90cc" # Counter-clockwise rotation for -y direction
    else:
        rotation_type = "90c" # Default to clockwise

    # Create depth directory if it doesn't exist
    os.makedirs(args.depth_dir, exist_ok=True)

    # Get all depth files from the disparity directory
    disp_dir = Path(args.disp_dir)
    disp_files = list(disp_dir.glob("*.npy"))
    disp_files = sorted(disp_files)

    if not disp_files:
        print(f"No disparity files found in {args.disp_dir}")
        return

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Process each depth file in parallel
    processed_frames = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i, disp_file in enumerate(disp_files):
            futures.append(executor.submit(process_depth_file, str(disp_file), args.depth_dir, rotation_type, i))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing disparity files"):
            result = future.result()
            if result:
                processed_frames.append(result)

    # Update transforms.json with depth file paths
    update_transforms_json(args.transforms_json, args.depth_dir, processed_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process depth maps: rotate, convert to PNG, update transforms.json.")
    parser.add_argument("--disp-dir", required=True, help="Directory containing input disparity .npy files.")
    parser.add_argument("--depth-dir", required=True, help="Directory to save processed depth maps (.npy and .png).")
    parser.add_argument("--shift-direction", required=True, choices=['x', '-x', 'y', '-y'], help="Stereo shift direction ('x', '-x', 'y', or '-y') to determine rotation.")
    parser.add_argument("--transforms-json", required=True, help="Path to the transforms.json file to update.")

    args = parser.parse_args()

    main(args) 