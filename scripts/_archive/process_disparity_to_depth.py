#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Disparity to Depth

This script:
1. Rotates disparity maps back to original orientation
2. Calculates depth maps using stereo camera formula
3. Saves depth maps as 16-bit PNG and NPY files
4. Updates transforms.json to include depth file paths

Usage:
    python process_disparity_to_depth.py [options]
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_shift_value(process_stereo_sh_path):
    """
    Get the DEFAULT_SHIFT value from process_stereo.sh script
    
    Args:
        process_stereo_sh_path: Path to the process_stereo.sh script
    
    Returns:
        float: DEFAULT_SHIFT value, or None if parsing fails
    """
    try:
        with open(process_stereo_sh_path, 'r') as f:
            for line in f:
                if line.strip().startswith('DEFAULT_SHIFT='):
                    # Extract the DEFAULT_SHIFT value
                    value_str = line.strip().split('=')[1].strip('"\'')
                    
                    # 确保能够处理字符串形式的浮点数
                    try:
                        return float(value_str)
                    except ValueError:
                        # 如果直接转换失败，尝试去除引号等字符再转换
                        cleaned_value = value_str.strip('"\'')
                        print(f"将字符串 '{value_str}' 转换为浮点数 '{cleaned_value}'")
                        return float(cleaned_value)
    except Exception as e:
        print(f"Could not parse DEFAULT_SHIFT from {process_stereo_sh_path}: {e}")
    return None

def get_scale_from_dataparser_transforms(dataparser_transforms_path):
    """
    Get the scale value from dataparser_transforms.json
    
    Args:
        dataparser_transforms_path: Path to dataparser_transforms.json
    
    Returns:
        float: Scale value, or 1.0 if parsing fails
    """
    try:
        with open(dataparser_transforms_path, 'r') as f:
            data = json.load(f)
        
        # Check if data is a list (multiple frames)
        if isinstance(data, list) and len(data) > 0:
            # Get scale from the first frame if it exists
            if 'scale' in data[0]:
                return float(data[0]['scale'])
            
            # Otherwise, calculate scale from transform matrix
            transform = data[0].get('transform', [])
            if transform and len(transform) >= 1 and len(transform[0]) >= 3:
                scale = np.sqrt(transform[0][0]**2 + transform[0][1]**2 + transform[0][2]**2)
                return float(scale)
        
        # If data is a dictionary (single frame)
        elif isinstance(data, dict):
            if 'scale' in data:
                return float(data['scale'])
    except Exception as e:
        print(f"Could not parse scale from {dataparser_transforms_path}: {e}")
    
    print(f"Using default scale value of 1.0")
    return 1.0

def get_camera_params_from_transforms(transforms_json_path):
    """
    Get camera parameters from transforms.json
    
    Args:
        transforms_json_path: Path to transforms.json
    
    Returns:
        dict: Camera parameters
    """
    try:
        with open(transforms_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract camera parameters
        camera_params = {
            'fl_x': data.get('fl_x', 0.0),
            'fl_y': data.get('fl_y', 0.0),
            'cx': data.get('cx', 0.0),
            'cy': data.get('cy', 0.0),
            'w': data.get('w', 0),
            'h': data.get('h', 0)
        }
        
        return camera_params
    except Exception as e:
        print(f"Could not parse camera parameters from {transforms_json_path}: {e}")
        return None

def rotate_disparity(disparity_path, output_path, rotation_type="90cc"):
    """
    Rotate a disparity map and save it
    
    Args:
        disparity_path: Input disparity map path (.npy file)
        output_path: Output disparity map path
        rotation_type: Type of rotation: "90cc" (90° counter-clockwise), 
                                         "90c" (90° clockwise),
                                         "180" (180 degrees)
    
    Returns:
        numpy.ndarray: Rotated disparity map
    """
    # Load the disparity map
    disparity = np.load(disparity_path)
    
    # Rotate the disparity map based on rotation type
    if rotation_type == "90c":
        # Rotate 90 degrees clockwise
        rotated_disparity = np.rot90(disparity, k=-1)
    elif rotation_type == "90cc":
        # Rotate 90 degrees counter-clockwise
        rotated_disparity = np.rot90(disparity, k=1)
    elif rotation_type == "180":
        # Rotate 180 degrees
        rotated_disparity = np.rot90(disparity, k=2)
    else:
        # No rotation
        rotated_disparity = disparity
    
    # Save the rotated disparity map
    np.save(output_path, rotated_disparity)
    
    return rotated_disparity

def disparity_to_depth(disparity, focal_length, baseline):
    """
    Convert disparity to depth using stereo camera formula
    
    Args:
        disparity: Disparity map
        focal_length: Focal length in pixels
        baseline: Baseline distance in world units
    
    Returns:
        numpy.ndarray: Depth map
    """
    # Avoid division by zero
    valid_mask = disparity > 0.01
    
    # Initialize depth map with zeros
    depth = np.zeros_like(disparity, dtype=np.float32)
    
    # Apply stereo formula: depth = focal_length * baseline / disparity
    depth[valid_mask] = focal_length * baseline / disparity[valid_mask]
    
    return depth

def save_depth_as_png(depth, output_path):
    """
    Save depth map as 16-bit PNG
    
    Args:
        depth: Depth map in meters
        output_path: Output path for PNG file
        scale_factor: Scale factor (default: 1000.0, but not used anymore)
    """
    # Map depth values directly to 16-bit range where 100 units = 65535
    # This means 1 unit of depth = 655.35 pixel value
    depth_scaled = (depth * 65.535).astype(np.uint16)
    
    # Clip values to ensure they don't exceed 65535 (which corresponds to depth of 100)
    depth_scaled = np.clip(depth_scaled, 0, 65535)
    
    # Save as 16-bit PNG
    cv2.imwrite(str(output_path), depth_scaled)

def update_transforms_json(transforms_json_path, frames_data):
    """
    Update transforms.json to include depth file paths
    
    Args:
        transforms_json_path: Path to transforms.json
        frames_data: List of dictionaries with frame_idx and depth_file_path
    """
    try:
        # Read existing transforms.json
        with open(transforms_json_path, 'r') as f:
            data = json.load(f)
        
        # Check if 'frames' exists in data
        if 'frames' not in data:
            print(f"Error: 'frames' not found in {transforms_json_path}")
            return False
        
        # Update frames with depth file paths
        for frame_info in frames_data:
            frame_idx = frame_info['frame_idx']
            depth_file_path = frame_info['depth_file_path']
            
            if frame_idx < len(data['frames']):
                data['frames'][frame_idx]['depth_file_path'] = depth_file_path
            else:
                print(f"Warning: Frame index {frame_idx} out of range")
        
        # Write updated transforms.json
        with open(transforms_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error updating transforms.json: {e}")
        return False

def process_disparity_file(disp_file, args, focal_length, baseline, clockwise):
    """
    Process a single disparity file
    
    Args:
        disp_file: Path to disparity file
        args: Command line arguments
        focal_length: Focal length in pixels
        baseline: Baseline distance in world units
        clockwise: Whether to rotate clockwise (for 90-degree rotations)
    
    Returns:
        dict: Frame data for transforms.json update
    """
    # Extract frame index from filename
    match = re.search(r'(\d+)', disp_file.stem)
    if not match:
        print(f"Warning: Could not extract frame index from {disp_file.name}")
        return None
    
    # Extract the frame index and convert to integer
    frame_idx = int(match.group(1))
    
    # Define output paths with frame_XXXX format to match original images
    rotated_disp_path = Path(args.rotated_disp_dir) / disp_file.name
    depth_npy_path = Path(args.depth_dir) / f"frame_{frame_idx:04d}_depth.npy"
    depth_png_path = Path(args.depth_dir) / f"frame_{frame_idx:04d}.png"
    
    # Process disparity map based on rotation flag
    if args.rotate_180:
        # Rotate 180 degrees for X-direction stereo
        disparity = rotate_disparity(disp_file, rotated_disp_path, rotation_type="180")
        print(f"Rotated disparity map for {disp_file.name} by 180 degrees")
    else:
        # Standard 90-degree rotation (Y-direction stereo)
        rotation_type = "90c" if not clockwise else "90cc"  # Inverse rotation
        disparity = rotate_disparity(disp_file, rotated_disp_path, rotation_type=rotation_type)
    
    # Convert disparity to depth
    depth = disparity_to_depth(disparity, focal_length, baseline)
    
    # Calculate depth statistics (excluding zeros)
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        min_depth = np.min(valid_depth)
        max_depth = np.max(valid_depth)
        median_depth = np.median(valid_depth)
        
        # Print depth statistics
        print(f"Depth statistics for frame_{frame_idx:04d}.png:")
        print(f"  - Min depth: {min_depth:.4f}")
        print(f"  - Max depth: {max_depth:.4f}")
        print(f"  - Median depth: {median_depth:.4f}")
        
        # Check if max depth exceeds 100
        if max_depth > 1000:
            print(f"  - WARNING: Maximum depth ({max_depth:.4f}) exceeds 1000!")
    else:
        print(f"No valid depth values found in frame_{frame_idx:04d}.png")
    
    # Save depth as NPY
    np.save(depth_npy_path, depth)
    
    # Save depth as 16-bit PNG
    save_depth_as_png(depth, depth_png_path)
    
    # Return frame data for transforms.json update (0-indexed for transforms.json)
    return {
        'frame_idx': frame_idx - 1,  # Convert to 0-indexed for transforms.json
        'depth_file_path': str(depth_png_path.relative_to(Path(args.transforms_json).parent))
    }

def calculate_depth_stats(depth_file):
    """
    Calculate depth statistics for a single depth file
    
    Args:
        depth_file: Path to depth file (.npy)
    
    Returns:
        dict: Dictionary with min, max, and median depth values
    """
    try:
        depth = np.load(depth_file)
        valid_depth = depth[depth > 0]
        
        if len(valid_depth) > 0:
            return {
                'file': depth_file.name,
                'min': float(np.min(valid_depth)),
                'max': float(np.max(valid_depth)),
                'median': float(np.median(valid_depth))
            }
        else:
            return {
                'file': depth_file.name,
                'min': None,
                'max': None,
                'median': None
            }
    except Exception as e:
        print(f"Error processing depth file {depth_file}: {e}")
        return {
            'file': depth_file.name,
            'min': None,
            'max': None,
            'median': None
        }

def process_disparity_files(args):
    """
    Process all disparity files
    
    Args:
        args: Command line arguments
    """
    # Get shift value from process_stereo.sh
    shift_value = get_shift_value(args.process_stereo_sh)
    if shift_value is None:
        print("Could not get shift value, using default value of 0.7")
        shift_value = 0.7
    
    # Determine rotation direction based on shift value
    clockwise = shift_value < 0
    
    # Check rotation type
    if args.rotate_180:
        print("Using 180-degree rotation for X-direction stereo images")
    else:
        rotation_direction = "clockwise" if clockwise else "counter-clockwise"
        print(f"Shift value is {shift_value}, will rotate disparity maps 90 degrees {rotation_direction}")
    
    # Get scale from dataparser_transforms.json
    scale = get_scale_from_dataparser_transforms(args.dataparser_transforms)
    print(f"Scale value from dataparser_transforms.json: {scale}")
    
    # Calculate absolute baseline
    baseline = abs(shift_value) / scale
    print(f"Calculated baseline: {baseline}")
    
    # Get camera parameters from transforms.json
    camera_params = get_camera_params_from_transforms(args.transforms_json)
    if camera_params is None:
        print("Could not get camera parameters, exiting")
        return
    
    print(f"Camera parameters: {camera_params}")
    focal_length = camera_params['fl_x']
    
    # Create output directories
    os.makedirs(args.rotated_disp_dir, exist_ok=True)
    os.makedirs(args.depth_dir, exist_ok=True)
    
    # Get all disparity files
    disp_dir = Path(args.disp_dir)
    disp_files = list(disp_dir.glob("*.npy"))
    
    if not disp_files:
        print(f"No disparity files found in {args.disp_dir}")
        return
    
    print(f"Found {len(disp_files)} disparity files")
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Process each disparity file in parallel
    frames_data = []
    
    # Initialize lists to store depth statistics
    all_min_depths = []
    all_max_depths = []
    all_median_depths = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                process_disparity_file, 
                disp_file, 
                args, 
                focal_length, 
                baseline, 
                clockwise
            ): disp_file.name for disp_file in disp_files
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing disparity files"):
            result = future.result()
            if result:
                frames_data.append(result)
    
    # Update transforms.json
    if frames_data:
        success = update_transforms_json(args.transforms_json, frames_data)
        if success:
            print(f"Updated transforms.json with {len(frames_data)} depth file paths")
        else:
            print("Failed to update transforms.json")
    
    # Calculate and print overall depth statistics
    print("\nProcessing completed!")
    
    # Load all depth maps to calculate overall statistics
    depth_files = list(Path(args.depth_dir).glob("*_depth.npy"))
    if depth_files:
        print(f"Calculating depth statistics for {len(depth_files)} depth files...")
        
        # Process depth files in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks
            futures = [executor.submit(calculate_depth_stats, depth_file) for depth_file in depth_files]
            
            # Process results as they complete
            depth_stats = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating depth statistics"):
                result = future.result()
                if result and result['min'] is not None:
                    depth_stats.append(result)
        
        # Aggregate statistics
        if depth_stats:
            all_min_depths = [stat['min'] for stat in depth_stats]
            all_max_depths = [stat['max'] for stat in depth_stats]
            all_median_depths = [stat['median'] for stat in depth_stats]
            
            print("\nOverall depth statistics:")
            print(f"  - Min depth (across all frames): {min(all_min_depths):.4f}")
            print(f"  - Max depth (across all frames): {max(all_max_depths):.4f}")
            print(f"  - Average min depth: {np.mean(all_min_depths):.4f}")
            print(f"  - Average max depth: {np.mean(all_max_depths):.4f}")
            print(f"  - Average median depth: {np.mean(all_median_depths):.4f}")
            
            # Check if max depth exceeds 100
            if max(all_max_depths) > 1000:
                print(f"\nWARNING: Maximum depth ({max(all_max_depths):.4f}) exceeds 1000!")
                print("This may cause issues with the depth scaling in the PNG files.")
                print("Consider adjusting the baseline or scale factor.")
        else:
            print("No valid depth statistics found.")

def main():
    parser = argparse.ArgumentParser(description='Process disparity maps to depth maps')
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input/output directories
    parser.add_argument('--disp-dir', type=str, default=os.path.join(script_dir, 'stereo/disp'),
                        help='Directory containing disparity maps (.npy files)')
    parser.add_argument('--rotated-disp-dir', type=str, default=os.path.join(script_dir, 'stereo/disp_rotated'),
                        help='Directory to save rotated disparity maps')
    parser.add_argument('--depth-dir', type=str, default=os.path.join(script_dir, 'stereo/depth'),
                        help='Directory to save depth maps')
    
    # Configuration files
    parser.add_argument('--process-stereo-sh', type=str, default=os.path.join(script_dir, 'process_stereo.sh'),
                        help='Path to process_stereo.sh for DEFAULT_SHIFT value')
    parser.add_argument('--transforms-json', type=str, default=os.path.join(script_dir, 'transforms.json'),
                        help='Path to transforms.json for camera parameters and updating with depth paths')
    
    # Get config path from process_stereo.sh
    process_stereo_sh_path = os.path.join(script_dir, 'process_stereo.sh')
    config_path = None
    try:
        with open(process_stereo_sh_path, 'r') as f:
            for line in f:
                if line.strip().startswith('DEFAULT_CONFIG_PATH='):
                    config_path = line.strip().split('=')[1].strip('"\'')
                    break
    except:
        pass
    
    # Set default dataparser_transforms.json path
    default_dataparser_path = os.path.join(script_dir, 'outputs/clinical_3_testscript/nerfacto/2025-03-10_180503/dataparser_transforms.json')
    if config_path:
        config_dir = os.path.dirname(config_path)
        default_dataparser_path = os.path.join(config_dir, 'dataparser_transforms.json')
    
    parser.add_argument('--dataparser-transforms', type=str, default=default_dataparser_path,
                        help='Path to dataparser_transforms.json for scale value')
    parser.add_argument('--rotate-180', action='store_true',
                        help='Rotate disparity maps 180 degrees (use when shift direction is X)')
    
    args = parser.parse_args()
    
    # Process disparity files
    process_disparity_files(args)

if __name__ == "__main__":
    main() 