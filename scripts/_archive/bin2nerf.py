#!/usr/bin/env python3
"""
Convert COLMAP cameras.bin to text format and generate NeRF transforms.json
"""

import os
import json
import argparse
import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import subprocess

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """Read the next bytes from a binary file"""
    data = fid.read(num_bytes)
    return struct.unpack(format_char_sequence, data)

def read_cameras_binary(path_to_model_file: str) -> Dict[int, Camera]:
    """Read COLMAP's cameras.bin file"""
    cameras = {}
    
    # COLMAP camera models and their parameter counts
    camera_models = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12)
    }
    
    with open(path_to_model_file, "rb") as fid:
        num_cameras, = read_next_bytes(fid, 8, "Q")
        print(f"\nFound {num_cameras} cameras in the binary file")
        
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            if model_id not in camera_models:
                print(f"WARNING: Camera model {model_id} not recognized!")
                continue
                
            model_name, num_params = camera_models[model_id]
            params = read_next_bytes(fid, num_bytes=8*num_params, 
                                   format_char_sequence="d" * num_params)
            
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params)
            )
            
            print(f"\nCamera {camera_id}:")
            print(f"Model: {model_name}")
            print(f"Resolution: {width}x{height}")
            print("Parameters:", params)
    
    return cameras

def write_cameras_text(cameras: Dict[int, Camera], output_path: str) -> None:
    """Write cameras to a text file in COLMAP format"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras)))
        
        for camera_id, camera in cameras.items():
            # Format parameters as space-separated values
            params_str = " ".join([str(p) for p in camera.params])
            
            # Write camera line: id, model, width, height, params
            f.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")
    
    print(f"Camera parameters written to text file: {output_path}")

def run_colmap2nerf(images_dir: str, text_dir: str, output_path: str) -> None:
    """Run colmap2nerf.py to generate transforms.json"""
    
    # Check if colmap2nerf.py exists in the current directory
    colmap2nerf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colmap2nerf.py")
    
    if not os.path.exists(colmap2nerf_path):
        print(f"ERROR: colmap2nerf.py not found at {colmap2nerf_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run colmap2nerf.py
    cmd = [
        "python", 
        colmap2nerf_path,
        "--images", images_dir,
        "--text", text_dir,
        "--out", output_path,
        "--keep_colmap_coords"  # 保持COLMAP原始坐标系，跳过标准化
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated transforms.json at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running colmap2nerf.py: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP cameras.bin to text format and generate NeRF transforms.json")
    parser.add_argument("--cameras-bin", type=str, 
                        default="./undistorted/sparse/cameras.bin",
                        help="Path to COLMAP cameras.bin file")
    parser.add_argument("--images-dir", type=str, 
                        default="./undistorted/images",
                        help="Directory containing downsampled images")
    parser.add_argument("--text-dir", type=str, 
                        default="./undistorted/text",
                        help="Directory to save text files")
    parser.add_argument("--output-json", type=str, 
                        default="./transforms.json",
                        help="Path to save transforms.json")
    
    args = parser.parse_args()
    
    # Read cameras.bin
    print(f"Reading cameras from {args.cameras_bin}")
    cameras = read_cameras_binary(args.cameras_bin)
    
    # Write cameras.txt
    cameras_txt_path = os.path.join(args.text_dir, "cameras.txt")
    print(f"Writing cameras to {cameras_txt_path}")
    write_cameras_text(cameras, cameras_txt_path)
    
    # Generate transforms.json
    print(f"Generating transforms.json")
    run_colmap2nerf(args.images_dir, args.text_dir, args.output_json)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 