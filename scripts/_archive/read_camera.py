import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """从二进制文件中读取下一组字节"""
    data = fid.read(num_bytes)
    return struct.unpack(format_char_sequence, data)

def read_cameras_binary(path_to_model_file: str) -> dict:
    """读取COLMAP的cameras.bin文件"""
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
            print("Parameters:")
            
            # 根据不同的相机模型打印参数含义
            if model_name == "PINHOLE":
                print(f"fx: {params[0]}")
                print(f"fy: {params[1]}")
                print(f"cx: {params[2]}")
                print(f"cy: {params[3]}")
            elif model_name == "OPENCV":
                print(f"fx: {params[0]}")
                print(f"fy: {params[1]}")
                print(f"cx: {params[2]}")
                print(f"cy: {params[3]}")
                print(f"k1: {params[4]}")
                print(f"k2: {params[5]}")
                print(f"p1: {params[6]}")
                print(f"p2: {params[7]}")
            else:
                for i, param in enumerate(params):
                    print(f"  param{i}: {param}")
    
    return cameras

if __name__ == "__main__":
    camera_file = "/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/undistorted/sparse/cameras.bin"
    cameras = read_cameras_binary(camera_file)