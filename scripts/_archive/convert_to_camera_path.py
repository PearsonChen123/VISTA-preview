import json
import argparse
import math
from collections import OrderedDict

def convert_to_camera_path(dataparser_transforms, original_transforms):
    # Extract necessary information from original_transforms
    w = original_transforms['w']
    h = original_transforms['h']
    fl_x = original_transforms['fl_x']
    fl_y = original_transforms['fl_y']
    
    # Calculate FOV and aspect ratio
    fov = 2 * math.atan(h / (2 * fl_y)) * 180 / math.pi
    aspect = w / h

    camera_path = []
    for transform in dataparser_transforms:
        matrix = transform['transform']
        # Convert 3x4 matrix to 4x4
        matrix.append([0, 0, 0, 1])
        
        camera_path.append({
            "camera_to_world": matrix,
            "fov": fov,
            "aspect": aspect
        })

    result = OrderedDict([
        ("camera_type", "perspective"),
        ("render_height", float(h)),
        ("render_width", float(w)),
        ("fps", 1.0),
        ("seconds", float(len(dataparser_transforms))),
        ("is_cycle", False),
        ("smoothness_value", 0.0),
        ("camera_path", camera_path)
    ])

    return result

def main():
    parser = argparse.ArgumentParser(description="Convert dataparser transforms to camera path.")
    parser.add_argument("dataparser_file", help="Path to dataparser_transforms_origin.json")
    parser.add_argument("original_transform_file", help="Path to original transforms.json")
    parser.add_argument("output_file", help="Path to output camera_path.json")
    args = parser.parse_args()

    with open(args.dataparser_file, 'r') as f:
        dataparser_transforms = json.load(f)
    
    with open(args.original_transform_file, 'r') as f:
        original_transforms = json.load(f)
    
    camera_path = convert_to_camera_path(dataparser_transforms, original_transforms)
    
    with open(args.output_file, 'w') as f:
        json.dump(camera_path, f, indent=2)

if __name__ == "__main__":
    main()