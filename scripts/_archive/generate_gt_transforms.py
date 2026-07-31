import argparse
import numpy as np
import json
from pathlib import Path
import sys

# Add the pose_opt directory to the path to allow direct import
# This assumes the script is run from its own directory
sys.path.append(str(Path(__file__).parent / 'pose_opt'))

try:
    # We are borrowing this function from the optimization script
    from test_optimize import load_gt_poses
except ImportError:
    print("Error: Could not import 'load_gt_poses'.")
    print("Please ensure this script is located in 'keyframe_3/test_images_100/'")
    print("and the 'pose_opt' directory is present with 'test_optimize.py'.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a transforms.json file directly from ground truth (GT) poses."
    )
    parser.add_argument('--gt-dir', type=str, required=True, 
                        help='Path to the GT frame_data directory')
    parser.add_argument('--template-json', type=str, required=True,
                        help='Path to an existing transforms.json to use as a template for intrinsics etc.')
    parser.add_argument('--output-json', type=str, default='transforms_gt.json',
                        help='Path for the output JSON file.')
    parser.add_argument('--scale-factor', type=float, default=0.5,
                        help='Factor to scale the GT translation by (e.g., 0.5 for half size).')
    args = parser.parse_args()

    template_path = Path(args.template_json).expanduser().resolve()
    gt_dir_path = Path(args.gt_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not template_path.exists():
        print(f"Error: Template JSON not found at {template_path}")
        sys.exit(1)
        
    if not gt_dir_path.exists():
        print(f"Error: Ground truth directory not found at {gt_dir_path}")
        sys.exit(1)

    print(f"Loading template from: {template_path}")
    with open(template_path, 'r') as f:
        out_data = json.load(f)

    num_frames_in_template = len(out_data['frames'])
    print(f"Template has {num_frames_in_template} frames. Loading corresponding GT poses...")

    # Load the GT poses (these are in COLMAP's c2w format)
    gt_poses_colmap = load_gt_poses(gt_dir_path, num_poses=num_frames_in_template)

    if len(gt_poses_colmap) != num_frames_in_template:
        print("Warning: Number of loaded GT poses does not match number of frames in template.")
        print("The output JSON will only contain entries for which GT poses were found.")

    # This is the matrix used in `colmap2nerf.py --keep_colmap_coords`
    flip_mat = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    new_frames_data = []
    # We iterate through the template frames to maintain the original file order and names
    for i, frame_template in enumerate(out_data['frames']):
        if i < len(gt_poses_colmap):
            pose_c2w_colmap = gt_poses_colmap[i].copy() # Make a copy to modify

            # Scale the translation part of the pose
            pose_c2w_colmap[:3, 3] *= args.scale_factor

            # Convert to the NeRF format used by `--keep_colmap_coords`
            pose_nerf_format = pose_c2w_colmap @ flip_mat

            new_frame = frame_template.copy()
            new_frame['transform_matrix'] = pose_nerf_format.tolist()
            new_frames_data.append(new_frame)

    out_data['frames'] = new_frames_data
    
    print(f"\nSaving {len(new_frames_data)} GT-based poses to {output_path}...")
    print(f"Translation scale factor applied: {args.scale_factor}")
    
    with open(output_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main() 