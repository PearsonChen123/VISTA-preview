#!/usr/bin/env python3
"""
Run DEFORM-Stereo on RGBD-500 rotated stereo pairs
Converts disparity to depth and unrotates back to original orientation
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import cv2

# Add DEFORM-Stereo to path
DEFOMSTEREO_PATH = Path("/mnt/h/DEFOM-Stereo")
sys.path.insert(0, str(DEFOMSTEREO_PATH))
sys.path.insert(0, str(DEFOMSTEREO_PATH / "core"))

from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    """Load image and convert to tensor"""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def unrotate_image(img, rotation_type):
    """
    Reverse the rotation applied to an image

    Args:
        img: numpy array of image
        rotation_type: "90cc", "90c", "180", or "none"

    Returns:
        Unrotated image
    """
    if rotation_type == "90cc":
        # Reverse 90 counter-clockwise -> rotate 90 clockwise
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_type == "90c":
        # Reverse 90 clockwise -> rotate 90 counter-clockwise
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_type == "180":
        # Reverse 180 -> rotate 180
        return cv2.rotate(img, cv2.ROTATE_180)
    else:
        return img


def disparity_to_depth(disparity, K_matrix, baseline):
    """
    Convert disparity to depth using camera intrinsics

    Args:
        disparity: disparity map (pixels)
        K_matrix: 3x3 camera intrinsic matrix
        baseline: stereo baseline (meters)

    Returns:
        depth map (meters)
    """
    # Extract focal length from K matrix
    fx = K_matrix[0, 0]

    # Depth = (focal_length * baseline) / disparity
    # Avoid division by zero
    depth = np.zeros_like(disparity)
    valid_mask = disparity > 0
    depth[valid_mask] = (fx * baseline) / disparity[valid_mask]

    return depth


def load_camera_intrinsics(intrinsic_file):
    """
    Load camera intrinsics from K.txt

    Format:
    Line 1: 9 values for 3x3 K matrix (space separated)
    Line 2: baseline value

    Returns:
        K_matrix: 3x3 numpy array
        baseline: float
    """
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()

        # Parse K matrix
        k_values = np.array([float(x) for x in lines[0].strip().split()])
        K_matrix = k_values.reshape(3, 3)

        # Parse baseline
        baseline = float(lines[1].strip())

    return K_matrix, baseline


def main():
    parser = argparse.ArgumentParser(
        description='Run DEFORM-Stereo on rotated stereo pairs and unrotate results'
    )
    parser.add_argument('--restore_ckpt', type=str, required=True,
                        help='Path to DEFORM-Stereo checkpoint')
    parser.add_argument('--left_dir', type=str,
                        default='/mnt/h/RGBD-500/stereo/left_rotated',
                        help='Directory with rotated left images')
    parser.add_argument('--right_dir', type=str,
                        default='/mnt/h/RGBD-500/stereo/right_rotated',
                        help='Directory with rotated right images')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/h/RGBD-500/stereo/depth_defomstereo',
                        help='Output directory for unrotated depth maps')
    parser.add_argument('--intrinsic_file', type=str,
                        default='/mnt/h/RGBD-500/K.txt',
                        help='Path to camera intrinsic file (K.txt)')
    parser.add_argument('--shift_direction', type=str, default='x',
                        choices=['x', '-x', 'y', '-y'],
                        help='Shift direction used in stereo generation')
    parser.add_argument('--save_disparity', action='store_true',
                        help='Also save disparity maps')
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save depth visualizations')

    # DEFORM-Stereo model parameters
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8,
                        help='number of scaling updates')
    parser.add_argument('--dinov2_encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"],
                        default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--scale_list', type=float, nargs='+',
                        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument('--scale_corr_radius', type=int, default=2)
    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3])
    parser.add_argument('--context_norm', type=str, default="batch",
                        choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--n_gru_layers', type=int, default=3)

    args = parser.parse_args()

    # Setup paths
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional output directories
    if args.save_disparity:
        disp_dir = output_dir.parent / "disp_defomstereo"
        disp_dir.mkdir(parents=True, exist_ok=True)

    if args.save_visualization:
        vis_dir = output_dir.parent / "depth_defomstereo_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DEFORM-Stereo Inference for RGBD-500")
    print("=" * 70)
    print(f"Left images:    {left_dir}")
    print(f"Right images:   {right_dir}")
    print(f"Output depth:   {output_dir}")
    print(f"Checkpoint:     {args.restore_ckpt}")
    print(f"Shift direction: {args.shift_direction}")
    print("=" * 70)
    print()

    # Load camera intrinsics
    print("Loading camera intrinsics...")
    K_matrix, baseline = load_camera_intrinsics(args.intrinsic_file)
    print(f"K matrix:\n{K_matrix}")
    print(f"Baseline: {baseline} meters")
    print()

    # Determine rotation type for unrotation
    # These match the rotation applied in process_stereo_foundation.sh
    if args.shift_direction.lower() == "x":
        rotation_type = "180"
    elif args.shift_direction.lower() == "-x":
        rotation_type = "none"
    elif args.shift_direction.lower() == "y":
        rotation_type = "90cc"
    elif args.shift_direction.lower() == "-y":
        rotation_type = "90c"
    else:
        rotation_type = "none"

    print(f"Unrotation type: {rotation_type}")
    print()

    # Load DEFORM-Stereo model
    print("Loading DEFORM-Stereo model...")
    model = DEFOMStereo(args)

    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    print()

    # Get image pairs
    left_images = sorted(left_dir.glob("*.png"))
    right_images = sorted(right_dir.glob("*.png"))

    if len(left_images) == 0:
        print(f"Error: No images found in {left_dir}")
        return 1

    if len(left_images) != len(right_images):
        print(f"Warning: Number of left ({len(left_images)}) and right ({len(right_images)}) images differ")
        # Take minimum
        n_images = min(len(left_images), len(right_images))
        left_images = left_images[:n_images]
        right_images = right_images[:n_images]

    print(f"Processing {len(left_images)} stereo pairs...")
    print()

    # Process each stereo pair
    with torch.no_grad():
        for left_path, right_path in tqdm(list(zip(left_images, right_images)),
                                          desc="Running stereo matching"):
            # Load images
            image1 = load_image(str(left_path))
            image2 = load_image(str(right_path))

            # Pad to multiple of 32
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            # Run stereo matching
            disp_pr = model(image1, image2,
                          iters=args.valid_iters,
                          scale_iters=args.scale_iters,
                          test_mode=True)

            # Unpad
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()

            # Convert disparity to depth
            depth = disparity_to_depth(disp_pr, K_matrix, baseline)

            # Unrotate depth back to original orientation
            depth_unrotated = unrotate_image(depth, rotation_type)

            # Save depth map
            stem = left_path.stem
            output_path = output_dir / f"{stem}.npy"
            np.save(output_path, depth_unrotated.astype(np.float32))

            # Optional: Save disparity
            if args.save_disparity:
                disp_unrotated = unrotate_image(disp_pr, rotation_type)
                disp_path = disp_dir / f"{stem}.npy"
                np.save(disp_path, disp_unrotated.astype(np.float32))

            # Optional: Save visualization
            if args.save_visualization:
                # Only visualize valid depth
                valid_depth = depth_unrotated[depth_unrotated > 0]

                if len(valid_depth) > 0:
                    # Use 5-95 percentile for visualization
                    depth_min = np.percentile(valid_depth, 5)
                    depth_max = np.percentile(valid_depth, 95)

                    # Normalize to 0-255
                    depth_vis = np.zeros_like(depth_unrotated, dtype=np.uint8)
                    depth_mask = depth_unrotated > 0
                    depth_normalized = np.clip(
                        (depth_unrotated[depth_mask] - depth_min) / (depth_max - depth_min + 1e-6),
                        0, 1
                    )
                    depth_vis[depth_mask] = (depth_normalized * 255).astype(np.uint8)

                    # Apply colormap
                    depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                    depth_vis_colored[~depth_mask] = 0

                    vis_path = vis_dir / f"{stem}.png"
                    cv2.imwrite(str(vis_path), depth_vis_colored)

    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Depth maps saved to: {output_dir}")
    if args.save_disparity:
        print(f"Disparity maps saved to: {disp_dir}")
    if args.save_visualization:
        print(f"Visualizations saved to: {vis_dir}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
