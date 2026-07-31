#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stereo Image Rotation Script
Rotates stereo images based on the DEFAULT_SHIFT value in process_stereo.sh:
- If DEFAULT_SHIFT is positive, rotate 90 degrees counter-clockwise
- If DEFAULT_SHIFT is negative, rotate 90 degrees clockwise
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

def rotate_image(image_path, output_path, clockwise=False):
    """
    Rotate an image and save it
    
    Args:
        image_path: Input image path
        output_path: Output image path
        clockwise: Whether to rotate clockwise, default is False (counter-clockwise)
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return False
    
    # Rotate the image based on direction
    if clockwise:
        # Rotate 90 degrees clockwise (ROTATE_90_CLOCKWISE)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Rotate 90 degrees counter-clockwise (ROTATE_90_COUNTERCLOCKWISE)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the rotated image
    cv2.imwrite(str(output_path), rotated_image)
    return True

def process_directory(input_dir, output_dir, clockwise=False):
    """
    Process all images in a directory
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        clockwise: Whether to rotate clockwise
    """
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files in {input_dir}")
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Use multiprocessing to process images
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for img_path in image_files:
            out_path = output_path / img_path.name
            futures.append(executor.submit(rotate_image, img_path, out_path, clockwise))
        
        # Show progress bar
        for _ in tqdm(futures, total=len(futures), desc=f"Rotating images ({'clockwise' if clockwise else 'counter-clockwise'})"):
            pass
    
    print(f"Completed rotation of images in {input_dir}, results saved to {output_dir}")

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
                    return float(value_str)
    except Exception as e:
        print(f"Could not parse DEFAULT_SHIFT from {process_stereo_sh_path}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description='Rotate stereo images')
    parser.add_argument('--left-dir', type=str, default='./stereo/left',
                        help='Left image directory path')
    parser.add_argument('--right-dir', type=str, default='./stereo/right',
                        help='Right image directory path')
    parser.add_argument('--output-left-dir', type=str, default='./stereo/left_rotated',
                        help='Output directory for rotated left images')
    parser.add_argument('--output-right-dir', type=str, default='./stereo/right_rotated',
                        help='Output directory for rotated right images')
    parser.add_argument('--process-stereo-sh', type=str, default='./process_stereo.sh',
                        help='Path to process_stereo.sh script to get DEFAULT_SHIFT value')
    parser.add_argument('--shift', type=float, default=None,
                        help='Manually specify shift value, overrides value from process_stereo.sh if provided')
    
    args = parser.parse_args()
    
    # Get shift value
    shift_value = args.shift
    if shift_value is None:
        shift_value = get_shift_value(args.process_stereo_sh)
        if shift_value is None:
            print("Could not get shift value, defaulting to counter-clockwise rotation")
            shift_value = 0.1  # Default to positive value, counter-clockwise rotation
    
    # Determine rotation direction based on shift value
    clockwise = shift_value < 0
    rotation_direction = "clockwise" if clockwise else "counter-clockwise"
    print(f"Shift value is {shift_value}, will rotate {rotation_direction}")
    
    # Process left images
    print("Processing left images...")
    process_directory(args.left_dir, args.output_left_dir, clockwise)
    
    # Process right images
    print("Processing right images...")
    process_directory(args.right_dir, args.output_right_dir, clockwise)
    
    print("All image rotation completed!")
    print(f"Rotated left images saved to: {args.output_left_dir}")
    print(f"Rotated right images saved to: {args.output_right_dir}")

if __name__ == "__main__":
    main() 