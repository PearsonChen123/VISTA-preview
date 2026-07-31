#!/usr/bin/env python3

"""
Test script for opt_pose.py functionality
"""

from pathlib import Path
import sys

# Add current directory to path so we can import pose_opt module
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pose_opt.opt_pose import read_images_binary_full, write_images_binary_full

def test_read_write():
    """Test reading and writing images.bin file"""
    
    # Test file paths
    original_bin = Path("undistorted/sparse/images.bin")
    test_output = Path("test_images_copy.bin")
    
    if not original_bin.exists():
        print(f"Error: {original_bin} not found")
        return False
    
    try:
        # Read original file
        print(f"Reading {original_bin}...")
        images = read_images_binary_full(original_bin)
        print(f"Found {len(images)} images")
        
        # Print first few image names for verification
        for i, (image_id, img) in enumerate(list(images.items())[:5]):
            print(f"  Image {image_id}: {img.name} (camera_id: {img.camera_id})")
        
        # Write copy
        print(f"Writing copy to {test_output}...")
        write_images_binary_full(images, test_output)
        
        # Read copy back to verify
        print("Verifying written file...")
        images_copy = read_images_binary_full(test_output)
        
        if len(images) == len(images_copy):
            print("✓ File size matches")
        else:
            print("✗ File size mismatch")
            return False
        
        # Check first image data
        first_id = next(iter(images.keys()))
        img_orig = images[first_id]
        img_copy = images_copy[first_id]
        
        if img_orig.name == img_copy.name:
            print("✓ Image names match")
        else:
            print("✗ Image names don't match")
            return False
            
        print("✓ Read/write test passed")
        
        # Clean up
        test_output.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing opt_pose.py read/write functionality...")
    success = test_read_write()
    
    if success:
        print("\nAll tests passed! opt_pose.py should work correctly.")
    else:
        print("\nTests failed. Check the implementation.") 