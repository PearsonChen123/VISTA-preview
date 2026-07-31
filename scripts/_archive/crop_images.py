import os
import argparse
from PIL import Image
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2

def detect_circle(image_path):
    """
    Detect the circular region in an endoscopic image
    
    Args:
        image_path: Path to the image
        
    Returns:
        tuple: (center_x, center_y, radius) of the detected circle, or None if no circle is detected
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Hough Circle Transform to detect circles
        # Adjust parameters based on your specific images
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(blurred.shape[0], blurred.shape[1]),  # Only find one circle
            param1=50,
            param2=30,
            minRadius=min(blurred.shape[0], blurred.shape[1]) // 6,  # Minimum expected radius
            maxRadius=min(blurred.shape[0], blurred.shape[1]) // 2   # Maximum expected radius
        )
        
        if circles is not None:
            # Convert to integer coordinates
            circles = np.round(circles[0, :]).astype(int)
            
            # Get the largest circle (usually the endoscope image border)
            if len(circles) > 0:
                # Sort by radius (descending)
                circles = sorted(circles, key=lambda x: x[2], reverse=True)
                center_x, center_y, radius = circles[0]
                return (center_x, center_y, radius)
        
        return None
        
    except Exception as e:
        print(f"Error detecting circle in {image_path}: {str(e)}")
        return None

def crop_from_circle_center(img, center_x, center_y, crop_width, crop_height):
    """
    Crop an image with the specified center coordinates
    
    Args:
        img: PIL Image to crop
        center_x, center_y: Center coordinates for cropping
        crop_width, crop_height: Dimensions of the crop
        
    Returns:
        PIL Image: Cropped image
    """
    width, height = img.size
    
    # Calculate crop boundaries
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height
    
    # Ensure crop boundaries are within image dimensions
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    
    return img.crop((left, top, right, bottom))

def crop_center(img, crop_width, crop_height):
    """
    Crop the center region of an image (fallback method)
    """
    width, height = img.size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))

def process_image(args):
    """
    Process a single image
    
    Args:
        args: Tuple containing (src_path, dst_path, crop_size, circle_data)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if len(args) == 4:
        src_path, dst_path, crop_size, circle_data = args
    else:
        src_path, dst_path, crop_size = args
        circle_data = None
    
    try:
        # Open original image
        with Image.open(src_path) as img:
            # Crop based on circle center if available
            if circle_data is not None:
                center_x, center_y, _ = circle_data
                cropped_img = crop_from_circle_center(img, center_x, center_y, crop_size, crop_size)
            else:
                # Fallback to center crop
                cropped_img = crop_center(img, crop_size, crop_size)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Save cropped image
            cropped_img.save(dst_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")
        return False

def process_images(src_dir, dst_dir, crop_size=1080, detect_circles=True):
    """
    Process images by cropping them to a square of specified size
    
    Args:
        src_dir: Source directory containing images
        dst_dir: Destination directory for cropped images
        crop_size: Size of the square crop (default: 1080)
        detect_circles: Whether to detect circles in images (default: True)
    """
    # Create destination directory (if it doesn't exist)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    src_path = Path(src_dir)
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(list(src_path.glob(f"*{ext}")))
        image_files.extend(list(src_path.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No images found. Check the source directory and file extensions.")
        return
    
    # Prepare arguments for parallel processing
    args_list = []
    for img_path in image_files:
        dst_path = os.path.join(dst_dir, img_path.name)
        circle_data = None
        if detect_circles:
            circle_data = detect_circle(img_path)
        args_list.append((str(img_path), dst_path, crop_size, circle_data))
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Process images in parallel
    successful = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(
            executor.map(process_image, args_list),
            total=len(args_list),
            desc="Cropping images"
        ))
        successful = sum(results)
    
    print(f"Image processing completed! Successfully processed {successful} of {len(image_files)} images.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Crop images to a square of specified size")
    parser.add_argument("--src", type=str, default="./images",
                        help="Source directory containing images")
    parser.add_argument("--dst", type=str, default="./images_720",
                        help="Destination directory for cropped images")
    parser.add_argument("--size", type=int, default=720,
                        help="Size of the square crop (default: 720)")
    parser.add_argument("--no-detect", action="store_true",
                        help="Disable automatic circle detection (use center crop)")
    
    args = parser.parse_args()
    
    # Process images with the specified parameters
    process_images(args.src, args.dst, args.size, not args.no_detect)

if __name__ == "__main__":
    main()