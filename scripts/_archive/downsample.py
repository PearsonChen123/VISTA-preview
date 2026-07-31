import os
import cv2
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

def process_image(args):
    """
    Process an image by downsampling it to the target size
    
    Args:
        args: Tuple containing (input_path, output_path, target_size)
    
    Returns:
        bool: True if successful, False otherwise
    """
    input_path, output_path, target_size = args
    
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise Exception("Failed to read image")
            
        # Resize using LANCZOS interpolation
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Save with highest quality
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, target_size):
    """
    Process all images in a directory by downsampling them to the target size
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        target_size: Target size for the square images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                image_files.append((input_path, output_path, target_size))

    print(f"Found {len(image_files)} images to process")
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Process images in parallel with progress bar
    successful = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(
            executor.map(process_image, image_files),
            total=len(image_files),
            desc="Downsampling images"
        ))
        successful = sum(results)
    
    print(f"Image processing completed. Successfully downsampled {successful} of {len(image_files)} images.")
    print(f"Downsampled images saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Downsample images to a specified size")
    parser.add_argument("--input-dir", type=str, 
                        default="./images_720",
                        help="Input directory containing images")
    parser.add_argument("--output-dir", type=str, 
                        default="./images_600",
                        help="Output directory for downsampled images")
    parser.add_argument("--target-size", type=int, default=600,
                        help="Target size for the square images (default: 600)")
    
    args = parser.parse_args()
    
    # Process images
    print(f"Downsampling images from {args.input_dir} to {args.output_dir} with target size {args.target_size}x{args.target_size}")
    process_directory(args.input_dir, args.output_dir, args.target_size)

if __name__ == "__main__":
    main() 