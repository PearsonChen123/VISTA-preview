import numpy as np
import os
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt

def process_and_save_depth():
    # Define paths
    input_dir = "/mnt/g/algorithm_backup/keyframe_3/keyframe_3/data/scene_points"
    output_dir = "/mnt/g/algorithm_backup/keyframe_3/test_images_100/GT-depth"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of tiff files and sort them
    try:
        tiff_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tiff')])
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    # Select the first 100 frames (no skipping)
    files_to_process = tiff_files[:100]
    
    print(f"Found {len(tiff_files)} total TIFF files. Processing the first {len(files_to_process)} depth maps.")

    # Lists to store depth statistics
    average_depths = []
    valid_pixel_counts = []
    total_pixel_counts = []

    for i, filename in enumerate(files_to_process):
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Read the TIFF file
            raw = tiff.imread(input_path)
            
            # Process the upper half of the image (left image)
            img_l = raw[:raw.shape[0] // 2, :, :]
            
            # Extract depth info (third channel) and handle potential NaNs from source
            depth_map = img_l[:, :, 2].astype(np.float32)

            # Better downsampling method that properly handles NaN and zero values
            def downsample_depth_properly(depth_img):
                """
                Proper downsampling for depth images with NaN/zero handling
                """
                h, w = depth_img.shape
                new_h, new_w = h // 2, w // 2
                downsampled = np.zeros((new_h, new_w), dtype=np.float32)
                
                for i in range(new_h):
                    for j in range(new_w):
                        # Get 2x2 region
                        y1, y2 = i*2, (i+1)*2
                        x1, x2 = j*2, (j+1)*2
                        
                        region_values = depth_img[y1:y2, x1:x2]
                        
                        # Only consider valid pixels (not NaN and > 0)
                        valid_mask = ~np.isnan(region_values) & (region_values > 0)
                        valid_pixels = region_values[valid_mask]
                        
                        if len(valid_pixels) > 0:
                            downsampled[i, j] = np.mean(valid_pixels)
                        else:
                            downsampled[i, j] = np.nan  # No valid pixels, set as NaN
                
                return downsampled

            # Use the improved downsampling method
            final_depth = downsample_depth_properly(depth_map)

            # Calculate depth statistics for this image
            valid_depth_mask = (final_depth > 0) & ~np.isnan(final_depth)
            valid_depths = final_depth[valid_depth_mask]
            
            total_pixels = final_depth.size
            valid_pixels = np.sum(valid_depth_mask)
            avg_depth = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0
            
            # Store statistics
            average_depths.append(avg_depth)
            valid_pixel_counts.append(valid_pixels)
            total_pixel_counts.append(total_pixels)

            # --- Save the results ---
            base_filename = os.path.splitext(filename)[0]
            
            # 1. Save as .npy file
            npy_path = os.path.join(output_dir, f"{base_filename}_depth.npy")
            np.save(npy_path, final_depth)

            # 2. Save a visual representation
            vis_path = os.path.join(output_dir, f"{base_filename}_depth_vis.png")
            
            # For visualization, replace NaNs with a value that can be mapped to a color (e.g., 0)
            # and use a colormap on the valid depth values.
            depth_for_vis = np.nan_to_num(final_depth, nan=0.0)
            # Normalize to 0-255 for visualization. Using percentiles to avoid outliers dominating the color range.
            if np.any(depth_for_vis > 0):
                vmax = np.percentile(depth_for_vis[depth_for_vis > 0], 95)
                vmin = np.percentile(depth_for_vis[depth_for_vis > 0], 5)
                normalized_depth = (depth_for_vis - vmin) / (vmax - vmin)
                normalized_depth = np.clip(normalized_depth, 0, 1)
            else:
                normalized_depth = depth_for_vis # all zeros

            colored_depth = (plt.cm.viridis(normalized_depth) * 255).astype(np.uint8)
            cv2.imwrite(vis_path, cv2.cvtColor(colored_depth, cv2.COLOR_RGBA2BGRA))

            print(f"Processed {filename} ({i+1}/{len(files_to_process)}). Average depth: {avg_depth:.3f}, Valid pixels: {valid_pixels}/{total_pixels}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            # Add placeholder values for failed processing
            average_depths.append(0.0)
            valid_pixel_counts.append(0)
            total_pixel_counts.append(0)

    # Print overall statistics
    print("\n" + "="*60)
    print("Depth Processing Statistics")
    print("="*60)
    
    valid_averages = [d for d in average_depths if d > 0]
    if valid_averages:
        overall_avg_depth = np.mean(valid_averages)
        min_depth = np.min(valid_averages)
        max_depth = np.max(valid_averages)
        std_depth = np.std(valid_averages)
        
        total_valid_pixels = sum(valid_pixel_counts)
        total_all_pixels = sum(total_pixel_counts)
        valid_pixel_ratio = total_valid_pixels / total_all_pixels if total_all_pixels > 0 else 0
        
        print(f"Successfully processed: {len(valid_averages)}/{len(files_to_process)} images")
        print(f"Overall average depth: {overall_avg_depth:.3f}")
        print(f"Depth range: {min_depth:.3f} - {max_depth:.3f}")
        print(f"Depth std deviation: {std_depth:.3f}")
        print(f"Total valid pixels: {total_valid_pixels:,} / {total_all_pixels:,} ({valid_pixel_ratio:.1%})")
        print(f"Average valid pixels per image: {total_valid_pixels // len(files_to_process):,}")
    else:
        print("No valid depth values found in any processed images.")
    
    print("="*60)
    print("Processing complete.")

if __name__ == '__main__':
    process_and_save_depth() 