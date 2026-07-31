import cv2
import os
import glob

def process_images():
    # Define paths
    input_dir = '/mnt/g/algorithm_backup/keyframe_3/test_images_100/left'
    output_dir = '/mnt/g/algorithm_backup/keyframe_3/test_images_100/images'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of image files, sort them
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Select the first 100 frames
    files_to_process = image_files[:100]

    print(f"Found {len(image_files)} total images. Processing {len(files_to_process)} images from the first 100 frames.")

    for filename in files_to_process:
        try:
            # Construct full file path
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Could not read image {filename}. Skipping.")
                continue

            # Downsample using Gaussian pyramid
            downsampled_image = cv2.pyrDown(image)

            # Save the new image
            cv2.imwrite(output_path, downsampled_image)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    print("Processing complete.")

if __name__ == '__main__':
    process_images() 