#!/bin/bash

# Script name: process_stereo.sh
# Function: Automate the process from exporting camera paths to rendering stereo images and generating disparity maps

# =============================================
# Default paths and parameters
# =============================================

# Base directory
BASE_DIR="/mnt/f/algorithm_F/post-op/8Q3_left"

# Nerfstudio paths
DEFAULT_TRANSFORMS_JSON="${BASE_DIR}/transforms.json"
DEFAULT_CONFIG_PATH="${BASE_DIR}/outputs/8Q3_left/depth-nerfacto/2025-04-01_142739/config.yml" 
DEFAULT_DATAPARSER_TRANSFORMS="${BASE_DIR}/outputs/8Q3_left/depth-nerfacto/2025-04-01_142739/dataparser_transforms.json"
DEFAULT_CAMERA_PATH_DIR="${BASE_DIR}/cam_path"
DEFAULT_STEREO_LEFT_DIR="${BASE_DIR}/stereo/left"
DEFAULT_STEREO_RIGHT_DIR="${BASE_DIR}/stereo/right"
# Stereo parameters
DEFAULT_SHIFT=0.7  # Relative baseline length
DEFAULT_SHIFT_DIRECTION="y"  # Options: "x" for left-right, "y" for up-down

# Rotated stereo image paths
DEFAULT_STEREO_LEFT_ROTATED_DIR="${BASE_DIR}/stereo/left_rotated"
DEFAULT_STEREO_RIGHT_ROTATED_DIR="${BASE_DIR}/stereo/right_rotated"
DEFAULT_STEREO_DISP_DIR="${BASE_DIR}/stereo/disp"

# Disparity estimation model paths
DEFAULT_SELECTIVE_IGEV_DIR="/mnt/f/algorithm_F/Selective-Stereo/Selective-IGEV"
DEFAULT_SELECTIVE_IGEV_MODEL="/mnt/f/algorithm_F/Selective-Stereo/Selective-IGEV/pretrained_models/middlebury_finetune.pth"
DEFAULT_VALID_ITERS=32

# Depth processing paths
DEFAULT_ROTATED_DISP_DIR="${BASE_DIR}/stereo/disp_rotated"
DEFAULT_DEPTH_DIR="${BASE_DIR}/stereo/depth"

# =============================================
# Function: Display usage instructions
# =============================================
show_usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                         Show this help message"
    echo "  -c, --config FILE                  Path to the config.yml file of your trained model"
    echo "                                     (default: $DEFAULT_CONFIG_PATH)"
    echo "  -d, --camera-path-dir DIR          Directory where camera path files will be saved"
    echo "                                     (default: $DEFAULT_CAMERA_PATH_DIR)"
    echo "  --left-dir DIR                     Directory where left eye rendered images will be saved"
    echo "                                     (default: $DEFAULT_STEREO_LEFT_DIR)"
    echo "  --right-dir DIR                    Directory where right eye rendered images will be saved"
    echo "                                     (default: $DEFAULT_STEREO_RIGHT_DIR)"
    echo "  -s, --shift VALUE                  Amount of shift to apply for stereo effect"
    echo "                                     (default: $DEFAULT_SHIFT)"
    echo "  --shift-direction DIR              Direction of shift: 'x' for left-right, 'y' for up-down"
    echo "                                     (default: $DEFAULT_SHIFT_DIRECTION)"
    echo "  --igev-dir DIR                     Directory containing the Selective-IGEV code"
    echo "                                     (default: $DEFAULT_SELECTIVE_IGEV_DIR)"
    echo "  --igev-model FILE                  Path to the Selective-IGEV model checkpoint"
    echo "                                     (default: $DEFAULT_SELECTIVE_IGEV_MODEL)"
    echo "  --depth-dir DIR                    Directory to save depth maps"
    echo "                                     (default: $DEFAULT_DEPTH_DIR)"
    echo "  --transforms-json FILE             Path to transforms.json file"
    echo "                                     (default: $DEFAULT_TRANSFORMS_JSON)"
    echo
    echo "Example:"
    echo "  $0 --config /path/to/config.yml --shift 0.2 --shift-direction x"
}

# =============================================
# Parse command line arguments
# =============================================
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--camera-path-dir)
            CAMERA_PATH_DIR="$2"
            shift 2
            ;;
        --left-dir)
            STEREO_LEFT_DIR="$2"
            shift 2
            ;;
        --right-dir)
            STEREO_RIGHT_DIR="$2"
            shift 2
            ;;
        -s|--shift)
            SHIFT="$2"
            shift 2
            ;;
        --shift-direction)
            SHIFT_DIRECTION="$2"
            shift 2
            ;;
        --igev-dir)
            SELECTIVE_IGEV_DIR="$2"
            shift 2
            ;;
        --igev-model)
            SELECTIVE_IGEV_MODEL="$2"
            shift 2
            ;;
        --depth-dir)
            DEPTH_DIR="$2"
            shift 2
            ;;
        --transforms-json)
            TRANSFORMS_JSON="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# =============================================
# Set parameters, using default values or command line arguments
# =============================================
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}
CAMERA_PATH_DIR=${CAMERA_PATH_DIR:-$DEFAULT_CAMERA_PATH_DIR}
STEREO_LEFT_DIR=${STEREO_LEFT_DIR:-$DEFAULT_STEREO_LEFT_DIR}
STEREO_RIGHT_DIR=${STEREO_RIGHT_DIR:-$DEFAULT_STEREO_RIGHT_DIR}
SHIFT=${SHIFT:-$DEFAULT_SHIFT}
SHIFT_DIRECTION=${SHIFT_DIRECTION:-$DEFAULT_SHIFT_DIRECTION}

# Set rotated image and disparity paths based on stereo directories
STEREO_LEFT_ROTATED_DIR="$(dirname "$STEREO_LEFT_DIR")/left_rotated"
STEREO_RIGHT_ROTATED_DIR="$(dirname "$STEREO_RIGHT_DIR")/right_rotated"
STEREO_DISP_DIR="$(dirname "$STEREO_LEFT_DIR")/disp"

# Set Selective-IGEV paths
SELECTIVE_IGEV_DIR=${SELECTIVE_IGEV_DIR:-$DEFAULT_SELECTIVE_IGEV_DIR}
SELECTIVE_IGEV_MODEL=${SELECTIVE_IGEV_MODEL:-$DEFAULT_SELECTIVE_IGEV_MODEL}
VALID_ITERS=${VALID_ITERS:-$DEFAULT_VALID_ITERS}

# Set depth processing paths
ROTATED_DISP_DIR=${ROTATED_DISP_DIR:-$DEFAULT_ROTATED_DISP_DIR}
DEPTH_DIR=${DEPTH_DIR:-$DEFAULT_DEPTH_DIR}
DATAPARSER_TRANSFORMS=${DATAPARSER_TRANSFORMS:-$DEFAULT_DATAPARSER_TRANSFORMS}
TRANSFORMS_JSON=${TRANSFORMS_JSON:-$DEFAULT_TRANSFORMS_JSON}

# Define paths for intermediate files
TRANSFORMS_TRAIN_JSON="$CAMERA_PATH_DIR/transforms_train.json"
TRANSFORMS_TRAIN_RIGHT_JSON="$CAMERA_PATH_DIR/transforms_train_right.json"
CAMERA_PATH_JSON="$CAMERA_PATH_DIR/camera_path.json"
CAMERA_PATH_RIGHT_JSON="$CAMERA_PATH_DIR/camera_path_right.json"
ORIGINAL_TRANSFORM_FILE="transforms.json"

# =============================================
# Check if config file exists
# =============================================
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file does not exist: $CONFIG_PATH"
    exit 1
fi

# =============================================
# Create necessary directories
# =============================================
mkdir -p "$CAMERA_PATH_DIR"
mkdir -p "$STEREO_LEFT_DIR"
mkdir -p "$STEREO_RIGHT_DIR"
mkdir -p "$STEREO_LEFT_ROTATED_DIR"
mkdir -p "$STEREO_RIGHT_ROTATED_DIR"
mkdir -p "$STEREO_DISP_DIR"
mkdir -p "$ROTATED_DISP_DIR"
mkdir -p "$DEPTH_DIR"

# =============================================
# Export camera paths
# =============================================
echo "Exporting camera paths..."
python export_camera_poses_safe.py \
    --load-config "$CONFIG_PATH" \
    --output-dir "$CAMERA_PATH_DIR" \
    --combine-train-eval \
    --reference-transforms "$TRANSFORMS_JSON"

# Check if export was successful
if [ ! -f "$TRANSFORMS_TRAIN_JSON" ]; then
    echo "Error: Camera path export failed. transforms_train.json not found."
    exit 1
fi

# =============================================
# Generate stereo camera paths
# =============================================
echo "Generating stereo camera paths..."

# Create a temporary Python script to generate stereo paths
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOF
import numpy as np
import json
import sys

src_json_file = "$TRANSFORMS_TRAIN_JSON"
dest_json_file = "$TRANSFORMS_TRAIN_RIGHT_JSON"
shift = $SHIFT
shift_direction = "$SHIFT_DIRECTION"

def get_translated_matrix(transform_matrix, shift, shift_direction):
    transform_matrix = np.array(transform_matrix + [[0, 0, 0, 1]])  # Add fourth row [0, 0, 0, 1]
    
    # Create shift matrix based on direction
    if shift_direction.lower() == "x":
        # Left-right shift
        shift_matrix = np.array([[1.0, 0.0, 0.0, shift],
                                [0.0, 1.0, 0.0, 0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    else:
        # Up-down shift (default)
        shift_matrix = np.array([[1.0, 0.0, 0.0, 0],
                                [0.0, 1.0, 0.0, shift],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    
    result = np.linalg.inv(shift_matrix @ np.linalg.inv(transform_matrix))
    return result[:3, :4].tolist()  # Return first three rows

# Open source JSON file
with open(src_json_file, 'r') as src_file:
    data = json.load(src_file)

# Process each frame in the JSON data
for frame in data:
    # Calculate new transform matrix
    t_matrix = frame["transform"]
    frame["transform"] = get_translated_matrix(transform_matrix=t_matrix, shift=shift, shift_direction=shift_direction)

# Write modified JSON data to destination file
with open(dest_json_file, 'w') as dest_file:
    json.dump(data, dest_file, indent=2)

print(f"Stereo camera paths generated and saved to {dest_json_file}")
EOF

# Execute the temporary Python script
python "$TMP_SCRIPT"
rm "$TMP_SCRIPT"

# Check if stereo path generation was successful
if [ ! -f "$TRANSFORMS_TRAIN_RIGHT_JSON" ]; then
    echo "Error: Stereo camera path generation failed. transforms_train_right.json not found."
    exit 1
fi

# =============================================
# Convert transforms to camera paths
# =============================================
echo "Converting transforms to camera paths..."

# Use colmap_convert_to_camera_path.py to generate camera paths
python colmap_convert_to_camera_path.py "$TRANSFORMS_TRAIN_JSON" "$ORIGINAL_TRANSFORM_FILE" "$CAMERA_PATH_JSON" --config "$CONFIG_PATH"
python colmap_convert_to_camera_path.py "$TRANSFORMS_TRAIN_RIGHT_JSON" "$ORIGINAL_TRANSFORM_FILE" "$CAMERA_PATH_RIGHT_JSON" --config "$CONFIG_PATH"

# Check if camera path conversion was successful
if [ ! -f "$CAMERA_PATH_JSON" ] || [ ! -f "$CAMERA_PATH_RIGHT_JSON" ]; then
    echo "Error: Camera path conversion failed. Camera path files not found."
    exit 1
fi

# =============================================
# Render stereo images
# =============================================
echo "Rendering stereo images..."

# Render left eye view
echo "Rendering left eye view..."
ns-render camera-path --load-config "$CONFIG_PATH" --camera-path-filename "$CAMERA_PATH_JSON" --output-path "$STEREO_LEFT_DIR" --output-format images --image-format png

# Render right eye view
echo "Rendering right eye view..."
ns-render camera-path --load-config "$CONFIG_PATH" --camera-path-filename "$CAMERA_PATH_RIGHT_JSON" --output-path "$STEREO_RIGHT_DIR" --output-format images --image-format png

echo "Stereo rendering process completed successfully!"
echo "Left eye images saved to: $STEREO_LEFT_DIR"
echo "Right eye images saved to: $STEREO_RIGHT_DIR"

# =============================================
# Rotate stereo images for disparity estimation
# =============================================
echo "Checking rotation direction based on shift direction..."

# Handle rotation differently based on shift direction
if [[ "${SHIFT_DIRECTION,,}" == "x" ]]; then
    echo "Shift direction is X (horizontal). Rotating images 180 degrees..."
    
    # Create rotation script for 180-degree rotation
    TMP_ROTATION_SCRIPT=$(mktemp)
    cat > "$TMP_ROTATION_SCRIPT" << EOF
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def rotate_image(img_path, output_dir):
    """Rotate a single image 180 degrees"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        
        # Rotate 180 degrees
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        
        # Save rotated image
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), rotated_img)
        return img_path.name
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return None

def rotate_180(input_dir, output_dir):
    """Rotate all images in a directory 180 degrees using parallel processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG images in the input directory
    image_files = list(Path(input_dir).glob("*.png"))
    
    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel rotation")
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks
        futures = [executor.submit(rotate_image, img_path, output_dir) for img_path in image_files]
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Rotating images in {input_dir}"):
            result = future.result()
            if result:
                pass  # Successfully rotated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate stereo images 180 degrees")
    parser.add_argument("--left-dir", required=True, help="Directory with left eye images")
    parser.add_argument("--right-dir", required=True, help="Directory with right eye images")
    parser.add_argument("--output-left-dir", required=True, help="Output directory for rotated left eye images")
    parser.add_argument("--output-right-dir", required=True, help="Output directory for rotated right eye images")
    
    args = parser.parse_args()
    
    # Rotate left and right images
    rotate_180(args.left_dir, args.output_left_dir)
    rotate_180(args.right_dir, args.output_right_dir)
EOF

    # Execute the rotation script
    python "$TMP_ROTATION_SCRIPT" \
        --left-dir "$STEREO_LEFT_DIR" \
        --right-dir "$STEREO_RIGHT_DIR" \
        --output-left-dir "$STEREO_LEFT_ROTATED_DIR" \
        --output-right-dir "$STEREO_RIGHT_ROTATED_DIR"
    
    # Remove temporary script
    rm "$TMP_ROTATION_SCRIPT"
else
    echo "Shift direction is Y (vertical). Using 90-degree rotation..."
    
    # Rotate images using the rotate_stereo_images.py script
    python rotate_stereo_images.py --left-dir "$STEREO_LEFT_DIR" --right-dir "$STEREO_RIGHT_DIR" \
        --output-left-dir "$STEREO_LEFT_ROTATED_DIR" --output-right-dir "$STEREO_RIGHT_ROTATED_DIR" \
        --process-stereo-sh "$0" --shift "$SHIFT"
fi

# =============================================
# Generate disparity maps using Selective-IGEV
# =============================================
echo "Generating disparity maps using Selective-IGEV..."

# Activate IGEV conda environment, run script, then deactivate
echo "Activating IGEV conda environment..."
eval "$(conda shell.bash hook)"
conda activate Selective_Stereo

# Run Selective-IGEV script.py
echo "Running Selective-IGEV for disparity estimation..."
python "$SELECTIVE_IGEV_DIR/script.py" \
    --restore_ckpt "$SELECTIVE_IGEV_MODEL" \
    -l "$STEREO_LEFT_ROTATED_DIR" \
    -r "$STEREO_RIGHT_ROTATED_DIR" \
    --output_directory "$STEREO_DISP_DIR" \
    --valid_iters $VALID_ITERS

# Deactivate IGEV conda environment
echo "Deactivating IGEV conda environment..."
conda deactivate

echo "Disparity estimation completed successfully!"
echo "Disparity maps saved to: $STEREO_DISP_DIR"

# =============================================
# Process disparity maps to depth maps
# =============================================
echo "Processing disparity maps to depth maps..."

# Activate nerfstudio conda environment
echo "Activating nerfstudio conda environment..."
eval "$(conda shell.bash hook)"
conda activate nerfstudio

# Set rotation flag based on shift direction
if [[ "${SHIFT_DIRECTION,,}" == "x" ]]; then
    ROTATION_FLAG="--rotate-180"
else
    ROTATION_FLAG=""
fi

# Run process_disparity_to_depth.py
python process_disparity_to_depth.py \
    --disp-dir "$STEREO_DISP_DIR" \
    --rotated-disp-dir "$ROTATED_DISP_DIR" \
    --depth-dir "$DEPTH_DIR" \
    --process-stereo-sh "$0" \
    --transforms-json "$TRANSFORMS_JSON" \
    --dataparser-transforms "$DATAPARSER_TRANSFORMS" \
    $ROTATION_FLAG

echo "Depth map generation completed successfully!"
echo "Depth maps saved to: $DEPTH_DIR"

# =============================================
# Train depth-nerfacto model
# =============================================
echo "Training depth-nerfacto model..."

# Get the directory containing transforms.json
TRANSFORMS_DIR=$(dirname "$TRANSFORMS_JSON")

# Run depth-nerfacto training
echo "Starting depth-nerfacto training..."
ns-train depth-nerfacto --data "$TRANSFORMS_DIR" --pipeline.model.camera-optimizer.mode off --max-num-iterations 4000

# Deactivate nerfstudio conda environment
echo "Deactivating nerfstudio conda environment..."
conda deactivate

echo "Depth-nerfacto training completed!"
echo "The entire stereo processing pipeline has been completed successfully!" 
