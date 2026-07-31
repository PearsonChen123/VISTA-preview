#!/bin/bash

# Script name: process_stereo_foundation.sh
# Function: Automate the process from exporting camera paths to rendering stereo images and generating disparity maps using Foundation Stereo

# =============================================
# Default paths and parameters
# =============================================

# Base directory
BASE_DIR="/mnt/h/RGBD-500"

# Nerfstudio paths
DEFAULT_TRANSFORMS_JSON="${BASE_DIR}/transforms.json"
DEFAULT_CONFIG_PATH="${BASE_DIR}/outputs/unnamed/zipnerf/2025-09-26_021009/config.yml" 
DEFAULT_DATAPARSER_TRANSFORMS="${BASE_DIR}/outputs/unnamed/zipnerf/2025-09-26_021009/dataparser_transforms.json"
DEFAULT_CAMERA_PATH_DIR="${BASE_DIR}/cam_path"
DEFAULT_STEREO_LEFT_DIR="${BASE_DIR}/stereo/left"
DEFAULT_STEREO_RIGHT_DIR="${BASE_DIR}/stereo/right"
# Stereo parameters
DEFAULT_SHIFT=0.2  # Relative baseline length
DEFAULT_SHIFT_DIRECTION="y"  # Options: "x" for left-right, "y" for up-down

# Rotated stereo image paths
DEFAULT_STEREO_LEFT_ROTATED_DIR="${BASE_DIR}/stereo/left_rotated"
DEFAULT_STEREO_RIGHT_ROTATED_DIR="${BASE_DIR}/stereo/right_rotated"
DEFAULT_STEREO_DISP_DIR="${BASE_DIR}/stereo/disp"

# Disparity estimation model paths
DEFAULT_FOUNDATION_STEREO_DIR="/mnt/f/algorithm_F/FoundationStereo/scripts"
DEFAULT_FOUNDATION_STEREO_MODEL="/mnt/f/algorithm_F/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
DEFAULT_VALID_ITERS=64
DEFAULT_CUDA_SWITCH_SCRIPT="/mnt/f/algorithm_F/cuda_switch.sh"
DEFAULT_INTRINSIC_FILE="${BASE_DIR}/K.txt"

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
    echo "  --foundation-dir DIR               Directory containing the Foundation Stereo scripts"
    echo "                                     (default: $DEFAULT_FOUNDATION_STEREO_DIR)"
    echo "  --foundation-model FILE            Path to the Foundation Stereo model checkpoint"
    echo "                                     (default: $DEFAULT_FOUNDATION_STEREO_MODEL)"
    echo "  --cuda-switch FILE                 Path to the CUDA switch script"
    echo "                                     (default: $DEFAULT_CUDA_SWITCH_SCRIPT)"
    echo "  --intrinsic-file FILE              Path to the camera intrinsic file (K.txt)"
    echo "                                     (default: $DEFAULT_INTRINSIC_FILE)"
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
        --foundation-dir)
            FOUNDATION_STEREO_DIR="$2"
            shift 2
            ;;
        --foundation-model)
            FOUNDATION_STEREO_MODEL="$2"
            shift 2
            ;;
        --cuda-switch)
            CUDA_SWITCH_SCRIPT="$2"
            shift 2
            ;;
        --intrinsic-file)
            INTRINSIC_FILE="$2"
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

# Set Foundation Stereo paths
FOUNDATION_STEREO_DIR=${FOUNDATION_STEREO_DIR:-$DEFAULT_FOUNDATION_STEREO_DIR}
FOUNDATION_STEREO_MODEL=${FOUNDATION_STEREO_MODEL:-$DEFAULT_FOUNDATION_STEREO_MODEL}
VALID_ITERS=${VALID_ITERS:-$DEFAULT_VALID_ITERS}
CUDA_SWITCH_SCRIPT=${CUDA_SWITCH_SCRIPT:-$DEFAULT_CUDA_SWITCH_SCRIPT}
INTRINSIC_FILE=${INTRINSIC_FILE:-$DEFAULT_INTRINSIC_FILE}

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
#ns-render camera-path --load-config "$CONFIG_PATH" --camera-path-filename "$CAMERA_PATH_JSON" --output-path "$STEREO_LEFT_DIR" --output-format images --image-format png

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
# Generate camera intrinsic file (K.txt)
# =============================================
echo "Generating camera intrinsic file (K.txt)..."

# Extract camera intrinsics from transforms.json
TMP_INTRINSIC_SCRIPT=$(mktemp)
cat > "$TMP_INTRINSIC_SCRIPT" << EOF
import json
import numpy as np

# Load transforms.json to get camera parameters
with open('$TRANSFORMS_JSON', 'r') as f:
    transforms_data = json.load(f)

# Directly use the intrinsic parameters from transforms.json
fl_x = transforms_data.get('fl_x')
fl_y = transforms_data.get('fl_y')
cx = transforms_data.get('cx')
cy = transforms_data.get('cy')

if not all([fl_x, fl_y, cx, cy]):
    print("Error: Could not find camera intrinsic parameters in transforms.json")
    exit(1)

# Create camera intrinsic matrix using the values directly from transforms.json
K = np.array([
    [fl_x, 0, cx],
    [0, fl_y, cy],
    [0, 0, 1]
])

print(f"Using camera intrinsics directly from transforms.json:")
print(f"  - fl_x: {fl_x}")
print(f"  - fl_y: {fl_y}")
print(f"  - cx: {cx}")
print(f"  - cy: {cy}")

# Get scale from dataparser_transforms.json
try:
    with open('$DATAPARSER_TRANSFORMS', 'r') as f:
        dataparser_data = json.load(f)
    
    # Get scale value
    scale = dataparser_data.get('scale')
    
    if scale is None:
        print("Warning: Could not find scale in dataparser_transforms.json, using default scale of 1.0")
        scale = 1.0
    else:
        print(f"Found scale in dataparser_transforms.json: {scale}")
    
    # Calculate baseline based on shift and scale
    shift = $SHIFT
    baseline = shift / scale
    print(f"Calculated baseline: {baseline} (shift: {shift} / scale: {scale})")
    
except Exception as e:
    print(f"Error reading dataparser_transforms.json: {e}")
    print("Using default scale of 1.0")
    scale = 1.0
    baseline = $SHIFT

# Define the output path for the original intrinsics
ORIGINAL_INTRINSIC_FILE = '$CAMERA_PATH_DIR/K_origin.txt'

# Save K matrix to file in Foundation Stereo format:
# - First line: 9 elements of K matrix in a single row (space separated)
# - Second line: baseline value (use absolute value for -y direction)
with open(ORIGINAL_INTRINSIC_FILE, 'w') as f:
    # Write K matrix as a single line (9 elements space-separated)
    k_flat = K.flatten()
    k_string = ' '.join([f"{val}" for val in k_flat])
    f.write(k_string + '\n')
    
    # Use absolute value of baseline for -y direction
    shift_direction = "$SHIFT_DIRECTION"
    if shift_direction.lower() == "-y":
        baseline_to_write = abs(baseline)
        print(f"Using absolute baseline for -y direction: {baseline_to_write} (original: {baseline})")
    else:
        baseline_to_write = baseline
    
    # Write baseline on second line
    f.write(f"{baseline_to_write}\n")

print(f"Original camera intrinsic matrix saved to '{ORIGINAL_INTRINSIC_FILE}'")
print(f"K matrix: {k_string}")
print(f"Baseline: {baseline}")
EOF

# Execute the script to generate K.txt
python "$TMP_INTRINSIC_SCRIPT"
rm "$TMP_INTRINSIC_SCRIPT"

# Check if K_origin.txt was created successfully
ORIGINAL_INTRINSIC_FILE="$CAMERA_PATH_DIR/K_origin.txt"
if [ ! -f "$ORIGINAL_INTRINSIC_FILE" ]; then
    echo "Error: Failed to generate original camera intrinsic file: $ORIGINAL_INTRINSIC_FILE"
    exit 1
fi

# =============================================
# Rotate camera intrinsics and generate final K.txt
# =============================================
echo "Rotating camera intrinsics for final K.txt..."

# Get the path to one of the rotated images to determine dimensions
# Use the left rotated directory, find the first png file
FIRST_ROTATED_IMAGE=$(find "$STEREO_LEFT_ROTATED_DIR" -maxdepth 1 -name '*.png' -print -quit)

if [ -z "$FIRST_ROTATED_IMAGE" ]; then
    echo "Error: No rotated images found in $STEREO_LEFT_ROTATED_DIR to determine dimensions."
    # Fallback: Try the right directory
    FIRST_ROTATED_IMAGE=$(find "$STEREO_RIGHT_ROTATED_DIR" -maxdepth 1 -name '*.png' -print -quit)
    if [ -z "$FIRST_ROTATED_IMAGE" ]; then
        echo "Error: No rotated images found in $STEREO_RIGHT_ROTATED_DIR either. Cannot rotate intrinsics."
        exit 1
    fi
fi

echo "Using image $FIRST_ROTATED_IMAGE to determine rotated dimensions."

# Create a temporary Python script to rotate intrinsics
TMP_ROTATE_K_SCRIPT=$(mktemp)
cat > "$TMP_ROTATE_K_SCRIPT" << EOF
import numpy as np
import json
import sys
import cv2
import os

original_intrinsic_file = "$ORIGINAL_INTRINSIC_FILE"
final_intrinsic_file = "$INTRINSIC_FILE"
shift_direction = "$SHIFT_DIRECTION"
rotated_image_path = "$FIRST_ROTATED_IMAGE"

def rotate_camera_intrinsics(K, rotation_type, image_shape):
    height, width = image_shape[:2] # Use only height and width
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"\n【旋转相机内参】")
    print(f"  原始内参矩阵:")
    print(f"    fx = {fx}, fy = {fy}")
    print(f"    cx = {cx}, cy = {cy}")
    print(f"  原始图像尺寸: {width} x {height}")

    if rotation_type == "90cc":
        new_width, new_height = height, width
        # 90 degrees counter-clockwise rotation
        new_cx = cy
        new_cy = width - 1.0 - cx
        new_K = np.array([
            [fy, 0, new_cx],
            [0, fx, new_cy],
            [0, 0, 1]
        ])
        print(f"  90cc 旋转后图像尺寸: {new_width} x {new_height}")
    elif rotation_type == "90c":
        new_width, new_height = height, width
        # 90 degrees clockwise rotation
        new_cx = height - 1.0 - cy
        new_cy = cx
        new_K = np.array([
            [fy, 0, new_cx],
            [0, fx, new_cy],
            [0, 0, 1]
        ])
        print(f"  90c 旋转后图像尺寸: {new_width} x {new_height}")
    elif rotation_type == "180":
        new_width, new_height = width, height
        # 180 degrees rotation
        new_cx = width - 1.0 - cx
        new_cy = height - 1.0 - cy
        new_K = np.array([
            [fx, 0, new_cx],
            [0, fy, new_cy],
            [0, 0, 1]
        ])
        print(f"  180 旋转后图像尺寸: {new_width} x {new_height}")
    else: # no rotation (none)
        new_K = K
        new_width, new_height = width, height
        print(f"  未执行旋转")

    print(f"  旋转后内参矩阵 ({rotation_type}):")
    print(f"    fx = {new_K[0, 0]}, fy = {new_K[1, 1]}")
    print(f"    cx = {new_K[0, 2]}, cy = {new_K[1, 2]}")
    return new_K

try:
    # Read original intrinsics
    with open(original_intrinsic_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            print(f"Error: Invalid format in {original_intrinsic_file}")
            sys.exit(1)
        k_values = np.array([float(x) for x in lines[0].strip().split()]).reshape(3, 3)
        baseline = float(lines[1].strip())

    # Get original image dimensions from transforms.json
    with open('$TRANSFORMS_JSON', 'r') as f:
        transforms_data = json.load(f)
    
    original_width = int(transforms_data.get('w'))
    original_height = int(transforms_data.get('h'))
    image_shape = (original_height, original_width, 3)  # (height, width, channels)
    
    print(f"Using original image dimensions from transforms.json: {original_width} x {original_height}")

    # Determine rotation type based on shift direction
    # K矩阵旋转方向应该与图像旋转方向一致，与process_depth_maps.py相反
    if shift_direction.lower() == "x":
        rotation_type = "180"
    elif shift_direction.lower() == "-x":
        rotation_type = "none"
    elif shift_direction.lower() == "y":
        rotation_type = "90cc"  # 与process_depth_maps.py的90c相反
    elif shift_direction.lower() == "-y":
        rotation_type = "90c"   # 与process_depth_maps.py的90cc相反
    else:
        rotation_type = "90c" # Default fallback

    # Rotate intrinsics
    rotated_k = rotate_camera_intrinsics(k_values, rotation_type, image_shape)

    # Write final K.txt
    with open(final_intrinsic_file, 'w') as f:
        k_flat = rotated_k.flatten()
        k_string = ' '.join([f"{val}" for val in k_flat])
        f.write(k_string + '\n')
        
        # Use absolute value of baseline for -y direction
        if shift_direction.lower() == "-y":
            baseline_to_write = abs(baseline)
            print(f"Using absolute baseline for -y direction: {baseline_to_write} (original: {baseline})")
        else:
            baseline_to_write = baseline
        
        f.write(f"{baseline_to_write}\n") # Write the baseline

    print(f"\nRotated camera intrinsics saved to: {final_intrinsic_file}")
    print(f"  K matrix: {k_string}")
    print(f"  Baseline: {baseline}")

except Exception as e:
    print(f"Error rotating intrinsics: {e}")
    sys.exit(1)

EOF

# Execute the temporary Python script to rotate K
python3 "$TMP_ROTATE_K_SCRIPT"
EXIT_CODE=$?
rm "$TMP_ROTATE_K_SCRIPT"

# Check if rotation was successful
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to rotate camera intrinsics."
    exit 1
fi
if [ ! -f "$INTRINSIC_FILE" ]; then
    echo "Error: Final intrinsic file $INTRINSIC_FILE was not generated."
    exit 1
fi

# =============================================
# Generate disparity maps using Foundation Stereo
# =============================================
echo "Generating depth maps using Foundation Stereo..."

# First, switch to CUDA 12.8 for Foundation Stereo
echo "Switching to CUDA 12.8 environment for Foundation Stereo..."
source "$CUDA_SWITCH_SCRIPT" 12.8

# Verify the CUDA version after switching
nvcc --version

# Set CUDA environment variables directly for redundancy
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate Foundation Stereo conda environment
echo "Activating Foundation Stereo conda environment..."
eval "$(conda shell.bash hook)"
conda activate foundation

# Run Foundation Stereo batch_process.py for depth map generation directly
echo "Running Foundation Stereo for depth estimation..."
cd /mnt/f/algorithm_F # Change to the base directory
python "$FOUNDATION_STEREO_DIR/batch_process.py" \
    --left_dir "$STEREO_LEFT_ROTATED_DIR" \
    --right_dir "$STEREO_RIGHT_ROTATED_DIR" \
    --intrinsic_file "$INTRINSIC_FILE" \
    --ckpt_dir "$FOUNDATION_STEREO_MODEL" \
    --out_dir "$STEREO_DISP_DIR" \
    --valid_iters $VALID_ITERS

# Check if disparity maps were generated
if [ ! "$(ls -A "$STEREO_DISP_DIR")" ]; then
    echo "Warning: No disparity maps were generated in $STEREO_DISP_DIR"
    exit 1
else
    echo "Disparity estimation completed successfully!"
    echo "Disparity maps saved to: $STEREO_DISP_DIR"
fi

# Deactivate Foundation Stereo conda environment
echo "Deactivating Foundation Stereo conda environment..."
conda deactivate

# Switch back to CUDA 11.8 for nerfstudio
echo "Switching back to CUDA 11.8 environment for nerfstudio..."
source "$CUDA_SWITCH_SCRIPT" 11.8

# Verify the CUDA version after switching
nvcc --version

# Set CUDA environment variables directly for redundancy
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# =============================================
# Process depth maps directly for nerfstudio
# =============================================
echo "Processing disparity maps into rotated depth maps for nerfstudio..."
echo "Source disparity directory: $STEREO_DISP_DIR"
echo "Target depth directory: $DEPTH_DIR"

# Activate nerfstudio conda environment
echo "Activating nerfstudio conda environment..."
eval "$(conda shell.bash hook)"
conda activate nerfstudio

# Construct the path to the depth processing script using BASE_DIR
PROCESS_DEPTH_MAPS_SCRIPT="$BASE_DIR/process_depth_maps.py"

# Run the external Python script for depth processing
echo "Running $PROCESS_DEPTH_MAPS_SCRIPT script..."
echo "Debug: SHIFT_DIRECTION = '$SHIFT_DIRECTION'"
echo "Debug: STEREO_DISP_DIR = '$STEREO_DISP_DIR'"
echo "Debug: DEPTH_DIR = '$DEPTH_DIR'"
echo "Debug: TRANSFORMS_JSON = '$TRANSFORMS_JSON'"

if [ ! -f "$PROCESS_DEPTH_MAPS_SCRIPT" ]; then
    echo "Error: $PROCESS_DEPTH_MAPS_SCRIPT not found!"
    conda deactivate
    exit 1
fi

python3 "$PROCESS_DEPTH_MAPS_SCRIPT" \
    --disp-dir "$STEREO_DISP_DIR" \
    --depth-dir "$DEPTH_DIR" \
    --shift-direction="$SHIFT_DIRECTION" \
    --transforms-json "$TRANSFORMS_JSON"

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Error: process_depth_maps.py script failed."
    conda deactivate
    exit 1
fi

# Deactivate nerfstudio conda environment (moved from inside the removed block)
echo "Deactivating nerfstudio conda environment..."
conda deactivate

echo "Depth map processing for nerfstudio completed successfully!"
echo "Original disparity maps: $STEREO_DISP_DIR"
echo "Processed and rotated depth maps: $DEPTH_DIR"

# =============================================
# Train depth-nerfacto model
# =============================================
echo "Training depth-nerfacto model..."

# Get the directory containing transforms.json
# TRANSFORMS_DIR=$(dirname "$TRANSFORMS_JSON") # No longer needed if using BASE_DIR

# Activate nerfstudio environment again for training
echo "Activating nerfstudio conda environment for training..."
eval "$(conda shell.bash hook)"
conda activate nerfstudio

# Run depth-nerfacto training, using BASE_DIR as the data path
echo "Starting depth-nerfacto training in $BASE_DIR..."
cd "$BASE_DIR" # Change to the base directory before training
ns-train depth-nerfacto --data "$BASE_DIR" --max-num-iterations 6000

# Deactivate nerfstudio conda environment
echo "Deactivating nerfstudio conda environment..."
conda deactivate

echo "Depth-nerfacto training completed!"

echo "The entire stereo processing pipeline has been completed successfully!" 
