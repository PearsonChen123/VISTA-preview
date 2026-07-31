#!/bin/bash

# Script name: convert_to_stereo_camera_paths.sh
# Function: Convert left and right eye transforms files to camera_path format

# Default paths
DEFAULT_LEFT_TRANSFORMS_FILE="/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/transforms_train.json"
DEFAULT_RIGHT_TRANSFORMS_FILE="/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/transforms_train_right.json"
DEFAULT_ORIGINAL_TRANSFORM_FILE="/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/undistorted/transforms.json"
DEFAULT_LEFT_OUTPUT_FILE="/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/camera_path.json"
DEFAULT_RIGHT_OUTPUT_FILE="/mnt/d/research/nerfstudio/nerfstudio/data/clinical_3/cam_path/camera_path_right.json"

# Function: Display usage instructions
show_usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                         Show this help message"
    echo "  -l, --left-transforms FILE         Path to the left eye transforms JSON file"
    echo "                                     (default: $DEFAULT_LEFT_TRANSFORMS_FILE)"
    echo "  -r, --right-transforms FILE        Path to the right eye transforms JSON file"
    echo "                                     (default: $DEFAULT_RIGHT_TRANSFORMS_FILE)"
    echo "  -o, --original-transforms FILE     Path to the original transforms JSON file"
    echo "                                     (default: $DEFAULT_ORIGINAL_TRANSFORM_FILE)"
    echo "  -c, --config FILE                  Path to Nerfstudio config.yml (enables auto normalization)"
    echo "                                     (optional)"
    echo "  --left-output FILE                 Path to the output left camera path JSON file"
    echo "                                     (default: $DEFAULT_LEFT_OUTPUT_FILE)"
    echo "  --right-output FILE                Path to the output right camera path JSON file"
    echo "                                     (default: $DEFAULT_RIGHT_OUTPUT_FILE)"
    echo
    echo "Example:"
    echo "  $0 --left-transforms path/to/left.json --right-transforms path/to/right.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--left-transforms)
            LEFT_TRANSFORMS_FILE="$2"
            shift 2
            ;;
        -r|--right-transforms)
            RIGHT_TRANSFORMS_FILE="$2"
            shift 2
            ;;
        -o|--original-transforms)
            ORIGINAL_TRANSFORM_FILE="$2"
            shift 2
            ;;
        --left-output)
            LEFT_OUTPUT_FILE="$2"
            shift 2
            ;;
        --right-output)
            RIGHT_OUTPUT_FILE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set parameters, using default values or command line arguments
LEFT_TRANSFORMS_FILE=${LEFT_TRANSFORMS_FILE:-$DEFAULT_LEFT_TRANSFORMS_FILE}
RIGHT_TRANSFORMS_FILE=${RIGHT_TRANSFORMS_FILE:-$DEFAULT_RIGHT_TRANSFORMS_FILE}
ORIGINAL_TRANSFORM_FILE=${ORIGINAL_TRANSFORM_FILE:-$DEFAULT_ORIGINAL_TRANSFORM_FILE}
LEFT_OUTPUT_FILE=${LEFT_OUTPUT_FILE:-$DEFAULT_LEFT_OUTPUT_FILE}
RIGHT_OUTPUT_FILE=${RIGHT_OUTPUT_FILE:-$DEFAULT_RIGHT_OUTPUT_FILE}
CONFIG_PATH=${CONFIG_PATH:-}

CONFIG_ARG=()
if [ -n "$CONFIG_PATH" ]; then
    CONFIG_ARG=(--config "$CONFIG_PATH")
fi

# Check if files exist
if [ ! -f "$ORIGINAL_TRANSFORM_FILE" ]; then
    echo "Error: Original transforms file does not exist: $ORIGINAL_TRANSFORM_FILE"
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Create output directories (if they don't exist)
mkdir -p "$(dirname "$LEFT_OUTPUT_FILE")"
mkdir -p "$(dirname "$RIGHT_OUTPUT_FILE")"

# Convert left eye camera path
if [ -f "$LEFT_TRANSFORMS_FILE" ]; then
    echo "Converting left eye transforms to camera path..."
    python ./colmap_convert_to_camera_path.py \
        "$LEFT_TRANSFORMS_FILE" \
        "$ORIGINAL_TRANSFORM_FILE" \
        "$LEFT_OUTPUT_FILE" \
        "${CONFIG_ARG[@]}"
    
    # Check Python script execution result
    if [ $? -eq 0 ]; then
        echo "Left eye camera path conversion completed successfully."
        echo "Output saved to: $LEFT_OUTPUT_FILE"
    else
        echo "Error: Left eye camera path conversion failed."
        exit 1
    fi
else
    echo "Warning: Left eye transforms file does not exist: $LEFT_TRANSFORMS_FILE"
    echo "Skipping left eye camera path conversion."
fi

# Convert right eye camera path
if [ -f "$RIGHT_TRANSFORMS_FILE" ]; then
    echo "Converting right eye transforms to camera path..."
    python ./colmap_convert_to_camera_path.py \
        "$RIGHT_TRANSFORMS_FILE" \
        "$ORIGINAL_TRANSFORM_FILE" \
        "$RIGHT_OUTPUT_FILE" \
        "${CONFIG_ARG[@]}"
    
    # Check Python script execution result
    if [ $? -eq 0 ]; then
        echo "Right eye camera path conversion completed successfully."
        echo "Output saved to: $RIGHT_OUTPUT_FILE"
    else
        echo "Error: Right eye camera path conversion failed."
        exit 1
    fi
else
    echo "Warning: Right eye transforms file does not exist: $RIGHT_TRANSFORMS_FILE"
    echo "Skipping right eye camera path conversion."
fi

echo "Stereo camera paths conversion process completed." 
