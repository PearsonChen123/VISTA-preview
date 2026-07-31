#!/bin/bash

# Script to apply confidence mask to existing depth maps
# Filters /mnt/h/RGBD-500/stereo/depth using confidence from SEDNet
# Saves filtered results to /mnt/h/RGBD-500/stereo/depth_with_conf

BASE_DIR="/mnt/h/RGBD-500"

# Input directories
DEPTH_DIR="${BASE_DIR}/stereo/depth"
CONFIDENCE_DIR="${BASE_DIR}/stereo/confidence"

# Output directory
OUTPUT_DIR="${BASE_DIR}/stereo/depth_with_conf"

# Default threshold
CONFIDENCE_THRESHOLD=0.2

# Parse arguments
USE_PERCENTILE=""
SAVE_VIS=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --threshold)
            CONFIDENCE_THRESHOLD="$2"
            shift 2
            ;;
        --percentile)
            USE_PERCENTILE="--use_percentile"
            shift
            ;;
        --save-vis)
            SAVE_VIS="--save_visualization"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --threshold FLOAT    Confidence threshold (default: 0.5)"
            echo "                       Without --percentile: keep pixels with confidence >= threshold"
            echo "                       With --percentile: keep best N% (e.g., 0.1 = best 10%)"
            echo "  --percentile         Use percentile mode (threshold as fraction of best pixels)"
            echo "  --save-vis           Save visualization images"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Keep pixels with confidence >= 0.5"
            echo "  $0 --threshold 0.5"
            echo ""
            echo "  # Keep best 10% pixels by uncertainty"
            echo "  $0 --threshold 0.1 --percentile"
            echo ""
            echo "  # Keep best 20% with visualization"
            echo "  $0 --threshold 0.2 --percentile --save-vis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -d "$DEPTH_DIR" ]; then
    echo "Error: Depth directory not found: $DEPTH_DIR"
    exit 1
fi

if [ ! -d "$CONFIDENCE_DIR" ]; then
    echo "Error: Confidence directory not found: $CONFIDENCE_DIR"
    echo "Please run SEDNet inference first to generate confidence maps"
    exit 1
fi

# Display configuration
echo "========================================"
echo "Apply Confidence Mask to Depth Maps"
echo "========================================"
echo "Input depth directory:  $DEPTH_DIR"
echo "Confidence directory:   $CONFIDENCE_DIR"
echo "Output directory:       $OUTPUT_DIR"
echo "Threshold:              $CONFIDENCE_THRESHOLD"
if [ -n "$USE_PERCENTILE" ]; then
    echo "Mode:                   Percentile (keep best ${CONFIDENCE_THRESHOLD})"
else
    echo "Mode:                   Confidence threshold (>= ${CONFIDENCE_THRESHOLD})"
fi
echo "========================================"
echo ""

# Activate nerfstudio environment
eval "$(conda shell.bash hook)"
conda activate nerfstudio

# Run filtering
python "${BASE_DIR}/apply_confidence_mask.py" \
    --depth_dir "$DEPTH_DIR" \
    --confidence_dir "$CONFIDENCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --threshold "$CONFIDENCE_THRESHOLD" \
    $USE_PERCENTILE \
    $SAVE_VIS

EXIT_CODE=$?

# Deactivate environment
conda deactivate

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Filtering completed successfully!"
    echo "========================================"
    echo "Filtered depth maps saved to:"
    echo "  $OUTPUT_DIR"
else
    echo ""
    echo "Error: Filtering failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
