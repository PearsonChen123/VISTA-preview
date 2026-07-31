#!/bin/bash

# Script to run MonSter on RGBD-500 rotated stereo pairs
# Converts disparity to depth and unrotates back to original orientation

BASE_DIR="/mnt/h/RGBD-500"
MONSTER_PATH="/mnt/h/MonSter"

# Input directories
LEFT_DIR="${BASE_DIR}/stereo/left_rotated"
RIGHT_DIR="${BASE_DIR}/stereo/right_rotated"

# Output directory
OUTPUT_DIR="${BASE_DIR}/stereo/depth_monster"

# Camera intrinsics (after rotation)
INTRINSIC_FILE="${BASE_DIR}/K.txt"

# Shift direction (from process_stereo_foundation.sh)
SHIFT_DIRECTION="x"

# MonSter checkpoint
# Default: mix_all (mix of all datasets) - best for zero-shot generalization
CHECKPOINT="${MONSTER_PATH}/mix_all.pth"

# Model parameters
VALID_ITERS=64
ENCODER="vitl"  # Options: vits, vitb, vitl, vitg

# Default: save visualization
SAVE_VIS="--save_visualization"

# Parse arguments for custom checkpoint
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --shift-direction)
            SHIFT_DIRECTION="$2"
            shift 2
            ;;
        --valid-iters)
            VALID_ITERS="$2"
            shift 2
            ;;
        --encoder)
            ENCODER="$2"
            shift 2
            ;;
        --save-disparity)
            SAVE_DISPARITY="--save_disparity"
            shift
            ;;
        --no-vis)
            SAVE_VIS=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH        Path to MonSter checkpoint"
            echo "                          (default: mix_all.pth)"
            echo "  --shift-direction DIR   Shift direction: x, -x, y, -y"
            echo "                          (default: y)"
            echo "  --valid-iters N         Number of iterations (default: 32)"
            echo "  --encoder TYPE          Encoder type: vits, vitb, vitl, vitg"
            echo "                          (default: vitl)"
            echo "  --save-disparity        Save disparity maps"
            echo "  --no-vis                Don't save visualizations (default: save)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0                                    # Run with visualization"
            echo "  $0 --save-disparity                   # Save both depth and disparity"
            echo "  $0 --no-vis                           # Don't save visualization"
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
if [ ! -d "$LEFT_DIR" ]; then
    echo "Error: Left image directory not found: $LEFT_DIR"
    exit 1
fi

if [ ! -d "$RIGHT_DIR" ]; then
    echo "Error: Right image directory not found: $RIGHT_DIR"
    exit 1
fi

if [ ! -f "$INTRINSIC_FILE" ]; then
    echo "Error: Camera intrinsic file not found: $INTRINSIC_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Please download the pretrained model first:"
    echo "  wget https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth -P $MONSTER_PATH"
    echo ""
    echo "Or download from: https://huggingface.co/cjd24/MonSter"
    exit 1
fi

# Display configuration
echo "========================================"
echo "Run MonSter on RGBD-500"
echo "========================================"
echo "Left images:       $LEFT_DIR"
echo "Right images:      $RIGHT_DIR"
echo "Output depth:      $OUTPUT_DIR"
echo "Checkpoint:        $CHECKPOINT"
echo "Intrinsics:        $INTRINSIC_FILE"
echo "Shift direction:   $SHIFT_DIRECTION"
echo "Valid iterations:  $VALID_ITERS"
echo "Encoder:           $ENCODER"
if [ -n "$SAVE_VIS" ]; then
    echo "Visualization:     Enabled"
else
    echo "Visualization:     Disabled"
fi
echo "========================================"
echo ""

# Activate monster conda environment
eval "$(conda shell.bash hook)"
conda activate monster

# Run MonSter
python "${BASE_DIR}/run_monster.py" \
    --restore_ckpt "$CHECKPOINT" \
    --left_dir "$LEFT_DIR" \
    --right_dir "$RIGHT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --intrinsic_file "$INTRINSIC_FILE" \
    --shift_direction "$SHIFT_DIRECTION" \
    --valid_iters "$VALID_ITERS" \
    --encoder "$ENCODER" \
    $SAVE_DISPARITY \
    $SAVE_VIS

EXIT_CODE=$?

# Deactivate environment
conda deactivate

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "MonSter processing completed!"
    echo "========================================"
    echo "Depth maps saved to:"
    echo "  $OUTPUT_DIR"
    if [ -n "$SAVE_VIS" ]; then
        echo "Visualizations saved to:"
        echo "  ${OUTPUT_DIR}_vis"
    fi
    if [ -n "$SAVE_DISPARITY" ]; then
        echo "Disparity maps saved to:"
        echo "  ${OUTPUT_DIR/depth/disp}"
    fi
else
    echo ""
    echo "Error: MonSter processing failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
