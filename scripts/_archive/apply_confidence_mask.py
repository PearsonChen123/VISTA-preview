#!/usr/bin/env python3
"""
Apply confidence mask to existing depth maps
Filters depth maps based on confidence threshold
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2


def filter_by_uncertainty_percentile(uncertainty, valid_mask, percentile_threshold):
    """
    Create binary mask based on uncertainty percentile

    Args:
        uncertainty: Uncertainty map (lower is better)
        valid_mask: Valid pixel mask
        percentile_threshold: Keep pixels with uncertainty <= this percentile

    Returns:
        Binary mask of pixels to keep
    """
    filtered_mask = np.zeros_like(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        return filtered_mask

    valid_uncertainty = uncertainty[valid_mask]

    # Calculate the uncertainty threshold at the given percentile
    uncert_threshold = np.percentile(valid_uncertainty, percentile_threshold * 100)

    # Keep pixels with uncertainty <= threshold
    filtered_mask = valid_mask & (uncertainty <= uncert_threshold)

    return filtered_mask


def apply_confidence_mask(depth_path, conf_path, threshold, output_path):
    """
    Apply confidence filtering to a depth map

    Args:
        depth_path: Path to depth .npy file
        conf_path: Path to confidence .npy file (from SEDNet inference)
        threshold: Confidence percentile threshold
        output_path: Path to save filtered depth map
    """
    # Load depth and confidence
    depth = np.load(depth_path).astype(np.float32)
    confidence = np.load(conf_path).astype(np.float32)

    # Valid mask (where depth > 0)
    valid_mask = depth > 0

    # Apply confidence filtering
    keep_mask = confidence >= threshold

    # Create filtered depth
    filtered_depth = np.copy(depth)
    filtered_depth[~keep_mask] = 0

    # Save result
    np.save(output_path, filtered_depth)

    return keep_mask.sum(), valid_mask.sum()


def apply_uncertainty_mask(depth_path, uncertainty_path, percentile, output_path):
    """
    Apply uncertainty-based filtering to a depth map

    Args:
        depth_path: Path to depth .npy file
        uncertainty_path: Path to uncertainty .npy file (raw uncertainty from SEDNet)
        percentile: Percentile threshold (e.g., 0.1 = keep best 10%)
        output_path: Path to save filtered depth map
    """
    # Load depth and uncertainty
    depth = np.load(depth_path).astype(np.float32)
    uncertainty = np.load(uncertainty_path).astype(np.float32)

    # Valid mask (where depth > 0)
    valid_mask = depth > 0

    # Filter by uncertainty percentile
    keep_mask = filter_by_uncertainty_percentile(uncertainty, valid_mask, percentile)

    # Create filtered depth
    filtered_depth = np.copy(depth)
    filtered_depth[~keep_mask] = 0

    # Save result
    np.save(output_path, filtered_depth)

    return keep_mask.sum(), valid_mask.sum()


def main():
    parser = argparse.ArgumentParser(
        description='Apply confidence mask to existing depth maps'
    )
    parser.add_argument('--depth_dir', type=str, required=True,
                        help='Directory containing depth .npy files')
    parser.add_argument('--confidence_dir', type=str, required=True,
                        help='Directory containing confidence .npy files (from SEDNet inference)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for filtered depth maps')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (0.0-1.0). Keep pixels with confidence >= threshold')
    parser.add_argument('--use_percentile', action='store_true',
                        help='If set, treat confidence as uncertainty and use percentile filtering. '
                             'threshold then means: 0.1 = keep best 10%%')
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save visualization images')

    args = parser.parse_args()

    # Setup paths
    depth_dir = Path(args.depth_dir)
    confidence_dir = Path(args.confidence_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of depth files
    depth_files = sorted(depth_dir.glob("*.npy"))
    if not depth_files:
        raise ValueError(f"No .npy files found in {depth_dir}")

    print(f"Found {len(depth_files)} depth maps to process")
    print(f"Confidence directory: {confidence_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {args.threshold}")
    if args.use_percentile:
        print(f"Mode: Percentile filtering (keep best {args.threshold*100:.1f}%)")
    else:
        print(f"Mode: Confidence threshold (keep pixels >= {args.threshold})")
    print()

    # Process each depth map
    total_kept = 0
    total_valid = 0

    for depth_path in tqdm(depth_files, desc="Applying confidence masks"):
        # Find corresponding confidence file
        stem = depth_path.stem
        conf_path = confidence_dir / f"{stem}.npy"

        if not conf_path.exists():
            print(f"Warning: Confidence file not found for {stem}, skipping")
            continue

        try:
            # Apply filtering
            if args.use_percentile:
                kept, valid = apply_uncertainty_mask(
                    depth_path, conf_path, args.threshold,
                    output_dir / f"{stem}.npy"
                )
            else:
                kept, valid = apply_confidence_mask(
                    depth_path, conf_path, args.threshold,
                    output_dir / f"{stem}.npy"
                )

            total_kept += kept
            total_valid += valid

            # Optional visualization
            if args.save_visualization:
                filtered_depth = np.load(output_dir / f"{stem}.npy")
                valid_depth = filtered_depth[filtered_depth > 0]

                if len(valid_depth) > 0:
                    depth_min = np.percentile(valid_depth, 5)
                    depth_max = np.percentile(valid_depth, 95)
                    depth_vis = np.zeros_like(filtered_depth, dtype=np.uint8)
                    depth_mask = filtered_depth > 0
                    depth_normalized = np.clip(
                        (filtered_depth[depth_mask] - depth_min) / (depth_max - depth_min + 1e-6),
                        0, 1
                    )
                    depth_vis[depth_mask] = (depth_normalized * 255).astype(np.uint8)

                    # Apply colormap
                    depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                    depth_vis_colored[~depth_mask] = 0

                    cv2.imwrite(str(output_dir / f"{stem}_vis.png"), depth_vis_colored)

        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue

    # Print statistics
    if total_valid > 0:
        keep_ratio = total_kept / total_valid * 100
        print(f"\nStatistics:")
        print(f"  Total valid pixels: {total_valid:,}")
        print(f"  Kept pixels: {total_kept:,} ({keep_ratio:.2f}%)")
        print(f"  Filtered out: {total_valid - total_kept:,} ({100-keep_ratio:.2f}%)")
    else:
        print("\nNo valid pixels found!")

    print(f"\nFiltered depth maps saved to: {output_dir}")


if __name__ == '__main__':
    main()
