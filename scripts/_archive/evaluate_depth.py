import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_scale_factor_max_delta(
    gt_valid: np.ndarray,
    pred_valid: np.ndarray,
    threshold: float = 1.05,
) -> float:
    """Find scale that maximizes δ<threshold coverage (default 5% band)."""
    if gt_valid.size == 0 or pred_valid.size == 0:
        return 1.0

    finite_mask = (
        np.isfinite(gt_valid)
        & np.isfinite(pred_valid)
        & (pred_valid > 0)
        & (gt_valid > 0)
    )
    if not np.any(finite_mask):
        return 1.0

    gt_valid = gt_valid[finite_mask].astype(np.float64, copy=False)
    pred_valid = pred_valid[finite_mask].astype(np.float64, copy=False)

    ratios = gt_valid / pred_valid
    positive_mask = ratios > 0
    if not np.any(positive_mask):
        return 1.0
    ratios = ratios[positive_mask]

    t = float(threshold)
    if t <= 1.0:
        raise ValueError('delta threshold must be greater than 1.0')

    lows = ratios / t
    highs = ratios * t

    values = np.concatenate([lows, highs])
    deltas = np.concatenate([
        np.ones_like(lows, dtype=np.int32),
        -np.ones_like(highs, dtype=np.int32),
    ])

    order = np.lexsort((-deltas, values))

    count = 0
    best_count = 0
    best_scale = 1.0

    for idx in order:
        count += deltas[idx]
        if count > best_count and values[idx] > 0:
            best_count = count
            best_scale = values[idx]

    if best_count == 0:
        denom = np.sum(pred_valid ** 2)
        if denom > 0:
            return float(np.sum(gt_valid * pred_valid) / denom)
        return 1.0

    return float(best_scale)

def compute_depth_errors(gt: np.ndarray, pred: np.ndarray, extra_mask: Optional[np.ndarray] = None):
    """
    Computes depth evaluation metrics.
    Args:
        gt (np.ndarray): Ground truth depth map.
        pred (np.ndarray): Predicted depth map.
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Create a mask for valid pixels: only evaluate where GT depth > 0 and not NaN
    # This excludes all pixels where GT depth is 0, negative, or invalid
    mask = (gt > 0) & np.isfinite(gt)
    mask &= np.isfinite(pred) & (pred > 0)
    if extra_mask is not None:
        mask &= extra_mask

    # Count total and valid pixels for information
    total_pixels = gt.size
    valid_pixels = np.sum(mask)

    if valid_pixels == 0:
        return None

    # Ensure prediction has no NaNs or infs in the masked region for calculations
    pred = np.nan_to_num(pred)

    # Filter arrays using the mask - only GT > 0 pixels are evaluated
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    # Scale predictions to maximize δ<1.05 (5% relative error) coverage
    scale_factor = compute_scale_factor_max_delta(gt_valid, pred_valid, threshold=1.05)

    pred_scaled = pred * scale_factor
    pred_valid_scaled = pred_valid * scale_factor
    
    # --- Metric Calculation ---
    # Abs_Rel
    abs_rel = np.mean(np.abs(gt_valid - pred_valid_scaled) / gt_valid)
    # Sq_Rel
    sq_rel = np.mean(((gt_valid - pred_valid_scaled)**2) / gt_valid)
    # RMSE
    rmse = np.sqrt(np.mean((gt_valid - pred_valid_scaled)**2))

    # Delta thresholds
    ratio = np.maximum(gt_valid / pred_valid_scaled, pred_valid_scaled / gt_valid)
    delta_5 = np.mean((ratio < 1.05).astype(float))
    delta_10 = np.mean((ratio < 1.10).astype(float))
    delta_25 = np.mean((ratio < 1.25).astype(float)) # Using 1.25 as a standard comparable threshold

    return {
        "scale_factor": scale_factor,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "delta < 5%": delta_5,
        "delta < 10%": delta_10,
        "delta < 25%": delta_25,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels
    }


def refine_mask_remove_outliers(
    pred: np.ndarray,
    base_mask: Optional[np.ndarray],
    mad_k: float = 3.5,
) -> np.ndarray:
    """Prediction-only robust outlier removal using MAD on log(pred).

    Pixels are considered valid if:
      - base_mask (if provided)
      - pred is finite and > 0
      - |log(pred) - median| <= mad_k * 1.4826 * MAD

    If filtering removes all pixels, falls back to the unfiltered valid mask.
    """
    valid = np.isfinite(pred) & (pred > 0)
    if base_mask is not None:
        valid &= base_mask

    if not np.any(valid):
        return valid

    vals = pred[valid]
    log_vals = np.log(vals)
    med = np.median(log_vals)
    mad = np.median(np.abs(log_vals - med))
    # Avoid division by zero; if MAD is 0, keep original mask
    if mad == 0 or not np.isfinite(mad):
        return valid

    robust_threshold = mad_k * 1.4826 * mad
    with np.errstate(divide='ignore', invalid='ignore'):
        log_full = np.log(np.where(valid, pred, 1.0))
        robust_ok = np.abs(log_full - med) <= robust_threshold

    refined = valid & robust_ok
    if not np.any(refined):
        return valid
    return refined

def create_visualization(
    gt_depth: np.ndarray,
    pred_depth_resized: np.ndarray,
    rgb_image: Optional[np.ndarray],
    output_path: str,
    gt_filename: str,
    pred_filename: str,
    valid_mask: Optional[np.ndarray] = None,
):
    """
    Create a visualization showing GT depth, predicted depth, and RGB image side by side.
    """
    # Apply scale factor for prediction visualization (same as in evaluation)
    combined_mask = (gt_depth > 0) & np.isfinite(gt_depth)
    pred_no_nan = np.nan_to_num(pred_depth_resized)
    combined_mask &= np.isfinite(pred_no_nan) & (pred_no_nan > 0)
    if valid_mask is not None:
        combined_mask &= valid_mask
    gt_valid = gt_depth[combined_mask]
    pred_valid = pred_no_nan[combined_mask]
    
    scale_factor = compute_scale_factor_max_delta(gt_valid, pred_valid, threshold=1.05)
    
    pred_scaled = pred_depth_resized * scale_factor
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB Image
    if rgb_image is not None:
        axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('RGB Image', fontsize=12)
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'RGB Not Found', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('RGB Image', fontsize=12)
        axes[0].axis('off')
    
    # GT Depth
    gt_vis = np.nan_to_num(gt_depth, nan=0.0)
    if valid_mask is not None:
        gt_vis = np.where(valid_mask, gt_vis, np.nan)
    if np.any(gt_vis > 0):
        gt_vmax = np.percentile(gt_vis[gt_vis > 0], 95)
        gt_vmin = np.percentile(gt_vis[gt_vis > 0], 5)
    else:
        gt_vmax, gt_vmin = 1.0, 0.0
    
    im1 = axes[1].imshow(gt_vis, cmap='viridis', vmin=gt_vmin, vmax=gt_vmax)
    axes[1].set_title(f'GT Depth\\n{os.path.basename(gt_filename)}', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Predicted Depth (scaled)
    pred_vis = np.nan_to_num(pred_scaled, nan=0.0)
    if valid_mask is not None:
        pred_vis = np.where(valid_mask, pred_vis, np.nan)
    im2 = axes[2].imshow(pred_vis, cmap='viridis', vmin=gt_vmin, vmax=gt_vmax)  # Use same scale as GT
    axes[2].set_title(f'Predicted Depth (scaled)\\n{os.path.basename(pred_filename)}\\nScale: {scale_factor:.3f}', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_colmap_depth(path: Path) -> np.ndarray:
    with path.open('rb') as f:
        header_bytes = []
        ampersands = 0
        while True:
            byte = f.read(1)
            if byte == b'':
                break
            header_bytes.append(byte)
            if byte == b'&':
                ampersands += 1
                if ampersands == 3:
                    break
        header = b''.join(header_bytes).decode('utf-8')
        parts = [p for p in header.split('&') if p]
        if len(parts) < 3:
            raise ValueError(f"Invalid COLMAP depth map header in {path}: '{header}'")
        width, height = int(parts[0]), int(parts[1])
        data = f.read()
    depth = np.frombuffer(data, dtype=np.float32)
    if depth.size != width * height:
        raise ValueError(
            f"Depth size mismatch for {path}: expected {width*height} values, got {depth.size}"
        )
    return depth.reshape((height, width))


def load_ground_truth_depth(
    path: Path,
    gt_format: str,
    invalid_values: Iterable[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if gt_format == 'npy':
        depth = np.load(path)
    elif gt_format == 'png':
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth PNG: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
    else:
        raise ValueError(f"Unsupported GT format '{gt_format}' for {path}")

    valid_mask: Optional[np.ndarray] = None
    invalid_values = list(invalid_values)
    if invalid_values:
        valid_mask = np.ones_like(depth, dtype=bool)
        for value in invalid_values:
            if np.isnan(value):
                valid_mask &= ~np.isnan(depth)
            else:
                valid_mask &= ~np.isclose(depth, value)

    return depth.astype(np.float32), valid_mask


def extract_stem(path: Path) -> str:
    name = path.name
    if name.endswith('.npy'):
        return name[:-4]
    if name.endswith('.bin'):
        name = name[:-4]
    return name.split('.')[0]


def find_rgb_path(rgb_dir: Optional[Path], stem: str) -> Optional[str]:
    if rgb_dir is None:
        return None
    for ext in ('.png', '.jpg', '.jpeg'):
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def _colmap_variant_from_name(path: Path) -> str:
    stem_parts = Path(path).stem.split('.')
    return stem_parts[-1] if len(stem_parts) >= 3 else ''


def collect_file_pairs(
    gt_dir: Path,
    pred_dir: Path,
    gt_format: str,
    pred_format: str,
    colmap_variant: str,
    rgb_dir: Optional[Path],
) -> List[Tuple[str, str, Optional[str]]]:
    if gt_format == 'png':
        gt_files = sorted(gt_dir.glob('*.png'))
    else:
        gt_files = sorted(gt_dir.glob('*.npy'))

    if not gt_files:
        raise FileNotFoundError(f"No ground-truth depth files found in {gt_dir}")

    if pred_format == 'colmap':
        if colmap_variant == 'auto':
            pred_candidates = sorted(pred_dir.glob('*.bin'))
        else:
            pattern = f"*.{colmap_variant}.bin"
            pred_candidates = sorted(pred_dir.glob(pattern))
        if not pred_candidates:
            raise FileNotFoundError(f"No COLMAP depth files found in {pred_dir}")
    elif pred_format == 'png':
        pred_candidates = sorted(pred_dir.glob('*.png'))
        if not pred_candidates:
            raise FileNotFoundError(f"No prediction depth files found in {pred_dir}")
    else:
        pred_candidates = sorted(pred_dir.glob('*.npy'))
        if not pred_candidates:
            raise FileNotFoundError(f"No prediction depth files found in {pred_dir}")

    pred_map: Dict[str, Path] = {}
    if pred_format == 'colmap' and colmap_variant == 'auto':
        for candidate in pred_candidates:
            stem = extract_stem(candidate)
            variant = _colmap_variant_from_name(candidate)
            existing = pred_map.get(stem)
            if existing is None:
                pred_map[stem] = candidate
            else:
                existing_variant = _colmap_variant_from_name(existing)
                if existing_variant != 'geometric' and variant == 'geometric':
                    pred_map[stem] = candidate
    else:
        for candidate in pred_candidates:
            pred_map[extract_stem(candidate)] = candidate

    pairs: List[Tuple[str, str, Optional[str]]] = []
    missing = []
    for gt_path in gt_files:
        stem = extract_stem(gt_path)
        pred_path = pred_map.get(stem)
        if pred_path is None:
            missing.append(stem)
            continue
        rgb_path = find_rgb_path(rgb_dir, stem) if rgb_dir else None
        pairs.append((str(gt_path), str(pred_path), rgb_path))

    if missing:
        print(f"Warning: missing predictions for {len(missing)} frames (e.g., {missing[:5]})")

    if not pairs and len(gt_files) == len(pred_candidates):
        print('Falling back to index-based pairing due to naming mismatch.')
        for gt_path, pred_path in zip(gt_files, pred_candidates):
            stem = extract_stem(gt_path)
            rgb_path = find_rgb_path(rgb_dir, stem) if rgb_dir else None
            pairs.append((str(gt_path), str(pred_path), rgb_path))

    return pairs

PROCESS_CONFIG: dict = {}


def _init_pool(config: dict):  # pragma: no cover - pool initializer
    global PROCESS_CONFIG
    PROCESS_CONFIG = config


def _run_evaluation(args_list: List[Tuple[int, str, str, Optional[str], bool]], workers: int, config: dict):
    _init_pool(config)
    if workers <= 1:
        return [process_single_pair(*args) for args in args_list]

    try:
        with Pool(processes=workers, initializer=_init_pool, initargs=(config,)) as pool:
            return pool.starmap(process_single_pair, args_list)
    except (PermissionError, OSError) as exc:
        print(f"Multiprocessing unavailable ({exc}); falling back to single-process mode.")
        _init_pool(config)
        return [process_single_pair(*args) for args in args_list]


def process_single_pair(i: int, gt_path: str, pred_path: str, rgb_path: Optional[str], create_vis: bool):
    cfg = PROCESS_CONFIG

    try:
        gt_depth, gt_valid_mask = load_ground_truth_depth(
            Path(gt_path),
            cfg['gt_format'],
            cfg['invalid_gt_values'],
        )

        if cfg['pred_format'] == 'colmap':
            pred_depth = load_colmap_depth(Path(pred_path))
        elif cfg['pred_format'] == 'png':
            pred_depth = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
            if pred_depth is None:
                raise ValueError(f"Failed to read prediction PNG: {pred_path}")
            if pred_depth.ndim == 3:
                pred_depth = pred_depth[..., 0]
            pred_depth = pred_depth.astype(np.float32)
        else:
            pred_depth = np.load(pred_path).astype(np.float32)

        if pred_depth.shape != gt_depth.shape:
            target_shape = (gt_depth.shape[1], gt_depth.shape[0])
            pred_depth = cv2.resize(pred_depth, target_shape, interpolation=cv2.INTER_LINEAR)

        percentile_mask = None
        if cfg['apply_percentile']:
            pred_valid = np.isfinite(pred_depth) & (pred_depth > 0)
            if np.any(pred_valid):
                lo, hi = np.percentile(
                    pred_depth[pred_valid],
                    [cfg['percentile_min'], cfg['percentile_max']],
                )
                percentile_mask = pred_valid & (pred_depth >= lo) & (pred_depth <= hi)
            else:
                percentile_mask = np.zeros_like(pred_depth, dtype=bool)

        rgb_image = None
        if rgb_path and os.path.exists(rgb_path):
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is not None and rgb_image.shape[:2] != gt_depth.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (gt_depth.shape[1], gt_depth.shape[0]))

        combined_mask = percentile_mask
        if gt_valid_mask is not None:
            combined_mask = gt_valid_mask if combined_mask is None else (combined_mask & gt_valid_mask)

        # Optional outlier removal before scaling and evaluation (prediction-only)
        if cfg.get('remove_outliers', False):
            combined_mask = refine_mask_remove_outliers(
                pred_depth,
                combined_mask,
                cfg.get('outlier_mad_k', 3.5),
            )

        errors = compute_depth_errors(gt_depth, pred_depth, combined_mask)
        if errors is None:
            raise ValueError("No valid pixels after masking")

        if create_vis and cfg['enable_vis']:
            plot_filename = f"comparison_{i:03d}_{Path(pred_path).stem}.png"
            plot_path = cfg['plots_dir'] / plot_filename
            create_visualization(
                gt_depth,
                pred_depth,
                rgb_image,
                str(plot_path),
                gt_path,
                pred_path,
                combined_mask,
            )

        return i, errors, True

    except Exception as exc:
        print(
            f"Error processing pair {i}: {os.path.basename(gt_path)} and {os.path.basename(pred_path)}. Error: {exc}"
        )
        return i, None, False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth predictions against ground truth.")
    parser.add_argument('--gt-dir', type=Path, default=Path('/mnt/h/RGBD-500/depth_500'))
    parser.add_argument('--gt-format', choices=['npy', 'png', 'auto'], default='png')
    parser.add_argument('--pred-dir', type=Path, default=Path('/mnt/h/RGBD-500/stereo/depth_GS'))
    parser.add_argument('--pred-format', choices=['npy', 'colmap', 'png'], default='npy')
    parser.add_argument('--colmap-variant', choices=['geometric', 'photometric', 'auto'], default='geometric')
    parser.add_argument('--rgb-dir', type=Path, default=None)
    parser.add_argument('--plots-dir', type=Path, default=Path('/mnt/h/RGBD-500/evaluation_plots'))
    parser.add_argument('--num-samples', type=int, default=None, help='Number of image pairs to evaluate (default: all)')
    parser.add_argument('--percentile-min', type=float, default=0.0,
                        help='Lower percentile bound for COLMAP depth filtering (default: 10)')
    parser.add_argument('--percentile-max', type=float, default=95.0,
                        help='Upper percentile bound for COLMAP depth filtering (default: 90)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--no-visualizations', action='store_true', help='Disable visualization rendering')
    parser.add_argument('--apply-percentile', action='store_true', help='Force enable percentile filtering')
    parser.add_argument('--no-percentile', action='store_true', help='Force disable percentile filtering')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Enable prediction-only robust outlier removal (MAD on log(pred)) before scaling and evaluation')
    parser.add_argument('--outlier-mad-k', type=float, default=3.5,
                        help='MAD scale factor k for prediction-only filtering on log(pred) (default: 3.5)')
    parser.add_argument(
        '--invalid-gt-values',
        type=float,
        nargs='*',
        default=None,
        help='Depth values to ignore in GT before evaluation (default: 0 for PNG GT)',
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()

    if args.percentile_min >= args.percentile_max:
        raise ValueError('percentile-min must be smaller than percentile-max')

    gt_dir = args.gt_dir.resolve()
    pred_dir = args.pred_dir.resolve()
    rgb_dir = args.rgb_dir.resolve() if args.rgb_dir else None
    plots_dir = args.plots_dir.resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_format = args.gt_format
    if gt_format == 'auto':
        if list(gt_dir.glob('*.npy')):
            gt_format = 'npy'
        elif list(gt_dir.glob('*.png')):
            gt_format = 'png'
        else:
            raise FileNotFoundError(f"No GT depth files found in {gt_dir} for auto-detection")

    invalid_gt_values = args.invalid_gt_values
    if invalid_gt_values is None:
        invalid_gt_values = [0.0] if gt_format == 'png' else []
    else:
        invalid_gt_values = list(invalid_gt_values)

    pairs = collect_file_pairs(gt_dir, pred_dir, gt_format, args.pred_format, args.colmap_variant, rgb_dir)
    if not pairs:
        print('No matching GT/prediction pairs found; aborting.')
        return

    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError('num-samples must be positive')
        pairs = pairs[:args.num_samples]

    workers = args.workers or min(cpu_count(), len(pairs))
    workers = max(1, workers)

    if args.apply_percentile and args.no_percentile:
        raise ValueError('Cannot set both --apply-percentile and --no-percentile')

    apply_percentile = args.pred_format == 'colmap'
    if args.apply_percentile:
        apply_percentile = True
    if args.no_percentile:
        apply_percentile = False

    config = {
        'gt_format': gt_format,
        'pred_format': args.pred_format,
        'apply_percentile': apply_percentile,
        'percentile_min': args.percentile_min,
        'percentile_max': args.percentile_max,
        'plots_dir': plots_dir,
        'enable_vis': not args.no_visualizations,
        'invalid_gt_values': invalid_gt_values,
        'remove_outliers': bool(args.remove_outliers),
        'outlier_mad_k': float(args.outlier_mad_k),
    }

    args_list: List[Tuple[int, str, str, Optional[str], bool]] = []
    for i, (gt_path, pred_path, rgb_path) in enumerate(pairs):
        create_vis = (i < 10) or (i % 10 == 0)
        args_list.append((i, gt_path, pred_path, rgb_path, create_vis))

    print(f"Evaluating {len(pairs)} depth map pairs using {workers} worker(s)...")
    results = _run_evaluation(args_list, workers, config)

    all_metrics = defaultdict(list)
    frame_results = []  # Store per-frame results
    successful_count = 0
    vis_count = 0

    for i, errors, success in results:
        if success and errors is not None:
            for key, value in errors.items():
                all_metrics[key].append(value)
            # Store frame-level data
            frame_results.append({
                'frame_id': i,
                'gt_path': pairs[i][0],
                'pred_path': pairs[i][1],
                'errors': errors,
            })
            successful_count += 1
            if config['enable_vis'] and ((i < 10) or (i % 10 == 0)):
                vis_count += 1
            
            # Print per-frame metrics in one line
            pred_name = Path(pairs[i][1]).name
            print(f"[{i:03d}] {pred_name:20s} | AbsRel:{errors['abs_rel']:.4f} RMSE:{errors['rmse']:.4f} δ5%:{errors['delta < 5%']:.3f} δ10%:{errors['delta < 10%']:.3f} δ25%:{errors['delta < 25%']:.3f} Scale:{errors['scale_factor']:.4f} Valid:{errors['valid_pixels']/errors['total_pixels']*100:.1f}%")

    if successful_count == 0:
        print('No valid pairs could be evaluated.')
        return

    print("\n" + "=" * 50)
    print("Average Depth Evaluation Metrics")
    print("(Evaluation mask accounts for GT>0, valid prediction, and optional percentile filtering)")
    print("=" * 50)

    total_valid_pixels = int(np.sum(all_metrics['valid_pixels']))
    total_all_pixels = int(np.sum(all_metrics['total_pixels']))
    valid_ratio = total_valid_pixels / total_all_pixels if total_all_pixels > 0 else 0.0

    print(f"Successfully evaluated: {successful_count}/{len(pairs)} image pairs.")
    print(f"Total pixels: {total_all_pixels:,}")
    print(f"Valid pixels: {total_valid_pixels:,} ({valid_ratio:.1%})")
    if successful_count > 0:
        print(f"Average valid pixels per image: {total_valid_pixels // successful_count:,}")
    print("-" * 50)

    def _mean(key: str) -> float:
        values = all_metrics.get(key, [])
        return float(np.mean(values)) if values else float('nan')

    print(f"Abs_Rel:  {_mean('abs_rel'):.4f}")
    print(f"Sq_Rel:   {_mean('sq_rel'):.4f}")
    print(f"RMSE:     {_mean('rmse'):.4f}")
    print(f"δ < 5%:   {_mean('delta < 5%'):.4f}")
    print(f"δ < 10%:  {_mean('delta < 10%'):.4f}")
    print(f"δ < 25%:  {_mean('delta < 25%'):.4f}")
    print(f"Avg. Scale Factor: {_mean('scale_factor'):.4f}")
    print("=" * 50)
    if config['enable_vis']:
        print(f"Visualization plots saved in: {plots_dir}")
        print(f"Total plots generated: {vis_count}")
        print("=" * 50)

    # Save results to CSV
    csv_path = plots_dir / 'evaluation_results.csv'

    # Sort frames by RMSE (worst to best)
    frame_results_sorted = sorted(frame_results, key=lambda x: x['errors']['rmse'], reverse=True)

    # Calculate worst 10%
    num_worst = max(1, int(len(frame_results_sorted) * 0.1))
    worst_frames = frame_results_sorted[:num_worst]

    # Write all results to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame_id', 'gt_filename', 'pred_filename', 'abs_rel', 'sq_rel', 'rmse',
                      'delta_5', 'delta_10', 'delta_25', 'scale_factor', 'valid_pixels', 'total_pixels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame in frame_results_sorted:
            writer.writerow({
                'frame_id': frame['frame_id'],
                'gt_filename': Path(frame['gt_path']).name,
                'pred_filename': Path(frame['pred_path']).name,
                'abs_rel': f"{frame['errors']['abs_rel']:.6f}",
                'sq_rel': f"{frame['errors']['sq_rel']:.6f}",
                'rmse': f"{frame['errors']['rmse']:.6f}",
                'delta_5': f"{frame['errors']['delta < 5%']:.6f}",
                'delta_10': f"{frame['errors']['delta < 10%']:.6f}",
                'delta_25': f"{frame['errors']['delta < 25%']:.6f}",
                'scale_factor': f"{frame['errors']['scale_factor']:.6f}",
                'valid_pixels': frame['errors']['valid_pixels'],
                'total_pixels': frame['errors']['total_pixels'],
            })

    # Write worst 10% to separate CSV
    worst_csv_path = plots_dir / 'worst_10percent_frames.csv'
    with open(worst_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['rank', 'frame_id', 'gt_filename', 'pred_filename', 'abs_rel', 'sq_rel', 'rmse',
                      'delta_5', 'delta_10', 'delta_25', 'scale_factor', 'valid_pixels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for rank, frame in enumerate(worst_frames, start=1):
            writer.writerow({
                'rank': rank,
                'frame_id': frame['frame_id'],
                'gt_filename': Path(frame['gt_path']).name,
                'pred_filename': Path(frame['pred_path']).name,
                'abs_rel': f"{frame['errors']['abs_rel']:.6f}",
                'sq_rel': f"{frame['errors']['sq_rel']:.6f}",
                'rmse': f"{frame['errors']['rmse']:.6f}",
                'delta_5': f"{frame['errors']['delta < 5%']:.6f}",
                'delta_10': f"{frame['errors']['delta < 10%']:.6f}",
                'delta_25': f"{frame['errors']['delta < 25%']:.6f}",
                'scale_factor': f"{frame['errors']['scale_factor']:.6f}",
                'valid_pixels': frame['errors']['valid_pixels'],
            })

    print(f"\nResults saved to: {csv_path}")
    print(f"Worst 10% frames ({num_worst} frames) saved to: {worst_csv_path}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate() 
