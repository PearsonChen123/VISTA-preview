#!/usr/bin/env python3
"""
高质量图像筛选和复制工具
使用多种先进的motion blur检测方法：
1. 拉普拉斯方差检测（快速预筛选）
2. 频域分析（FFT能量分布）
3. 边缘宽度分析
4. 梯度强度分析
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Tuple, List
import argparse
from tqdm import tqdm


class MotionBlurDetector:
    """综合motion blur检测器"""

    def __init__(self, lap_threshold=100, fft_threshold=10, edge_threshold=0.15):
        """
        初始化检测器

        Args:
            lap_threshold: 拉普拉斯方差阈值，越高越严格
            fft_threshold: FFT高频能量比例阈值（百分比）
            edge_threshold: 边缘宽度阈值
        """
        self.lap_threshold = lap_threshold
        self.fft_threshold = fft_threshold
        self.edge_threshold = edge_threshold

    def detect_blur_laplacian(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        使用拉普拉斯算子检测模糊
        基于图像清晰度的经典方法

        Returns:
            (is_sharp, score): 是否清晰，以及清晰度分数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > self.lap_threshold, laplacian_var

    def detect_blur_fft(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        使用FFT频域分析检测模糊
        模糊图像的高频分量会显著减少

        Returns:
            (is_sharp, score): 是否清晰，以及高频能量百分比
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # FFT变换
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # 计算中心和边缘的能量比
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2

        # 中心区域（低频）
        center_region = magnitude_spectrum[
            center_h - h//8:center_h + h//8,
            center_w - w//8:center_w + w//8
        ]

        # 总能量
        total_energy = np.sum(magnitude_spectrum)
        center_energy = np.sum(center_region)

        # 高频能量百分比
        high_freq_percentage = (1 - center_energy / total_energy) * 100

        return high_freq_percentage > self.fft_threshold, high_freq_percentage

    def detect_blur_edge_width(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        通过边缘宽度检测模糊
        模糊图像的边缘会变宽

        Returns:
            (is_sharp, score): 是否清晰，以及边缘锐度分数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀操作来测量边缘宽度
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 计算边缘密度比
        edge_pixels = np.sum(edges > 0)
        dilated_pixels = np.sum(dilated > 0)

        if dilated_pixels == 0:
            return False, 0.0

        # 边缘锐度：原始边缘占膨胀边缘的比例
        edge_sharpness = edge_pixels / dilated_pixels

        return edge_sharpness > (1 - self.edge_threshold), edge_sharpness

    def detect_blur_gradient(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        使用Sobel梯度检测模糊
        清晰图像有更强的梯度

        Returns:
            (is_sharp, score): 是否清晰，以及梯度强度分数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 梯度幅值
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude)

        return gradient_mean > 15, gradient_mean

    def is_sharp(self, image_path: str, verbose=False) -> Tuple[bool, dict]:
        """
        综合判断图像是否清晰（无motion blur）

        Args:
            image_path: 图像路径
            verbose: 是否输出详细信息

        Returns:
            (is_sharp, scores): 是否清晰，以及各项检测分数
        """
        image = cv2.imread(image_path)
        if image is None:
            return False, {}

        # 多种方法检测
        lap_sharp, lap_score = self.detect_blur_laplacian(image)
        fft_sharp, fft_score = self.detect_blur_fft(image)
        edge_sharp, edge_score = self.detect_blur_edge_width(image)
        grad_sharp, grad_score = self.detect_blur_gradient(image)

        scores = {
            'laplacian': lap_score,
            'fft_high_freq': fft_score,
            'edge_sharpness': edge_score,
            'gradient': grad_score
        }

        # 投票机制：至少3个方法认为清晰
        votes = sum([lap_sharp, fft_sharp, edge_sharp, grad_sharp])
        is_sharp = votes >= 3

        if verbose:
            print(f"  Laplacian: {lap_score:.2f} ({'✓' if lap_sharp else '✗'})")
            print(f"  FFT High-Freq: {fft_score:.2f}% ({'✓' if fft_sharp else '✗'})")
            print(f"  Edge Sharpness: {edge_score:.2f} ({'✓' if edge_sharp else '✗'})")
            print(f"  Gradient: {grad_score:.2f} ({'✓' if grad_sharp else '✗'})")
            print(f"  Vote: {votes}/4 -> {'SHARP' if is_sharp else 'BLURRED'}")

        return is_sharp, scores


def get_image_pairs(source_dir: str) -> List[Tuple[str, str]]:
    """
    获取所有RGB和深度图像对

    Returns:
        List of (rgb_path, depth_path) tuples
    """
    source_path = Path(source_dir)
    pairs = []

    # 查找所有color图像
    color_files = sorted(source_path.glob("*.color.png"))

    for color_file in color_files:
        # 对应的深度图
        depth_file = color_file.with_name(
            color_file.name.replace(".color.png", ".depth.png")
        )

        if depth_file.exists():
            pairs.append((str(color_file), str(depth_file)))

    return pairs


def copy_sharp_images(source_dir: str,
                     target_rgb_dir: str,
                     target_depth_dir: str,
                     detector: MotionBlurDetector,
                     dry_run: bool = False,
                     verbose: bool = False):
    """
    复制清晰的图像到目标目录

    Args:
        source_dir: 源目录
        target_rgb_dir: RGB图像目标目录
        target_depth_dir: 深度图目标目录
        detector: Motion blur检测器
        dry_run: 是否仅预览不实际复制
        verbose: 是否输出详细信息
    """
    # 创建目标目录
    Path(target_rgb_dir).mkdir(parents=True, exist_ok=True)
    Path(target_depth_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有图像对
    pairs = get_image_pairs(source_dir)
    print(f"找到 {len(pairs)} 对图像")

    # 统计
    sharp_count = 0
    blur_count = 0

    # 处理每一对图像
    for rgb_path, depth_path in tqdm(pairs, desc="处理图像"):
        frame_name = Path(rgb_path).name.replace(".color.png", "")

        # 检测motion blur
        is_sharp, scores = detector.is_sharp(rgb_path, verbose=verbose)

        if verbose:
            print(f"\n{frame_name}:")

        if is_sharp:
            sharp_count += 1

            if not dry_run:
                # 复制RGB
                target_rgb = Path(target_rgb_dir) / f"{frame_name}.png"
                shutil.copy2(rgb_path, target_rgb)

                # 复制深度图
                target_depth = Path(target_depth_dir) / f"{frame_name}.png"
                shutil.copy2(depth_path, target_depth)

            if verbose:
                print(f"  ✓ 复制 {frame_name}")
        else:
            blur_count += 1
            if verbose:
                print(f"  ✗ 跳过 {frame_name} (检测到motion blur)")

    # 输出统计结果
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"总图像数: {len(pairs)}")
    print(f"清晰图像: {sharp_count} ({sharp_count/len(pairs)*100:.1f}%)")
    print(f"模糊图像: {blur_count} ({blur_count/len(pairs)*100:.1f}%)")

    if dry_run:
        print(f"\n[预览模式] 未实际复制文件")
    else:
        print(f"\n已复制到:")
        print(f"  RGB: {target_rgb_dir}")
        print(f"  Depth: {target_depth_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="筛选并复制无motion blur的RGB和深度图像"
    )
    parser.add_argument(
        "--source", "-s",
        default="/mnt/h/7scenes/heads/seq-01",
        help="源目录路径"
    )
    parser.add_argument(
        "--target-rgb",
        default="/mnt/h/7scenes/heads/01-train/images",
        help="RGB图像目标目录"
    )
    parser.add_argument(
        "--target-depth",
        default="/mnt/h/7scenes/heads/01-train/depth",
        help="深度图目标目录"
    )
    parser.add_argument(
        "--lap-threshold",
        type=float,
        default=100,
        help="拉普拉斯方差阈值 (默认: 100, 更高=更严格)"
    )
    parser.add_argument(
        "--fft-threshold",
        type=float,
        default=10,
        help="FFT高频能量阈值百分比 (默认: 10)"
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.15,
        help="边缘宽度阈值 (默认: 0.15)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际复制文件"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细信息"
    )

    args = parser.parse_args()

    # 创建检测器
    detector = MotionBlurDetector(
        lap_threshold=args.lap_threshold,
        fft_threshold=args.fft_threshold,
        edge_threshold=args.edge_threshold
    )

    # 执行复制
    copy_sharp_images(
        source_dir=args.source,
        target_rgb_dir=args.target_rgb,
        target_depth_dir=args.target_depth,
        detector=detector,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
