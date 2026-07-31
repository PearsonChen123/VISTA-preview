import os
import json
import argparse
import numpy as np
import cv2


def load_intrinsics(transforms_json_path):
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)
    fx = float(data.get('fl_x'))
    fy = float(data.get('fl_y'))
    cx = float(data.get('cx'))
    cy = float(data.get('cy'))
    w = int(round(float(data.get('w'))))
    h = int(round(float(data.get('h'))))
    return fx, fy, cx, cy, w, h


def write_ply(points_xyz, colors_rgb, out_path):
    assert points_xyz.shape[0] == colors_rgb.shape[0]
    n = points_xyz.shape[0]

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(out_path, 'w') as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def backproject_depth_to_points(depth, rgb, fx, fy, cx, cy):
    h, w = depth.shape
    # 生成像素网格
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    u = u_coords[valid]
    v = v_coords[valid]
    z_valid = z[valid]

    x = (u - cx) * z_valid / fx
    y = (v - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=1).astype(np.float32)
    colors = rgb[valid].reshape(-1, 3).astype(np.uint8)
    return points, colors


def process_one(depth_path, rgb_path, transforms_path, out_path, resize_depth_to_rgb):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fx, fy, cx, cy, cam_w, cam_h = load_intrinsics(transforms_path)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f'无法读取RGB图像: {rgb_path}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_ext = os.path.splitext(depth_path)[1].lower()
    if depth_ext == '.npy':
        if not os.path.isfile(depth_path):
            raise FileNotFoundError(f'未找到深度npy文件: {depth_path}')
        depth = np.load(depth_path)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f'无法读取深度图: {depth_path}')

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    target_h, target_w = rgb.shape[:2]
    sx = target_w / float(cam_w)
    sy = target_h / float(cam_h)
    fx_scaled = fx * sx
    fy_scaled = fy * sy
    cx_scaled = cx * sx
    cy_scaled = cy * sy

    if resize_depth_to_rgb or depth.shape[:2] != (target_h, target_w):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth = depth.astype(np.float32)

    points, colors = backproject_depth_to_points(depth, rgb, fx_scaled, fy_scaled, cx_scaled, cy_scaled)
    write_ply(points, colors, out_path)

    print(f"点云保存完成: {out_path}")
    print(f"点数: {points.shape[0]} | 深度范围: [{float(np.min(depth[np.isfinite(depth) & (depth>0)])):.6f}, {float(np.max(depth)):.6f}]")


def main():
    parser = argparse.ArgumentParser(description='深度图 -> 彩色点云PLY 可视化脚本（支持单文件与目录批量）')
    parser.add_argument('--depth', type=str, help='深度单文件路径(.png/.npy)')
    parser.add_argument('--rgb', type=str, help='RGB单文件路径(.jpg/.png)')
    parser.add_argument('--out', type=str, help='输出PLY路径')
    parser.add_argument('--depth-dir', type=str, help='深度目录，内部文件按名称排序配对')
    parser.add_argument('--rgb-dir', type=str, help='RGB目录，内部文件按名称排序配对')
    parser.add_argument('--out-dir', type=str, default='/mnt/h/RGBD-500/pointclouds', help='批量时输出目录')
    parser.add_argument('--transforms', type=str, default='/mnt/h/RGBD-500/transforms.json')
    parser.add_argument('--resize-depth-to-rgb', action='store_true', help='将深度图调整到RGB分辨率')
    args = parser.parse_args()

    # 单文件模式
    if args.depth and args.rgb and args.out:
        process_one(args.depth, args.rgb, args.transforms, args.out, args.resize_depth_to_rgb)
        return

    # 目录批量模式
    if args.depth-dir is not None or args.rgb-dir is not None:  # 防止用户只给了一个
        if not (args.depth_dir and args.rgb_dir):
            raise ValueError('批量模式需要同时提供 --depth-dir 与 --rgb-dir')

        if not os.path.isdir(args.depth_dir):
            raise FileNotFoundError(f'深度目录不存在: {args.depth_dir}')
        if not os.path.isdir(args.rgb_dir):
            raise FileNotFoundError(f'RGB目录不存在: {args.rgb_dir}')

        # 列举并按文件名排序
        def sorted_files(d, exts):
            files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
            files.sort(key=lambda p: os.path.basename(p))
            return files

        depth_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.npy'}
        rgb_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

        depth_files = sorted_files(args.depth_dir, depth_exts)
        rgb_files = sorted_files(args.rgb_dir, rgb_exts)

        if len(depth_files) == 0 or len(rgb_files) == 0:
            raise RuntimeError('批量模式: 深度或RGB目录为空')

        n = min(len(depth_files), len(rgb_files))
        os.makedirs(args.out_dir, exist_ok=True)

        print(f"批量配对 {n} 对 (按文件名排序):")
        for i in range(n):
            d_path = depth_files[i]
            r_path = rgb_files[i]
            base = os.path.splitext(os.path.basename(r_path))[0]
            out_path = os.path.join(args.out_dir, f"{base}.ply")
            print(f"  [{i+1}/{n}] depth={os.path.basename(d_path)}  rgb={os.path.basename(r_path)} -> {os.path.basename(out_path)}")
            process_one(d_path, r_path, args.transforms, out_path, args.resize_depth_to_rgb)
        return

    raise ValueError('请使用单文件模式(--depth --rgb --out)或目录批量模式(--depth-dir --rgb-dir --out-dir)')


if __name__ == '__main__':
    main()


