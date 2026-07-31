#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import open3d as o3d
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm

def read_transforms_json(json_path):
    """读取transforms.json文件获取相机参数"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取内参
    fl_x = data.get('fl_x')
    fl_y = data.get('fl_y')
    cx = data.get('cx')
    cy = data.get('cy')
    w = data.get('w')
    h = data.get('h')
    
    # 获取指定帧的变换矩阵
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    return K, w, h, data

def read_depth_map(depth_path):
    """优先读取.npy格式深度图，如果不存在则尝试读取PNG"""
    # 检查是否有同名的.npy文件
    npy_path = depth_path.replace('.png', '.npy')
    if os.path.exists(npy_path):
        print(f"读取NPY格式深度图: {npy_path}")
        depth = np.load(npy_path)
    elif depth_path.endswith('.npy'):
        print(f"读取NPY格式深度图: {depth_path}")
        depth = np.load(depth_path)
    elif depth_path.endswith('.png'):
        # 读取16位PNG并转换回深度值
        print(f"读取PNG格式深度图: {depth_path}")
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"无法读取深度图: {depth_path}")
        
        scaling_factor = 1000.0 / 65535.0
        depth = depth_png.astype(np.float32) * scaling_factor
    else:
        raise ValueError(f"不支持的深度图格式: {depth_path}")
    
    return depth

def depth_to_points(depth, K, mask=None):
    """将深度图转换为点云"""
    h, w = depth.shape
    
    # 创建像素坐标网格
    v, u = np.indices((h, w))
    
    # 应用相机参数将像素坐标转换为3D点
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    
    # 创建点云数组 (x, y, z)
    xyz = np.stack([x, y, z], axis=-1)
    
    # 如果有mask，只保留mask中的点
    if mask is not None:
        xyz = xyz[mask]
    else:
        # 否则只保留深度大于0的点
        xyz = xyz[z > 0]
    
    return xyz

def denoise_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    """使用统计滤波对点云进行降噪"""
    print(f"使用统计滤波进行点云降噪 (邻居数量: {nb_neighbors}, 标准差比率: {std_ratio})")
    # 移除离群点
    pcd_cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    # 输出降噪结果
    num_points_before = len(pcd.points)
    num_points_after = len(pcd_cleaned.points)
    percent_removed = (num_points_before - num_points_after) / num_points_before * 100
    print(f"原始点云: {num_points_before} 点")
    print(f"降噪后点云: {num_points_after} 点")
    print(f"移除了 {percent_removed:.2f}% 的离群点")
    
    return pcd_cleaned

def downsample_point_cloud(pcd, voxel_size=0.05):
    """对点云进行均匀下采样"""
    print(f"使用体素下采样进行点云均匀采样 (体素尺寸: {voxel_size})")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 输出下采样结果
    num_points_before = len(pcd.points)
    num_points_after = len(pcd_down.points)
    percent_reduced = (num_points_before - num_points_after) / num_points_before * 100
    print(f"原始点云: {num_points_before} 点")
    print(f"下采样后点云: {num_points_after} 点")
    print(f"减少了 {percent_reduced:.2f}% 的点数")
    
    return pcd_down

def create_voxel_mesh(pcd, voxel_size=0.05):
    """创建体素网格（每个点变成小方块）"""
    print(f"创建体素网格 (体素尺寸: {voxel_size})")
    
    # 将Open3D点云转换为体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    # 获取体素网格的体素坐标和颜色
    voxels = voxel_grid.get_voxels()
    
    # 创建网格
    vertices = []
    faces = []
    vertex_colors = []
    
    # 生成立方体网格
    for i, voxel in enumerate(tqdm(voxels, desc="生成体素模型")):
        # 获取体素中心坐标
        x, y, z = voxel.grid_index
        x, y, z = x * voxel_size, y * voxel_size, z * voxel_size
        
        # 立方体顶点
        cube_vertices = [
            [x-voxel_size/2, y-voxel_size/2, z-voxel_size/2],  # 0
            [x+voxel_size/2, y-voxel_size/2, z-voxel_size/2],  # 1
            [x+voxel_size/2, y+voxel_size/2, z-voxel_size/2],  # 2
            [x-voxel_size/2, y+voxel_size/2, z-voxel_size/2],  # 3
            [x-voxel_size/2, y-voxel_size/2, z+voxel_size/2],  # 4
            [x+voxel_size/2, y-voxel_size/2, z+voxel_size/2],  # 5
            [x+voxel_size/2, y+voxel_size/2, z+voxel_size/2],  # 6
            [x-voxel_size/2, y+voxel_size/2, z+voxel_size/2]   # 7
        ]
        
        # 立方体面（每个面由两个三角形组成）
        cube_triangles = [
            [0, 1, 2], [0, 2, 3],  # 底面
            [4, 5, 6], [4, 6, 7],  # 顶面
            [0, 1, 5], [0, 5, 4],  # 前面
            [2, 3, 7], [2, 7, 6],  # 后面
            [0, 3, 7], [0, 7, 4],  # 左面
            [1, 2, 6], [1, 6, 5]   # 右面
        ]
        
        # 添加顶点和面
        base_idx = len(vertices)
        vertices.extend(cube_vertices)
        
        for tri in cube_triangles:
            faces.append([tri[0] + base_idx, tri[1] + base_idx, tri[2] + base_idx])
        
        # 添加颜色
        color = voxel.color if hasattr(voxel, 'color') else [0.7, 0.7, 0.7]
        vertex_colors.extend([color] * 8)  # 每个立方体有8个顶点
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_colors=np.array(vertex_colors)
    )
    
    print(f"生成了包含 {len(faces)} 个三角形的体素网格")
    return mesh

def save_ply(pcd, output_path):
    """保存点云为PLY文件"""
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd['points'])
        if 'colors' in pcd:
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd['colors'])
        pcd = pcd_o3d
    
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}")

def save_stl(mesh, output_path):
    """保存网格为STL文件"""
    # 直接使用trimesh对象保存为STL
    mesh.export(output_path)
    print(f"网格已保存到 {output_path}")

def create_point_cloud_from_frame(rgb_path, depth_path, transforms_json_path, frame_idx=0, min_depth=None, max_depth=None):
    """从单帧图像创建点云"""
    # 读取相机参数
    K, w, h, transforms_data = read_transforms_json(transforms_json_path)
    
    # 读取RGB图像
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"无法读取RGB图像: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 读取深度图（优先使用.npy格式）
    depth = read_depth_map(depth_path)
    
    # 分析深度图数值范围
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        raise ValueError("深度图中没有大于0的有效深度值")
    
    depth_min = np.min(valid_depth)
    depth_max = np.max(valid_depth)
    depth_mean = np.mean(valid_depth)
    depth_median = np.median(valid_depth)
    
    print(f"深度图统计信息:")
    print(f"  - 最小深度: {depth_min:.6f}")
    print(f"  - 最大深度: {depth_max:.6f}")
    print(f"  - 平均深度: {depth_mean:.6f}")
    print(f"  - 中位深度: {depth_median:.6f}")
    
    # 自动调整深度范围阈值
    if min_depth is None:
        # 设置为最小有效深度值的0.9倍或0.001，取较大值
        min_depth = max(depth_min * 0.9, 0.001)
    
    if max_depth is None:
        # 设置为最大有效深度值的1.1倍或1000，取较小值
        max_depth = min(depth_max * 1.1, 1000.0)
    
    print(f"使用深度范围: [{min_depth:.6f}, {max_depth:.6f}]")
    
    # 确保RGB和深度图尺寸一致
    if rgb.shape[:2] != depth.shape:
        print(f"调整RGB图像大小从 {rgb.shape[:2]} 到 {depth.shape}")
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
    
    # 创建有效点的掩码
    valid_mask = (depth > min_depth) & (depth < max_depth)
    valid_count = np.sum(valid_mask)
    if valid_count == 0:
        raise ValueError(f"在指定的深度范围 [{min_depth}, {max_depth}] 内没有有效点")
    
    print(f"有效点数量: {valid_count} (占总像素的 {valid_count / depth.size * 100:.2f}%)")
    
    # 创建点云
    points = depth_to_points(depth, K, valid_mask)
    colors = rgb[valid_mask] / 255.0
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 输出点云信息
    print(f"生成点云: {len(pcd.points)} 点")
    return pcd

def visualize(pcd=None, mesh=None):
    """可视化点云和网格"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    if pcd is not None:
        vis.add_geometry(pcd)
    
    if mesh is not None:
        # 如果是trimesh对象，转换为Open3D格式
        if isinstance(mesh, trimesh.Trimesh):
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            if mesh.visual.vertex_colors is not None:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3] / 255.0)
            vis.add_geometry(o3d_mesh)
        else:
            vis.add_geometry(mesh)
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="将单帧深度图转换为3D点云和体素网格")
    parser.add_argument("--rgb", default="./undistorted/images/frame_0034.png", help="RGB图像路径")
    parser.add_argument("--depth", default="./stereo/depth/00034.npy", help="深度图路径 (优先使用.npy格式)")
    parser.add_argument("--transforms", default="transforms.json", help="transforms.json路径")
    parser.add_argument("--frame-idx", type=int, default=0, help="帧索引")
    parser.add_argument("--output-prefix", default="frame_3d", help="输出文件前缀")
    parser.add_argument("--visualize", action="store_true", help="可视化点云和网格")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="体素尺寸 (越小体素越精细)")
    parser.add_argument("--min-depth", type=float, default=None, help="最小深度阈值 (默认为自动确定)")
    parser.add_argument("--max-depth", type=float, default=None, help="最大深度阈值 (默认为自动确定)")
    parser.add_argument("--denoise", action="store_true", default=True, help="是否进行点云降噪")
    parser.add_argument("--downsample", action="store_true", default=True, help="是否进行点云下采样")
    
    args = parser.parse_args()
    
    print(f"处理帧 {args.frame_idx}:")
    print(f"  RGB图像: {args.rgb}")
    print(f"  深度图: {args.depth}")
    
    try:
        # 创建点云
        pcd = create_point_cloud_from_frame(
            args.rgb, args.depth, args.transforms, args.frame_idx,
            min_depth=args.min_depth, max_depth=args.max_depth
        )
        
        # 保存原始点云
        raw_ply_path = f"{args.output_prefix}_raw.ply"
        save_ply(pcd, raw_ply_path)
        
        # 点云降噪
        if args.denoise:
            pcd = denoise_point_cloud(pcd, nb_neighbors=30, std_ratio=2.0)
            
            # 保存降噪后的点云
            clean_ply_path = f"{args.output_prefix}_clean.ply"
            save_ply(pcd, clean_ply_path)
        
        # 点云下采样（为STL生成做准备）
        if args.downsample:
            pcd_down = downsample_point_cloud(pcd, voxel_size=args.voxel_size)
        else:
            pcd_down = pcd
        
        # 保存最终点云
        ply_path = f"{args.output_prefix}.ply"
        save_ply(pcd_down, ply_path)
        
        # 创建体素网格
        mesh = create_voxel_mesh(pcd_down, voxel_size=args.voxel_size)
        
        # 保存STL
        stl_path = f"{args.output_prefix}.stl"
        save_stl(mesh, stl_path)
        
        # 可视化
        if args.visualize:
            visualize(pcd_down, mesh)
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
