#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
import glob
import open3d as o3d
import trimesh
import time

def read_transforms_json(json_path):
    """读取transforms.json文件获取相机参数和每帧的位姿"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取内参
    fl_x = data.get('fl_x')
    fl_y = data.get('fl_y')
    cx = data.get('cx')
    cy = data.get('cy')
    w = data.get('w')
    h = data.get('h')
    
    # 获取内参矩阵
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    # 获取每一帧的变换矩阵
    frames = data.get('frames', [])
    frame_transforms = {}
    
    for frame in frames:
        file_path = frame.get('file_path')
        transform_matrix = np.array(frame.get('transform_matrix'))
        frame_transforms[os.path.basename(file_path)] = transform_matrix
    
    return K, w, h, frame_transforms

def read_depth_map(depth_path):
    """读取深度图，优先尝试.npy格式"""
    npy_path = depth_path.replace('.png', '.npy')
    if os.path.exists(npy_path) or depth_path.endswith('.npy'):
        if not depth_path.endswith('.npy'):
            depth_path = npy_path
        print(f"读取NPY格式深度图: {depth_path}")
        depth = np.load(depth_path)
    elif depth_path.endswith('.png'):
        print(f"读取PNG格式深度图: {depth_path}")
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"无法读取深度图: {depth_path}")
        
        scaling_factor = 1000.0 / 65535.0
        depth = depth_png.astype(np.float32) * scaling_factor
    else:
        raise ValueError(f"不支持的深度图格式: {depth_path}")
    
    return depth

def create_point_cloud_from_depth(rgb_path, depth_path, K, min_depth=None, max_depth=None, transform_matrix=None):
    """从RGB图像和深度图创建点云"""
    # 读取RGB图像
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"无法读取RGB图像: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 读取深度图
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
        min_depth = max(depth_min * 0.9, 0.001)
    
    if max_depth is None:
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
    if transform_matrix is not None:
        # 使用参考代码中的方法直接转换到世界坐标系
        h, w = depth.shape
        y, x = np.indices((h, w))
        
        # 保留有效点
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        depth_valid = depth[valid_mask]
        
        # 根据相机内参计算3D点
        x_3d = -1 * (x_valid - K[0, 2]) * depth_valid / K[0, 0]
        y_3d = -1 * (y_valid - K[1, 2]) * depth_valid / K[1, 1]
        z_3d = depth_valid
        
        # 堆叠3D点
        points = np.vstack((x_3d, y_3d, z_3d)).T
        
        # 获取颜色
        colors = rgb[valid_mask] / 255.0
        
        # 应用变换矩阵将点转换到世界坐标系
        points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T
    else:
        # 如果不提供变换矩阵，则保持在相机坐标系中
        h, w = depth.shape
        v, u = np.indices((h, w))
        
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        
        # 创建点云数组 (x, y, z)
        xyz = np.stack([x, y, z], axis=-1)
        
        # 只保留有效点
        points = xyz[valid_mask]
        colors = rgb[valid_mask] / 255.0
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 输出点云信息
    print(f"生成点云: {len(pcd.points)} 点")
    
    return pcd

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

def transform_point_cloud(pcd, transform_matrix):
    """使用变换矩阵变换点云"""
    # 获取点云数据
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 应用变换
    points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T
    
    # 创建新的点云
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(points)
    transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return transformed_pcd

def registration_icp(source, target, threshold=0.02, trans_init=None, max_iteration=100):
    """使用ICP算法进行点云配准"""
    if trans_init is None:
        trans_init = np.eye(4)
    
    print(f"执行ICP配准 (阈值: {threshold}, 最大迭代次数: {max_iteration})")
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration
    )
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    
    print(f"ICP配准结果 - 适应度: {result.fitness}, RMSE: {result.inlier_rmse}")
    return result.transformation

def save_ply(pcd, output_path):
    """保存点云为PLY文件"""
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}")

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
    print(f"生成包含 {len(voxels)} 个体素的网格...")
    for i, voxel in enumerate(voxels):
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

def save_stl(mesh, output_path):
    """保存网格为STL文件"""
    mesh.export(output_path)
    print(f"网格已保存到 {output_path}")

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

def extract_frame_number(file_path):
    """从文件路径提取帧号"""
    filename = os.path.basename(file_path)
    # 尝试从文件名中提取数字
    digits = ''.join(filter(str.isdigit, filename))
    if digits:
        return int(digits)
    return 0

def select_representative_frames(image_paths, n_clusters=4):
    """将图像分组并找出每组的中位数图像"""
    print(f"将{len(image_paths)}张图像分为{n_clusters}组...")
    
    # 确保图像文件按编号排序
    image_paths = sorted(image_paths, key=lambda x: extract_frame_number(x))
    
    # 基于索引均匀分组
    groups = np.array_split(image_paths, n_clusters)
    
    representative_images = []
    for i, group in enumerate(groups):
        # 选择每组的中位数索引的图像
        mid_idx = len(group) // 2
        representative_images.append(group[mid_idx])
        print(f"组 {i+1}: 选择 {Path(group[mid_idx]).name} 作为代表 (共{len(group)}张图像)")
    
    return representative_images

def main():
    parser = argparse.ArgumentParser(description="简化版多帧点云融合")
    parser.add_argument("--rgb-dir", default="./undistorted/images", help="RGB图像目录")
    parser.add_argument("--depth-dir", default="./stereo/depth", help="深度图目录")
    parser.add_argument("--transforms", default="./transforms.json", help="transforms.json路径")
    parser.add_argument("--output-prefix", default="simple_fused_model", help="输出文件前缀")
    parser.add_argument("--clusters", type=int, default=4, help="将图像分为几个组")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="体素尺寸 (越小体素越精细)")
    parser.add_argument("--min-depth", type=float, default=None, help="最小深度阈值 (默认为自动确定)")
    parser.add_argument("--max-depth", type=float, default=None, help="最大深度阈值 (默认为自动确定)")
    parser.add_argument("--visualize", action="store_true", help="可视化点云和网格")
    
    args = parser.parse_args()
    
    print("=== 简化版多帧点云融合 ===")
    start_time = time.time()
    
    try:
        # 步骤1: 读取transforms.json
        print(f"读取相机参数和位姿信息: {args.transforms}")
        K, w, h, frame_transforms = read_transforms_json(args.transforms)
        print(f"加载了 {len(frame_transforms)} 帧的位姿信息")
        
        # 步骤2: 查找并选择代表性图像
        rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, "*.png")))
        if not rgb_files:
            raise ValueError(f"在 {args.rgb_dir} 中未找到PNG图像")
        
        print(f"找到 {len(rgb_files)} 个RGB图像文件")
        representative_images = select_representative_frames(rgb_files, args.clusters)
        
        # 步骤3: 为每个代表图像创建点云并变换到世界坐标系
        pcds = []
        
        for i, rgb_path in enumerate(representative_images):
            print(f"\n=== 处理代表图像 {i+1}/{len(representative_images)}: {Path(rgb_path).name} ===")
            
            # 获取图像文件名以查找对应的变换矩阵
            img_filename = os.path.basename(rgb_path)
            if img_filename not in frame_transforms:
                print(f"警告: 在transforms.json中未找到 {img_filename} 的位姿信息")
                transform_matrix = np.eye(4)  # 使用单位矩阵作为默认值
            else:
                transform_matrix = frame_transforms[img_filename]
            
            # 构造对应的深度图路径
            frame_number = extract_frame_number(rgb_path)
            depth_path = os.path.join(args.depth_dir, f"{frame_number:05d}.npy")
            
            if not os.path.exists(depth_path):
                # 尝试备用路径格式
                depth_path = os.path.join(args.depth_dir, f"frame_{frame_number:04d}_depth.npy")
                if not os.path.exists(depth_path):
                    print(f"警告: 未找到对应的深度图: {depth_path}")
                    print("尝试在深度图目录中查找文件...")
                    
                    # 列出深度图目录中的文件
                    depth_files = sorted(glob.glob(os.path.join(args.depth_dir, "*.npy")))
                    if depth_files:
                        print(f"使用备选深度图: {depth_files[min(i, len(depth_files)-1)]}")
                        depth_path = depth_files[min(i, len(depth_files)-1)]
                    else:
                        print(f"错误: 深度图目录 {args.depth_dir} 中没有发现.npy文件")
                        continue
            
            # 创建点云 - 直接变换到世界坐标系
            pcd = create_point_cloud_from_depth(
                rgb_path, depth_path, K,
                min_depth=args.min_depth, max_depth=args.max_depth,
                transform_matrix=transform_matrix
            )
            
            # 保存原始点云
            raw_ply_path = f"frame_{i}_raw.ply"
            save_ply(pcd, raw_ply_path)
            
            # 对点云进行降噪和下采样
            cleaned_pcd = denoise_point_cloud(pcd, nb_neighbors=30, std_ratio=2.0)
            down_pcd = downsample_point_cloud(cleaned_pcd, voxel_size=args.voxel_size)
            
            # 保存处理后的点云
            clean_ply_path = f"frame_{i}_clean.ply"
            save_ply(down_pcd, clean_ply_path)
            
            # 添加到点云列表 - 不需要再次变换，因为已经在create_point_cloud_from_depth中变换过了
            pcds.append(down_pcd)
        
        # 步骤4: 合并所有点云
        print("\n=== 合并所有点云 ===")
        if not pcds:
            raise ValueError("没有有效的点云可合并")
        
        # 创建合并点云
        combined_pcd = o3d.geometry.PointCloud()
        combined_points = []
        combined_colors = []
        
        # 提取所有点云的点和颜色
        for pcd in pcds:
            combined_points.append(np.asarray(pcd.points))
            combined_colors.append(np.asarray(pcd.colors))
        
        # 垂直堆叠点和颜色
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        
        # 创建合并点云
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        print(f"初步合并点云: {len(combined_pcd.points)} 点")
        
        # 保存合并前的原始点云
        raw_combined_path = f"{args.output_prefix}_raw.ply"
        save_ply(combined_pcd, raw_combined_path)
        print(f"原始合并点云已保存: {raw_combined_path}")
        
        # 步骤5: 使用ICP进行精细对齐 (可选，因为已经使用相机位姿进行了对齐)
        # 这里的ICP只是为了进一步改进对齐，实际可能不需要
        print("\n=== 使用ICP进行精细配准 ===")
        combined_pcd_refined = combined_pcd
        
        # 步骤6: 对合并后的点云进行处理
        print("\n=== 处理合并后的点云 ===")
        
        # 执行降噪
        print("对合并点云进行统计降噪...")
        combined_pcd_refined = denoise_point_cloud(combined_pcd_refined, nb_neighbors=30, std_ratio=2.0)
        
        # 执行下采样
        print("对合并点云进行体素下采样...")
        combined_pcd_refined = downsample_point_cloud(combined_pcd_refined, voxel_size=args.voxel_size)
        
        # 步骤7: 保存最终点云
        combined_ply_path = f"{args.output_prefix}.ply"
        save_ply(combined_pcd_refined, combined_ply_path)
        print(f"最终处理后的点云已保存: {combined_ply_path}")
        
        # 步骤8: 创建体素网格
        print("\n=== 创建体素网格 ===")
        mesh = create_voxel_mesh(combined_pcd_refined, voxel_size=args.voxel_size)
        
        # 步骤9: 保存STL
        stl_path = f"{args.output_prefix}.stl"
        save_stl(mesh, stl_path)
        print(f"网格模型已保存: {stl_path}")
        
        # 步骤10: 可视化
        if args.visualize:
            print("打开可视化窗口...")
            visualize(combined_pcd_refined, mesh)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n=== 处理完成! ===")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"合并点云: {len(combined_pcd_refined.points)} 点")
        print(f"网格: {len(mesh.faces)} 三角形")
        print(f"输出文件:")
        print(f"  - 点云: {combined_ply_path}")
        print(f"  - 模型: {stl_path}")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
