#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.optim as optim
from pathlib import Path
import argparse
import open3d as o3d
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
from sklearn.cluster import KMeans
import random

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

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

def create_point_cloud_from_frame(rgb_path, depth_path, transforms_json_path, min_depth=None, max_depth=None):
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
    
    # 存储原始RGB图像用于特征匹配
    result = {
        'pcd': pcd,
        'rgb': rgb,
        'depth': depth,
        'K': K,
        'valid_mask': valid_mask
    }
    
    return result

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

def detect_features(image, max_features=2000):
    """使用ORB检测图像特征点"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 创建ORB探测器
    orb = cv2.ORB_create(max_features)
    
    # 检测关键点和描述符
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_threshold=0.75):
    """匹配两组特征描述符"""
    # 创建BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # 匹配描述符
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # 应用比率测试进行过滤
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # 确保有两个匹配结果
            m, n = m_n
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches

def group_images_by_cluster(image_paths, n_clusters=4):
    """将图像分组并找出每组的中位数图像"""
    print(f"将{len(image_paths)}张图像分为{n_clusters}组...")
    
    # 确保图像文件按编号排序
    image_paths = sorted(image_paths, key=lambda x: int(Path(x).stem.split('_')[-1]))
    
    # 简单的方法：基于索引均匀分组
    groups = np.array_split(image_paths, n_clusters)
    
    representative_images = []
    for i, group in enumerate(groups):
        # 选择每组的中位数索引的图像
        mid_idx = len(group) // 2
        representative_images.append(group[mid_idx])
        print(f"组 {i+1}: 选择 {Path(group[mid_idx]).name} 作为代表 (共{len(group)}张图像)")
    
    return representative_images, groups

def find_3d_correspondences(frame1, frame2, matches, kp1, kp2):
    """基于2D特征匹配找到对应的3D点"""
    pts1 = []
    pts2 = []
    pts3d1 = []
    pts3d2 = []
    
    # 获取深度图和有效掩码
    depth1 = frame1['depth']
    depth2 = frame2['depth']
    valid_mask1 = frame1['valid_mask']
    valid_mask2 = frame2['valid_mask']
    
    # 获取相机内参
    K1 = frame1['K']
    K2 = frame2['K']
    
    # 遍历匹配对
    for match in matches:
        # 获取特征点索引
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        
        # 获取图像中的坐标 (四舍五入为整数)
        x1, y1 = int(round(kp1[idx1].pt[0])), int(round(kp1[idx1].pt[1]))
        x2, y2 = int(round(kp2[idx2].pt[0])), int(round(kp2[idx2].pt[1]))
        
        # 检查坐标是否在有效范围内
        h1, w1 = depth1.shape
        h2, w2 = depth2.shape
        
        if (0 <= x1 < w1 and 0 <= y1 < h1 and 
            0 <= x2 < w2 and 0 <= y2 < h2):
            
            # 检查深度值是否有效
            if valid_mask1[y1, x1] and valid_mask2[y2, x2]:
                # 直接从对应点云中获取3D点坐标
                # 这样处理更可靠，避免复杂的索引查找
                if hasattr(frame1['pcd'], 'points') and hasattr(frame2['pcd'], 'points'):
                    # 从深度图计算3D点坐标
                    z1 = depth1[y1, x1]
                    x3d1 = (x1 - K1[0, 2]) * z1 / K1[0, 0]
                    y3d1 = (y1 - K1[1, 2]) * z1 / K1[1, 1]
                    p3d1 = np.array([x3d1, y3d1, z1])
                    
                    z2 = depth2[y2, x2]
                    x3d2 = (x2 - K2[0, 2]) * z2 / K2[0, 0]
                    y3d2 = (y2 - K2[1, 2]) * z2 / K2[1, 1]
                    p3d2 = np.array([x3d2, y3d2, z2])
                    
                    # 保存2D和3D对应点
                    pts1.append((x1, y1))
                    pts2.append((x2, y2))
                    pts3d1.append(p3d1)
                    pts3d2.append(p3d2)
    
    print(f"从 {len(matches)} 个特征匹配中找到 {len(pts3d1)} 对有效的3D对应点")
    return np.array(pts1), np.array(pts2), np.array(pts3d1), np.array(pts3d2)

def rigid_transform_3d(A, B):
    """
    使用SVD计算从点集A到点集B的最优刚性变换
    返回旋转矩阵R和平移向量t，使得B ≈ R*A + t
    """
    assert A.shape == B.shape
    
    # 获取点的数量和维度
    n = A.shape[0]
    
    # 计算质心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # 减去质心
    AA = A - centroid_A
    BB = B - centroid_B
    
    # 计算协方差矩阵H
    H = np.dot(AA.T, BB)
    
    # 对H进行SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 特殊情况：如果行列式为负，需要修正
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # 计算平移向量
    t = centroid_B - np.dot(R, centroid_A)
    
    return R, t

def optimize_alignment_torch(pts_src, pts_dst, max_iterations=1000, lr=0.01):
    """使用PyTorch和Adam优化器优化点云对齐"""
    # 将numpy数组转换为PyTorch张量
    src_pts = torch.tensor(pts_src, dtype=torch.float32)
    dst_pts = torch.tensor(pts_dst, dtype=torch.float32)
    
    # 初始化可学习的旋转和平移参数
    # 使用轴角表示旋转，为了避免奇异性问题
    rotation = torch.zeros(3, requires_grad=True)
    translation = torch.zeros(3, requires_grad=True)
    scale = torch.ones(1, requires_grad=True)
    
    # 设置优化器
    optimizer = optim.Adam([rotation, translation, scale], lr=lr)
    
    # 辅助函数：将轴角旋转转换为旋转矩阵
    def axis_angle_to_matrix(axis_angle):
        """将轴角表示的旋转转换为旋转矩阵"""
        angle = torch.norm(axis_angle)
        if angle < 1e-6:
            return torch.eye(3)
        
        axis = axis_angle / angle
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        
        # 使用Rodrigues公式
        cross_mat = torch.zeros(3, 3)
        cross_mat[0, 1] = -axis[2]
        cross_mat[0, 2] = axis[1]
        cross_mat[1, 0] = axis[2]
        cross_mat[1, 2] = -axis[0]
        cross_mat[2, 0] = -axis[1]
        cross_mat[2, 1] = axis[0]
        
        R = torch.eye(3) + sin_angle * cross_mat + (1 - cos_angle) * torch.matmul(cross_mat, cross_mat)
        return R
    
    # 训练循环
    pbar = tqdm(range(max_iterations), desc="优化点云对齐")
    best_loss = float('inf')
    best_params = None
    
    for i in pbar:
        # 计算当前的旋转矩阵
        R = axis_angle_to_matrix(rotation)
        
        # 应用变换: scale * R * src + t
        transformed_src = torch.matmul(scale * src_pts, R.T) + translation
        
        # 计算损失 (MSE)
        loss = torch.mean(torch.sum((transformed_src - dst_pts) ** 2, dim=1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新进度条
        pbar.set_postfix(loss=loss.item())
        
        # 保存最佳参数
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                'R': R.detach().numpy(),
                'T': translation.detach().numpy(),
                'scale': scale.item()
            }
    
    print(f"最终RMSE误差: {np.sqrt(best_loss)}")
    return best_params

def transform_point_cloud(pcd, R, t, scale=1.0):
    """根据旋转矩阵R和平移向量t变换点云"""
    # 转换为numpy数组
    points = np.asarray(pcd.points)
    
    # 应用变换
    transformed_points = scale * np.dot(points, R.T) + t
    
    # 创建新的点云
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    
    # 复制颜色
    if pcd.has_colors():
        transformed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    return transformed_pcd

def refine_registration_icp(source, target, threshold=0.02, trans_init=None):
    """使用ICP算法精细配准两个点云"""
    print("使用ICP进行精细配准...")
    
    # 如果未提供初始变换，使用单位矩阵
    if trans_init is None:
        trans_init = np.eye(4)
    
    # 设置ICP参数
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=100
    )
    
    # 执行ICP配准
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    
    print(f"ICP配准结果 - 适应度: {result.fitness}, RMSE: {result.inlier_rmse}")
    return result.transformation

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

def visualize_matches(img1, img2, kp1, kp2, matches, pts3d1=None, pts3d2=None):
    """可视化两幅图像间的特征匹配"""
    # 创建匹配可视化图像
    match_img = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
        kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 显示图像
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'特征匹配 ({len(matches)} 个匹配点)')
    plt.tight_layout()
    
    # 如果有效的3D对应点不为空，在图像下方显示有效3D对应点数量
    if pts3d1 is not None and pts3d2 is not None:
        plt.xlabel(f'有效3D对应点: {len(pts3d1)} 对')
    
    # 保存和显示
    plt.savefig('feature_matches.png')
    plt.close()
    print(f"特征匹配可视化已保存到 'feature_matches.png'")
    
    # 如果有3D对应点，可视化3D对应点
    if pts3d1 is not None and pts3d2 is not None and len(pts3d1) > 0:
        # 创建两个点云
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts3d1)
        pcd1.paint_uniform_color([1, 0, 0])  # 红色
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts3d2)
        pcd2.paint_uniform_color([0, 0, 1])  # 蓝色
        
        # 保存匹配的点云
        o3d.io.write_point_cloud('matched_points_src.ply', pcd1)
        o3d.io.write_point_cloud('matched_points_dst.ply', pcd2)
        print(f"匹配的3D点已保存到 'matched_points_src.ply' 和 'matched_points_dst.ply'")

def evaluate_transformation(src_pts, dst_pts, R, t, scale=1.0):
    """评估变换的质量，返回RMSE误差"""
    # 应用变换
    transformed_pts = scale * np.dot(src_pts, R.T) + t
    
    # 计算误差
    errors = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    return rmse, errors

def find_best_transformation(frames, reference_idx=0, max_pairs=3):
    """尝试多种帧对组合，找到最佳的变换路径"""
    print("\n=== 寻找最佳变换路径 ===")
    
    reference_frame = frames[reference_idx]
    results = []
    
    # 尝试多种比率阈值
    ratio_thresholds = [0.75, 0.8, 0.85, 0.9, 0.95]
    
    # 测试多种配对方式
    for i, src_frame in enumerate(frames):
        if i == reference_idx:
            continue
            
        print(f"\n测试 帧{i} 到 参考帧{reference_idx} 的变换...")
        
        # 尝试不同的匹配阈值
        best_matches = None
        best_ratio = None
        best_count = 0
        
        for ratio in ratio_thresholds:
            # 特征匹配
            matches = match_features(src_frame['descriptors'], reference_frame['descriptors'], ratio_threshold=ratio)
            
            if len(matches) > best_count:
                best_count = len(matches)
                best_matches = matches
                best_ratio = ratio
        
        if best_count < 5:
            print(f"  - 在所有阈值下匹配点都太少 (最多 {best_count}), 尝试手动创建匹配...")
            # 如果匹配太少，尝试强制匹配
            try:
                # 使用ORB特征的均匀采样进行匹配
                matches = force_feature_matching(
                    src_frame['rgb'], reference_frame['rgb'],
                    src_frame['keypoints'], reference_frame['keypoints'],
                    src_frame['descriptors'], reference_frame['descriptors']
                )
                best_matches = matches
                best_count = len(matches)
                best_ratio = "强制匹配"
            except Exception as e:
                print(f"  - 强制匹配失败: {e}")
                pass
        
        if best_count < 3:  # 修复了错误的 'this' 变量，使用数字3作为最小匹配点数
            print(f"  - 最终匹配点太少 ({best_count}), 跳过但保留此帧")
            # 如果匹配点还是太少，将此帧添加到结果中，但使用单位变换
            # 这样至少可以包含这个帧的点云
            results.append({
                'src_idx': i,
                'dst_idx': reference_idx,
                'R': np.eye(3),  # 单位旋转
                't': np.zeros(3),  # 零平移
                'scale': 1.0,  # 单位缩放
                'rmse': float('inf'),  # 设置为无穷大，表示没有真正匹配
                'n_matches': 0,
                'n_3d_points': 0
            })
            continue
            
        print(f"  - 使用比率阈值 {best_ratio} 找到 {best_count} 个匹配点")
        
        # 找到3D对应点
        _, _, pts3d1, pts3d2 = find_3d_correspondences(
            src_frame, reference_frame, best_matches,
            src_frame['keypoints'], reference_frame['keypoints']
        )
        
        if len(pts3d1) < 3:
            print(f"  - 有效3D对应点太少 ({len(pts3d1)}), 添加单位变换")
            # 添加单位变换
            results.append({
                'src_idx': i,
                'dst_idx': reference_idx,
                'R': np.eye(3),
                't': np.zeros(3),
                'scale': 1.0,
                'rmse': float('inf'),
                'n_matches': best_count,
                'n_3d_points': len(pts3d1)
            })
            continue
            
        # 优化变换
        alignment_params = optimize_alignment_torch(pts3d1, pts3d2)
        
        # 评估变换质量
        R = alignment_params['R']
        t = alignment_params['T']
        scale = alignment_params['scale']
        rmse, _ = evaluate_transformation(pts3d1, pts3d2, R, t, scale)
        
        results.append({
            'src_idx': i,
            'dst_idx': reference_idx,
            'R': R,
            't': t,
            'scale': scale,
            'rmse': rmse,
            'n_matches': best_count,
            'n_3d_points': len(pts3d1)
        })
        
        print(f"  - RMSE: {rmse:.4f}, 匹配点: {best_count}, 3D点: {len(pts3d1)}")
    
    # 即使没有找到有效变换，也返回所有帧的单位变换
    if not results:
        print("未找到任何有效变换，将使用所有帧的原始点云")
        for i, frame in enumerate(frames):
            if i == reference_idx:
                continue
            results.append({
                'src_idx': i,
                'dst_idx': reference_idx,
                'R': np.eye(3),
                't': np.zeros(3),
                'scale': 1.0,
                'rmse': float('inf'),
                'n_matches': 0,
                'n_3d_points': 0
            })
    else:
        # 根据RMSE排序结果
        results.sort(key=lambda x: x['rmse'])
        
        # 打印最佳结果
        valid_results = [r for r in results if r['rmse'] != float('inf')]
        if valid_results:
            best_result = valid_results[0]
            print(f"\n最佳变换路径:")
            print(f"  - 帧{best_result['src_idx']} 到 参考帧{best_result['dst_idx']}")
            print(f"  - RMSE: {best_result['rmse']:.4f}")
            print(f"  - 3D点对: {best_result['n_3d_points']}")
        else:
            print("\n没有找到有效的变换路径，将使用原始点云")
    
    return results[:max_pairs]

def force_feature_matching(img1, img2, kp1, kp2, desc1, desc2, grid_size=10):
    """强制特征匹配，根据图像区域均匀采样特征点"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建网格
    grid_w1 = w1 // grid_size
    grid_h1 = h1 // grid_size
    grid_w2 = w2 // grid_size
    grid_h2 = h2 // grid_size
    
    # 将特征点分配到网格
    grid1 = {}
    for i, kp in enumerate(kp1):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        gx, gy = x // grid_w1, y // grid_h1
        if (gx, gy) not in grid1:
            grid1[(gx, gy)] = []
        grid1[(gx, gy)].append((i, kp))
    
    grid2 = {}
    for i, kp in enumerate(kp2):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        gx, gy = x // grid_w2, y // grid_h2
        if (gx, gy) not in grid2:
            grid2[(gx, gy)] = []
        grid2[(gx, gy)].append((i, kp))
    
    # 为每个网格创建匹配
    matches = []
    for (gx, gy) in grid1:
        if (gx, gy) in grid2:
            # 在相同网格中找到最佳匹配
            for idx1, kp1_item in grid1[(gx, gy)]:
                best_dist = float('inf')
                best_idx2 = -1
                
                for idx2, kp2_item in grid2[(gx, gy)]:
                    # 计算描述符距离
                    dist = cv2.norm(desc1[idx1], desc2[idx2], cv2.NORM_HAMMING)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx2 = idx2
                
                if best_idx2 >= 0:
                    # 创建DMatch对象
                    match = cv2.DMatch()
                    match.queryIdx = idx1
                    match.trainIdx = best_idx2
                    match.distance = best_dist
                    matches.append(match)
    
    print(f"强制匹配: 找到 {len(matches)} 个匹配点")
    return matches

def main():
    parser = argparse.ArgumentParser(description="从多帧图像融合点云并创建3D模型")
    parser.add_argument("--rgb-dir", default="./undistorted/images", help="RGB图像目录")
    parser.add_argument("--depth-dir", default="./stereo/depth", help="深度图目录")
    parser.add_argument("--transforms", default="./transforms.json", help="transforms.json路径")
    parser.add_argument("--output-prefix", default="fused_model", help="输出文件前缀")
    parser.add_argument("--clusters", type=int, default=4, help="将图像分为几个组")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="体素尺寸 (越小体素越精细)")
    parser.add_argument("--min-depth", type=float, default=None, help="最小深度阈值 (默认为自动确定)")
    parser.add_argument("--max-depth", type=float, default=None, help="最大深度阈值 (默认为自动确定)")
    parser.add_argument("--features", type=int, default=2000, help="每张图像检测的最大特征点数量")
    parser.add_argument("--visualize", action="store_true", help="可视化点云和网格")
    
    args = parser.parse_args()
    
    print("=== 多帧点云融合与3D重建 ===")
    
    try:
        # 步骤1: 查找并分组RGB图像
        rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, "*.png")))
        if not rgb_files:
            raise ValueError(f"在 {args.rgb_dir} 中未找到PNG图像")
        
        print(f"找到 {len(rgb_files)} 个RGB图像文件")
        
        # 将图像分为指定数量的组并选择每组的代表图像
        representative_images, image_groups = group_images_by_cluster(rgb_files, args.clusters)
        
        # 步骤2: 为每个代表图像创建点云
        frames = []
        for i, rgb_path in enumerate(representative_images):
            print(f"\n=== 处理代表图像 {i+1}/{len(representative_images)}: {Path(rgb_path).name} ===")
            
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
            
            # 创建点云
            frame_data = create_point_cloud_from_frame(
                rgb_path, depth_path, args.transforms,
                min_depth=args.min_depth, max_depth=args.max_depth
            )
            
            # 保存原始点云
            raw_pcd = frame_data['pcd']
            raw_ply_path = f"frame_{i}_raw.ply"
            save_ply(raw_pcd, raw_ply_path)
            
            # 对点云进行降噪和下采样
            cleaned_pcd = denoise_point_cloud(raw_pcd, nb_neighbors=30, std_ratio=2.0)
            down_pcd = downsample_point_cloud(cleaned_pcd, voxel_size=args.voxel_size)
            
            # 保存处理后的点云
            clean_ply_path = f"frame_{i}_clean.ply"
            save_ply(down_pcd, clean_ply_path)
            
            # 存储当前帧的数据
            frame_data['pcd'] = down_pcd  # 使用处理后的点云
            frames.append(frame_data)
            
            # 提取特征点
            keypoints, descriptors = detect_features(frame_data['rgb'], max_features=args.features)
            print(f"检测到 {len(keypoints)} 个特征点")
            
            # 保存特征点
            frame_data['keypoints'] = keypoints
            frame_data['descriptors'] = descriptors
        
        # 步骤3: 寻找最佳变换路径并融合点云
        print("\n=== 寻找最佳变换路径并融合点云 ===")
        
        # 确保有足够的帧
        if len(frames) < 2:
            raise ValueError(f"需要至少2个有效帧进行融合，但只找到了{len(frames)}个")
        
        # 寻找最佳变换路径
        best_transformations = find_best_transformation(frames, reference_idx=0)
        
        # 初始化融合点云为参考帧
        reference_frame = frames[0]
        # 正确的方式是创建一个新的点云对象并复制点和颜色
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(np.asarray(reference_frame['pcd'].points))
        combined_pcd.colors = o3d.utility.Vector3dVector(np.asarray(reference_frame['pcd'].colors))
        
        aligned_frames = [reference_frame]
        
        # 保存参考点云
        save_ply(combined_pcd, "reference_frame.ply")
        print(f"参考点云已保存: reference_frame.ply (点数: {len(combined_pcd.points)})")
        
        # 按照最佳变换路径进行点云融合
        for transform_info in best_transformations:
            src_idx = transform_info['src_idx']
            src_frame = frames[src_idx]
            
            print(f"\n=== 融合点云: 帧{src_idx} 到 参考帧 ===")
            
            # 获取变换参数
            R = transform_info['R']
            t = transform_info['t']
            scale = transform_info['scale']
            
            # 变换当前帧的点云
            transformed_pcd = transform_point_cloud(src_frame['pcd'], R, t, scale)
            
            # 使用ICP进一步精细配准
            print("执行ICP精细配准...")
            init_transform = np.eye(4)
            init_transform[:3, :3] = R
            init_transform[:3, 3] = t
            
            icp_transform = refine_registration_icp(
                transformed_pcd, combined_pcd, 
                threshold=0.02, trans_init=init_transform
            )
            
            # 应用ICP结果
            transformed_pcd = transformed_pcd.transform(icp_transform)
            
            # 保存对齐后的点云
            aligned_pcd_path = f"frame_{src_idx}_aligned.ply"
            save_ply(transformed_pcd, aligned_pcd_path)
            print(f"对齐后的点云已保存: {aligned_pcd_path}")
            
            # 更新帧数据
            src_frame['pcd'] = transformed_pcd
            aligned_frames.append(src_frame)
            
            # 将当前对齐后的点云与合并点云结合 - 重要修改
            print(f"合并点云: 合并前点数 = {len(combined_pcd.points)}")
            
            # 提取合并前的点和颜色
            combined_points_np = np.asarray(combined_pcd.points)
            combined_colors_np = np.asarray(combined_pcd.colors)
            
            # 提取当前帧的点和颜色
            current_points_np = np.asarray(transformed_pcd.points)
            current_colors_np = np.asarray(transformed_pcd.colors)
            
            # 垂直堆叠点和颜色
            all_points = np.vstack([combined_points_np, current_points_np])
            all_colors = np.vstack([combined_colors_np, current_colors_np])
            
            # 创建新的合并点云
            new_combined_pcd = o3d.geometry.PointCloud()
            new_combined_pcd.points = o3d.utility.Vector3dVector(all_points)
            new_combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
            
            # 更新合并点云
            combined_pcd = new_combined_pcd
            print(f"合并点云: 合并后点数 = {len(combined_pcd.points)}")
            
            # 保存临时合并结果
            temp_combined_path = f"combined_after_frame_{src_idx}.ply"
            save_ply(combined_pcd, temp_combined_path)
            print(f"临时合并点云已保存: {temp_combined_path}")
        
        # 步骤4: 对合并后的点云进行处理
        print("\n=== 处理合并后的点云 ===")
        
        # 保存合并前的原始点云
        raw_combined_path = f"{args.output_prefix}_raw.ply"
        save_ply(combined_pcd, raw_combined_path)
        print(f"原始合并点云已保存: {raw_combined_path}")
        
        # 执行降噪
        print("对合并点云进行统计降噪...")
        combined_pcd = denoise_point_cloud(combined_pcd, nb_neighbors=30, std_ratio=2.0)
        
        # 执行下采样
        print("对合并点云进行体素下采样...")
        combined_pcd = downsample_point_cloud(combined_pcd, voxel_size=args.voxel_size)
        
        # 步骤5: 保存最终点云
        combined_ply_path = f"{args.output_prefix}.ply"
        save_ply(combined_pcd, combined_ply_path)
        print(f"最终处理后的点云已保存: {combined_ply_path}")
        
        # 步骤6: 创建体素网格
        print("\n=== 创建体素网格 ===")
        mesh = create_voxel_mesh(combined_pcd, voxel_size=args.voxel_size)
        
        # 步骤7: 保存STL
        stl_path = f"{args.output_prefix}.stl"
        save_stl(mesh, stl_path)
        print(f"网格模型已保存: {stl_path}")
        
        # 步骤8: 可视化
        if args.visualize:
            print("打开可视化窗口...")
            visualize(combined_pcd, mesh)
        
        print(f"\n=== 处理完成! ===")
        print(f"合并点云: {len(combined_pcd.points)} 点")
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