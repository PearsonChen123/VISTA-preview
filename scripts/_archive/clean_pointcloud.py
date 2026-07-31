#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """使用统计离群值滤波器移除噪点"""
    print(f"执行统计离群值过滤（邻居数:{nb_neighbors}, 标准差比例:{std_ratio}）...")
    
    # 统计滤波器，基于点和邻居之间的距离分布
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    """使用半径离群值滤波器移除孤立点"""
    print(f"执行半径离群值过滤（半径:{radius}m, 最小邻居数:{nb_points}）...")
    
    # 半径过滤器，基于定义的半径内需要有最小数量的邻居
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cl


def dbscan_clustering(pcd, eps=0.02, min_points=10):
    """使用DBSCAN聚类算法去除噪声点"""
    print(f"执行DBSCAN聚类过滤（邻域半径:{eps}m, 最小点数:{min_points}）...")
    
    # DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    if len(labels) == 0:
        print("警告：DBSCAN未能找到任何聚类")
        return pcd
    
    # 统计各个聚类的点数
    max_label = labels.max()
    if max_label < 0:
        print("警告：DBSCAN未能找到任何聚类")
        return pcd
        
    print(f"DBSCAN聚类结果: {max_label + 1} 个聚类")
    
    # 计算每个聚类包含的点数
    cluster_counts = np.bincount(labels[labels >= 0])
    
    # 找出最大的聚类
    largest_cluster = np.argmax(cluster_counts)
    print(f"最大聚类为聚类 {largest_cluster}，包含 {cluster_counts[largest_cluster]} 个点")
    
    # 创建一个新点云，只包含最大聚类中的点
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster])
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[labels == largest_cluster])
    
    return filtered_pcd


def voxel_downsample(pcd, voxel_size=0.005):
    """体素下采样，减少点云数量"""
    print(f"执行体素下采样（体素大小:{voxel_size}m）...")
    return pcd.voxel_down_sample(voxel_size)


def crop_by_distance(pcd, center=[0, 0, 0], max_distance=1.0):
    """根据到中心点的距离裁剪点云"""
    print(f"裁剪点云（最大距离:{max_distance}m）...")
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 计算每个点到中心的距离
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
    # 创建一个掩码，只保留距离小于最大距离的点
    mask = distances < max_distance
    
    # 创建一个新点云，只包含掩码内的点
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    return filtered_pcd


def main():
    parser = argparse.ArgumentParser(description="点云离群点清理工具")
    parser.add_argument("input_file", help="输入点云文件路径（.ply格式）")
    parser.add_argument("--output_file", help="输出点云文件路径（不指定则使用input_file_cleaned.ply）")
    parser.add_argument("--statistical", action="store_true", help="应用统计离群值滤波")
    parser.add_argument("--radius", action="store_true", help="应用半径离群值滤波")
    parser.add_argument("--dbscan", action="store_true", help="应用DBSCAN聚类滤波")
    parser.add_argument("--voxel", action="store_true", help="应用体素下采样")
    parser.add_argument("--crop", action="store_true", help="根据距离裁剪点云")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="统计滤波的邻居数量")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="统计滤波的标准差比例")
    parser.add_argument("--radius_size", type=float, default=0.8, help="半径滤波的半径大小（米）")
    parser.add_argument("--min_neighbors", type=int, default=16, help="半径滤波的最小邻居数量")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="DBSCAN的邻域半径（米）")
    parser.add_argument("--dbscan_min_points", type=int, default=10, help="DBSCAN的最小点数")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="体素下采样的体素大小（米）")
    parser.add_argument("--max_distance", type=float, default=1.0, help="裁剪的最大距离（米）")
    parser.add_argument("--center_x", type=float, default=0.0, help="裁剪中心点的X坐标")
    parser.add_argument("--center_y", type=float, default=0.0, help="裁剪中心点的Y坐标")
    parser.add_argument("--center_z", type=float, default=0.0, help="裁剪中心点的Z坐标")
    parser.add_argument("--visualize", action="store_true", help="可视化点云处理结果")
    parser.add_argument("--all", action="store_true", help="应用所有滤波方法（按照推荐顺序）")
    
    args = parser.parse_args()
    
    # 如果没有指定任何滤波方法，则默认应用所有方法
    if not (args.statistical or args.radius or args.dbscan or args.voxel or args.crop):
        args.all = True
    
    # 如果没有指定输出文件路径，则使用默认值
    if not args.output_file:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    # 读取点云
    print(f"读取点云文件: {args.input_file}")
    pcd = o3d.io.read_point_cloud(args.input_file)
    
    original_points = len(pcd.points)
    print(f"原始点云包含 {original_points} 个点")
    
    # 应用各种滤波方法
    if args.all:
        # 推荐的处理流程：先体素下采样，然后DBSCAN聚类，再统计离群值滤波，最后半径离群值滤波
        pcd = voxel_downsample(pcd, args.voxel_size)
        pcd = dbscan_clustering(pcd, args.dbscan_eps, args.dbscan_min_points)
        pcd = statistical_outlier_removal(pcd, args.nb_neighbors, args.std_ratio)
        pcd = radius_outlier_removal(pcd, args.min_neighbors, args.radius_size)
        if args.crop:
            center = [args.center_x, args.center_y, args.center_z]
            pcd = crop_by_distance(pcd, center, args.max_distance)
    else:
        if args.voxel:
            pcd = voxel_downsample(pcd, args.voxel_size)
        
        if args.dbscan:
            pcd = dbscan_clustering(pcd, args.dbscan_eps, args.dbscan_min_points)
        
        if args.statistical:
            pcd = statistical_outlier_removal(pcd, args.nb_neighbors, args.std_ratio)
        
        if args.radius:
            pcd = radius_outlier_removal(pcd, args.min_neighbors, args.radius_size)
        
        if args.crop:
            center = [args.center_x, args.center_y, args.center_z]
            pcd = crop_by_distance(pcd, center, args.max_distance)
    
    # 显示处理后的点数
    filtered_points = len(pcd.points)
    removed_points = original_points - filtered_points
    removed_percentage = (removed_points / original_points) * 100
    
    print(f"清理后的点云包含 {filtered_points} 个点")
    print(f"已移除 {removed_points} 个点（{removed_percentage:.2f}%）")
    
    # 保存结果
    o3d.io.write_point_cloud(args.output_file, pcd)
    print(f"已将清理后的点云保存至: {args.output_file}")
    
    # 可选：可视化结果
    if args.visualize:
        print("正在可视化点云...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main() 