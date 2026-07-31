#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import open3d as o3d
import argparse
from pathlib import Path

def read_point_cloud(pcd_path):
    """读取点云文件"""
    print(f"读取点云: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        raise ValueError(f"点云文件为空或读取失败: {pcd_path}")
    print(f"点云包含 {len(pcd.points)} 个点")
    return pcd

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
    print(f"检测到 {len(keypoints)} 个特征点")
    
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
    
    print(f"找到 {len(good_matches)} 个有效匹配点")
    return good_matches

def find_3d_correspondences(pcd1, pcd2, rgb1, rgb2, kp1, kp2, matches):
    """根据2D特征匹配找到3D点云对应点"""
    # 点云1中的点
    points1 = np.asarray(pcd1.points)
    colors1 = np.asarray(pcd1.colors)
    
    # 点云2中的点
    points2 = np.asarray(pcd2.points)
    colors2 = np.asarray(pcd2.colors)
    
    # 图像尺寸
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    
    # 存储对应的3D点
    pts3d1 = []
    pts3d2 = []
    
    for match in matches:
        # 获取特征点在图像中的坐标
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        
        x1, y1 = int(round(kp1[idx1].pt[0])), int(round(kp1[idx1].pt[1]))
        x2, y2 = int(round(kp2[idx2].pt[0])), int(round(kp2[idx2].pt[1]))
        
        # 坐标不能超出图像边界
        if 0 <= x1 < w1 and 0 <= y1 < h1 and 0 <= x2 < w2 and 0 <= y2 < h2:
            # 图像坐标转换为点云索引
            idx1_3d = y1 * w1 + x1
            idx2_3d = y2 * w2 + x2
            
            # 确保索引有效
            if idx1_3d < len(points1) and idx2_3d < len(points2):
                # 获取对应的3D点
                pt1 = points1[idx1_3d]
                pt2 = points2[idx2_3d]
                
                # 检查点的有效性（非零、非NaN）
                if (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2)) and 
                    np.linalg.norm(pt1) > 0 and np.linalg.norm(pt2) > 0):
                    pts3d1.append(pt1)
                    pts3d2.append(pt2)
    
    print(f"找到 {len(pts3d1)} 对有效的3D对应点")
    return np.array(pts3d1), np.array(pts3d2)

def estimate_rigid_transform(A, B):
    """
    使用SVD计算从点集A到点集B的最优刚性变换
    返回旋转矩阵R和平移向量t，使得B ≈ R*A + t
    """
    if len(A) < 3 or len(B) < 3:
        raise ValueError("需要至少3对点来计算刚性变换")
    
    # 确保点的数量相同
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

def rigid_transform_3d_points(points, R, t):
    """应用刚性变换到点集"""
    return np.dot(points, R.T) + t

def visualize_matches(img1, img2, kp1, kp2, matches):
    """可视化两幅图像之间的特征匹配"""
    # 绘制匹配
    match_img = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
        kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 保存匹配图像
    cv2.imwrite("feature_matches.jpg", match_img)
    print("特征匹配可视化已保存为 'feature_matches.jpg'")
    
    # 显示匹配图像（可选）
    # cv2.imshow("特征匹配", match_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def refine_registration_icp(source, target, threshold=0.02, trans_init=None, max_iteration=100):
    """使用ICP算法精细配准两个点云"""
    print("使用ICP进行精细配准...")
    
    # 如果未提供初始变换，使用单位矩阵
    if trans_init is None:
        trans_init = np.eye(4)
    
    # 设置ICP参数
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration
    )
    
    # 执行ICP配准
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    
    print(f"ICP配准结果 - 适应度: {result.fitness}, RMSE: {result.inlier_rmse}")
    return result.transformation

def main():
    parser = argparse.ArgumentParser(description="基于特征点匹配的点云缝合")
    parser.add_argument("--pcd1", required=True, help="第一个点云文件路径")
    parser.add_argument("--pcd2", required=True, help="第二个点云文件路径")
    parser.add_argument("--rgb1", required=True, help="第一个点云对应的RGB图像路径")
    parser.add_argument("--rgb2", required=True, help="第二个点云对应的RGB图像路径")
    parser.add_argument("--output", default="fused.ply", help="输出融合点云的文件路径")
    parser.add_argument("--max-features", type=int, default=2000, help="最大特征点数量")
    parser.add_argument("--ratio", type=float, default=0.75, help="特征匹配的比率阈值")
    parser.add_argument("--icp-threshold", type=float, default=0.02, help="ICP配准的距离阈值")
    
    args = parser.parse_args()
    
    try:
        # 步骤1: 读取点云和RGB图像
        print("\n=== 读取点云和RGB图像 ===")
        pcd1 = read_point_cloud(args.pcd1)
        pcd2 = read_point_cloud(args.pcd2)
        
        rgb1 = cv2.imread(args.rgb1)
        rgb2 = cv2.imread(args.rgb2)
        
        if rgb1 is None or rgb2 is None:
            raise ValueError("无法读取RGB图像")
        
        # 转换为RGB颜色空间（OpenCV默认为BGR）
        rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
        
        # 步骤2: 进行特征点检测和匹配
        print("\n=== 特征点检测和匹配 ===")
        kp1, desc1 = detect_features(rgb1, args.max_features)
        kp2, desc2 = detect_features(rgb2, args.max_features)
        
        matches = match_features(desc1, desc2, args.ratio)
        
        # 可视化特征匹配
        visualize_matches(rgb1, rgb2, kp1, kp2, matches)
        
        # 步骤3: 找到3D对应点
        print("\n=== 寻找3D对应点 ===")
        pts3d1, pts3d2 = find_3d_correspondences(pcd1, pcd2, rgb1, rgb2, kp1, kp2, matches)
        
        if len(pts3d1) < 3:
            raise ValueError("找到的3D对应点太少，无法计算刚性变换")
        
        # 步骤4: 估计刚性变换
        print("\n=== 估计刚性变换 ===")
        R, t = estimate_rigid_transform(pts3d2, pts3d1)  # 从pcd2到pcd1的变换
        
        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        print("估计的变换矩阵:")
        print(transform)
        
        # 步骤5: 变换点云2
        print("\n=== 变换点云 ===")
        pcd2_transformed = o3d.geometry.PointCloud(pcd2)
        pcd2_transformed.transform(transform)
        
        # 步骤6: 使用ICP进行精细配准
        print("\n=== ICP精细配准 ===")
        icp_transform = refine_registration_icp(
            pcd2_transformed, pcd1, 
            threshold=args.icp_threshold, 
            trans_init=np.eye(4)
        )
        
        # 应用ICP变换
        pcd2_refined = o3d.geometry.PointCloud(pcd2_transformed)
        pcd2_refined.transform(icp_transform)
        
        # 步骤7: 合并点云
        print("\n=== 合并点云 ===")
        combined_pcd = o3d.geometry.PointCloud(pcd1)
        combined_pcd += pcd2_refined
        
        # 步骤8: 保存结果
        o3d.io.write_point_cloud(args.output, combined_pcd)
        print(f"融合点云已保存为: {args.output} (包含 {len(combined_pcd.points)} 个点)")
        
        # 可视化结果
        print("\n=== 可视化结果 ===")
        print("红色: 第一个点云")
        print("绿色: 变换后的第二个点云")
        
        # 为了可视化，设置不同颜色
        pcd1.paint_uniform_color([1, 0, 0])  # 红色
        pcd2_refined.paint_uniform_color([0, 1, 0])  # 绿色
        
        # 保存彩色的点云用于可视化
        vis_pcds = [pcd1, pcd2_refined]
        o3d.io.write_point_cloud("visualization.ply", combined_pcd)
        
        # 显示点云（可选）
        o3d.visualization.draw_geometries(vis_pcds)
        
        print("\n=== 处理完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 