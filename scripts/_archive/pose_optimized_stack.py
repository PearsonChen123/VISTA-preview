import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt


def load_transforms(transforms_path):
    with open(transforms_path, 'r') as f:
        return json.load(f)


def load_rgb_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((658, 658, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_features(image, max_features=2000):
    """Extract features and descriptors from the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return [], np.array([])
    return keypoints, descriptors


def match_features(desc1, desc2, ratio=0.5):
    """Match feature descriptors between two images"""
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # Use FLANN matcher for fast matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                         table_number=6,
                         key_size=12,
                         multi_probe_level=1)
    search_params = dict(checks=100)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches
    except Exception as e:
        print(f"Error matching features: {e}")
        # Fallback to brute force matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]  # Return top 50 matches


def depth_to_pointcloud(depth, rgb, K, transform_matrix, sample_rate=0.1):
    h, w = depth.shape
    y, x = np.mgrid[0:h, 0:w]
    mask = np.random.rand(h, w) < sample_rate
    x_3d = -1 * (x[mask] - K[0, 2]) * depth[y[mask], x[mask]] / K[0, 0]
    y_3d = -1 * (y[mask] - K[1, 2]) * depth[mask] / K[1, 1]
    z_3d = depth[mask]

    points = np.vstack((x_3d, y_3d, z_3d)).T

    colors = rgb[y[mask], x[mask]] / 255.0

    points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T

    return points, colors


def get_pixel_coordinates(keypoints):
    """Convert keypoints to pixel coordinate array"""
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def get_3d_points_from_keypoints(keypoints, depth, K):
    """Get 3D points from feature points and depth map"""
    points_3d = []
    valid_indices = []
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Ensure coordinates are within depth map range
        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
            z = depth[y, x]
            
            # Check if depth value is valid
            if z > 0:
                # Back-project to 3D space
                x_3d = -1 * (x - K[0, 2]) * z / K[0, 0]
                y_3d = -1 * (y - K[1, 2]) * z / K[1, 1]
                z_3d = z
                
                points_3d.append([x_3d, y_3d, z_3d])
                valid_indices.append(i)
    
    return np.array(points_3d, dtype=np.float32), valid_indices


def transform_points(points_3d, transform_matrix):
    """Transform 3D points using a transformation matrix"""
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # Apply transformation
    transformed_points = (transform_matrix @ points_homogeneous.T).T
    return transformed_points[:, :3]  # Return non-homogeneous coordinates


class CameraPoseOptimizer(nn.Module):
    def __init__(self, initial_poses, device='cpu'):
        super(CameraPoseOptimizer, self).__init__()
        
        # Convert initial poses to rotation matrices (3x3) and translation vectors (3x1)
        self.rotations = []
        self.translations = []
        
        for pose in initial_poses:
            R = torch.tensor(pose[:3, :3], dtype=torch.float32, device=device)
            t = torch.tensor(pose[:3, 3], dtype=torch.float32, device=device)
            
            # Convert rotation matrix to axis-angle representation
            rotation_angle = torch.acos(torch.clamp((R.trace() - 1) / 2, -1.0, 1.0))
            if rotation_angle.abs() < 1e-6:
                axis = torch.tensor([1.0, 0.0, 0.0], device=device)
            else:
                axis = torch.tensor([
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ], device=device)
                axis = axis / (2 * torch.sin(rotation_angle))
            
            axis_angle = axis * rotation_angle
            
            self.rotations.append(nn.Parameter(axis_angle))
            self.translations.append(nn.Parameter(t))
        
        # Register parameters
        for i, (rot, trans) in enumerate(zip(self.rotations, self.translations)):
            self.register_parameter(f'rotation_{i}', rot)
            self.register_parameter(f'translation_{i}', trans)
    
    def axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle representation to rotation matrix"""
        angle = torch.norm(axis_angle)
        if angle < 1e-6:
            return torch.eye(3, device=axis_angle.device)
        
        axis = axis_angle / angle
        
        K = torch.zeros((3, 3), device=axis_angle.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        R = torch.eye(3, device=axis_angle.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
        return R
    
    def get_transformation_matrix(self, index):
        """Get transformation matrix for the specified index"""
        R = self.axis_angle_to_rotation_matrix(self.rotations[index])
        t = self.translations[index]
        
        T = torch.eye(4, device=R.device)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def project_points(self, points_3d, K, transform_matrix, device):
        """Project 3D points to 2D plane"""
        # Convert to homogeneous coordinates
        points_homogeneous = torch.cat([points_3d, torch.ones((points_3d.shape[0], 1), device=device)], dim=1)
        
        # Apply transformation
        transformed_points = torch.matmul(transform_matrix, points_homogeneous.t()).t()
        
        # Project to camera plane
        x = -transformed_points[:, 0] / transformed_points[:, 2]
        y = -transformed_points[:, 1] / transformed_points[:, 2]
        
        # Apply camera intrinsics
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        
        return torch.stack([u, v], dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description='Specify center frame and range for point cloud stacking and pose optimization')
    parser.add_argument('--center_frame', type=int, help='Center frame index')
    parser.add_argument('--range', type=int, default= 10, help='Number of frames before and after center frame')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='Point cloud sampling rate')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--max_features', type=int, default=2000, help='Maximum number of features to extract per frame')
    parser.add_argument('--num_iterations', type=int, default=200, help='Number of iterations for pose optimization')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--use_cuda', action='store_true', help='Whether to use CUDA acceleration (if available)')
    parser.add_argument('--error_threshold', type=float, default=5.0, help='Maximum reprojection error threshold (pixels) for point filtering')
    parser.add_argument('--auto_select_window', action='store_true', help='Automatically select optimal frame window')
    return parser.parse_args()


def get_keypoint_info(keypoint):
    """Convert cv2.KeyPoint to serializable dictionary"""
    return {
        'pt': keypoint.pt,
        'size': keypoint.size,
        'angle': keypoint.angle,
        'response': keypoint.response,
        'octave': keypoint.octave,
        'class_id': keypoint.class_id
    }


def keypoint_info_to_keypoint(info):
    """Convert dictionary back to cv2.KeyPoint object"""
    try:
        # Create an empty KeyPoint object
        kp = cv2.KeyPoint()
        
        # Manually set attributes instead of using constructor
        kp.pt = (float(info['pt'][0]), float(info['pt'][1]))
        kp.size = float(info['size'])
        kp.angle = float(info['angle'])
        kp.response = float(info['response'])
        kp.octave = int(info['octave'])
        kp.class_id = int(info['class_id'])
        
        return kp
    except Exception as e:
        print(f"Failed to create KeyPoint using method 1: {e}")
        try:
            # Try another constructor form
            return cv2.KeyPoint(
                x=float(info['pt'][0]),
                y=float(info['pt'][1]),
                _size=float(info['size']),
                _angle=float(info['angle']),
                _response=float(info['response']),
                _octave=int(info['octave']),
                _class_id=int(info['class_id'])
            )
        except Exception as e2:
            print(f"Failed to create KeyPoint using method 2: {e2}")
            # Last attempt, suitable for some OpenCV versions
            x, y = info['pt']
            return cv2.KeyPoint(
                float(x), float(y), 
                float(info['size']), 
                float(info['angle']), 
                float(info['response']), 
                int(info['octave']), 
                int(info['class_id'])
            )


def ensure_numpy_arrays_contiguous(data):
    """Ensure all numpy arrays are contiguous (serializable)"""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data)
    elif isinstance(data, dict):
        return {k: ensure_numpy_arrays_contiguous(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_numpy_arrays_contiguous(item) for item in data]
    else:
        return data


def process_frame(idx, depth_files, image_dir, transforms, K, max_features):
    """Process single frame data, extract feature points and depth information"""
    depth = np.load(depth_files[idx])
    rgb_path = image_dir / transforms["frames"][idx]["file_path"].split("/")[-1]
    rgb = load_rgb_image(rgb_path)
    transform_matrix = np.array(transforms["frames"][idx]["transform_matrix"])
    
    # Extract feature points
    keypoints, descriptors = extract_features(rgb, max_features)
    
    # Get 3D coordinates corresponding to feature points
    points_3d, valid_indices = get_3d_points_from_keypoints(keypoints, depth, K)
    
    # Filter valid keypoints and descriptors
    if len(valid_indices) > 0:
        valid_keypoints = [keypoints[i] for i in valid_indices]
        if descriptors is not None and len(descriptors) > 0:
            valid_descriptors = descriptors[valid_indices]
        else:
            valid_descriptors = np.array([])
    else:
        valid_keypoints = []
        valid_descriptors = np.array([])
    
    # Convert KeyPoint objects to serializable dictionaries
    valid_keypoints_info = [get_keypoint_info(kp) for kp in valid_keypoints]
    
    # Ensure all numpy arrays are contiguous
    result = {
        'depth': depth,
        'rgb': rgb,
        'transform_matrix': transform_matrix,
        'keypoints_info': valid_keypoints_info,
        'descriptors': valid_descriptors,
        'points_3d': points_3d
    }
    
    return ensure_numpy_arrays_contiguous(result)


def plot_reprojection_errors(errors, output_dir):
    """Plot reprojection error changes during optimization"""
    if not errors:
        print("No error data to plot")
        return
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    iterations = [error['iteration'] for error in errors]
    mean_errors = [error['mean_error'] for error in errors]
    max_errors = [error['max_error'] for error in errors]
    median_errors = [error['median_error'] for error in errors]
    total_losses = [error['total_loss'] for error in errors]
    
    # Plot average reprojection error
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, mean_errors, 'b-', label='Mean Error')
    plt.plot(iterations, median_errors, 'g-', label='Median Error')
    plt.plot(iterations, max_errors, 'r-', label='Max Error', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Reprojection Error vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save image
    error_plot_path = os.path.join(output_dir, 'reprojection_errors.png')
    plt.savefig(error_plot_path)
    print(f"Reprojection error plot saved as {error_plot_path}")
    
    # Plot total loss
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, total_losses, 'b-', label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss Function vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save image
    loss_plot_path = os.path.join(output_dir, 'loss_function.png')
    plt.savefig(loss_plot_path)
    print(f"Loss function plot saved as {loss_plot_path}")
    
    # Save numerical data to CSV file
    csv_path = os.path.join(output_dir, 'optimization_stats.csv')
    with open(csv_path, 'w') as f:
        f.write("iteration,mean_error,median_error,max_error,total_loss\n")
        for it, mean, med, max_err, loss in zip(iterations, mean_errors, median_errors, max_errors, total_losses):
            f.write(f"{it},{mean},{med},{max_err},{loss}\n")
    print(f"Optimization statistics saved as {csv_path}")


def optimize_poses(frames_data, K, center_idx, device, num_iterations, learning_rate, max_error_threshold=5.0):
    """Optimize camera poses using Adam to minimize reprojection errors"""
    print("Starting camera pose optimization...")
    
    # Prepare initial poses
    initial_poses = [frame['transform_matrix'] for frame in frames_data]
    optimizer_model = CameraPoseOptimizer(initial_poses, device=device).to(device)
    
    # Prepare optimizer
    optimizer = optim.Adam(optimizer_model.parameters(), lr=learning_rate)
    
    # Center frame data
    center_frame = frames_data[center_idx]
    
    try:
        # Convert dictionary back to KeyPoint objects
        center_keypoints = [keypoint_info_to_keypoint(info) for info in center_frame['keypoints_info']]
        use_simplified_matching = False
    except Exception as e:
        print(f"Warning: Unable to create KeyPoint objects: {e}")
        print("Using simplified matching with only keypoint coordinates")
        # Extract coordinate information without using full KeyPoint objects
        center_keypoints_coords = [info['pt'] for info in center_frame['keypoints_info']]
        use_simplified_matching = True
    
    center_descriptors = center_frame['descriptors']
    center_points_3d = center_frame['points_3d']
    
    # Camera intrinsic matrix
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)
    
    # Record errors during optimization
    iteration_errors = []
    best_error = float('inf')
    best_iteration = 0
    
    # Optimization process
    for iteration in tqdm(range(num_iterations), desc="Optimizing poses"):
        optimizer.zero_grad()
        total_loss = 0
        
        # Statistics for current iteration
        all_reprojection_errors = []
        frame_mean_errors = {}
        frame_max_errors = {}
        total_matched_points = 0
        filtered_points_count = 0
        
        # Iterate through other frames to compute reprojection errors with center frame
        for i, frame in enumerate(frames_data):
            if i == center_idx:
                continue  # Skip center frame
            
            if not use_simplified_matching:
                # Convert dictionary back to KeyPoint objects
                try:
                    frame_keypoints = [keypoint_info_to_keypoint(info) for info in frame['keypoints_info']]
                    
                    # Compute feature matches between frames
                    matches = match_features(center_descriptors, frame['descriptors'])
                    
                    if len(matches) < 10:  # Skip if too few matches
                        continue
                    
                    # Get matched keypoints and 3D points
                    center_matched_points_3d = []
                    frame_matched_points_3d = []
                    center_matched_pixels = []
                    frame_matched_pixels = []
                    
                    for match in matches:
                        center_idx_match = match.queryIdx
                        frame_idx_match = match.trainIdx
                        
                        if center_idx_match < len(center_points_3d) and frame_idx_match < len(frame['points_3d']):
                            center_matched_points_3d.append(center_points_3d[center_idx_match])
                            frame_matched_points_3d.append(frame['points_3d'][frame_idx_match])
                            
                            center_matched_pixels.append(center_keypoints[center_idx_match].pt)
                            frame_matched_pixels.append(frame_keypoints[frame_idx_match].pt)
                except Exception as e:
                    print(f"Error processing frame {i}, skipping: {e}")
                    continue
            else:
                # Use simplified matching based on descriptor indices
                try:
                    # Extract current frame's keypoint coordinates
                    frame_keypoints_coords = [info['pt'] for info in frame['keypoints_info']]
                    
                    # Match using descriptor distances directly
                    center_matched_points_3d = []
                    frame_matched_points_3d = []
                    center_matched_pixels = []
                    frame_matched_pixels = []
                    
                    # Only proceed if there are enough descriptors
                    if (len(center_descriptors) > 0 and 
                        len(frame['descriptors']) > 0 and 
                        len(center_points_3d) > 0 and 
                        len(frame['points_3d']) > 0):
                        
                        # Compute descriptor distances
                        for i_c, desc_c in enumerate(center_descriptors):
                            if i_c >= len(center_keypoints_coords) or i_c >= len(center_points_3d):
                                continue
                                
                            best_dist = float('inf')
                            best_idx = -1
                            
                            for i_f, desc_f in enumerate(frame['descriptors']):
                                if i_f >= len(frame_keypoints_coords) or i_f >= len(frame['points_3d']):
                                    continue
                                    
                                # Compute Hamming distance
                                dist = np.sum(desc_c != desc_f)
                                
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = i_f
                            
                            # If found a good match below threshold
                            if best_idx != -1 and best_dist < 50:  # Adjustable threshold
                                center_matched_points_3d.append(center_points_3d[i_c])
                                frame_matched_points_3d.append(frame['points_3d'][best_idx])
                                
                                center_matched_pixels.append(center_keypoints_coords[i_c])
                                frame_matched_pixels.append(frame_keypoints_coords[best_idx])
                except Exception as e:
                    print(f"Error using simplified matching for frame {i}, skipping: {e}")
                    continue
            
            if len(center_matched_points_3d) < 5:  # Need at least 5 points for optimization
                continue
                
            # Convert to PyTorch tensors
            center_matched_points_3d = torch.tensor(center_matched_points_3d, dtype=torch.float32, device=device)
            frame_matched_points_3d = torch.tensor(frame_matched_points_3d, dtype=torch.float32, device=device)
            center_matched_pixels = torch.tensor(center_matched_pixels, dtype=torch.float32, device=device)
            frame_matched_pixels = torch.tensor(frame_matched_pixels, dtype=torch.float32, device=device)
            
            # Get current transformation matrices
            T_center = optimizer_model.get_transformation_matrix(center_idx)
            T_frame = optimizer_model.get_transformation_matrix(i)
            
            # Project center frame 3D points to current frame
            # First transform from center frame coordinates to world coordinates
            points_world = torch.matmul(T_center, 
                                     torch.cat([center_matched_points_3d, 
                                               torch.ones((center_matched_points_3d.shape[0], 1), device=device)], 
                                              dim=1).t()).t()
            
            # Then transform from world coordinates to current frame coordinates
            T_frame_inv = torch.inverse(T_frame)
            points_frame = torch.matmul(T_frame_inv, points_world.t()).t()
            
            # Project to current frame image plane
            x = -points_frame[:, 0] / points_frame[:, 2]
            y = -points_frame[:, 1] / points_frame[:, 2]
            
            # Apply camera intrinsics
            u = K_tensor[0, 0] * x + K_tensor[0, 2]
            v = K_tensor[1, 1] * y + K_tensor[1, 2]
            
            projected_pixels = torch.stack([u, v], dim=1)
            
            # Compute reprojection errors
            reprojection_error = torch.sum((projected_pixels - frame_matched_pixels) ** 2, dim=1)
            reprojection_distances = torch.sqrt(reprojection_error + 1e-6)  # Euclidean distance
            
            # Filter out points with large reprojection errors in every iteration
            # Get error values in numpy array
            current_errors = reprojection_distances.detach().cpu().numpy()
            
            # Create mask for valid points (error <= max_error_threshold)
            valid_mask = current_errors <= max_error_threshold
            filtered_points_count += np.sum(~valid_mask)
            
            # Only keep points with small errors
            if np.sum(valid_mask) < 5:  # If too few points remain, keep all
                # If we're in the first few iterations, keep all points
                if iteration < 10:
                    valid_mask = np.ones_like(valid_mask, dtype=bool)
                    filtered_points_count -= np.sum(~valid_mask)
                else:
                    # Skip this frame if there are too few valid points after several iterations
                    continue
            
            # Apply mask to all tensors
            center_matched_points_3d = center_matched_points_3d[valid_mask]
            frame_matched_points_3d = frame_matched_points_3d[valid_mask]
            center_matched_pixels = center_matched_pixels[valid_mask]
            frame_matched_pixels = frame_matched_pixels[valid_mask]
            reprojection_distances = reprojection_distances[valid_mask]
            
            # Recompute if no points remain
            if center_matched_points_3d.shape[0] < 5:
                continue
            
            # Use the reprojection distances as robust error
            robust_error = reprojection_distances
            
            # Collect error statistics
            frame_errors = reprojection_distances.detach().cpu().numpy()
            all_reprojection_errors.extend(frame_errors.tolist())
            frame_mean_errors[i] = float(np.mean(frame_errors))
            frame_max_errors[i] = float(np.max(frame_errors))
            total_matched_points += len(frame_errors)
            
            # Accumulate loss
            frame_loss = torch.mean(robust_error)
            total_loss += frame_loss
        
        if total_loss > 0:
            # Calculate and record error statistics for current iteration
            if all_reprojection_errors:
                mean_error = np.mean(all_reprojection_errors)
                max_error = np.max(all_reprojection_errors)
                median_error = np.median(all_reprojection_errors)
                
                # Record error information
                error_info = {
                    'iteration': iteration,
                    'mean_error': mean_error,
                    'max_error': max_error,
                    'median_error': median_error,
                    'total_loss': float(total_loss.detach().cpu().numpy()),
                    'frame_mean_errors': frame_mean_errors,
                    'frame_max_errors': frame_max_errors,
                    'matched_points': total_matched_points,
                    'filtered_points': filtered_points_count
                }
                iteration_errors.append(error_info)
                
                # Print current iteration statistics
                if iteration % 10 == 0 or iteration == num_iterations - 1:
                    print(f"\nIteration {iteration+1}/{num_iterations}:")
                    print(f"  Mean reprojection error: {mean_error:.2f} pixels")
                    print(f"  Max reprojection error: {max_error:.2f} pixels")
                    print(f"  Median reprojection error: {median_error:.2f} pixels")
                    print(f"  Matched points: {total_matched_points}")
                    print(f"  Filtered points (error > {max_error_threshold} pixels): {filtered_points_count}")
                    print(f"  Total loss: {float(total_loss.detach().cpu().numpy()):.4f}")
                
                # Check if this is the best error
                if mean_error < best_error:
                    best_error = mean_error
                    best_iteration = iteration
            
            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()
    
    # Print final optimization results
    print("\nOptimization complete:")
    if iteration_errors:
        initial_error = iteration_errors[0]['mean_error']
        final_error = iteration_errors[-1]['mean_error']
        print(f"  Initial mean reprojection error: {initial_error:.2f} pixels")
        print(f"  Final mean reprojection error: {final_error:.2f} pixels")
        print(f"  Error reduction: {initial_error - final_error:.2f} pixels ({100 * (initial_error - final_error) / initial_error:.2f}%)")
        print(f"  Best error at iteration {best_iteration+1}: {best_error:.2f} pixels")
    
    # Return optimized poses and error records
    optimized_poses = []
    for i in range(len(frames_data)):
        T = optimizer_model.get_transformation_matrix(i).detach().cpu().numpy()
        optimized_poses.append(T)
    
    return optimized_poses, iteration_errors


def evaluate_window_reprojection_error(start_idx, window_size, depth_files, image_dir, transforms, K, max_features, device, silent=False):
    """Evaluate initial reprojection error for a specific window of frames"""
    end_idx = start_idx + window_size
    center_idx = window_size // 2  # Relative center in window
    
    # Process frame data for the window
    process_frame_partial = partial(
        process_frame, 
        depth_files=depth_files, 
        image_dir=image_dir,
        transforms=transforms,
        K=K,
        max_features=max_features
    )
    
    # Process frames in parallel
    frames_data = []
    if not silent:
        print(f"Processing window from frame {start_idx} to {end_idx-1}")
    
    # 使用并行处理帧数据
    try:
        # 创建一个内部进程池来处理此窗口的帧
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), window_size)) as inner_pool:
            frames_data = inner_pool.map(process_frame_partial, range(start_idx, end_idx))
    except Exception as e:
        if not silent:
            print(f"Parallel processing error, falling back to sequential: {e}")
        for idx in range(start_idx, end_idx):
            frames_data.append(process_frame_partial(idx))
    
    # 获取中心帧
    center_frame = frames_data[center_idx]
    
    try:
        # Convert dictionary back to KeyPoint objects
        center_keypoints = [keypoint_info_to_keypoint(info) for info in center_frame['keypoints_info']]
        use_simplified_matching = False
    except Exception as e:
        if not silent:
            print(f"Warning: Unable to create KeyPoint objects: {e}")
            print("Using simplified matching with only keypoint coordinates")
        # Extract coordinate information without using full KeyPoint objects
        center_keypoints_coords = [info['pt'] for info in center_frame['keypoints_info']]
        use_simplified_matching = True
    
    center_descriptors = center_frame['descriptors']
    center_points_3d = center_frame['points_3d']
    
    # 定义一个帧匹配计算函数，用于并行处理
    def compute_frame_errors(i, frame):
        if i == center_idx:
            return []  # 跳过中心帧
        
        try:
            frame_matched_points = []
            
            if not use_simplified_matching:
                # 使用KeyPoint对象匹配
                try:
                    frame_keypoints = [keypoint_info_to_keypoint(info) for info in frame['keypoints_info']]
                    matches = match_features(center_descriptors, frame['descriptors'])
                    
                    if len(matches) < 10:
                        return []
                    
                    center_matched_points_3d = []
                    frame_matched_points_3d = []
                    center_matched_pixels = []
                    frame_matched_pixels = []
                    
                    for match in matches:
                        center_idx_match = match.queryIdx
                        frame_idx_match = match.trainIdx
                        
                        if center_idx_match < len(center_points_3d) and frame_idx_match < len(frame['points_3d']):
                            center_matched_points_3d.append(center_points_3d[center_idx_match])
                            frame_matched_points_3d.append(frame['points_3d'][frame_idx_match])
                            
                            center_matched_pixels.append(center_keypoints[center_idx_match].pt)
                            frame_matched_pixels.append(frame_keypoints[frame_idx_match].pt)
                except Exception as e:
                    if not silent:
                        print(f"Error processing frame {i}, skipping: {e}")
                    return []
            else:
                # 使用简化匹配
                try:
                    frame_keypoints_coords = [info['pt'] for info in frame['keypoints_info']]
                    
                    center_matched_points_3d = []
                    frame_matched_points_3d = []
                    center_matched_pixels = []
                    frame_matched_pixels = []
                    
                    if (len(center_descriptors) > 0 and 
                        len(frame['descriptors']) > 0 and 
                        len(center_points_3d) > 0 and 
                        len(frame['points_3d']) > 0):
                        
                        for i_c, desc_c in enumerate(center_descriptors):
                            if i_c >= len(center_keypoints_coords) or i_c >= len(center_points_3d):
                                continue
                                
                            best_dist = float('inf')
                            best_idx = -1
                            
                            for i_f, desc_f in enumerate(frame['descriptors']):
                                if i_f >= len(frame_keypoints_coords) or i_f >= len(frame['points_3d']):
                                    continue
                                
                                dist = np.sum(desc_c != desc_f)
                                
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = i_f
                            
                            if best_idx != -1 and best_dist < 50:
                                center_matched_points_3d.append(center_points_3d[i_c])
                                frame_matched_points_3d.append(frame['points_3d'][best_idx])
                                
                                center_matched_pixels.append(center_keypoints_coords[i_c])
                                frame_matched_pixels.append(frame_keypoints_coords[best_idx])
                except Exception as e:
                    if not silent:
                        print(f"Error using simplified matching for frame {i}: {e}")
                    return []
            
            if len(center_matched_points_3d) < 5:
                return []
                
            # 计算重投影误差
            center_matched_points_3d_np = np.array(center_matched_points_3d)
            frame_matched_pixels_np = np.array(frame_matched_pixels)
            
            center_transform = center_frame['transform_matrix']
            frame_transform = frame['transform_matrix']
            
            center_points_world = transform_points(center_matched_points_3d_np, center_transform)
            
            frame_transform_inv = np.linalg.inv(frame_transform)
            center_points_frame = transform_points(center_points_world, frame_transform_inv)
            
            x = -center_points_frame[:, 0] / center_points_frame[:, 2]
            y = -center_points_frame[:, 1] / center_points_frame[:, 2]
            
            u = K[0, 0] * x + K[0, 2]
            v = K[1, 1] * y + K[1, 2]
            
            projected_pixels = np.stack([u, v], axis=1)
            
            reprojection_errors = np.sqrt(np.sum((projected_pixels - frame_matched_pixels_np) ** 2, axis=1))
            
            return reprojection_errors.tolist()
        except Exception as e:
            if not silent:
                print(f"Error computing frame errors for frame {i}: {e}")
            return []
    
    # 并行计算所有帧的重投影误差
    all_errors = []
    
    # 创建计算任务
    frame_error_tasks = [(i, frame) for i, frame in enumerate(frames_data) if i != center_idx]
    
    # 并行处理匹配和误差计算
    try:
        # 使用ThreadPool而不是ProcessPool来避免潜在的内存和序列化问题
        from multiprocessing.pool import ThreadPool
        with ThreadPool(processes=min(multiprocessing.cpu_count(), len(frame_error_tasks))) as thread_pool:
            all_frame_errors = thread_pool.starmap(compute_frame_errors, frame_error_tasks)
        
        # 合并所有误差
        for errors in all_frame_errors:
            all_errors.extend(errors)
    except Exception as e:
        if not silent:
            print(f"Parallel error calculation failed, falling back to sequential: {e}")
        # 退回到顺序计算
        for i, frame in frame_error_tasks:
            errors = compute_frame_errors(i, frame)
            all_errors.extend(errors)
    
    if len(all_errors) == 0:
        return float('inf')  # 没有有效匹配时返回无穷大
    
    mean_error = np.mean(all_errors)
    if not silent:
        print(f"Window starting at frame {start_idx}: Mean reprojection error = {mean_error:.2f} pixels")
    
    return mean_error


def main():
    args = parse_args()

    base_dir = Path.cwd()
    transforms_path = base_dir / "transforms.json"
    depth_dir = base_dir / "stereo/depth"
    image_dir = base_dir / "undistorted/images"

    transforms = load_transforms(transforms_path)
    depth_files = sorted(list(depth_dir.glob("*_depth.npy")))

    K = np.array([[300, 0, 329], [0, 300, 329], [0, 0, 1]])
    
    # Check if required to auto-select window
    if args.auto_select_window:
        if args.center_frame is not None:
            print("Warning: Both --auto_select_window and --center_frame provided. Auto-selection will be used.")
            
        print("Auto-selecting optimal frame window...")
        window_size = 2 * args.range + 1
        
        # Check if CUDA should be used
        device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Initialize multiprocessing pool for parallel processing
        with multiprocessing.Pool() as pool:
            # Create a list of evaluation tasks
            tasks = []
            for start_idx in range(0, len(depth_files) - window_size + 1):
                tasks.append((start_idx, window_size, depth_files, image_dir, transforms, K, args.max_features, device, True))
            
            # Execute tasks in parallel
            print(f"Evaluating {len(tasks)} possible windows...")
            
            # 使用多进程并行评估所有窗口
            results = pool.starmap(evaluate_window_reprojection_error, tasks)
            
            # 找出最佳窗口
            best_error = float('inf')
            best_start_idx = 0
            for i, error in enumerate(results):
                if error < best_error:
                    best_error = error
                    best_start_idx = tasks[i][0]  # start_idx
            
        # Set center frame to the middle of the best window
        args.center_frame = best_start_idx + args.range
        print(f"Selected optimal window starting at frame {best_start_idx} (center frame: {args.center_frame})")
        print(f"Initial reprojection error: {best_error:.2f} pixels")
    
    if args.center_frame is None:
        raise ValueError("No center frame specified. Please provide --center_frame or use --auto_select_window.")

    start_idx = max(0, args.center_frame - args.range)
    end_idx = min(len(depth_files), args.center_frame + args.range + 1)
    
    # Determine the relative index of the center frame within the processing range
    center_frame_relative_idx = args.center_frame - start_idx
    
    print(f"Processing frames from {start_idx} to {end_idx-1}, total {end_idx-start_idx} frames")

    # Process each frame's data using multiprocessing
    print("Extracting features and matching...")
    
    # Prepare function parameters
    process_frame_partial = partial(
        process_frame, 
        depth_files=depth_files, 
        image_dir=image_dir,
        transforms=transforms,
        K=K,
        max_features=args.max_features
    )
    
    try:
        # Process all frames in parallel
        with multiprocessing.Pool() as pool:
            frames_data = pool.map(process_frame_partial, range(start_idx, end_idx))
    except Exception as e:
        print(f"Multiprocessing error: {e}")
        print("Falling back to single-process mode...")
        frames_data = []
        for idx in tqdm(range(start_idx, end_idx), desc="Processing frames"):
            frames_data.append(process_frame_partial(idx))
    
    print(f"Processed {len(frames_data)} frames total")
    
    # Check if CUDA should be used
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Pose optimization
    optimized_poses, iteration_errors = optimize_poses(
        frames_data, 
        K, 
        center_frame_relative_idx,
        device,
        args.num_iterations,
        args.learning_rate,
        max_error_threshold=args.error_threshold
    )
    
    # Plot error changes during optimization
    if iteration_errors:
        plot_reprojection_errors(iteration_errors, args.output_dir)
    
    # Generate point cloud using optimized poses
    all_points, all_colors = [], []
    
    for idx, frame in enumerate(frames_data):
        # Use optimized pose
        optimized_transform = optimized_poses[idx]
        
        points, colors = depth_to_pointcloud(
            frame['depth'], 
            frame['rgb'], 
            K, 
            optimized_transform, 
            args.sample_rate
        )
        
        all_points.append(points)
        all_colors.append(colors)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Create and save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save the optimized point cloud
    output_path = Path(args.output_dir) / "optimized_stack_pointcloud.ply"
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Optimized point cloud saved as {output_path}")
    
    # Save the optimized camera poses
    optimized_transforms = transforms.copy()
    for i, idx in enumerate(range(start_idx, end_idx)):
        optimized_transforms["frames"][idx]["transform_matrix"] = optimized_poses[i].tolist()
    
    optimized_transforms_path = Path(args.output_dir) / "optimized_transforms.json"
    with open(optimized_transforms_path, 'w') as f:
        json.dump(optimized_transforms, f, indent=2)
    
    print(f"Optimized camera poses saved as {optimized_transforms_path}")


if __name__ == "__main__":
    main() 