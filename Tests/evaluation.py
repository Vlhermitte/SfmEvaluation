import numpy as np
import open3d as o3d


def evaluate_camera_pose(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray):
    """
    Evaluation of position error, not translation error.
    Args:
        R_gt: Ground truth rotation matrix (3x3) in camera coordinate frame
        t_gt: Ground truth translation vector (3x1) in camera coordinate frame
        R_est: Estimated rotation matrix (3x3) in camera coordinate frame
        t_est: Estimated translation vector (3x1) in camera coordinate frame

    Returns:
        position_error: Euclidean distance between estimated and ground truth camera position
    """
    # Compute estimated camera position in world coordinates
    p_est = -R_est.T @ t_est

    # Compute ground truth camera position in world coordinates
    p_gt = -R_gt.T @ t_gt

    # Compute position error (Euclidean distance)
    position_error = np.linalg.norm(p_est - p_gt)

    return position_error

def evaluate_registration(estimated_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud, transformation: np.ndarray):
    """
    Evaluate the registration between estimated and ground truth point clouds.
    Args:
        estimated_pcd: Estimated point cloud
        gt_pcd: Ground truth point cloud
        transformation: Estimated transformation matrix (4x4)

    Returns:
        fitness: Fitness score
        inlier_rmse: Inlier RMSE
    """
    # Apply transformation
    estimated_pcd.transform(transformation)

    # Evaluate registration
    evaluation = o3d.pipelines.registration.evaluate_registration(estimated_pcd, gt_pcd,
                                                                 max_correspondence_distance=np.inf)

    fitness = evaluation.fitness
    inlier_rmse = evaluation.inlier_rmse

    return fitness, inlier_rmse

def evaluate_points_cloud(estimated_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud):
    """
    Given 2 points clouds, evaluate the accuracy of the estimated point cloud.
    The evaluation is done by evaluation the distance from the estimated point cloud to nearest neighbor in the ground truth point cloud.
    Args:
        estimated_pcd: Estimated point cloud
        gt_pcd: Ground truth point cloud

    Returns:
        mean_distance: Mean distance between estimated point cloud and ground truth point cloud
    """
    # compute_point_cloud_distance info : https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Point-Cloud-Distance
    distances = estimated_pcd.compute_point_cloud_distance(gt_pcd)
    mean_distance = np.mean(distances)

    return mean_distance