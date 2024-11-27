import numpy as np
import open3d as o3d

from Tests import read_write_model

def display_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def quat2rotmat(qvec):
    rotmat = np.array(
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2], 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
         2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ).reshape(3, 3)
    return rotmat

def matrix2Rt(A):
    R = A[0:3, 0:3]
    t = A[0:3, 3]
    return R, t

def rotationError(R_gt, R_est) -> float:
    """
    Evaluate the rotation error between the ground truth and the estimated rotation matrix.
    Args:
        R_gt: 3x3 ground truth rotation matrix
        R_est: 3x3 estimated rotation matrix

    Returns:
        The rotation error in radians.
    """
    R = R_gt @ R_est.T
    return np.arccos((np.trace(R) - 1) / 2)

def translationError(t_gt, t_est) -> float:
    """
    Evaluate the translation error between the ground truth and the estimated translation vector.
    The error is the Euclidean L2 norm between the two vectors.
    Args:
        t_gt: 3x1 ground truth translation vector
        t_est: 3x1 estimated translation vector

    Returns:
        The translation error in meters.
    """
    return np.linalg.norm(t_gt - t_est)

def evaluate_R_t(R_gt, t_gt, R_est, t_est) -> tuple:
    """
    Evaluate the rotation and translation errors between the ground truth and the estimated poses.
    Args:
        R_gt: 3x3 ground truth rotation matrix
        t_gt: 3x1 ground truth translation vector
        R_est: 3x3 estimated rotation matrix
        t_est: 3x1 estimated translation vector

    Returns:
        A tuple containing the rotation and translation errors in radians and meters respectively.
    """
    R_error = rotationError(R_gt, R_est)
    t_error = translationError(t_gt, t_est)
    return R_error, t_error

def perform_icp(source_pcd, target_pcd, voxel_size=0.05, visualisation=False):
    """
    Perform ICP alignment between two point clouds.
    Args:
        source_pcd: Source point cloud.
        target_pcd: Target point cloud.
        voxel_size: Voxel size for downsampling.
        visualisation: Enable visualisation or not
    Returns:
        Transformation matrix and aligned source point cloud.
    """

    # Downsample point clouds
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals (necessary for colored ICP or point-to-plane ICP)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Initial transformation (identity)
    initial_transformation = np.eye(4)

    # Run ICP (Point-to-Point)
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, max_correspondence_distance=voxel_size * 2,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply transformation to original source point cloud
    source_transformed = source_pcd.transform(icp_result.transformation)

    # Print result
    print("ICP Alignment Results:")
    print(f"Transformation Matrix:\n{icp_result.transformation}")
    print(f"Fitness: {icp_result.fitness}")
    print(f"Inlier RMSE: {icp_result.inlier_rmse}")

    # Visualize alignment
    if visualisation:
        source_transformed.paint_uniform_color([1, 0, 0])  # Source in red
        target_pcd.paint_uniform_color([0, 1, 0])  # Target in green
        o3d.visualization.draw_geometries([source_transformed, target_pcd])

    return icp_result.transformation, source_transformed

if __name__ == '__main__':
    model_path = '../images/out/sparse/0'
    db_path = '../images/out/database.db'

    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    # Estimated model
    cameras, images, points3D = read_write_model.read_model(model_path, '.bin')

    # Ground truth model
    gt_cameras, gt_images, gt_points3D = read_write_model.read_model(gt_model_path, '.txt')

    estimated_R = []
    estimated_t = []
    for key, img in images.items():
        img_name = img.name

        # Get estimated R and t
        R = quat2rotmat(img.qvec)
        t = img.tvec

        estimated_R.append(R)
        estimated_t.append(t)

    # Create Open3D point cloud
    points = []
    colors = []
    for point3D in points3D.values():
        points.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
        colors.append([point3D.rgb[0] / 255.0, point3D.rgb[1] / 255.0, point3D.rgb[2] / 255.0])
    estimated_pcd = o3d.geometry.PointCloud()
    estimated_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    estimated_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    # display_point_cloud(estimated_pcd)

    gt_R = []
    gt_t = []
    for key, img in gt_images.items():
        img_name = img.name

        # Get estimated R and t
        R = quat2rotmat(img.qvec)
        t = img.tvec

        gt_R.append(R)
        gt_t.append(t)

    # Create Open3D point cloud for ground truth
    gt_points = []
    gt_colors = []
    for point3D in gt_points3D.values():
        gt_points.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
        gt_colors.append([point3D.rgb[0] / 255.0, point3D.rgb[1] / 255.0, point3D.rgb[2] / 255.0])
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(np.array(gt_points))
    gt_pcd.colors = o3d.utility.Vector3dVector(np.array(gt_colors))
    # display_point_cloud(gt_pcd)

    # Compare the estimated poses with the ground truth poses
    print("Comparing estimated poses with ground truth poses:")
    for i in range(len(estimated_R)):
        R_error, t_error = evaluate_R_t(gt_R[i], gt_t[i], estimated_R[i], estimated_t[i])
        print(f"Image {i}: Rotation Error: {R_error:.4f} rad, Translation Error: {t_error:.4f} m")


    # Perform ICP alignment between the estimated and ground truth point clouds
    # TODO : Find a correct voxel value and check why alignment is poor
    transformation, aligned_pcd = perform_icp(estimated_pcd, gt_pcd, voxel_size=1, visualisation=True)

    # Display aligned estimated_pcd
    # display_point_cloud(aligned_pcd)

