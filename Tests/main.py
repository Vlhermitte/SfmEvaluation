import numpy as np
import open3d as o3d
from typing import Tuple

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

def evaluate_R_t(R_gt, t_gt, R_est, t_est) -> Tuple[float, float]:
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

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_down, pcd_fpfh

def perform_global_registration(estimated_pcd, gt_pcd, estimated_pcd_fpfh, gt_pcd_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.4
    print(":: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        estimated_pcd, gt_pcd, estimated_pcd_fpfh, gt_pcd_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def perform_fast_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Fast global registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    return result

def perform_local_registration(estimated_pcd, gt_pcd, result, voxel_size):
    distance = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance)
    result = o3d.pipelines.registration.registration_icp(
        estimated_pcd, gt_pcd, distance, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

def evaluate_registration(icp_result) -> None:
    print("ICP Alignment Results:")
    print(f"Transformation Matrix:\n{icp_result.transformation}")
    print(f"Fitness: {icp_result.fitness}")  # The higher, the better
    print(f"Inlier RMSE: {icp_result.inlier_rmse}")  # The lower, the better

def visualize_registration(source, target, result_icp) -> None:
    # Visualize alignment
    source_transformed = source.transform(result_icp.transformation)
    source_transformed.paint_uniform_color([1, 0, 0])  # Source in red
    target.paint_uniform_color([0, 1, 0])  # Target in green
    o3d.visualization.draw_geometries([source_transformed, target])

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
    voxel_size = 0.8

    # Preprocess point clouds
    estimated_pcd, estimated_pcd_down, estimated_pcd_fpfh = preprocess_point_cloud(estimated_pcd, voxel_size=voxel_size)
    gt_pcd, gt_pcd_down, gt_pcd_fpfh = preprocess_point_cloud(gt_pcd, voxel_size=voxel_size)

    # Perform global registration
    result_ransac = perform_global_registration(
        estimated_pcd_down, gt_pcd_down, estimated_pcd_fpfh, gt_pcd_fpfh, voxel_size=voxel_size
    )

    # Perform refined registration
    result_icp = perform_local_registration(
        estimated_pcd, gt_pcd, result_ransac, voxel_size=voxel_size
    )

    evaluate_registration(result_icp)

    # Display aligned estimated_pcd
    visualize_registration(estimated_pcd, gt_pcd, result_icp)
