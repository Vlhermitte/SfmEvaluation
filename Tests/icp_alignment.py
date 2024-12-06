import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def interpolate_missing_points(pcd1, pcd2):
    # Ensure pcd1 has more points than pcd2
    if len(pcd1.points) < len(pcd2.points):
        pcd1, pcd2 = pcd2, pcd1

    # Convert point clouds to numpy arrays
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Calculate the number of points to interpolate
    missing_points_count = len(points1) - len(points2)

    # Build a k-d tree for the smaller point cloud (pcd2)
    tree = cKDTree(points2)

    # Sample random points from pcd2 and interpolate
    sampled_indices = np.random.choice(len(points2), size=missing_points_count, replace=True)
    sampled_points = points2[sampled_indices]

    # For each sampled point, find its nearest neighbor in pcd2
    _, nearest_indices = tree.query(sampled_points, k=1)
    interpolated_points = points2[nearest_indices]

    # Add interpolated points to pcd2
    new_points2 = np.vstack([points2, interpolated_points])

    # Create updated point cloud for pcd2
    pcd2_new = o3d.geometry.PointCloud()
    pcd2_new.points = o3d.utility.Vector3dVector(new_points2)

    return pcd1, pcd2_new

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd, pcd_down, pcd_fpfh

def perform_global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999))
    return result

def perform_global_registration_correspondence(source, target, correspondences, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds with correspondence.")
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target, correspondences, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)
    )
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

def perform_local_registration(source, target, init_transformation, voxel_size):
    distance = voxel_size * 1.5
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=100)
    )
    return result

def perform_colored_icp(source, target, init_transformation, voxel_size):
    distance = voxel_size * 0.4
    print(":: Colored point cloud registration")
    print("   We use a strict distance threshold %.3f." % distance)
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, distance, init_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=100)
    )
    return result

def perform_generalized_icp(source, target, init_transformation, voxel_size):
    # Set the distance threshold based on the voxel size
    distance_threshold = voxel_size * 0.4

    # Print information
    print(":: Performing Generalized ICP (GICP) registration")

    # Initialize GICP-based registration
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=100))

    return result