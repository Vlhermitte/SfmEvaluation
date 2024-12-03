import open3d as o3d
import numpy as np

# TODO: When preprocessing the point cloud, interpolate the missing points to have the same number of points in both point clouds
# TODO: Then use registration_ransac_based_on_correspondence() to perform the global registration

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
    distance = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
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

def perform_registration(source, target, voxel_size):
    source, source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target, target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_global = perform_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = perform_local_registration(source, target, result_global.transformation, voxel_size)
    return result_icp