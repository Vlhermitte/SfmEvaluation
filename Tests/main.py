import copy
import numpy as np
import open3d as o3d

from evaluation import evaluate_registration, quaternion2rotation
from read_write_model import read_model
from probreg import cpd
from icp_alignment import perform_local_registration, perform_fast_global_registration, preprocess_point_cloud

def display_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def visualize_registration(source, target, result) -> None:
    # Visualize alignment
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result])

def perform_registration(source, target, voxel_size):
    # source, target = interpolate_missing_points(source, target)
    source, source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target, target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Perform CPD registration (global registration)
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name='affine', maxiter=2, use_color=True, use_cuda=False)
    transformation = np.hstack((tf_param.b, tf_param.t.reshape(3, 1)))
    transformation = np.vstack((transformation, np.array([0, 0, 0, 1])))

    # Perform fast global registration (global registration)
    # result_global = perform_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # transformation = result_global.transformation

    # Perform ICP registration (local registration)
    result_icp = perform_local_registration(source, target, transformation, voxel_size)

    # result_icp = perform_generalized_icp(source_down, target_down, result_global.transformation, voxel_size)
    return result_icp



if __name__ == '__main__':
    model_path = '../images/out/sparse/0'
    db_path = '../images/out/database.db'

    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    # Estimated model
    cameras, images, points3D = read_model(model_path, '.bin')

    # Ground truth model
    gt_cameras, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    estimated_R = []
    estimated_t = []
    for key, img in images.items():
        img_name = img.name

        # Get estimated R and t
        R = quaternion2rotation(img.qvec)
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
        R = quaternion2rotation(img.qvec)
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

    # Remove outliers
    gt_pcd, ind = gt_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Perform alignment between the estimated and ground truth point clouds
    voxel_size = 0.05
    result_icp = perform_registration(estimated_pcd, gt_pcd, voxel_size)

    result = copy.deepcopy(estimated_pcd)
    result = result.transform(result_icp.transformation)

    evaluate_registration(result_icp)
    visualize_registration(estimated_pcd, gt_pcd, result)

