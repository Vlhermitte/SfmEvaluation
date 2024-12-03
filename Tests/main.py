import numpy as np
import open3d as o3d
from typing import Tuple

from evaluation import *
from read_write_model import read_model
from pcd_alignment import perform_registration

def display_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

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

    # Compare the estimated poses with the ground truth poses
    print("Comparing estimated poses with ground truth poses:")
    for i in range(len(estimated_R)):
        R_error, t_error = evaluate_R_t(gt_R[i], gt_t[i], estimated_R[i], estimated_t[i])
        print(f"Image {i}: Rotation Error: {R_error:.4f} rad, Translation Error: {t_error:.4f} m")


    # Perform ICP alignment between the estimated and ground truth point clouds
    voxel_size = 0.01
    result_icp = perform_registration(estimated_pcd, gt_pcd, voxel_size)

    evaluate_registration(result_icp)

    # Display aligned estimated_pcd
    visualize_registration(estimated_pcd, gt_pcd, result_icp)
