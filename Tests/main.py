from typing import Tuple
import numpy as np
import open3d as o3d

from geometry import quaternion2rotation
from read_write_model import read_model
from trajectory import align
from icp_alignment import perform_registration

class Camera:
    def __init__(self, image, type, qvec, tvec):
        self.image = image
        self.type = type
        self.qvec = qvec
        self.tvec = tvec
        self.pose = np.hstack((quaternion2rotation(qvec), tvec.reshape(3, 1)))
        self.pose = np.vstack((self.pose, np.array([0, 0, 0, 1])))


def display_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def visualize_registration(source, target, result) -> None:
    # Visualize alignment
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result])

def get_cam_pcd(camera_type, images, points3D) -> Tuple[list, o3d.geometry.PointCloud]:
    cameras = []
    for key, img in images.items():
        img_name = img.name
        qvec = img.qvec
        tvec = img.tvec
        camera = Camera(img_name, camera_type, qvec, tvec)
        cameras.append(camera)

    # Create Open3D point cloud
    points = []
    colors = []
    for point3D in points3D.values():
        points.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
        colors.append([point3D.rgb[0] / 255.0, point3D.rgb[1] / 255.0, point3D.rgb[2] / 255.0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return cameras, pcd



if __name__ == '__main__':
    model_path = '../images/out/sparse/0'
    db_path = '../images/out/database.db'

    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras, estimated_pcd = get_cam_pcd(est_cameras_type, images, est_points3D)
    gt_cameras, gt_pcd = get_cam_pcd(gt_cameras_type, gt_images, gt_points3D)

    # Align trajectories
    estimated_trajectory = [camera.pose for camera in cameras]
    gt_trajectory = [camera.pose for camera in gt_cameras]

    aligned_trajectory = align(estimated_trajectory, gt_trajectory)
