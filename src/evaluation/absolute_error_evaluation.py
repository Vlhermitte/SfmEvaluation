import numpy as np
import cv2
import pycolmap
from typing import List, Dict

from utils.common import detect_colmap_format
from utils.alignment import ransac_kabsch


def evaluate_camera_pose(
        gt_images: List[pycolmap.Image],
        est_images: List[pycolmap.Image],
        R_threshold: float = 5,
        t_threshold: float = 0.1
) -> Dict:
    """
    Evaluate the absolute pose error between the ground truth and estimated camera poses.
    Args:
        gt_images (List[pycolmap.Image]): List of ground truth images.
        est_images (List[pycolmap.Image]): List of estimated images.
        R_threshold (float): Rotation threshold in degrees.
        t_threshold (float): Translation threshold in meters.
    Returns:
        dict: A dictionary with the rotation_error and translation_error lists.
    """
    assert len(gt_images) == len(est_images), "Number of estimated and ground truth cameras should be the same."
    # Sort the cameras in estimated cameras based on the image name
    #gt_images = sorted(gt_images, key=lambda image: image.name.split('.')[0])
    #est_images = sorted(est_images, key=lambda image: image.name.split('.')[0])

    # Evaluation loop
    rotation_errors = []
    translation_errors = []
    accuracy = 0
    for gt_image, est_image in zip(gt_images, est_images):
        if gt_image.registered and est_image.registered:

            T_gt = gt_image.cam_from_world.matrix()
            T_est = est_image.cam_from_world.matrix()

            # Compute translation error
            t_err = np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])
            translation_errors.append(t_err)

            # Compute rotation error
            R_err = T_gt[:3, :3] @ T_est[:3, :3].T
            angle = cv2.Rodrigues(R_err)[0]
            angle = np.linalg.norm(angle)
            # Convert to degrees
            angle = np.degrees(angle)
            rotation_errors.append(angle)

            if angle <= R_threshold and t_err <= t_threshold:
                accuracy += 1

    accuracy = accuracy / len(est_images)

    return {'rotation_error': rotation_errors, 'translation_error': translation_errors, 'accuracy': accuracy}



if __name__ == '__main__':
    from data.read_write_model import read_model
    from utils.common import get_cameras_info
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type=str,
        required=False,
        default="../../data/datasets/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../../data/results/glomap/ETH3D/courtyard/colmap/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    est_cameras_type, images, est_points3D = read_model(est_model_path, detect_colmap_format(est_model_path))
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, detect_colmap_format(gt_model_path))

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    if len(est_cameras) != len(gt_cameras):
        # For every gt_cameras not in est_cameras, add a corresponding camera with in_valid=False
        for gt_camera in gt_cameras:
            if gt_camera not in est_cameras:
                # Note: it is important to have a non-zero tvec to avoid division by zero in the trajectory alignment
                est_cameras.append(
                    Camera(gt_camera.image, gt_camera.type, np.array([0, 0, 0, 1]), np.array([1, 1, 1]), is_valid=False)
                )

    est_cameras = sorted(est_cameras, key=lambda camera: camera.image)
    gt_cameras = sorted(gt_cameras, key=lambda camera: camera.image)

    # Evaluate pose error
    results = evaluate_camera_pose(est_cameras, gt_cameras, perform_alignment=True)

    gt_poses = np.array([gt_camera.pose[:3, 3] for gt_camera in gt_cameras])
    est_poses = np.array([est_camera.pose[:3, 3] for est_camera in est_cameras])
    plot_trajectory(gt_poses)
    plot_trajectory(est_poses)

    print(f"ATE: {ate}")
