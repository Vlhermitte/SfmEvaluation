import numpy as np
import cv2
from typing import List, Dict

from utils.common import Camera
from utils.alignment import estimate_alignment


def evaluate_camera_pose(est_cameras: List[Camera], gt_cameras: List[Camera], perform_alignment: bool = True) -> Dict:
    """
    Evaluate the absolute pose error between the ground truth and estimated camera poses.
    Args:
        est_cameras (List[Camera]): List of estimated camera poses.
        gt_cameras (List[Camera]): List of ground truth camera poses.
        perform_alignment (bool): Whether to estimate the alignment between the estimated and ground truth poses.

    Returns:
        dict: A dictionary with the rotation_error and translation_error lists.
    """
    # Sort the cameras in estimated cameras based on the image name
    est_cameras = sorted(est_cameras, key=lambda camera: camera.image)
    gt_cameras = sorted(gt_cameras, key=lambda camera: camera.image)

    # Camera pose is in world-to-cam but we need cam-to-world
    est_poses = []
    for est_camera in est_cameras:
        pose = np.vstack((est_camera.pose, np.ones((1, 4))))
        est_poses.append(np.linalg.inv(pose))

    gt_poses = []
    for gt_camera in gt_cameras:
        pose = np.vstack((gt_camera.pose, np.ones((1, 4))))
        gt_poses.append(np.linalg.inv(pose))

    if perform_alignment:
        # Alignment needs a list of pose correspondences with confidences
        alignment_transformation, alignment_scale = estimate_alignment(est_poses, gt_poses, estimate_scale=True)

        if alignment_transformation is None:
            print("Alignment requested but failed. Setting all pose errors to infinity.")
            alignment_transformation = np.eye(4)
            alignment_scale = 1.0
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.0

    # Evaluation loop
    rotation_errors = []
    translation_errors = []
    for est_pose, gt_pose in zip(est_poses, gt_poses):
        gt_pose = alignment_transformation @ gt_pose

        # Compute translation error
        t_err = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
        t_err = t_err / alignment_scale
        translation_errors.append(t_err)

        # Compute rotation error
        R_rel = gt_pose[:3, :3] @ est_pose[:3, :3].T
        angle = cv2.Rodrigues(R_rel)[0]
        angle = np.linalg.norm(angle)
        # Convert to degrees
        angle = np.degrees(angle)
        rotation_errors.append(angle)

    return {'rotation_error': rotation_errors, 'translation_error': translation_errors}



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
        default="../../data/results/glomap/courtyard/colmap/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    est_cameras_type, images, est_points3D = read_model(est_model_path)
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Evaluate pose error
    results = evaluate_camera_pose(est_cameras, gt_cameras, perform_alignment=True)

    print(f"Rotation error: {results['rotation_error']}\n")
    print(f"Translation error: {results['translation_error']}")
