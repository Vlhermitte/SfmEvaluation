import numpy as np
import cv2
from typing import List, Dict

from utils.common import Camera, detect_colmap_format
from utils.alignment import ransac_kabsch


def evaluate_camera_pose(
        est_cameras: List[Camera],
        gt_cameras: List[Camera],
        R_threshold: float = 5,
        t_threshold: float = 0.1,
        perform_alignment: bool = True) -> Dict:
    """
    Evaluate the absolute pose error between the ground truth and estimated camera poses.
    Args:
        est_cameras (List[Camera]): List of estimated camera poses.
        gt_cameras (List[Camera]): List of ground truth camera poses.
        R_threshold (float): Rotation threshold in degrees.
        t_threshold (float): Translation threshold in meters.
        perform_alignment (bool): Whether to estimate the alignment between the estimated and ground truth poses.

    Returns:
        dict: A dictionary with the rotation_error and translation_error lists.
    """
    assert len(est_cameras) == len(gt_cameras), "Number of estimated and ground truth cameras should be the same."
    # Sort the cameras in estimated cameras based on the image name
    est_cameras = sorted(est_cameras, key=lambda camera: camera.image)
    gt_cameras = sorted(gt_cameras, key=lambda camera: camera.image)

    # Camera pose is in world-to-cam but we need cam-to-world
    est_poses = []
    for est_camera in est_cameras:
        pose = est_camera.pose
        est_poses.append(np.linalg.inv(pose))

    gt_poses = []
    for gt_camera in gt_cameras:
        pose = gt_camera.pose
        gt_poses.append(np.linalg.inv(pose))

    if perform_alignment:
        # Alignment needs a list of pose correspondences with confidences
        alignment_transformation, alignment_scale = ransac_kabsch(
            est_poses=est_poses,
            gt_poses=gt_poses,
            inlier_threshold_r=R_threshold,
            inlier_threshold_t=t_threshold,
            estimate_scale=True
        )

        if alignment_transformation is None:
            print("Alignment failed. Setting all pose errors to infinity.")
            alignment_transformation = None
            alignment_scale = 1.0
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.0

    # Evaluation loop
    rotation_errors = []
    translation_errors = []
    accuracy = 0
    for est_camera, gt_camera in zip(est_cameras, gt_cameras):
        if alignment_transformation is None or est_camera.is_valid is False:
            rotation_errors.append(np.inf)
            translation_errors.append(np.inf)
        else:
            gt_pose = alignment_transformation @ gt_camera.pose

            # Compute translation error
            t_err = np.linalg.norm(gt_pose[:3, 3] - est_camera.pose[:3, 3])
            t_err = t_err / alignment_scale
            translation_errors.append(t_err)

            # Compute rotation error
            R_err = gt_pose[:3, :3] @ est_camera.pose[:3, :3].T
            angle = cv2.Rodrigues(R_err)[0]
            angle = np.linalg.norm(angle)
            # Convert to degrees
            angle = np.degrees(angle)
            rotation_errors.append(angle)

            if angle <= R_threshold and t_err <= t_threshold:
                accuracy += 1

    accuracy = accuracy / len(est_cameras)

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

    # Evaluate pose error
    results = evaluate_camera_pose(est_cameras, gt_cameras, perform_alignment=True)

    print(f"Rotation error: {results['rotation_error']}\n")
    print(f"Translation error: {results['translation_error']}")
