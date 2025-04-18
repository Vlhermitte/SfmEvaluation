import numpy as np
import cv2
import pycolmap
import os
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
    gt_images = sorted(gt_images, key=lambda image: os.path.basename(image.name.split('.')[0]))
    est_images = sorted(est_images, key=lambda image: os.path.basename(image.name.split('.')[0]))

    # Evaluation loop
    rotation_errors = []
    translation_errors = []
    accuracy = 0
    for gt_image, est_image in zip(gt_images, est_images):
        if gt_image.registered and est_image.registered:
            T_gt = gt_image.cam_from_world.matrix()
            T_est = est_image.cam_from_world.matrix()

            T_gt = np.vstack((T_gt, [0, 0, 0, 1]))
            T_est = np.vstack((T_est, [0, 0, 0, 1]))

            R_gt = T_gt[:3, :3]
            R_est = T_est[:3, :3]
            R_error = R_est @ R_gt.T
            t_gt = T_gt[:3, 3]
            t_est = T_est[:3, 3]
            c_gt = -R_gt.T * t_gt
            c_est = -R_est.T * t_est

            angle = np.arccos((np.trace(R_error) - 1) / 2)
            angle = np.degrees(angle)
            c_error = np.linalg.norm(c_est - c_gt)
            if angle <= R_threshold and c_error <= t_threshold:
                accuracy += 1

            rotation_errors.append(np.degrees(angle))
            translation_errors.append(c_error)
        else:
            # Assign high values for missing cameras
            rotation_errors.append(180.0)
            translation_errors.append(100.0)

    accuracy = accuracy / len(est_images)

    return {'rotation_errors': rotation_errors, 'translation_errors': translation_errors, 'accuracy': accuracy}
