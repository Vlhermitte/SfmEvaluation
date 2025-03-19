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
    **NOT IN USE FOR NOW (MIGHT GET DEPRECATED)**

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
