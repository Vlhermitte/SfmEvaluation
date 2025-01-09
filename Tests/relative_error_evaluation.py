import numpy as np
import cv2
from common import Camera
from typing import List, Dict


def evaluate_rotation_matrices(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Evaluate the relative rotation error between the ground truth and estimated rotation matrices.

    The rotation error is computed as the angle between the two rotation matrices.
    The angle is computed using the trace of the relative rotation matrix, which is
    defined as:
        R_rel = R_est.T @ R_gt
    The angle is then computed as:
        angle = arccos((trace(R_rel) - 1) / 2)

    Note: Because the rotation matrix comes from a quaternion, there is an ambiguity
    in the angle computation. To account for this, the function returns the minimum
    angle between the two possible solutions.

    Args:
        R_gt (np.ndarray): Ground truth rotation matrix (3x3), world orientation in camera coordinate frame.
        R_est (np.ndarray): Estimated rotation matrix (3x3), world orientation in camera coordinate frame.

    Returns:
        float: The angle (in degrees) between the ground truth and estimated rotation matrices.
    """
    assert R_gt.shape == (3, 3), f'Ground truth R shape is {R_gt.shape}, expected (3, 3)'
    assert R_est.shape == (3, 3), f'Estimated R shape is {R_est.shape}, expected (3, 3)'
    assert np.allclose(np.linalg.det(R_gt), 1), 'Ground truth R determinant is not 1'
    assert np.allclose(np.linalg.det(R_est), 1), 'Estimated R determinant is not 1'

    # Compute the angle between the two rotation matrices
    R_rel = R_est.T @ R_gt
    angle = cv2.Rodrigues(R_rel)[0]
    angle = np.linalg.norm(angle)
    # Convert to degrees
    angle = np.degrees(angle)

    return angle

def evaluate_translation_error(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the relative translation error between the ground truth and estimated translation vectors.

    The translation error is computed as the Euclidean distance between the two translation vectors.

    Args:
        t_gt (np.ndarray): Ground truth translation vector (3x1), world origin in camera coordinate frame.
        t_est (np.ndarray): Estimated translation vector (3x1), world origin in camera coordinate frame.

    Returns:
        float: The Euclidean distance (translation error) between the ground truth and estimated translation vectors.
    """
    assert t_gt.shape == (3, 1) or t_gt.shape == (3,), f'Ground truth t shape is {t_gt.shape}, expected (3, 1) or (3,)'
    assert t_est.shape == (3, 1) or t_est.shape == (3,), f'Estimated t shape is {t_est.shape}, expected (3, 1) or (3,)'

    # Normalize the translation vectors (because of the scaling ambiguity)
    t_gt_norm = t_gt / np.linalg.norm(t_gt)
    t_est_norm = t_est / np.linalg.norm(t_est)

    # Compute the translation error (Euclidean distance)
    translation_error = np.linalg.norm(t_gt_norm - t_est_norm)

    return translation_error

def evaluate_relative_errors(gt_cameras: List[Camera], est_cameras: List[Camera]) -> Dict[str, List]:
    """
    Evaluate the relative rotation and translation errors on image pairs.

    Args:
        gt_cameras (Cameras): Ground truth cameras.
        est_cameras (Cameras): Estimated cameras.

    Returns:
        dict: A dictionary with the relative_rotation_error and relative_translation_error lists.
    """
    assert len(gt_cameras) == len(est_cameras), f'Number of cameras in ground truth ({len(gt_cameras)}) and estimated ({len(est_cameras)}) models do not match'
    results = {'relative_rotation_error': [], 'relative_translation_error': []}

    for i in range(len(est_cameras) - 1):
        j = i+1
        # Compute relative transformations
        R_rel_est = est_cameras[j].R @ est_cameras[i].R.T   # R.T = R^-1
        R_rel_gt = gt_cameras[j].R @ gt_cameras[i].R.T

        t_rel_est = est_cameras[j].t - (R_rel_est @ est_cameras[i].t)
        t_rel_gt = gt_cameras[j].t - (R_rel_gt @ gt_cameras[i].t)

        # Evaluate errors
        rotation_error = evaluate_rotation_matrices(R_rel_gt, R_rel_est)
        translation_error = evaluate_translation_error(t_rel_gt, t_rel_est)

        results['relative_rotation_error'].append(rotation_error)
        results['relative_translation_error'].append(translation_error)

    return results
