import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict
import pycolmap


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

def evaluate_translation_angle(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the angular error (in degrees) between the ground truth and estimated translation vectors.

    Args:
        t_gt (np.ndarray): Ground truth translation vector (3x1), world origin in camera coordinate frame.
        t_est (np.ndarray): Estimated translation vector (3x1), world origin in camera coordinate frame.

    Returns:
        float: The angular error (in degrees) between the ground truth and estimated translation vectors.
    """
    assert t_gt.shape == (3, 1) or t_gt.shape == (3,), f'Ground truth t shape is {t_gt.shape}, expected (3, 1) or (3,)'
    assert t_est.shape == (3, 1) or t_est.shape == (3,), f'Estimated t shape is {t_est.shape}, expected (3, 1) or (3,)'

    # Normalize the translation vectors (to ensure they are unit vectors)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-10) # 1e-10 to avoid division by zero
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-10)

    # Compute the cosine of the angle using the dot product
    cos_theta = np.clip(np.dot(t_gt_norm, t_est_norm), -1.0, 1.0)  # Clip to avoid numerical issues

    # Compute the angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def evaluate_relative_errors(gt_images: List[pycolmap.Image], est_images: List[pycolmap.Image]) -> Dict[str, List]:
    """
    Evaluate the relative rotation and translation errors on image pairs.

    Args:
        gt_images (Cameras): Ground truth cameras.
        est_images (Cameras): Estimated cameras.

    Returns:
        dict: A dictionary with the relative_rotation_error and relative_translation_error lists.
    """
    assert len(gt_images) == len(est_images), "Number of estimated and ground truth cameras should be the same."

    # Sort the cameras in estimated cameras based on the image name to match the ground truth
    gt_images = sorted(gt_images, key=lambda image: image.name.split('.')[0])
    est_images = sorted(est_images, key=lambda image: image.name.split('.')[0])

    results = {'relative_rotation_error': [], 'relative_translation_angle': [], 'number_of_missing_cameras': 0}

    # Evaluate on all possible pairs from all images
    pairs = [(i, j) for i in range(len(gt_images)) for j in range(i + 1, len(gt_images))]
    for i, j in tqdm(pairs, desc='Evaluating relative errors'):
        if est_images[i].registered and est_images[j].registered:
            # cam_from_world is the transformation from world to camera (https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Image.cam_from_world)
            T_i_gt = gt_images[i].cam_from_world.matrix()
            T_j_gt = gt_images[j].cam_from_world.matrix()
            T_i_est = est_images[i].cam_from_world.matrix()
            T_j_est = est_images[j].cam_from_world.matrix()

            R_i_gt = T_i_gt[:3, :3]
            R_j_gt = T_j_gt[:3, :3]
            R_i_est = T_i_est[:3, :3]
            R_j_est = T_j_est[:3, :3]

            t_i_gt = T_i_gt[:3, 3]
            t_j_gt = T_j_gt[:3, 3]
            t_i_est = T_i_est[:3, 3]
            t_j_est = T_j_est[:3, 3]

            # Compute relative rotation between pairs (R_ij = R_j * R_i^T)
            R_ij_gt = R_j_gt @ R_i_gt.T
            R_ij_est = R_j_est @ R_i_est.T

            # Compute relative translation between pairs (t_ij = t_j - R_ij * t_i)
            t_ij_gt = t_j_gt - (R_ij_gt @ t_i_gt)
            t_ij_est = t_j_est - (R_ij_est @ t_i_est)

            # Evaluate errors
            rotation_error = evaluate_rotation_matrices(R_ij_gt, R_ij_est)
            translation_angle = evaluate_translation_angle(t_ij_gt, t_ij_est)

            results['relative_rotation_error'].append(rotation_error)
            results['relative_translation_angle'].append(translation_angle)
        else:
            # Assign high values for missing cameras
            results['relative_rotation_error'].append(180.0)
            results['relative_translation_angle'].append(180.0)
    results['number_of_missing_cameras'] = len([image for image in est_images if not image.registered])

    return results
