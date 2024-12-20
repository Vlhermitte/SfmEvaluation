import numpy as np
from common import Camera
from typing import List, Dict

def evaluate_camera_pose(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the positional accuracy of an estimated camera pose against the ground truth.

    The camera pose consists of a rotation matrix (R) and a translation vector (t),
    which together define the transformation from the world coordinate frame to
    the camera coordinate frame. This function computes the Euclidean distance
    between the camera centers derived from the ground truth and the estimated poses.

    Camera center in the world coordinate frame is computed as:
        p = -R^T @ t
    where:
        - R is the rotation matrix (3x3),
        - t is the translation vector (3x1).

    Args:
        R_gt (np.ndarray): Ground truth rotation matrix (3x3), transforming points
                           from world to camera coordinates.
        t_gt (np.ndarray): Ground truth translation vector (3x1), representing the
                           camera position in the world frame.
        R_est (np.ndarray): Estimated rotation matrix (3x3), transforming points
                            from world to camera coordinates.
        t_est (np.ndarray): Estimated translation vector (3x1), representing the
                            camera position in the world frame.

    Returns:
        float: The Euclidean distance (position error) between the ground truth
               and estimated camera centers in the world coordinate frame.
    """
    assert R_gt.shape == (3, 3), f'Ground truth R shape is {R_gt.shape}, expected (3, 3)'
    assert R_est.shape == (3, 3), f'Estimated R shape is {R_est.shape}, expected (3, 3)'
    assert t_gt.shape == (3, 1) or t_gt.shape == (3,), f'Ground truth t shape is {t_gt.shape}, expected (3, 1) or (3,)'
    assert t_est.shape == (3, 1) or t_est.shape == (3,), f'Estimated t shape is {t_est.shape}, expected (3, 1) or (3,)'
    assert np.allclose(np.linalg.det(R_gt), 1), 'Ground truth R determinant is not 1'
    assert np.allclose(np.linalg.det(R_est), 1), 'Estimated R determinant is not 1'

    # Compute estimated camera position (world coordinates)
    C_est = -R_est.T @ (t_est / np.linalg.norm(t_est))

    # Compute ground truth camera position
    C_gt = -R_gt.T @ (t_gt / np.linalg.norm(t_gt))

    # Compute position error (Euclidean distance)
    position_error = np.linalg.norm(C_est - C_gt)

    return position_error

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
    trace_R_rel = np.trace(R_rel)
    trace_R_rel = np.clip(trace_R_rel, -1, 3)

    # Consider ambiguity due to quaternion representation
    angle = np.arccos((trace_R_rel - 1) / 2) * 180 / np.pi
    angle_opposite = np.arccos((-trace_R_rel - 1) / 2)

    return min(angle, angle_opposite)

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
    results = {'relative_rotation_error': [], 'relative_translation_error': []}

    for i in range(len(est_cameras) - 1):
        for j in range(i + 1, len(est_cameras)):
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

def report_metrics(results) -> tuple[dict, dict]:
    """
    Compute and report comprehensive metrics for camera pose estimation.

    Args:
        results (dict): Dictionary containing lists of errors
            - rotation_error: List of rotation errors in degrees
            - translation_error: List of translation errors
            - position_error: List of position errors
    """

    # Compute summary statistics
    stats = {}
    for metric_name, values in results.items():
        stats[metric_name] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # Compute error distributions
    rotation_bins = [0, 5, 10, 15, 20, float('inf')]  # in degrees
    translation_bins = [0, 0.05, 0.10, 0.15, 0.20, float('inf')]

    def compute_histogram(values, bins):
        hist, _ = np.histogram(values, bins=bins)
        return hist.tolist()

    distributions = {
        'rotation_hist': compute_histogram(results['relative_rotation_error'], rotation_bins),
        'translation_hist': compute_histogram(results['relative_translation_error'], translation_bins)
    }

    # Print summary
    print("\nCamera Pose Estimation Results")
    print("==============================")

    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Mean: {metric_stats['mean']:.3f}")
        print(f"  Median: {metric_stats['median']:.3f}")
        print(f"  Std Dev: {metric_stats['std']:.3f}")
        print(f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]")

    return stats, distributions
