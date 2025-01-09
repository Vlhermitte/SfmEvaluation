import numpy as np
from common import Camera, get_cameras_info
from typing import List, Dict
from read_write_model import read_model
from alignment import estimate_alignment
from interpolation import interpolate_missing_cameras


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

def report_metrics(results, verbose: bool = False) -> tuple[dict, dict]:
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
    if verbose:
        print("\nCamera Pose Estimation Results")
        print("==============================")

        for metric_name, metric_stats in stats.items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Mean: {metric_stats['mean']:.3f}")
            print(f"  Median: {metric_stats['median']:.3f}")
            print(f"  Std Dev: {metric_stats['std']:.3f}")
            print(f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]")

    return stats, distributions

def run_evaluation(est_model_path: str, gt_model_path: str, verbose: bool = False) -> Tuple[dict, dict]:
    # Estimated model
    est_cameras_type, images, est_points3D = read_model(est_model_path)
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    if len(est_cameras) != len(gt_cameras):
        # Interpolate missing cameras in the estimated set using neighboring cameras
        if verbose:
            print('Missing cameras in the estimated model. Interpolating...')
        est_cameras = interpolate_missing_cameras(est_cameras, gt_cameras)

    # Sort the cameras in estimated cameras based on the image name to match the ground truth
    gt_camera_order = {camera.image: idx for idx, camera in enumerate(gt_cameras)}
    est_cameras = sorted(
        est_cameras,
        key=lambda camera: gt_camera_order.get(camera.image, float('inf'))
    )

    # Evaluating
    results = evaluate_relative_errors(est_cameras=est_cameras, gt_cameras=gt_cameras)
    stats, distributions = report_metrics(results, verbose=verbose)

    # Plot the error distributions
    plot_error_distributions(results, save_path=est_model_path)
    plot_cumulative_errors(results, save_path=est_model_path)
