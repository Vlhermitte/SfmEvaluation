import argparse
import logging
import json
import numpy as np
from typing import Tuple, List

from utils.common import Camera, get_cameras_info, detect_colmap_format
from data.read_write_model import read_model
from evaluation.absolute_error_evaluation import evaluate_camera_pose
from evaluation.relative_error_evaluation import evaluate_relative_errors


def run_rel_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> Tuple[dict, dict]:
    if len(est_cameras) != len(gt_cameras):
        if verbose:
            print('Missing cameras in the estimated model. Adding dummy cameras with invalid poses.')
        # For every gt_cameras not in est_cameras, add a corresponding camera with in_valid=False
        for gt_camera in gt_cameras:
            if gt_camera not in est_cameras:
                # Note: it is important to have a non-zero tvec to avoid division by zero in the trajectory alignment
                est_cameras.append(
                    Camera(gt_camera.image, gt_camera.type, np.array([0, 0, 0, 1]), np.array([1, 1, 1]), is_valid=False)
                )

    # Evaluating
    results = evaluate_relative_errors(est_cameras=est_cameras, gt_cameras=gt_cameras)

    return results

def run_abs_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> dict:
    if len(est_cameras) != len(gt_cameras):
        if verbose:
            print('Missing cameras in the estimated model. Adding dummy cameras with invalid poses.')
        # For every gt_cameras not in est_cameras, add a corresponding camera with in_valid=False
        for gt_camera in gt_cameras:
            if gt_camera not in est_cameras:
                # Note: it is important to have a non-zero tvec to avoid division by zero in the trajectory alignment
                est_cameras.append(
                    Camera(gt_camera.image, gt_camera.type, np.array([0, 0, 0, 1]), np.array([1, 1, 1]), is_valid=False)
                )

    # Evaluate pose error
    results = evaluate_camera_pose(
        est_cameras=est_cameras,
        gt_cameras=gt_cameras,
        R_threshold=5,
        t_threshold=10,
        perform_alignment=True
    )

    if verbose:
        print(f"Rotation error: {results['rotation_error']}\n")
        print(f"Translation error: {results['translation_error']}")

    return results

def compute_auc(R_error, t_error, max_threshold=30):
    """
    Compute the Area Under the Curve (AUC) for given errors and thresholds.

    Parameters:
        errors (list or np.ndarray): List of errors (e.g., rotation or translation errors).
        thresholds (list or np.ndarray): List of thresholds.

    Returns:
        List: AUC value.
    """
    R_error = np.array(R_error) if R_error is not None else np.zeros_like(t_error)
    t_error = np.array(t_error) if t_error is not None else np.zeros_like(R_error)
    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((R_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    auc = np.mean(np.cumsum(normalized_histogram)) * 100
    return auc, normalized_histogram



if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type = str,
        required = False,
        default="../data/datasets/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../data/results/glomap/ETH3D/courtyard/colmap/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=False,
        help="Print more information"
    )
    args = parser.parse_args()
    verbose = args.verbose

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    try:
        _logger.info(f"Reading ground truth model {gt_model_path}")
        gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, ext=detect_colmap_format(gt_model_path))
    except:
        _logger.error(f"Warning: Evaluation failed for {gt_model_path}. Please check the input model paths and try again.")

    try:
        _logger.info(f"Reading estimated model {est_model_path}")
        est_cameras_type, images, est_points3D = read_model(est_model_path, ext=detect_colmap_format(est_model_path))
    except:
        _logger.error(f"Warning: Evaluation failed for {est_model_path}. Please check the input model paths and try again.")
        exit(1)

    # Create list of Camera objects for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Find how many camera were not registered in the estimated model
    number_of_missing_cameras = len(gt_cameras) - len(est_cameras)

    ####################################################################################################################
    # Relative errors evaluation                                                                                       #
    ####################################################################################################################
    rel_results = run_rel_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)

    with open(f'{est_model_path}/relative_results.json', 'w') as f:
        json.dump(rel_results, f, indent=4)

    # Define thresholds
    # Compute AUC for rotation and translation errors
    Auc_30, normalized_histogram = compute_auc(rel_results['relative_rotation_error'], rel_results['relative_translation_angle'])
    Auc_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    Auc_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    Auc_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)
    if verbose:
        print(f"Auc_3  (%): {Auc_3}")
        print(f"Auc_5  (%): {Auc_5}")
        print(f"Auc_10 (%): {Auc_10}")
        print(f"Auc_30 (%): {Auc_30}")

    # RRE auc
    RRE_30, normalized_histogram = compute_auc(rel_results['relative_rotation_error'], None)
    RRE_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    RRE_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    RRE_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)
    if verbose:
        print(f'RRE_3  (%): {RRE_3}')
        print(f'RRE_5  (%): {RRE_5}')
        print(f'RRE_10 (%): {RRE_10}')
        print(f'RRE_30 (%): {RRE_30}')

    # RTE auc
    RTE_30, normalized_histogram = compute_auc(None, rel_results['relative_translation_angle'])
    RTE_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    RTE_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    RTE_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)
    if verbose:
        print(f'RTE_3  (%): {RTE_3}')
        print(f'RTE_5  (%): {RTE_5}')
        print(f'RTE_10 (%): {RTE_10}')
        print(f'RTE_30 (%): {RTE_30}')

    # Write auc to txt file
    with open(f'{est_model_path}/rel_auc.json', 'w') as f:
        json.dump({
            'Missing_cameras': number_of_missing_cameras,
            'Auc_3': Auc_3,
            'Auc_5': Auc_5,
            'Auc_10': Auc_10,
            'Auc_30': Auc_30,
            'RRE_3': RRE_3,
            'RRE_5': RRE_5,
            'RRE_10': RRE_10,
            'RRE_30': RRE_30,
            'RTE_3': RTE_3,
            'RTE_5': RTE_5,
            'RTE_10': RTE_10,
            'RTE_30': RTE_30
        }, f, indent=4)

    ####################################################################################################################
    # Absolute errors evaluation                                                                                       #
    ####################################################################################################################
    # TODO: Fix the absolute error evaluation (The rotation error seems correct but the translation error is not)
    abs_results = run_abs_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)

    with open(f'{est_model_path}/absolute_results.json', 'w') as f:
        json.dump(abs_results, f, indent=4)



