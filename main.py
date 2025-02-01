import argparse
import os
import logging
import json
import numpy as np
from typing import Tuple, List
from evaluation.utils.common import Camera, get_cameras_info
from evaluation.utils.read_write_model import read_model
from evaluation.core.absolute_error_evaluation import evaluate_camera_pose
from evaluation.core.relative_error_evaluation import evaluate_relative_errors

def detect_colmap_format(path: str) -> str:
    for ext in ['.txt', '.bin']:
        if os.path.isfile(os.path.join(path, "cameras" + ext)) and os.path.isfile(os.path.join(path, "images" + ext)):
            print("Detected model format: '" + ext + "'")
            return ext
    raise ValueError("No .txt or .bin format not found in the specified path")

def run_rel_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> Tuple[dict, dict]:
    if len(est_cameras) != len(gt_cameras):
        # Interpolate missing cameras in the estimated set using neighboring cameras
        # if verbose:
        #     print('Missing cameras in the estimated model. Interpolating...')
        # est_cameras = interpolate_missing_cameras(est_cameras, gt_cameras)
        if verbose:
            print('Missing cameras in the estimated model. Assigning high values for missing cameras...')
        # For every gt_cameras not in est_cameras, assign high values
        for gt_camera in gt_cameras:
            if gt_camera not in est_cameras:
                est_cameras.append(
                    Camera(gt_camera.image, gt_camera.type, np.array([0, 0, 0, 1]), np.array([1000, 1000, 1000]))
                )

    # Sort the cameras in estimated cameras based on the image name to match the ground truth
    est_cameras = sorted(est_cameras, key=lambda camera: camera.image)
    gt_cameras = sorted(gt_cameras, key=lambda camera: camera.image)

    # Evaluating
    results = evaluate_relative_errors(est_cameras=est_cameras, gt_cameras=gt_cameras)

    with open(f'{est_model_path}/relative_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def run_abs_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> dict:
    # Evaluate pose error
    results = evaluate_camera_pose(est_cameras, gt_cameras, perform_alignment=True)

    if verbose:
        print(f"Rotation error: {results['rotation_error']}\n")
        print(f"Translation error: {results['translation_error']}")

    with open(f'{est_model_path}/absolute_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def report_metrics(results, verbose: bool = False) -> tuple[dict, dict]:
    """
    Compute mean, median, and percentages below thresholds.
    """
    rotation_errors = results['relative_rotation_error']
    translation_errors = results['relative_translation_error']

    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)

    # Compute mean and median
    mean_rotation_error = np.mean(rotation_errors)
    median_rotation_error = np.median(rotation_errors)
    mean_translation_error = np.mean(translation_errors)
    median_translation_error = np.median(translation_errors)

    # Compute percentage below thresholds
    thresholds = [1, 2, 5, 10]
    rotation_percentages = [np.sum(rotation_errors < threshold) / len(rotation_errors) * 100 for threshold in thresholds]
    translation_percentages = [np.sum(translation_errors < threshold) / len(translation_errors) * 100 for threshold in thresholds]

    # Save as json file
    stats = {
        'mean_rotation_error': mean_rotation_error,
        'median_rotation_error': median_rotation_error,
        'mean_translation_error': mean_translation_error,
        'median_translation_error': median_translation_error,
        'rotation_percentages': rotation_percentages,
        'translation_percentages': translation_percentages
    }

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
        default="datasets/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="results/glomap/ETH3D/courtyard/sparse/0",
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
        _logger.info(f"Reading estimated model {est_model_path}")
        # Estimated model
        est_cameras_type, images, est_points3D = read_model(est_model_path, ext=detect_colmap_format(est_model_path))
        # Ground truth model
        gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, ext=detect_colmap_format(gt_model_path))
    except:
        _logger.error(f"Warning: Evaluation failed for {est_model_path}. Please check the input model paths and try again.")
        exit(1)

    # Create list of Camera objects for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Find how many camera were not registered in the estimated model
    number_of_missing_cameras = len(gt_cameras) - len(est_cameras)

    rel_results = run_rel_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)

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
    with open(f'{est_model_path}/auc.json', 'w') as f:
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


    # abs_results = run_abs_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)
