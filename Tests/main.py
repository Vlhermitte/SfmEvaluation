import argparse
import logging
import json
import numpy as np
from typing import Tuple, List
from common import Camera, get_cameras_info
from plotting import plot_percentage_below_thresholds
from read_write_model import read_model
from absolute_error_evaluation import evaluate_camera_pose
from interpolation import interpolate_missing_cameras
from relative_error_evaluation import evaluate_relative_errors

def run_rel_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> Tuple[dict, dict]:
    if len(est_cameras) != len(gt_cameras):
        # Interpolate missing cameras in the estimated set using neighboring cameras
        # if verbose:
        #     print('Missing cameras in the estimated model. Interpolating...')
        # est_cameras = interpolate_missing_cameras(est_cameras, gt_cameras)
        if verbose:
            print('Missing cameras in the estimated model. Assigning high values for missing cameras...')
        # For every est_camera not in gt_cameras, assign high values
        for est_camera in est_cameras:
            if est_camera not in gt_cameras:
                est_camera.R = np.eye(3)
                est_camera.t = np.array([1000, 1000, 1000])

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
    R_error = np.array(R_error)
    t_error = np.array(t_error)
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
    auc = np.mean(np.cumsum(normalized_histogram))
    return auc, normalized_histogram



if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type = str,
        required = False,
        default="../datasets/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../results/vggsfm/courtyard/sparse",
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
        est_cameras_type, images, est_points3D = read_model(est_model_path)
    except:
        _logger.error(f"Warning: Absolute error evaluation failed for {est_model_path}. Please check the input model paths and try again.")
        exit(1)

    # Ground truth model
    try:
        _logger.info(f"Reading ground truth model {gt_model_path}")
        gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')
    except:
        _logger.error(f"Warning: Absolute error evaluation failed for {gt_model_path}. Please check the input model paths and try again.")
        exit(1)

    # Create list of Camera objects for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Find how many camera were not registered in the estimated model
    number_of_missing_cameras = len(gt_cameras) - len(est_cameras)

    rel_results = run_rel_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)

    # Define thresholds
    #angle_thresholds = [5, 15, 30]
    # Compute AUC for rotation and translation errors
    Auc_30, normalized_histogram = compute_auc(rel_results['relative_rotation_error'], rel_results['relative_translation_angle'])
    Auc_3 = np.mean(np.cumsum(normalized_histogram[:3]))
    Auc_5 = np.mean(np.cumsum(normalized_histogram[:5]))
    Auc_10 = np.mean(np.cumsum(normalized_histogram[:10]))
    if verbose:
        print(f"Auc_3  (%): {Auc_3 * 100}")
        print(f"Auc_5  (%): {Auc_5 * 100}")
        print(f"Auc_10 (%): {Auc_10 * 100}")
        print(f"Auc_30 (%): {Auc_30 * 100}")

    # Write auc to txt file
    with open(f'{est_model_path}/auc.txt', 'w') as f:
        f.write(f'Number of unregistered images: {number_of_missing_cameras}\n')
        f.write(f'Auc_3  (%): {Auc_3 * 100}\n')
        f.write(f'Auc_5  (%): {Auc_5 * 100}\n')
        f.write(f'Auc_10 (%): {Auc_10 * 100}\n')
        f.write(f'Auc_30 (%): {Auc_30 * 100}\n')


    # abs_results = run_abs_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)
