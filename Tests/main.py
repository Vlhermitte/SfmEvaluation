import argparse
import logging
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

def run_abs_err_evaluation(est_cameras: List[Camera], gt_cameras: List[Camera], verbose: bool = False) -> dict:
    # Evaluate pose error
    results = evaluate_camera_pose(est_cameras, gt_cameras, perform_alignment=True)

    if verbose:
        print(f"Rotation error: {results['rotation_error']}\n")
        print(f"Translation error: {results['translation_error']}")

    import json
    with open(f'{est_model_path}/results.json', 'w') as f:
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
        default="../results/acezero/courtyard/sparse",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

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

    run_rel_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)

    run_abs_err_evaluation(est_cameras=est_cameras, gt_cameras=gt_cameras)
