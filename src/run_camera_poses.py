import argparse
import os
import logging
import json
import numpy as np
import pycolmap
from copy import deepcopy
from typing import Tuple, List
from pathlib import Path

from utils.common import detect_colmap_format, read_model
from evaluation.absolute_error_evaluation import evaluate_camera_pose
from evaluation.relative_error_evaluation import evaluate_relative_errors

def run_rel_err_evaluation(gt_model: pycolmap.Reconstruction, est_model: pycolmap.Reconstruction, verbose: bool = False) -> Tuple[dict, dict]:
    gt_images = [image for image in gt_model.images.values()]
    est_images = [image for image in est_model.images.values()]

    if len(gt_images) != len(est_images):
        if verbose:
            print('Missing cameras in the estimated model. Adding dummy cameras with invalid poses.')

        est_image_name = [est_image.name.split('.')[0] for est_image in est_images]
        for gt_image in gt_images:
            if gt_image.name.split('.')[0] not in est_image_name:
                dummy_image = pycolmap.Image()
                dummy_image.name = gt_image.name
                dummy_image.registered = False
                est_images.append(dummy_image)

    # Evaluating
    results = evaluate_relative_errors(gt_images=gt_images, est_images=est_images)

    return results

def run_abs_err_evaluation(gt_model: pycolmap.Reconstruction, est_model: pycolmap.Reconstruction, verbose: bool = False) -> dict:
    transform = pycolmap.align_reconstructions_via_proj_centers(
        src_reconstruction=est_model,
        tgt_reconstruction=gt_model,
        max_proj_center_error=0.1,
    )

    if transform is None:
        if verbose:
            print('Failed to align the estimated model with the ground truth model. Skipping the evaluation.')
        return {'rotation_errors': 'failed', 'translation_errors': 'failed'}
    # Apply the alignment to the estimated model
    est_model.transform(transform.matrix())

    gt_images = [image for image in gt_model.images.values()]
    est_images = [image for image in est_model.images.values()]

    if len(gt_images) != len(est_images):
        if verbose:
            print('Missing cameras in the estimated model. Adding dummy cameras with invalid poses.')

        est_image_name = [est_image.name.split('.')[0] for est_image in est_images]
        for gt_image in gt_images:
            if gt_image.name.split('.')[0] not in est_image_name:
                dummy_image = pycolmap.Image()
                dummy_image.name = gt_image.name
                dummy_image.registered = False
                est_images.append(dummy_image)

    abs_results = evaluate_camera_pose(
        gt_images=gt_images,
        est_images=est_images,
        R_threshold=5,
        t_threshold=0.1
    )

    abs_results['number_of_missing_cameras'] = len(gt_model.images) - len(est_model.images)

    return abs_results

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

def export_rel_results(rel_results: dict, output_path: Path):
    with open(f'{output_path}/relative_results.json', 'w') as f:
        json.dump(rel_results, f, indent=4)

    # Define thresholds
    # Compute AUC for rotation and translation errors
    Auc_30, normalized_histogram = compute_auc(rel_results['relative_rotation_error'],
                                               rel_results['relative_translation_angle'])
    Auc_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    Auc_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    Auc_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)

    # RRE auc
    RRE_30, normalized_histogram = compute_auc(rel_results['relative_rotation_error'], None)
    RRE_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    RRE_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    RRE_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)

    # RTE auc
    RTE_30, normalized_histogram = compute_auc(None, rel_results['relative_translation_angle'])
    RTE_3 = np.mean(np.cumsum(normalized_histogram[:3]) * 100)
    RTE_5 = np.mean(np.cumsum(normalized_histogram[:5]) * 100)
    RTE_10 = np.mean(np.cumsum(normalized_histogram[:10]) * 100)

    number_of_missing_cameras = rel_results['number_of_missing_cameras']

    # Write auc to txt file
    with open(f'{output_path}/rel_auc.json', 'w') as f:
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

def export_abs_results(abs_results: dict, output_path: Path):
    with open(f'{output_path}/absolute_results.json', 'w') as f:
        json.dump(abs_results, f, indent=4)

    if abs_results['rotation_errors'] == 'failed' or abs_results['translation_errors'] == 'failed':
        rotation_errors = np.NAN
        translation_errors = np.NAN
        number_of_missing_cameras = np.NAN
        accuracy = np.NAN
    else:
        # For each rotation error, compute mean and std
        rotation_errors = abs_results['rotation_errors']
        translation_errors = abs_results['translation_errors']
        number_of_missing_cameras = abs_results['number_of_missing_cameras']
        accuracy = abs_results['accuracy']

    min_rotation_error = np.min(rotation_errors)
    max_rotation_error = np.max(rotation_errors)
    mean_rotation_error = np.mean(rotation_errors)
    median_rotation_error = np.median(rotation_errors)
    rotation_99_percentile = np.percentile(rotation_errors, 99)
    rotation_95_percentile = np.percentile(rotation_errors, 95)

    min_translation_error = np.min(translation_errors)
    max_translation_error = np.max(translation_errors)
    mean_translation_error = np.mean(translation_errors)
    median_translation_error = np.median(translation_errors)
    translation_99_percentile = np.percentile(translation_errors, 99)
    translation_95_percentile = np.percentile(translation_errors, 95)



    # Write mean and std to txt file
    with open(f'{output_path}/abs_errors_summary.json', 'w') as f:
        json.dump({
            'Missing_cameras': number_of_missing_cameras,
            'Accuracy': accuracy,
            'Rotation_errors': {
                'Min': min_rotation_error,
                'Max': max_rotation_error,
                'Mean': mean_rotation_error,
                'Median': median_rotation_error,
                '95_percentile': rotation_95_percentile,
                '99_percentile': rotation_99_percentile,
            },
            'Translation_errors': {
                'Min': min_translation_error,
                'Max': max_translation_error,
                'Mean': mean_translation_error,
                'Median': median_translation_error,
                '95_percentile': translation_95_percentile,
                '99_percentile': translation_99_percentile

            }
        }, f, indent=4)

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

    gt_sparse_model = read_model(gt_model_path)
    est_sparse_model = read_model(est_model_path)

    if est_sparse_model is None or gt_sparse_model is None:
        _logger.error("Error reading the models. Please check the paths and formats.")
        exit(1)

    # Find how many camera were not registered in the estimated model
    number_of_missing_cameras = len(gt_sparse_model.images) - len(est_sparse_model.images)

    ####################################################################################################################
    # Relative errors evaluation                                                                                       #
    ####################################################################################################################
    rel_results = run_rel_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)
    export_rel_results(rel_results, Path(est_model_path).parent)


    ####################################################################################################################
    # Absolute errors evaluation                                                                                       #
    ####################################################################################################################
    abs_results = run_abs_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)
    export_abs_results(abs_results, Path(est_model_path).parent)





