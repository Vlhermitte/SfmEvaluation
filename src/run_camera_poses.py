import argparse
import os
import logging
import json
import numpy as np
import pycolmap
from copy import deepcopy
from typing import Tuple, List

from utils.common import detect_colmap_format
from evaluation.absolute_error_evaluation import evaluate_camera_pose
from evaluation.relative_error_evaluation import evaluate_relative_errors


def run_rel_err_evaluation(gt_model: pycolmap.Reconstruction, est_model: pycolmap.Reconstruction, verbose: bool = False) -> Tuple[dict, dict]:
    gt_images = []
    for gt_image in gt_model.images.values():
        gt_image.name = os.path.basename(gt_image.name)
        gt_images.append(gt_image)

    est_images = []
    for est_image in est_model.images.values():
        est_image.name = os.path.basename(est_image.name)
        est_images.append(est_image)

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
    comparison_results = pycolmap.compare_reconstructions(
        reconstruction1=gt_model,
        reconstruction2=est_model,
        alignment_error='proj_center'
    )

    sim3d_transform = comparison_results['rec2_from_rec1']

    if sim3d_transform is None:
        if verbose:
            print('Failed to align the estimated model with the ground truth model. Skipping the evaluation.')
        return {'rotation_errors': 'failed', 'translation_errors': 'failed'}

    rotation_errors = []
    translation_errors = []
    for error in comparison_results['errors']:
        rotation_errors.append(error.rotation_error_deg)
        translation_errors.append(error.proj_center_error)

    abs_results = {
        'rotation_errors': rotation_errors,
        'translation_errors': translation_errors,
    }

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
        gt_sparse_model = pycolmap.Reconstruction()
        ext = detect_colmap_format(gt_model_path)
        if ext == '.bin':
            gt_sparse_model.read_binary(gt_model_path)
        elif ext == '.txt':
            gt_sparse_model.read_text(gt_model_path)
        else:
            _logger.error(f"Warning: The ground truth model format is not supported. Please use .bin or .txt format.")
            exit(1)
    except Exception as e:
        _logger.error(f"Error: {e}")
        _logger.error(f"Warning: Evaluation failed for {gt_model_path}. Please check the input model paths and try again.")
        exit(1)

    try:
        _logger.info(f"Reading estimated model {est_model_path}")
        est_sparse_model = pycolmap.Reconstruction()
        ext = detect_colmap_format(est_model_path)
        if ext == '.bin':
            est_sparse_model.read_binary(est_model_path)
        elif ext == '.txt':
            est_sparse_model.read_text(est_model_path)
        else:
            _logger.error(f"Warning: The estimated model format is not supported. Please use .bin or .txt format.")
            exit(1)
    except Exception as e:
        _logger.error(f"Error: {e}")
        _logger.error(f"Warning: Evaluation failed for {est_model_path}. Please check the input model paths and try again.")
        exit(1)

    # Find how many camera were not registered in the estimated model
    number_of_missing_cameras = len(gt_sparse_model.images) - len(est_sparse_model.images)

    ####################################################################################################################
    # Relative errors evaluation                                                                                       #
    ####################################################################################################################
    rel_results = run_rel_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)

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
    abs_results = run_abs_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)

    with open(f'{est_model_path}/absolute_results.json', 'w') as f:
        json.dump(abs_results, f, indent=4)



