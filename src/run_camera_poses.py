import argparse
import os
import logging
import json
import numpy as np
import pycolmap
from copy import deepcopy
from typing import Tuple, List
from pathlib import Path

from utils.common import detect_colmap_format
from data.read_write_model import (
    read_images_binary, read_images_text, read_cameras_binary, read_cameras_text
)
from evaluation.absolute_error_evaluation import evaluate_camera_pose
from evaluation.relative_error_evaluation import evaluate_relative_errors

def read_model(model_path: Path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    assert model_path.exists(), f"Error: The ground truth model path {gt_model_path} does not exist."

    try:
        model = pycolmap.Reconstruction()
        ext = detect_colmap_format(model_path)

        if (model_path / f'cameras{ext}').exists() and (model_path / f'images{ext}').exists() and (model_path / f'points3D{ext}').exists():
            model.read_binary(model_path) if ext == '.bin' else model.read_text(model_path)
            for image in model.images.values():
                image.name = os.path.splitext(os.path.basename(image.name))[0]
        else:
            # Read manually in case points3D file is missing (THIS MAY CAUSE PROBLEMS FOR ABSOLUTE ERROR EVALUATION
            cameras = read_cameras_binary(model_path / 'cameras.bin') if ext == '.bin' else read_cameras_text(model_path / 'cameras.txt')
            for cam in cameras.values():
                camera = pycolmap.Camera(
                    camera_id=cam.id,
                    model=cam.model,
                    width=cam.width,
                    height=cam.height,
                    params=cam.params
                )
                model.add_camera(camera)

            images = read_images_binary(model_path / 'images.bin') if ext == '.bin' else read_images_text(model_path / 'images.txt')
            for img in images.values():
                quat_xyzw = img.qvec[1:] + img.qvec[:1]
                # Sometimes the COLMAP model contains the full path. This causes problem when comparing the gt model with the estimated model,
                # especially during the alignment process. So, we only keep the basename of the image name.
                basename_without_ext = os.path.splitext(os.path.basename(img.name))[0]
                image = pycolmap.Image(
                    image_id=img.id,
                    name=basename_without_ext,
                    camera_id=img.camera_id,
                    cam_from_world=pycolmap.Rigid3d(quat_xyzw, img.tvec),
                    registered = True
                )
                model.add_image(image)

        model.check()
    except Exception as e:
        print(f"Error: Failed to read the model. {e}")
        exit(1)

    return model

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
    comparison_results = pycolmap.compare_reconstructions(
        reconstruction1=gt_model,
        reconstruction2=est_model,
        alignment_error='proj_center'
    )

    if comparison_results is None:
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
        'missing_cameras': len(gt_model.images) - len(est_model.images)
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

    gt_sparse_model = read_model(gt_model_path)
    est_sparse_model = read_model(est_model_path)

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



