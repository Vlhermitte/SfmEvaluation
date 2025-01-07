from typing import Tuple
import copy
import numpy as np
import argparse

from read_write_model import read_model
from common import Camera, get_cameras_info
from evaluation import evaluate_relative_errors, report_metrics
from plotting import plot_error_distributions, plot_cumulative_errors

def run_evaluation(est_model_path: str, gt_model_path: str):
    # Estimated model
    est_cameras_type, images, est_points3D = read_model(est_model_path)
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    est_cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Check if any missing cameras in the estimated model
    missing_cameras = []
    for gt_camera in gt_cameras:
        if gt_camera not in est_cameras:
            missing_cameras.append(gt_camera)

    # TODO: Handle missing cameras. For now I simply add the gt camera to the estimated cameras as a placeholder.
    for missing_camera in missing_cameras:
        # Add the missing camera to the estimated cameras
        est_cameras.append(copy.deepcopy(missing_camera))

    # Sort the cameras in estimated cameras based on the image name to match the ground truth
    gt_camera_order = {camera.image: idx for idx, camera in enumerate(gt_cameras)}
    est_cameras = sorted(
        est_cameras,
        key=lambda camera: gt_camera_order.get(camera.image, float('inf'))
    )

    # Evaluating
    results = evaluate_relative_errors(est_cameras=est_cameras, gt_cameras=gt_cameras)
    stats, distributions = report_metrics(results)

    # Plot the error distributions
    plot_error_distributions(results, save_path=est_model_path)
    plot_cumulative_errors(results, save_path=est_model_path)

if __name__ == '__main__':
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
        default="../results/glomap/courtyard/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    run_evaluation(gt_model_path=gt_model_path, est_model_path=est_model_path)

