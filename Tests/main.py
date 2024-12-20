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
    est_cameras_type, images, est_points3D = read_model(est_model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Evaluating
    results = evaluate_relative_errors(cameras, gt_cameras)
    stats, distributions = report_metrics(results)

    # Plot the error distributions
    plot_error_distributions(results)
    plot_cumulative_errors(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type = str,
        required = False,
        default="../images/ETH3D/courtyard/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../results/colmap/courtyard/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )
    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    run_evaluation(gt_model_path=gt_model_path, est_model_path=est_model_path)

