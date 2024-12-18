from typing import Tuple
import copy
import numpy as np

from geometry import quaternion2rotation
from read_write_model import read_model
from common import Camera, get_cameras_info



if __name__ == '__main__':
    model_path = '../results/colmap/courtyard/sparse/0'
    db_path = '../results/colmap/courtyard/sample_reconstruction.db'

    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    aligned_path = '../results/colmap/courtyard/sparse/aligned'

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)