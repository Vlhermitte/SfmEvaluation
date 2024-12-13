from typing import Tuple
import copy
import numpy as np

from geometry import quaternion2rotation
from read_write_model import read_model

class Camera:
    def __init__(self, image, type, qvec, tvec):
        self.image = image
        self.type = type
        self.qvec = qvec
        self.tvec = tvec
        self.pose = np.hstack((quaternion2rotation(qvec), tvec.reshape(3, 1)))
        self.pose = np.vstack((self.pose, np.array([0, 0, 0, 1])))

    def __repr__(self):
        return f'Camera: {self.image} Type: {self.type}'

    def __str__(self):
        return f'Camera: {self.image}'


def get_camera_info(camera_type, images) -> list:
    cameras = []
    for key, img in images.items():
        img_name = img.name
        qvec = img.qvec
        tvec = img.tvec
        camera = Camera(img_name, camera_type, qvec, tvec)
        cameras.append(camera)

    return cameras



if __name__ == '__main__':
    model_path = '../results/courtyard/sparse/0'
    db_path = '../results/courtyard/sample_reconstruction.db'

    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    aligned_path = '../results/courtyard/sparse/aligned'

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras = get_camera_info(est_cameras_type, images)
    gt_cameras = get_camera_info(gt_cameras_type, gt_images)