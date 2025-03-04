import numpy as np
import os

from utils.geometry import quaternion2rotation

class Camera:
    """
    Camera class that stores camera information.
    The camera is of format colmap

    Attributes:
    - image: str, image name
    - type: CameraType, camera type
    - R: np.array, rotation matrix
    - t: np.array, translation vector
    - K: np.array, intrinsic matrix
    - P: np.array, projection matrix
    - pose: np.array, pose matrix
    - params: np.array, camera parameters (COLMAP sensor parameters, see https://colmap.github.io/cameras.html)
    """
    def __init__(self, image, type, qvec, tvec, is_valid=True):
        self.image = image
        self.type = type
        self.R = quaternion2rotation(qvec)
        self.t = tvec
        self.K = np.eye(3)
        self.K[0, 0] = type.params[0]
        self.K[1, 1] = type.params[1]
        self.K[0, 2] = type.width / 2
        self.K[1, 2] = type.height / 2
        self.K[2, 2] = 1
        self.P = self.K @ np.hstack((self.R, self.t.reshape(-1, 1)))
        self.pose = np.hstack((self.R, self.t.reshape(-1, 1)))
        self.params = type.params
        self.is_valid = is_valid # flag used if we add a dummy camera to the estimated cameras

    def __eq__(self, other):
        return self.image == other.image

    def __repr__(self):
        return f'Camera: {self.image} Type: {self.type.model} {", NOT VALID" if not self.is_valid else ""}'

    def __str__(self):
        return f'Camera: {self.image}'


def get_cameras_info(cameras_type, images) -> list:
    cameras = []
    for key, img in images.items():
        # take only the last part of the path as the image name
        img_name = img.name.split('/')[-1]
        qvec = img.qvec
        tvec = img.tvec
        type = cameras_type[img.camera_id]
        camera = Camera(img_name, type, qvec, tvec)
        cameras.append(camera)

    return cameras

def detect_colmap_format(path: str) -> str:
    for ext in ['.txt', '.bin']:
        if os.path.isfile(os.path.join(path, "cameras" + ext)) and os.path.isfile(os.path.join(path, "images" + ext)):
            # print("Detected model format: '" + ext + "'")
            return ext
    raise ValueError("No .txt or .bin format not found in the specified path")