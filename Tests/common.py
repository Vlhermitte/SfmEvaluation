import numpy as np

from geometry import quaternion2rotation

class Camera:
    """
    Camera class that stores camera information.
    The camera is of format colmap
    If the camera is of type THIN_PRISM_FISHEYE, the distortion parameters are as follows:
    params[4] - k1: radial distortion coefficient
    params[5] - k2: radial distortion coefficient
    params[6] - p1: tangential distortion coefficient
    params[7] - p2: tangential distortion coefficient
    params[8] - k3: radial distortion coefficient
    params[9] - k4: radial distortion coefficient
    params[10] - sx1: horizontal shear distortion coefficient
    params[11] - sy1: vertical shear distortion coefficient
    """
    def __init__(self, image, type, qvec, tvec):
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

        if type.model == 'THIN_PRISM_FISHEYE':
            self.distortion_params = type.params[4:]
        elif type.model == 'RADIAL':
            self.distortion_params = type.params[4:]

    def __repr__(self):
        return f'Camera: {self.image} Type: {self.type}'

    def __str__(self):
        return f'Camera: {self.image}'


def get_cameras_info(cameras_type, images) -> list:
    cameras = []
    for key, img in images.items():
        img_name = img.name
        qvec = img.qvec
        tvec = img.tvec
        type = cameras_type[img.camera_id]
        camera = Camera(img_name, type, qvec, tvec)
        cameras.append(camera)

    return cameras