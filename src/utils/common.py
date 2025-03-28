import numpy as np
import os
import warnings

from utils.geometry import quaternion2rotation
from data.read_write_model import (
    read_images_binary, read_images_text, read_cameras_binary, read_cameras_text
)

class Camera:
    """
    DEPRECATED: Use pycolmap.Camera instead
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
        warnings.warn("Camera class is deprecated. Use pycolmap.Camera instead", DeprecationWarning)
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
        self.pose = np.vstack((
            np.hstack((self.R, self.t.reshape(-1, 1))),
            np.array([0, 0, 0, 1])
        ))
        self.params = type.params
        self.is_valid = is_valid # flag used if we add a dummy camera to the estimated cameras

    def __eq__(self, other):
        return self.image == other.image

    def __repr__(self):
        return f'Camera: {self.image} Type: {self.type.model} {", NOT VALID" if not self.is_valid else ""}'

    def __str__(self):
        return f'Camera: {self.image}'

def get_cameras_info(cameras_type, images) -> list:
    """
    DEPRECATED: Use pycolmap.get_cameras_info instead
    """
    warnings.warn("get_cameras_info is deprecated. Use pycolmap.get_cameras_info instead", DeprecationWarning)
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


def read_model(model_path: Path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    try:
        assert model_path.exists(), f"Error: The ground truth model path {model_path} does not exist."
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

    except Exception as e:
        print(f"Error: Failed to read the {model_path}. {e}")
        return None

    return model

def detect_colmap_format(path: str) -> str:
    for ext in ['.txt', '.bin']:
        if os.path.isfile(os.path.join(path, "cameras" + ext)) and os.path.isfile(os.path.join(path, "images" + ext)):
            # print("Detected model format: '" + ext + "'")
            return ext
    raise ValueError("No .txt or .bin format not found in the specified path")