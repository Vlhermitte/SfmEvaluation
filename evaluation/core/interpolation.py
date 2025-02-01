import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import copy
from typing import List, Optional
from evaluation.utils.common import Camera
from geometry import rotation2quaternion


def interpolate_missing_cameras(est_cameras: List[Camera], gt_cameras: List[Camera]) -> List[Camera]:
    """
    Interpolate missing cameras in the estimated set using neighboring cameras.

    Args:
        est_cameras: List of estimated Camera objects
        gt_cameras: List of ground truth Camera objects

    Returns:
        List of Camera objects with interpolated values for missing cameras
    """
    # Create a deep copy to avoid modifying original data
    est_cameras = copy.deepcopy(est_cameras)

    # Create a mapping of image names to indices for estimated cameras
    est_camera_dict = {cam.image: idx for idx, cam in enumerate(est_cameras)}

    # Find missing cameras
    missing_cameras = []
    for idx, gt_cam in enumerate(gt_cameras):
        if gt_cam.image not in est_camera_dict:
            missing_cameras.append((idx, gt_cam))

    # For each missing camera, find nearest available cameras before and after
    for missing_idx, missing_cam in missing_cameras:
        # Find nearest available cameras
        prev_idx = next_idx = None
        prev_cam = next_cam = None

        # Search backward for previous camera
        for i in range(missing_idx - 1, -1, -1):
            if gt_cameras[i].image in est_camera_dict:
                prev_idx = i
                prev_cam = est_cameras[est_camera_dict[gt_cameras[i].image]]
                break

        # Search forward for next camera
        for i in range(missing_idx + 1, len(gt_cameras)):
            if gt_cameras[i].image in est_camera_dict:
                next_idx = i
                next_cam = est_cameras[est_camera_dict[gt_cameras[i].image]]
                break

        # Interpolate based on available neighboring cameras
        interpolated_cam = interpolate_camera(
            prev_cam, next_cam, missing_cam,
            missing_idx - prev_idx if prev_idx is not None else None,
            next_idx - missing_idx if next_idx is not None else None
        )

        # If interpolation wasn't possible, fall back to ground truth
        if interpolated_cam is None:
            print(f'Warning: No neighboring cameras found for {missing_cam.image}. Using ground truth.')
            interpolated_cam = copy.deepcopy(missing_cam)

        # Insert the interpolated camera at the correct position
        est_cameras.append(interpolated_cam)

    # Sort cameras to match ground truth order
    gt_camera_order = {camera.image: idx for idx, camera in enumerate(gt_cameras)}
    est_cameras = sorted(
        est_cameras,
        key=lambda camera: gt_camera_order.get(camera.image, float('inf'))
    )

    return est_cameras


def interpolate_camera(prev_cam: Optional[Camera], next_cam: Optional[Camera],
                       missing_cam: Camera, dist_to_prev: Optional[int],
                       dist_to_next: Optional[int]) -> Optional[Camera]:
    """
    Interpolate a camera position and orientation based on neighboring cameras.

    Args:
        prev_cam: Previous available camera or None
        next_cam: Next available camera or None
        missing_cam: Ground truth camera used for intrinsics and type
        dist_to_prev: Distance to previous camera in sequence
        dist_to_next: Distance to next camera in sequence

    Returns:
        Interpolated Camera object or None if interpolation not possible
    """
    if prev_cam is None and next_cam is None:
        return None

    # If only one neighbor is available, copy its pose
    if prev_cam is None:
        return copy.deepcopy(next_cam)
    if next_cam is None:
        return copy.deepcopy(prev_cam)

    # Calculate interpolation weight based on distances
    total_dist = dist_to_prev + dist_to_next
    weight = dist_to_next / total_dist  # Weight for previous camera

    # Interpolate translation
    t_interp = weight * prev_cam.t + (1 - weight) * next_cam.t

    # Interpolate rotation using SLERP
    times = [0, 1]  # Normalized time points
    rots = Rotation.from_matrix(np.stack([prev_cam.R, next_cam.R]))
    slerp = Slerp(times, rots)
    R_interp = slerp(1 - weight).as_matrix()  # Interpolate at weighted position

    # Convert interpolated rotation to quaternion for Camera constructor
    qvec = rotation2quaternion(R_interp)

    # Create interpolated camera using the same type as the missing camera
    # This preserves intrinsic parameters and distortion model
    interpolated_cam = Camera(
        image=missing_cam.image,
        type=missing_cam.type,
        qvec=qvec,
        tvec=t_interp
    )

    return interpolated_cam

