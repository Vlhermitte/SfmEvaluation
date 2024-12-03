import numpy as np
import open3d as o3d

from typing import Tuple


def quaternion2rotation(Q: np.ndarray) -> np.ndarray:
    R = np.array(
        [1 - 2 * Q[2] ** 2 - 2 * Q[3] ** 2, 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3],
         2 * Q[3] * Q[1] + 2 * Q[0] * Q[2], 2 * Q[1] * Q[2] + 2 * Q[0] * Q[3],
         1 - 2 * Q[1] ** 2 - 2 * Q[3] ** 2, 2 * Q[2] * Q[3] - 2 * Q[0] * Q[1],
         2 * Q[3] * Q[1] - 2 * Q[0] * Q[2], 2 * Q[2] * Q[3] + 2 * Q[0] * Q[1],
         1 - 2 * Q[1] ** 2 - 2 * Q[2] ** 2]
    ).reshape(3, 3)
    return R

def matrix2Rt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = A[0:3, 0:3]
    t = A[0:3, 3]
    return R, t

def rotationError(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Evaluate the rotation error between the ground truth and the estimated rotation matrix.
    Args:
        R_gt: 3x3 ground truth rotation matrix
        R_est: 3x3 estimated rotation matrix

    Returns:
        The rotation error in radians.
    """
    R = R_gt @ R_est.T
    return np.arccos((np.trace(R) - 1) / 2)

def translationError(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the translation error between the ground truth and the estimated translation vector.
    The error is the Euclidean L2 norm between the two vectors.
    Args:
        t_gt: 3x1 ground truth translation vector
        t_est: 3x1 estimated translation vector

    Returns:
        The translation error in meters.
    """
    return np.linalg.norm(t_gt - t_est)

def evaluate_R_t(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the rotation and translation errors between the ground truth and the estimated poses.
    Args:
        R_gt: 3x3 ground truth rotation matrix
        t_gt: 3x1 ground truth translation vector
        R_est: 3x3 estimated rotation matrix
        t_est: 3x1 estimated translation vector

    Returns:
        A tuple containing the rotation and translation errors in radians and meters respectively.
    """
    R_error = rotationError(R_gt, R_est)
    t_error = translationError(t_gt, t_est)
    return R_error, t_error

def evaluate_registration(icp_result: o3d.pipelines.registration.RegistrationResult) -> None:
    print("ICP Alignment Results:")
    print(f"Transformation Matrix:\n{icp_result.transformation}")
    print(f"Fitness: {icp_result.fitness}")  # The higher, the better
    print(f"Inlier RMSE: {icp_result.inlier_rmse}")  # The lower, the better