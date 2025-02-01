from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation

def quaternion2rotation(Q: np.ndarray) -> np.ndarray:
    R = np.array(
        [1 - 2 * Q[2] ** 2 - 2 * Q[3] ** 2, 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3],
         2 * Q[3] * Q[1] + 2 * Q[0] * Q[2], 2 * Q[1] * Q[2] + 2 * Q[0] * Q[3],
         1 - 2 * Q[1] ** 2 - 2 * Q[3] ** 2, 2 * Q[2] * Q[3] - 2 * Q[0] * Q[1],
         2 * Q[3] * Q[1] - 2 * Q[0] * Q[2], 2 * Q[2] * Q[3] + 2 * Q[0] * Q[1],
         1 - 2 * Q[1] ** 2 - 2 * Q[2] ** 2]
    ).reshape(3, 3)
    return R

def rotation2quaternion(R: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(R)
    return rot.as_quat()

def matrix2Rt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = A[0:3, 0:3]
    t = A[0:3, 3]
    return R, t