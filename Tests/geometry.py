from typing import Tuple
import numpy as np

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