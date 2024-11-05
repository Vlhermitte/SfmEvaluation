import numpy as np

from Tests import read_write_model


def quat2rotmat(qvec):
    rotmat = np.array(
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2], 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
         2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ).reshape(3, 3)
    return rotmat

def matrix2Rt(A):
    R = A[0:3, 0:3]
    t = A[0:3, 3]
    return R, t

def rotationError(R_gt, R_est) -> float:
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

def translationError(t_gt, t_est) -> float:
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

if __name__ == '__main__':
    model_path = '../results/Ignatius/sparse/0/'
    db_path = '../results/Ignatius/sample_reconstruction.db'

    cameras, images, points3D = read_write_model.read_model(model_path, '.bin')

    for key, img in images.items():
        img_name = img.name
        R = quat2rotmat(img.qvec)
        t = img.tvec