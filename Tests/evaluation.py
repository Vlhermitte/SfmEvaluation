import numpy as np

def evaluate_camera_pose(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the positional accuracy of an estimated camera pose against the ground truth.

    The camera pose consists of a rotation matrix (R) and a translation vector (t),
    which together define the transformation from the world coordinate frame to
    the camera coordinate frame. This function computes the Euclidean distance
    between the camera centers derived from the ground truth and the estimated poses.

    Camera center in the world coordinate frame is computed as:
        p = -R^T @ t
    where:
        - R is the rotation matrix (3x3),
        - t is the translation vector (3x1).

    Args:
        R_gt (np.ndarray): Ground truth rotation matrix (3x3), transforming points
                           from world to camera coordinates.
        t_gt (np.ndarray): Ground truth translation vector (3x1), representing the
                           camera position in the world frame.
        R_est (np.ndarray): Estimated rotation matrix (3x3), transforming points
                            from world to camera coordinates.
        t_est (np.ndarray): Estimated translation vector (3x1), representing the
                            camera position in the world frame.

    Returns:
        float: The Euclidean distance (position error) between the ground truth
               and estimated camera centers in the world coordinate frame.
    """
    assert R_gt.shape == (3, 3), f'Ground truth R shape is {R_gt.shape}, expected (3, 3)'
    assert R_est.shape == (3, 3), f'Estimated R shape is {R_est.shape}, expected (3, 3)'
    assert t_gt.shape == (3, 1) or t_gt.shape == (3,), f'Ground truth t shape is {t_gt.shape}, expected (3, 1) or (3,)'
    assert t_est.shape == (3, 1) or t_est.shape == (3,), f'Estimated t shape is {t_est.shape}, expected (3, 1) or (3,)'
    assert np.allclose(np.linalg.det(R_gt), 1), 'Ground truth R determinant is not 1'
    assert np.allclose(np.linalg.det(R_est), 1), 'Estimated R determinant is not 1'

    # Compute estimated camera position (world coordinates)
    C_est = -R_est.T @ t_est

    # Compute ground truth camera position
    C_gt = -R_gt.T @ t_gt

    # Compute position error (Euclidean distance)
    position_error = np.linalg.norm(C_est - C_gt)

    return position_error

def evaluate_rotation_matrices(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Evaluate the relative rotation error between the ground truth and estimated rotation matrices.

    The rotation error is computed as the angle between the two rotation matrices.
    The angle is computed using the trace of the relative rotation matrix, which is
    defined as:
        R_rel = R_est.T @ R_gt
    The angle is then computed as:
        angle = arccos((trace(R_rel) - 1) / 2)

    Note: Because the rotation matrix comes from a quaternion, there is an ambiguity
    in the angle computation. To account for this, the function returns the minimum
    angle between the two possible solutions.

    Args:
        R_gt (np.ndarray): Ground truth rotation matrix (3x3), transforming points
                           from world to camera coordinates.
        R_est (np.ndarray): Estimated rotation matrix (3x3), transforming points
                            from world to camera coordinates.

    Returns:
        float: The angle (in degrees) between the ground truth and estimated rotation matrices.
    """
    assert R_gt.shape == (3, 3), f'Ground truth R shape is {R_gt.shape}, expected (3, 3)'
    assert R_est.shape == (3, 3), f'Estimated R shape is {R_est.shape}, expected (3, 3)'
    assert np.allclose(np.linalg.det(R_gt), 1), 'Ground truth R determinant is not 1'
    assert np.allclose(np.linalg.det(R_est), 1), 'Estimated R determinant is not 1'

    # Compute the angle between the two rotation matrices
    R_rel = R_est.T @ R_gt
    trace_R_rel = np.trace(R_rel)
    trace_R_rel = np.clip(trace_R_rel, -1, 3)

    # Consider ambiguity due to quaternion representation
    angle = np.arccos((trace_R_rel - 1) / 2) * 180 / np.pi
    angle_opposite = np.arccos((-trace_R_rel - 1) / 2)

    return min(angle, angle_opposite)

def evaluate_translation_error(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """
    Evaluate the relative translation error between the ground truth and estimated translation vectors.

    The translation error is computed as the Euclidean distance between the two translation vectors.

    Args:
        t_gt (np.ndarray): Ground truth translation vector (3x1), representing the camera position in the world frame.
        t_est (np.ndarray): Estimated translation vector (3x1), representing the camera position in the world frame.

    Returns:
        float: The Euclidean distance (translation error) between the ground truth and estimated translation vectors.
    """
    assert t_gt.shape == (3, 1) or t_gt.shape == (3,), f'Ground truth t shape is {t_gt.shape}, expected (3, 1) or (3,)'
    assert t_est.shape == (3, 1) or t_est.shape == (3,), f'Estimated t shape is {t_est.shape}, expected (3, 1) or (3,)'

    # Normalize the translation vectors (because of the scaling ambiguity)
    t_gt_norm = t_gt / np.linalg.norm(t_gt)
    t_est_norm = t_est / np.linalg.norm(t_est)

    # Compute the translation error (Euclidean distance)
    translation_error = np.linalg.norm(t_gt_norm - t_est_norm)

    return translation_error

if __name__ == '__main__':
    from geometry import quaternion2rotation
    from common import get_cameras_info
    from read_write_model import read_model

    model_path = '../results/colmap/courtyard/sparse/0'
    # model_path = '../results/vggsfm/courtyard/sparse'
    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras = get_cameras_info(est_cameras_type, images)
    gt_cameras = get_cameras_info(gt_cameras_type, gt_images)

    # Evaluate camera pose
    results = {'position_error': [], 'rotation_error': [], 'translation_error': []}
    for camera, gt_camera in zip(cameras, gt_cameras):
        R_est = camera.R
        R_gt = gt_camera.R
        position_error = evaluate_camera_pose(R_gt, gt_camera.t, R_est, camera.t)
        rotation_error = evaluate_rotation_matrices(R_gt, R_est)
        translation_error = evaluate_translation_error(gt_camera.t, camera.t)
        results['position_error'].append(position_error)
        results['rotation_error'].append(rotation_error)
        results['translation_error'].append(translation_error)

    print(f'Average position error: {np.mean(results["position_error"]):.2f} meters')
    print(f'Average rotation error: {np.mean(results["rotation_error"]):.2f} degrees')
    print(f'Average translation error: {np.mean(results["translation_error"]):.2f} meters')