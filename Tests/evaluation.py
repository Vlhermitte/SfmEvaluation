import numpy as np

def evaluate_camera_pose(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray):
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
    p_est = -R_est.T @ t_est

    # Compute ground truth camera position
    p_gt = -R_gt.T @ t_gt

    # Compute position error (Euclidean distance)
    position_error = np.linalg.norm(p_est - p_gt)

    return position_error

if __name__ == '__main__':
    from geometry import quaternion2rotation
    from main import get_camera_info
    from read_write_model import read_model

    model_path = '../results/courtyard/sparse/aligned'
    db_path = '../results/courtyard/sample_reconstruction.db'
    gt_model_path = '../images/ETH3D/courtyard/dslr_calibration_jpg'

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(model_path, '.bin')
    # Ground truth model
    gt_cameras_type, gt_images, gt_points3D = read_model(gt_model_path, '.txt')

    # Create Open3D point cloud and get R and t for estimated and ground truth models
    cameras = get_camera_info(est_cameras_type, images)
    gt_cameras = get_camera_info(gt_cameras_type, gt_images)

    # Evaluate camera pose
    results = []
    for camera, gt_camera in zip(cameras, gt_cameras):
        R = quaternion2rotation(camera.qvec)
        R_gt = quaternion2rotation(gt_camera.qvec)
        position_error = evaluate_camera_pose(R_gt, gt_camera.tvec, R, camera.tvec)
        results.append(position_error)

    print(f'Average position error: {np.mean(results):.2f} meters')