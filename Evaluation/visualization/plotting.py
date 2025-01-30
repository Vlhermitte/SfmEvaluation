import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev


def plot_percentage_below_thresholds(rotation_errors, translation_errors, thresholds, save_path=None):
    """
    Plot the percentage of errors below different thresholds.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Percentage of errors below different thresholds')

    for i, (errors, title) in enumerate(zip([rotation_errors, translation_errors], ['Rotation', 'Translation'])):
        ax[i].set_title(f'{title} errors')
        ax[i].set_xlabel('Threshold')
        ax[i].set_ylabel('Percentage of errors')
        ax[i].set_xticks(thresholds)
        ax[i].set_ylim(0, 100)

        for threshold in thresholds:
            percentage = np.sum(errors < threshold) / len(errors) * 100
            ax[i].bar(threshold, percentage, width=0.1, color='blue')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_trajectory(trajectory, save_path=None):
    """
    Plot the trajectory of the camera.
    """
    # Create a B-spline representation of the trajectory
    tck, u = splprep([trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]], s=0)
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_fine, y_fine, z_fine, color='r', linestyle='-')
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory (B-spline)')
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    import argparse
    from Evaluation.main import read_model, get_cameras_info, detect_colmap_format
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type=str,
        required=False,
        default="../datasets/ETH3D/terrains/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        # default="../datasets/House/sparse",
        default="../results/House/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )

    args = parser.parse_args()

    gt_model_path = args.gt_model_path
    est_model_path = args.est_model_path

    # Estimated model
    est_cameras_type, images, est_points3D = read_model(est_model_path, ext=detect_colmap_format(est_model_path))
    est_cameras = get_cameras_info(est_cameras_type, images)
    est_cameras = sorted(est_cameras, key=lambda camera: camera.image)

    # Extract the camera poses
    est_trajectory = np.array([camera.pose[:3, 3] for camera in est_cameras])

    plot_trajectory(est_trajectory, save_path='plots/est_trajectory.png')
