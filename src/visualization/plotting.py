import numpy as np
import plotly.graph_objects as go
import pycolmap
from scipy.interpolate import splprep, splev
from typing import Optional
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Open3D is not installed. Point cloud visualization with .ply files will not work.")

def init_figure() -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig

def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    text: Optional[str] = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    corners = np.concatenate((corners, np.ones((corners.shape[0], 1))), axis=1) # Homogeneous coordinates
    corners = corners @ np.linalg.inv(K).T
    corners = (corners / 2) @ R.T + t

    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>"),
    )
    fig.add_trace(pyramid)


def plot_cameras(fig: go.Figure, reconstruction: pycolmap.Reconstruction, color: str = "rgb(0, 0, 255)"):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        world_t_camera = image.cam_from_world.inverse()
        camera = reconstruction.cameras[image.camera_id]
        plot_camera(
            fig,
            world_t_camera.rotation.matrix(),
            world_t_camera.translation,
            camera.calibration_matrix(),
            name=str(image.image_id),
            text=str(image),
            color=color
        )


def plot_trajectory(
        fig: go.Figure,
        reconstruction: pycolmap.Reconstruction,
        line_color: str = "rgb(255, 0, 0)"
):
    """
    Plot a smooth trajectory line of the cameras
    """

    # Sort the images to get a consistent order (here, by image_id).
    sorted_images = sorted(reconstruction.images.values(), key=lambda img: img.image_id)

    # Extract camera centers.
    positions = np.array([img.cam_from_world.inverse().translation for img in sorted_images])
    if positions.shape[0] < 2:
        print("Not enough cameras to plot a trajectory.")
        return

    # Compute a smooth spline interpolation between camera centers.
    try:
        # s=0 forces the spline to pass through all points; adjust s if smoothing is desired.
        tck, u = splprep([positions[:, 0], positions[:, 1], positions[:, 2]], s=0)
        # Evaluate the spline on a dense set of points.
        u_new = np.linspace(0, 1, num=200)
        x_new, y_new, z_new = splev(u_new, tck)
    except Exception as e:
        print("Error computing spline interpolation:", e)
        return

    # Plot the smooth trajectory line.
    trajectory_line = go.Scatter3d(
        x=x_new,
        y=y_new,
        z=z_new,
        mode="lines",
        line=dict(color=line_color, width=2),
        name="Trajectory",
        hoverinfo="none"
    )
    fig.add_trace(trajectory_line)

def plot_points(
    fig: go.Figure,
    reconstruction: pycolmap.Reconstruction,
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
    ply_file: Optional[Path] = None,
):
    """Plot a set of 3D points."""
    if ply_file is not None and ply_file.exists() and len(reconstruction.points3D.items()) == 0 and 'o3d' in globals():
        pcd = o3d.io.read_point_cloud(str(ply_file))
        # Limit to 100k points (compute voxel size)
        if len(pcd.points) > 100000:
            # Get the bounding box
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            # Compute the dimensions and volume of the bounding box
            dims = max_bound - min_bound
            volume = dims[0] * dims[1] * dims[2]
            # Desired number of points
            target_points = 100000
            # Calculate voxel size
            voxel_size = (volume / target_points)
            pcd = pcd.voxel_down_sample(voxel_size)
        xyzs = np.asarray(pcd.points)
        pcolor = np.asarray(pcd.colors)
    else:
        # Filter outliers
        bbs = reconstruction.compute_bounding_box(0.001, 0.999)
        # Filter points, use original reproj error here
        p3Ds = [
            p3D
            for _, p3D in reconstruction.points3D.items()
            if (
                    (p3D.xyz >= bbs[0]).all()
                    and (p3D.xyz <= bbs[1]).all()
                    and p3D.error <= 6.0
            )
        ]
        xyzs = np.array([p3D.xyz for p3D in p3Ds])
        pcolor = [p3D.color for p3D in p3Ds]

    x, y, z = xyzs.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        marker=dict(size=ps, color=pcolor, line_width=0.0, colorscale=colorscale),
    )
    fig.add_trace(tr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type=str,
        required=False,
        default="../../data/datasets/ETH3D/delivery_area/dslr_calibration_jpg",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../../data/results/flowmap/ETH3D/delivery_area/colmap/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )

    args = parser.parse_args()

    gt_model_path = Path(args.gt_model_path)
    est_model_path = Path(args.est_model_path)

    # Load the ground truth and estimated models.
    from run_camera_poses import read_model
    gt_sparse_model = read_model(gt_model_path)
    est_sparse_model = read_model(est_model_path)

    comparison_results = pycolmap.compare_reconstructions(
        reconstruction1=gt_sparse_model,
        reconstruction2=est_sparse_model,
        alignment_error='proj_center'
    )

    if comparison_results is not None:
        est_sparse_model.transform(comparison_results['rec2_from_rec1'].inverse())

    fig = init_figure()

    # plot_trajectory(fig, gt_sparse_model)
    plot_cameras(fig, gt_sparse_model)
    plot_points(fig, gt_sparse_model)

    # plot_trajectory(fig, est_sparse_model, line_color="rgb(100, 255, 100)")
    plot_cameras(fig, est_sparse_model, color="rgb(227, 130, 66)")
    # plot_points(fig, est_sparse_model, ply_file=est_model_path / "points3D.ply")

    fig.show()





