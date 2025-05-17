import numpy as np
import plotly.graph_objects as go
import pycolmap
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from typing import Optional
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Open3D is not installed. Point cloud visualization with .ply files will not work.")


def plot_fscore(
    scene,
    fscore,
    dist_threshold,
    edges_source,
    cum_source,
    edges_target,
    cum_target,
    plot_stretch,
    mvs_outpath,
    show_figure=False,
):
    f = plt.figure()
    plt_size = [14, 7]
    pfontsize = "medium"

    ax = plt.subplot(111)
    label_str = "precision"
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "recall"
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)
    plt.rcParams["figure.figsize"] = plt_size
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))
    plt.title("Precision and Recall: " + scene + ", " + "%02.2f f-score" %
              (fscore * 100))
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)

    plt.ylabel("# of points (%)", fontsize=15)
    plt.xlabel("Meters", fontsize=15)
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)
    plt.legend(loc=4)
    leg = plt.legend(loc="lower right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)
    png_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.png".format(
        scene, "%04d" % (dist_threshold * 10000))
    pdf_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.pdf".format(
        scene, "%04d" % (dist_threshold * 10000))

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()

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
    reconstruction: Optional[pycolmap.Reconstruction] = None,
    pcd: Optional[o3d.geometry.PointCloud] = None,
    ps: int = 2,
    color: Optional[str] = None,
    colorscale: Optional[str] = None,
    name: Optional[str] = None
):
    """Plot a set of 3D points."""
    if pcd is not None:
        xyzs = np.asarray(pcd.points)
        pcolor = np.asarray(pcd.colors) * 255
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
        pcolor = [p3D.color for p3D in p3Ds] if color is None else color

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

def compute_camera_bbox(reconstruction: pycolmap.Reconstruction):
    # List to store camera centers
    camera_centers = []

    # Iterate over images in the reconstruction
    for image in reconstruction.images.values():
        # Skip images that do not have a valid pose (if applicable)
        if not image.registered:
            continue

        # If the image pose is stored as a quaternion (qvec) and translation (tvec),
        # convert the quaternion to a rotation matrix.
        T = image.cam_from_world.matrix()
        R = T[:3, :3]
        t = T[:3, 3]
        # Compute the camera center using: center = -R^T * t
        center = -R.T @ t
        camera_centers.append(center)

    # Convert list to a NumPy array for easier manipulation
    camera_centers = np.array(camera_centers)

    # Compute the bounding box around the cameras
    min_bound = np.min(camera_centers, axis=0) + np.min(camera_centers, axis=0) * 0.1
    max_bound = np.max(camera_centers, axis=0) + np.max(camera_centers, axis=0) * 0.1

    return min_bound, max_bound

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-model-path",
        type=str,
        required=False,
        default="../../data/results/colmap/TanksAndTemples/Truck/colmap/sparse/0",
        help="path to the ground truth model containing .bin or .txt colmap format model"
    )
    parser.add_argument(
        "--est-model-path",
        type=str,
        required=False,
        default="../../data/results/acezero/TanksAndTemples/Truck/colmap/sparse/0",
        help="path to the estimated model containing .bin or .txt colmap format model"
    )

    args = parser.parse_args()

    gt_model_path = Path(args.gt_model_path)
    est_model_path = Path(args.est_model_path)

    # Load the ground truth and estimated models.
    from run_camera_poses import read_model
    gt_sparse_model = read_model(gt_model_path)
    est_sparse_model = read_model(est_model_path)

    if len(est_sparse_model.points3D.values()) == 0:
        pcd = o3d.io.read_point_cloud(str(est_model_path / "points3D.ply"))
        pcd = pcd.uniform_down_sample(100)
        pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

        # remove any points far away from the cameras
        bbs = est_sparse_model.compute_bounding_box(0.001, 0.999)
        pcd_points = np.asarray(pcd.points)
        mask = np.all(np.logical_and(bbs[0] <= pcd_points, pcd_points <= bbs[1]), axis=1)
        pcd.points = o3d.utility.Vector3dVector(pcd_points[mask])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    else:
        points3D = est_sparse_model.points3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([p3D.xyz for p3D in points3D.values()]))
        pcd.colors = o3d.utility.Vector3dVector(np.array([p3D.color for p3D in points3D.values()]) / 255.0)

        if len(pcd.points) > 1e5:
            pcd = pcd.uniform_down_sample(10)

    print(len(pcd.points))

    comparison_results = pycolmap.compare_reconstructions(
        reconstruction1=gt_sparse_model,
        reconstruction2=est_sparse_model,
        alignment_error='proj_center'
    )

    if comparison_results is not None:
        transform = comparison_results['rec2_from_rec1'].inverse()
        est_sparse_model.transform(transform)
        transform = np.vstack((transform.matrix(), [0, 0, 0, 1]))
        if pcd is not None:
            pcd.transform(transform)
    fig = init_figure()

    # plot_trajectory(fig, gt_sparse_model)
    plot_cameras(fig, gt_sparse_model, color="rgb(0, 0, 255)")
    #plot_points(fig, gt_sparse_model)

    #plot_trajectory(fig, est_sparse_model, line_color="rgb(100, 255, 100)")
    plot_cameras(fig, est_sparse_model, color="rgb(255, 0, 0)")
    # plot_points(fig, est_sparse_model, pcd=pcd)

    fig.show()





