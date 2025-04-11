# Tanks and Temples Triangulation Evaluation
# This code is based on the Tanks and Temples evaluation code.
# But it is modified to work with pycolmap and more recent versions of Open3D.

import numpy as np
import pycolmap
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from pathlib import Path

from utils.alignment import reconstruction_alignment, icp_refinement
from utils.common import read_model
from visualization.plotting import plot_fscore

def compute_pcd_to_pcd_distance(pcd0, pcd1):
    pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    distance1 = pcd0.compute_point_cloud_distance(pcd1)
    distance2 = pcd1.compute_point_cloud_distance(pcd0)
    return distance1, distance2

def write_color_distances(path, pcd, distances, max_distance, viz=False):
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)
    if viz:
        o3d.visualization.draw_geometries([pcd])

def get_f1_score_histo2(distance1, distance2, threshold, stretch=5):
    recall = float(sum(d < threshold for d in distance2)) / float(
        len(distance2) + 1e-10)
    precision = float(sum(d < threshold for d in distance1)) / float(
        len(distance1) + 1e-10)
    fscore = 2 * recall * precision / (recall + precision + 1e-10)
    num = len(distance1)
    bins = np.arange(0, threshold * stretch, threshold / 100)
    hist, edges_source = np.histogram(distance1, bins)
    cum_source = np.cumsum(hist).astype(float) / num + 1e-10

    num = len(distance2)
    bins = np.arange(0, threshold * stretch, threshold / 100)
    hist, edges_target = np.histogram(distance2, bins)
    cum_target = np.cumsum(hist).astype(float) / num + 1e-10

    metrics = {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "edges_source": edges_source,
        "cum_source": cum_source,
        "edges_target": edges_target,
        "cum_target": cum_target,
    }
    return metrics

def run_triangulation_evaluation(
        colmap_sparse_ref: pycolmap.Reconstruction,
        est_sparse_reconstruction: pycolmap.Reconstruction,
        gt_pcd: o3d.geometry.PointCloud,
        est_pcd: o3d.geometry.PointCloud,
        cropfile: Path = None,
        init_alignment: np.ndarray = None,
        verbose: bool = False,
        viz: bool = False
):
    assert gt_pcd is not None, "gt_pcd is not provided."
    assert colmap_sparse_ref is not None, "colmap_sparse_ref is not provided."
    assert est_sparse_reconstruction is not None, "est_sparse_reconstruction is not provided."
    assert est_pcd is not None, "est_dense_pcd is not provided."

    if init_alignment is not None:
        print("Using provided initial alignment.") if verbose else None
        # Make sure we have a 3x4 matrix
        init_alignment = init_alignment[:3, :4]
        # Apply the provided initial alignment to the provided colmap reconstruction reference
        colmap_sparse_ref.transform(pycolmap.Sim3d(init_alignment))

    # Align est_reconstruction to colmap_ref
    transform = pycolmap.align_reconstructions_via_proj_centers(
        src_reconstruction=est_sparse_reconstruction,
        tgt_reconstruction=colmap_sparse_ref,
        max_proj_center_error=0.1,
    )

    if transform is None:
        print("pycolmap.align_reconstructions_via_proj_centers() failed.")
        transform = pycolmap.Sim3d() # Identity transformation

    transform = transform.matrix()  # 3x4 matrix
    transform_h = np.vstack([transform, [0, 0, 0, 1]])  # 4x4 matrix

    # Uniform downsample if more than 10M points
    if len(est_pcd.points) > 10e6:
        print("More than 10e6 points, downsampling dense point cloud.") if verbose else None
        est_pcd = est_pcd.uniform_down_sample(10)

    est_pcd.transform(transform_h)

    if viz:
        gt_pcd_viz = copy.deepcopy(gt_pcd)
        gt_pcd_viz = gt_pcd_viz.uniform_down_sample(100)
        o3d.visualization.draw_geometries([est_pcd, gt_pcd_viz])

    # Refine alignment using ICP
    if cropfile is not None:
        print("Cropping point clouds.") if verbose else None
        crop_volume = o3d.visualization.read_selection_polygon_volume(str(cropfile))
        est_pcd = crop_volume.crop_point_cloud(est_pcd)
        gt_pcd = crop_volume.crop_point_cloud(gt_pcd)

    reg_p2p_0 = icp_refinement(src_pcd=est_pcd, tgt_pcd=gt_pcd, method="voxel")
    reg_p2p = icp_refinement(
        src_pcd=est_pcd, tgt_pcd=gt_pcd, init_transformation=reg_p2p_0.transformation, method="uniform"
    )

    if reg_p2p.fitness < 0.1:
        print("ICP refinement failed.")
        reg_p2p = o3d.pipelines.registration.RegistrationResult()
        reg_p2p.transformation = np.eye(4)

    est_pcd.transform(reg_p2p.transformation)

    # Draw the point clouds
    if viz:
        gt_pcd_viz = copy.deepcopy(gt_pcd)
        gt_pcd_viz = gt_pcd_viz.uniform_down_sample(100)
        o3d.visualization.draw_geometries([est_pcd, gt_pcd_viz])

    # Compute point cloud to point cloud distance
    distance1, distance2 = compute_pcd_to_pcd_distance(est_pcd, gt_pcd)

    return distance1, distance2


if __name__ == '__main__':
    scene = "Barn"
    scene_dir = Path(f"../../data/datasets/TanksAndTemples/{scene}")
    est_path = Path(f"../../data/results/vggsfm/TanksAndTemples/{scene}")

    gt_pcd = o3d.io.read_point_cloud(str(scene_dir / f"{scene_dir.name}.ply"))
    colmap_sparse_ref = read_model(scene_dir / "sparse/0")
    est_sparse_reconstruction = read_model(est_path / "colmap/dense/sparse")
    est_dense_pcd = o3d.io.read_point_cloud(str(est_path / f"colmap/dense/fused.ply"))
    cropfile = scene_dir / f"{scene_dir.name}.json"
    # Tanks and Temples applies the alignment to the estimated reconstruction
    init_alignment = np.loadtxt(scene_dir / f"{scene_dir.name}_trans.txt")

    threshold = 0.1
    viz = False
    distance1, distance2 = run_triangulation_evaluation(
        colmap_sparse_ref=colmap_sparse_ref,
        est_sparse_reconstruction=est_sparse_reconstruction,
        gt_pcd=gt_pcd,
        est_pcd=est_dense_pcd,
        cropfile=cropfile,
        init_alignment=init_alignment,
        verbose=True,
        viz=viz,
    )

    write_color_distances(est_path / f"{scene_dir.name}_precision.ply", est_dense_pcd, distance1, threshold, viz=viz)
    write_color_distances(est_path / f"{scene_dir.name}_recall.ply", gt_pcd, distance2, threshold, viz=viz)
    metrics = get_f1_score_histo2(distance1, distance2, threshold, stretch=5)

    precision = metrics["precision"]
    recall = metrics["recall"]
    fscore = metrics["fscore"]
    edges_source = metrics["edges_source"]
    cum_source = metrics["cum_source"]
    edges_target = metrics["edges_target"]
    cum_target = metrics["cum_target"]
    plot_fscore(
        scene=str(scene_dir.name),
        fscore=fscore,
        dist_threshold=threshold,
        edges_source=edges_source,
        cum_source=cum_source,
        edges_target=edges_target,
        cum_target=cum_target,
        plot_stretch=5,
        mvs_outpath=str(est_path),
    )