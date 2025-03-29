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

def write_color_distances(path, pcd, distances, max_distance, show_figure=False):
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)
    if show_figure:
        o3d.visualization.draw_geometries([pcd])

def get_f1_score_histo2(distance1, distance2, threshold, stretch=5):
    recall = float(sum(d < threshold for d in distance2)) / float(
        len(distance2) + 1e-10)
    precision = float(sum(d < threshold for d in distance1)) / float(
        len(distance1) + 1e-10)
    fscore = 2 * recall * precision / (recall + precision)
    num = len(distance1)
    bins = np.arange(0, threshold * stretch, threshold / 100)
    hist, edges_source = np.histogram(distance1, bins)
    cum_source = np.cumsum(hist).astype(float) / num + 1e-10

    num = len(distance2)
    bins = np.arange(0, threshold * stretch, threshold / 100)
    hist, edges_target = np.histogram(distance2, bins)
    cum_target = np.cumsum(hist).astype(float) / num + 1e-10

    return precision, recall, fscore, edges_source, cum_source, edges_target, cum_target

def run_tanks_and_temples_evaluation(
        scene_dir: Path,
        est_path: Path,
):
    if not isinstance(scene_dir, Path):
        scene_dir = Path(scene_dir)
    if not isinstance(est_path, Path):
        est_path = Path(est_path)

    assert scene_dir.is_dir(), f"scene_dir does not exist: {scene_dir}"
    assert est_path.exists(), f"est_path does not exist: {est_path}"

    gt_pcd = o3d.io.read_point_cloud(str(scene_dir / f"{scene_dir.name}.ply"))
    colmap_sparse_ref = read_model(scene_dir / "sparse/0")
    est_sparse_reconstruction = read_model(est_path / "colmap/sparse/0")
    cropfile = scene_dir / f"{scene_dir.name}.json"
    init_alignment_h = np.loadtxt(scene_dir / f"{scene_dir.name}_trans.txt")
    init_alignment = init_alignment_h[:3, :4]
    threshold = 0.1

    # Apply the provided initial alignment to the provided colmap reconstruction reference
    colmap_sparse_ref.transform(pycolmap.Sim3d(init_alignment))

    # Align est_reconstruction to colmap_ref
    transform = pycolmap.align_reconstructions_via_proj_centers(
        src_reconstruction=est_sparse_reconstruction,
        tgt_reconstruction=colmap_sparse_ref,
        max_proj_center_error=threshold
    )

    if transform is None:
        print("pycolmap.align_reconstructions_via_proj_centers() failed.")
        transform = pycolmap.Sim3d() # Identity transformation

    transform = transform.matrix()  # 3x4 matrix
    transform_h = np.vstack([transform, [0, 0, 0, 1]])  # 4x4 matrix

    # Converte est_reconstruction.Point3D to open3d.PointCloud
    est_pcd = o3d.geometry.PointCloud()
    if est_sparse_reconstruction.num_points3D() == 0:
        print("No points in est_sparse_reconstruction.")
        est_pcd = o3d.io.read_point_cloud(str(est_path / "colmap/sparse/0/points3D.ply"))
        if len(est_pcd.points) > 1e6:
            est_pcd = est_pcd.uniform_down_sample(int(len(est_pcd.points) / 1e6))
    else:
        bbs = est_sparse_reconstruction.compute_bounding_box(0.001, 0.999)
        for _, point3d in est_sparse_reconstruction.points3D.items():
            if (point3d.xyz >= bbs[0]).all() and (point3d.xyz <= bbs[1]).all() and point3d.error <= 6.0:
                est_pcd.points.append(point3d.xyz)
                est_pcd.colors.append(point3d.color / 255.0)
    est_pcd.transform(transform_h)

    # Refine alignment using ICP
    crop_volume = o3d.visualization.read_selection_polygon_volume(str(cropfile))

    est_pcd_cropped = copy.deepcopy(est_pcd)
    est_pcd_cropped = crop_volume.crop_point_cloud(est_pcd_cropped)
    gt_pcd_cropped = copy.deepcopy(gt_pcd)
    gt_pcd_cropped = crop_volume.crop_point_cloud(gt_pcd_cropped)

    reg_p2p_0 = icp_refinement(src_pcd=est_pcd_cropped, tgt_pcd=gt_pcd_cropped, method="voxel")
    reg_p2p = icp_refinement(
        src_pcd=est_pcd_cropped, tgt_pcd=gt_pcd_cropped, init_transformation=reg_p2p_0.transformation, method="uniform"
    )

    if reg_p2p.fitness < 0.1:
        print("ICP refinement failed.")
        reg_p2p = o3d.pipelines.registration.RegistrationResult()
        reg_p2p.transformation = np.eye(4)

    est_pcd_cropped.transform(reg_p2p.transformation)

    # Compute point cloud to point cloud distance
    distance1, distance2 = compute_pcd_to_pcd_distance(est_pcd_cropped, gt_pcd_cropped)

    # Plot color distances
    max_distance = threshold
    write_color_distances(est_path / "Ignatius_precision.ply", est_pcd_cropped, distance1, max_distance)
    write_color_distances(est_path / "Ignatius_recall.ply", gt_pcd_cropped, distance2, max_distance)

    stretch = 2
    metrics = get_f1_score_histo2(distance1, distance2, threshold, stretch=stretch)
    precision, recall, fscore, edges_source, cum_source, edges_target, cum_target = metrics

    plot_fscore(
        scene=str(scene_dir.name),
        fscore=fscore,
        dist_threshold=threshold,
        edges_source=edges_source,
        cum_source=cum_source,
        edges_target=edges_target,
        cum_target=cum_target,
        plot_stretch=stretch,
        mvs_outpath=str(est_path),
    )


if __name__ == '__main__':
    scene_dir = Path("../../data/datasets/TanksAndTemples/Ignatius")
    est_path = Path("../../data/results/glomap/TanksAndTemples/Ignatius")

    run_tanks_and_temples_evaluation(scene_dir, est_path)