import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
import open3d as o3d
import numpy as np
import pycolmap

from utils.common import detect_colmap_format, read_model
from run_camera_poses import (
    run_rel_err_evaluation, run_abs_err_evaluation, export_rel_results, export_abs_results
)
from run_nerfstudio import sanity_check_colmap, run_nerfstudio
from run_triangulation import parse_mlp_matrices
from evaluation.triangulation_evaluation import run_triangulation_evaluation, write_color_distances, get_f1_score_histo2
from visualization.plotting import plot_fscore

from config import (
        ETH3D_DATA_PATH, ETH3D_SCENES,
        MIP_NERF_360_DATA_PATH, MIP_NERF_360_SCENES,
        TANKS_AND_TEMPLES_DATA_PATH, TANKS_AND_TEMPLES_SCENES,
        GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH,
        COLMAP_FORMAT, COLMAP_DENSE_FORMAT
    )

def extract_pcd_from_model(model: pycolmap.Reconstruction) -> o3d.geometry.PointCloud:
    """
    Extract the point cloud from the model.
    :param model: pycolmap.Reconstruction, the model.
    :return: o3d.geometry.PointCloud, the point cloud.
    """
    est_pcd = o3d.geometry.PointCloud()
    bbs = model.compute_bounding_box(0.001, 0.999)
    for _, point3d in model.points3D.items():
        if (point3d.xyz >= bbs[0]).all() and (point3d.xyz <= bbs[1]).all() and point3d.error <= 6.0:
            est_pcd.points.append(point3d.xyz)
            est_pcd.colors.append(point3d.color / 255.0)
    return est_pcd


class Evaluator:
    """
    Easy to use class to run any of the following evaluations:
    - Camera poses evaluation
    - Novel view synthesis evaluation
    - Triangulation evaluation
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(logging.StreamHandler())

    @staticmethod
    def run_camera_evaluator(gt_sparse_model, est_sparse_model) -> Tuple[list, list]:
        """
        Run the camera poses evaluation.
        :param gt_sparse_model: pycolmap.Reconstruction, the ground truth model.
        :param est_sparse_model: pycolmap.Reconstruction, the estimated model.
        :return: rel_results, abs_results.
        """
        rel_results, abs_results = None, None

        # Run the evaluation
        if (gt_sparse_model and est_sparse_model) is not None:
            rel_results = run_rel_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)
            abs_results = run_abs_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)

        return rel_results, abs_results

    def run_novel_view_synthesis_evaluator(self, image_path: Path, colamp_model: pycolmap.Reconstruction, method: str="nerfacto") -> Tuple[float, float, float]:
        """
        Run the novel view synthesis evaluation using the nerfstudio tool.
        :param image_path: Path, the path to the images.
        :param colamp_model: pycolmap.Reconstruction, the colmap model.
        :param method: str, the method to use for the evaluation. Default is "nerfacto".
        :return: ssim, psnr, lpips.
        """
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        assert image_path.exists() and image_path.is_dir(), f"Error: The image path {image_path} does not exist or is not a directory."
        assert colamp_model is not None, "Error: The colmap model path is None."
        assert method in ["nerfacto", "splatfacto"], "Error: The method is not valid. Choose between 'nerfacto' and 'splatfacto'."

        sanity_check_colmap(colamp_model)

        # Run the evaluation
        run_nerfstudio(dataset_path=image_path.parent, results_path=colamp_model, method=method)

        # The results are saved in colamp_model/nerfstudio/<method>/run/eval.json
        if os.path.exists(f"{colamp_model}/nerfstudio/{method}/run/eval.json"):
            with open(f"{colamp_model}/nerfstudio/{method}/run/eval.json", "r") as f:
                results = json.load(f)["results"]
                ssim = results["ssim"]
                psnr = results["psnr"]
                lpips = results["lpips"]
                return ssim, psnr, lpips
        else:
            self._logger.error(f"Error: The evaluation results file does not exist. Maybe the evaluation failed.")
            return None

    @staticmethod
    def run_triangulation_evaluator(
            colmap_sparse_ref: pycolmap.Reconstruction,
            est_sparse_model: pycolmap.Reconstruction,
            gt_scan_pcd: o3d.geometry.PointCloud,
            est_dense_pcd: Optional[o3d.geometry.PointCloud]=None,
            cropfile: Optional[np.ndarray]=None,
            init_alignment: Optional[np.ndarray]=None,
            threshold=0.1,
            verbose=False,
            viz=False
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run the triangulation evaluation.
        :param colmap_sparse_ref: pycolmap.Reconstruction, the colmap sparse model.
        :param est_sparse_model: pycolmap.Reconstruction, the estimated sparse model.
        :param gt_scan_pcd: o3d.geometry.PointCloud, the ground truth scan point cloud.
        :param est_dense_pcd: o3d.geometry.PointCloud, the estimated dense point cloud (Optional).
        :param cropfile: np.ndarray, the crop file (Optional).
        :param init_alignment: np.ndarray, the initial alignment (Optional).
        :param threshold: float, the threshold.
        :param verbose: bool, whether to print the logs.
        :param viz: bool, whether to visualize the results.
        :return: distance1, distance2, metrics.
        """
        distance1, distance2 = run_triangulation_evaluation(
            colmap_sparse_ref=colmap_sparse_ref,
            est_sparse_reconstruction=est_sparse_model,
            est_pcd=est_dense_pcd,
            gt_pcd=gt_scan_pcd,
            cropfile=cropfile,
            init_alignment=init_alignment,
            verbose=verbose,
            viz=viz
        )

        return distance1, distance2



if __name__ == '__main__':
    evaluator = Evaluator()
    triangulation_threshold = 0.1
    for results in [GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH]:
        print("Evaluating", results)
        for scene in ETH3D_SCENES:
            print(scene)
            gt_model_path = ETH3D_DATA_PATH / scene / "dslr_calibration_jpg"
            est_model_path = results / "ETH3D" / scene / COLMAP_FORMAT
            image_path = ETH3D_DATA_PATH / scene / "images"
            if not est_model_path.exists():
                print(f"Error: The model path {est_model_path} does not exist.")
                continue

            gt_sparse_model = read_model(gt_model_path)
            est_sparse_model = read_model(est_model_path)

            # Camera poses evaluation
            rel_results, abs_results = evaluator.run_camera_evaluator(gt_sparse_model, est_sparse_model)
            if rel_results is not None:
                export_rel_results(rel_results, est_model_path)
            if abs_results is not None:
                export_abs_results(abs_results, est_model_path)

            # Novel view synthesis evaluation
            # ssim, psnr, lpips = evaluator.run_novel_view_synthesis_evaluator(image_path, est_model_path)

            # Triangulation evaluation
            dense_path = results / "ETH3D" / scene / COLMAP_DENSE_FORMAT
            if dense_path.exists():
                print("Using dense reconstruction")
                # We use the sparse model produced during the dense reconstruction (colmap stereo_fusion)
                est_sparse_model = read_model(str(dense_path / "sparse"))
                est_pcd = o3d.io.read_point_cloud(str(dense_path / "fused.ply"))
            elif est_sparse_model is not None:
                # Flowmap doesn't create a point3D.bin file but a points3D.ply file
                if (est_model_path / "points3D.ply").exists():
                    est_pcd = o3d.io.read_point_cloud(str(est_model_path / "points3D.ply"))
                else:
                    print(
                        "No dense points cloud provided. Falling back on sparse reconstruction. This might give poor results.")
                    est_pcd = extract_pcd_from_model(est_sparse_model)
            else:
                print("No sparse reconstruction found.")
                continue

            # Read the XML file
            mlp_file = ETH3D_DATA_PATH / scene / "dslr_scan_eval/scan_alignment.mlp"
            with open(mlp_file, "r") as file:
                mlp_lines = file.readlines()

            mlp_lines = [line.strip() for line in mlp_lines]
            matrices = parse_mlp_matrices(mlp_lines)
            init_alignment = matrices["scan1.ply"]

            # ETH3D applies the alignment to the ground truth scan
            gt_scan_pcd = o3d.io.read_point_cloud(str(ETH3D_DATA_PATH / scene / "dslr_scan_eval/scan1.ply"))
            gt_scan_pcd.transform(init_alignment)
            distance1, distance2 = evaluator.run_triangulation_evaluator(
                colmap_sparse_ref=gt_sparse_model,
                est_sparse_model=est_sparse_model,
                gt_scan_pcd=gt_scan_pcd,
                est_dense_pcd=est_pcd,
                threshold=triangulation_threshold,
                verbose=True,
            )

            # write_color_distances(est_model_path / f"{scene}_precision.ply", est_pcd, distance1, triangulation_threshold)
            # write_color_distances(est_model_path / f"{scene}_recall.ply", gt_scan_pcd, distance2, triangulation_threshold)

            metrics = get_f1_score_histo2(distance1, distance2, triangulation_threshold, stretch=5)
            precision = metrics["precision"]
            recall = metrics["recall"]
            fscore = metrics["fscore"]
            edges_source = metrics["edges_source"]
            cum_source = metrics["cum_source"]
            edges_target = metrics["edges_target"]
            cum_target = metrics["cum_target"]
            plot_fscore(
                scene=str(scene),
                fscore=fscore,
                dist_threshold=triangulation_threshold,
                edges_source=edges_source,
                cum_source=cum_source,
                edges_target=edges_target,
                cum_target=cum_target,
                plot_stretch=5,
                mvs_outpath=str(est_model_path),
            )

        for scene in MIP_NERF_360_SCENES:
            print(scene)
            est_model_path = results / "MipNerf360" / scene / COLMAP_FORMAT
            if not est_model_path.exists():
                print(f"Error: The model path {est_model_path} does not exist.")
                continue

            gt_sparse_model = read_model(MIP_NERF_360_DATA_PATH / scene / "sparse/0")
            est_sparse_model = read_model(est_model_path)
            image_path = MIP_NERF_360_DATA_PATH / scene / "images"

            # Camera poses evaluation
            rel_results, abs_results = evaluator.run_camera_evaluator(
                gt_sparse_model=gt_sparse_model,
                est_sparse_model=est_sparse_model
            )
            if rel_results is not None:
                export_rel_results(rel_results, est_model_path)
            if abs_results is not None:
                export_abs_results(abs_results, est_model_path)

            # Novel view synthesis evaluation
            ssim, psnr, lpips = evaluator.run_novel_view_synthesis_evaluator(image_path, est_sparse_model)

        for scene in TANKS_AND_TEMPLES_SCENES:
            print(scene)
            scene_dir = TANKS_AND_TEMPLES_DATA_PATH / scene
            est_model_path = results / "TanksAndTemples" / scene / COLMAP_FORMAT
            if not est_model_path.exists() or not scene_dir.exists():
                continue

            gt_pcd = o3d.io.read_point_cloud(str(scene_dir / f"{scene_dir.name}.ply"))
            colmap_sparse_ref = read_model(scene_dir / "sparse/0")
            est_sparse_model = read_model(str(est_model_path))

            est_pcd = None
            dense_path = results / "TanksAndTemples" / scene / COLMAP_DENSE_FORMAT
            if dense_path.exists():
                print("Using dense reconstruction")
                # We use the sparse model produced during the dense reconstruction (colmap stereo_fusion)
                est_sparse_model = read_model(str(dense_path / "sparse"))
                est_pcd = o3d.io.read_point_cloud(str(dense_path / "fused.ply"))
            elif est_sparse_model is not None:
                # Flowmap doesn't create a point3D.bin file but a points3D.ply file
                if (est_model_path / "points3D.ply").exists():
                    est_pcd = o3d.io.read_point_cloud(str(est_model_path / "points3D.ply"))
                else:
                    print("No dense points cloud provided. Falling back on sparse reconstruction. This might give poor results.")
                    est_pcd = extract_pcd_from_model(est_sparse_model)
            else:
                print("No sparse reconstruction found.")
                continue

            cropfile = scene_dir / f"{scene_dir.name}.json"
            # Tanks and Temples applies the alignment to the estimated reconstruction
            init_alignment = np.loadtxt(scene_dir / f"{scene_dir.name}_trans.txt")

            distance1, distance2 = run_triangulation_evaluation(
                colmap_sparse_ref=colmap_sparse_ref,
                est_sparse_reconstruction=est_sparse_model,
                gt_pcd=gt_pcd,
                est_pcd=est_pcd,
                cropfile=cropfile,
                init_alignment=init_alignment,
            )

            # write_color_distances(est_model_path / f"{scene}_precision.ply", est_pcd, distance1, triangulation_threshold)
            # write_color_distances(est_model_path / f"{scene}_recall.ply", gt_scan_pcd, distance2, triangulation_threshold)

            metrics = get_f1_score_histo2(distance1, distance2, triangulation_threshold, stretch=5)
            precision = metrics["precision"]
            recall = metrics["recall"]
            fscore = metrics["fscore"]
            edges_source = metrics["edges_source"]
            cum_source = metrics["cum_source"]
            edges_target = metrics["edges_target"]
            cum_target = metrics["cum_target"]
            plot_fscore(
                scene=str(scene),
                fscore=fscore,
                dist_threshold=triangulation_threshold,
                edges_source=edges_source,
                cum_source=cum_source,
                edges_target=edges_target,
                cum_target=cum_target,
                plot_stretch=5,
                mvs_outpath=str(est_model_path),
            )
