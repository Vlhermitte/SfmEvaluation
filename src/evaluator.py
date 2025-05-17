import copy
import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
import open3d as o3d
import numpy as np
import pycolmap
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.common import detect_colmap_format, read_model
from run_camera_poses import (
    run_rel_err_evaluation, run_abs_err_evaluation, export_rel_results, export_abs_results
)
from run_nerfstudio import sanity_check_colmap, run_nerfstudio

from config import (
        ETH3D_DATA_PATH, ETH3D_SCENES,
        MIP_NERF_360_DATA_PATH, MIP_NERF_360_SCENES,
        TANKS_AND_TEMPLES_DATA_PATH, TANKS_AND_TEMPLES_SCENES,
        LAMAR_HGE_DATA_PATH, LAMAR_HGE_SCENES,
        LAMAR_CAB_DATA_PATH, LAMAR_CAB_SCENES,
        LAMAR_LIN_DATA_PATH, LAMAR_LIN_SCENES,
        COLMAP_RESULTS_PATH, GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH,
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


def evaluate_scene(
        scene: str,
        gt_model_path: Path,
        est_model_path: Path,
        evaluator
):
    """
    Evaluate camera poses for a single scene.
    """
    print(f"  Scene: {scene}")
    if not est_model_path.exists():
        print(f"  Warning: Estimated model path {est_model_path} does not exist.")
        return

    # find numeric subdirectories
    directories = sorted(
        (d for d in os.listdir(est_model_path) if d.isdigit()),
        key=lambda x: int(x)
    )
    gt_sparse_model = read_model(gt_model_path)

    num_img = 0
    est_sparse_model = None
    for directory in directories:
        model_dir = est_model_path / directory
        if not (model_dir / 'images.bin').exists() or (model_dir / 'images.txt').exists():
            print(f"  Warning: The model at {model_dir} does not exist.")
            continue
        curr_est_sparse_model = read_model(model_dir)
        curr_num_img = curr_est_sparse_model.num_images()
        # We keep the largest model
        if curr_num_img > num_img:
            est_sparse_model = copy.deepcopy(curr_est_sparse_model)
            num_img = curr_num_img

    if est_sparse_model is None:
        print("  Error: no valid model found")
        return
    rel_results, abs_results = evaluator.run_camera_evaluator(
        gt_sparse_model=gt_sparse_model,
        est_sparse_model=est_sparse_model
    )
    if rel_results is not None:
        export_rel_results(rel_results, model_dir.parent)
    if abs_results is not None:
        export_abs_results(abs_results, model_dir.parent)


def main():
    evaluator = Evaluator()

    results_paths = [
        COLMAP_RESULTS_PATH,
        GLOMAP_RESULTS_PATH,
        VGGSFM_RESULTS_PATH,
        FLOWMAP_RESULTS_PATH,
        ACEZERO_RESULTS_PATH
    ]

    # Configuration for each dataset
    datasets = [
        {
            'name': 'ETH3D',
            'scenes': ETH3D_SCENES,
            'get_gt': lambda scene: ETH3D_DATA_PATH / scene / 'dslr_calibration_jpg',
            'get_est': lambda results, scene: results / 'ETH3D' / scene / 'colmap' / 'sparse'
        },
        {
            'name': 'LaMAR HGE',
            'scenes': LAMAR_HGE_SCENES,
            'get_gt': lambda scene: LAMAR_HGE_DATA_PATH / scene / 'sparse/0',
            'get_est': lambda results, scene: results / 'LaMAR/HGE/sessions/map/raw_data/' / scene / 'colmap' / 'sparse'
        },
        {
            'name': 'LaMAR CAB',
            'scenes': LAMAR_CAB_SCENES,
            'get_gt': lambda scene: LAMAR_CAB_DATA_PATH / scene / 'sparse/0',
            'get_est': lambda results, scene: results / 'LaMAR/CAB/sessions/map/raw_data/' / scene / 'colmap' / 'sparse'
        },
        {
            'name': 'LaMAR LIN',
            'scenes': LAMAR_LIN_SCENES,
            'get_gt': lambda scene: LAMAR_LIN_DATA_PATH / scene / 'sparse/0',
            'get_est': lambda results, scene: results / 'LaMAR/LIN/sessions/map/raw_data/' / scene / 'colmap' / 'sparse'
        }
    ]

    for results in results_paths:
        print(f"Evaluating {results}")
        for ds in datasets:
            print(ds['name'])
            for scene in ds['scenes']:
                gt_model = ds['get_gt'](scene)
                est_model = ds['get_est'](results, scene)
                evaluate_scene(
                    scene=scene,
                    gt_model_path=gt_model,
                    est_model_path=est_model,
                    evaluator=evaluator,
                )


if __name__ == '__main__':
    main()
