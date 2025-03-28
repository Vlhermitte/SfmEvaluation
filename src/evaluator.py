import os
import json
import logging
from pathlib import Path
from typing import Tuple

from utils.common import detect_colmap_format, read_model
from run_camera_poses import (
    run_rel_err_evaluation, run_abs_err_evaluation, export_rel_results, export_abs_results
)
from run_nerfstudio import sanity_check_colmap, run_nerfstudio
from src.evaluation.triangulation_evaluation import evaluate_multiview


class Evaluator:
    """
    Easy to use class to run any of the following evaluations:
    - Camera poses evaluation
    - Novel view synthesis evaluation
    - Triangulation evaluation
    """
    def __init__(self, gt_model_path: Path, est_model_path: Path, image_path: Path):
        self.gt_model_path = Path(gt_model_path)
        self.est_model_path = Path(est_model_path)
        self.image_path = Path(image_path)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(logging.StreamHandler())

    def run_camera_evaluator(self) -> Tuple[list, list]:
        """
        Run the camera poses evaluation.
        :return: rel_results, abs_results.
        """
        gt_sparse_model = read_model(self.gt_model_path)
        est_sparse_model = read_model(self.est_model_path)
        rel_results, abs_results = None, None

        # Run the evaluation
        if (gt_sparse_model and est_sparse_model) is not None:
            rel_results = run_rel_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)
            abs_results = run_abs_err_evaluation(gt_model=gt_sparse_model, est_model=est_sparse_model)

        return rel_results, abs_results

    def run_novel_view_synthesis_evaluator(self, method: str="nerfacto") -> Tuple[float, float, float]:
        """
        Run the novel view synthesis evaluation using the nerfstudio tool.
        :param method: str, the method to use for the evaluation. Default is "nerfacto".
        :return: ssim, psnr, lpips.
        """
        sanity_check_colmap(self.est_model_path)

        # Run the evaluation
        run_nerfstudio(dataset_path=self.image_path.parent, results_path=self.est_model_path, method=method)

        # The results are saved in self.est_model_path/nerfstudio/<method>/run/eval.json
        if os.path.exists(f"{self.est_model_path}/nerfstudio/{method}/run/eval.json"):
            with open(f"{self.est_model_path}/nerfstudio/{method}/run/eval.json", "r") as f:
                results = json.load(f)["results"]
                ssim = results["ssim"]
                psnr = results["psnr"]
                lpips = results["lpips"]
                return ssim, psnr, lpips
        else:
            self._logger.error(f"Error: The evaluation results file does not exist. Maybe the evaluation failed.")
            return None

    def run_triangulation_evaluator(self, ply_path: Path=None, mlp_path: Path=None) -> Tuple[list, list, list, list]:
        """
        Run the triangulation evaluation.
        :param ply_path: Path, the path to the .ply file.
        :param mlp_path: Path, the path to the .mlp file.
        :return: tolerances, completenesses, accuracies, f1_scores.
        """
        # If the paths are not provided, try to find the .ply and .mlp files
        if ply_path is None:
            # Find a .ply file. If none then error
            ply_files = list(self.est_model_path.glob("*.ply"))
            if len(ply_files) == 0:
                self._logger.error(f"Error: No .ply file found in {self.est_model_path}")
                raise FileNotFoundError(f"No .ply file found in {self.est_model_path}")
            else:
                ply_path = ply_files[0]
        else:
            ply_path = Path(ply_path)
        if mlp_path is None:
            mlp_path = Path(self.gt_model_path / "dslr_scan_eval" / "scan.mlp")
        else:
            mlp_path = Path(mlp_path)

        if not ply_path.exists():
            raise FileNotFoundError(f"Error: The .ply file does not exist in {ply_path}")
        if not mlp_path.exists():
            raise FileNotFoundError(f"Error: The .mlp file does not exist in {mlp_path}")

        # Run the evaluation
        evaluate_multiview(ply_path, mlp_path)
        if os.path.exists(ply_path.parent / "multiview_results.txt"):
            self._logger.info(f"Results saved in {ply_path.parent / 'multiview_results.txt'}")
            # Read file
            with open(ply_path.parent / "multiview_results.txt", "r") as f:
                # Keep only line 6, 7, 8, 9
                lines = f.readlines()
                results = lines[5:]
                tolerances = list(results[0].split(":")[1].strip().split())
                completenesses = list(results[1].split(":")[1].strip().split())
                accuracies = list(results[2].split(":")[1].strip().split())
                f1_scores = list(results[3].split(":")[1].strip().split())
                return tolerances, completenesses, accuracies, f1_scores
        else:
            self._logger.error(f"Error: The evaluation results file does not exist. Maybe the evaluation failed.")
            return None


if __name__ == '__main__':
    from config import (
        ETH3D_DATA_PATH, ETH3D_SCENES,
        MIP_NERF_360_DATA_PATH, MIP_NERF_360_SCENES,
        GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH, COLMAP_FORMAT
    )

    for results in [GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH]:
        print("Evaluating", results)
        for scene in ETH3D_SCENES:
            print(scene)
            est_model_path = results / "ETH3D" / scene / COLMAP_FORMAT
            evaluator = Evaluator(
                gt_model_path=ETH3D_DATA_PATH /scene / "dslr_calibration_jpg",
                est_model_path=est_model_path,
                image_path=ETH3D_DATA_PATH / scene / "images"
            )

            # Camera poses evaluation
            rel_results, abs_results = evaluator.run_camera_evaluator()
            if rel_results is not None:
                export_rel_results(rel_results, est_model_path)
            if abs_results is not None:
                export_abs_results(abs_results, est_model_path)

            # Novel view synthesis evaluation
            # ssim, psnr, lpips = evaluator.run_novel_view_synthesis_evaluator()

            # Triangulation evaluation
            # tolerances, completenesses, accuracies, f1_scores = evaluator.run_triangulation_evaluator()

        for scene in MIP_NERF_360_SCENES:
            print(scene)
            est_model_path = results / "MipNerf360" / scene / COLMAP_FORMAT
            if not est_model_path.exists():
                print(f"Error: The model path {est_model_path} does not exist.")
                continue
            evaluator = Evaluator(
                gt_model_path=MIP_NERF_360_DATA_PATH / scene / "sparse/0",
                est_model_path=est_model_path,
                image_path=MIP_NERF_360_DATA_PATH / scene / "images"
            )

            # Camera poses evaluation
            rel_results, abs_results = evaluator.run_camera_evaluator()
            if rel_results is not None:
                export_rel_results(rel_results, est_model_path)
            if abs_results is not None:
                export_abs_results(abs_results, est_model_path)

            # Novel view synthesis evaluation
            # ssim, psnr, lpips = evaluator.run_novel_view_synthesis_evaluator()