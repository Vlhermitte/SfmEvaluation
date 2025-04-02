import sys
import argparse
import logging
import numpy as np
import open3d as o3d
from pathlib import Path

from evaluation.triangulation_evaluation import run_triangulation_evaluation
from utils.common import read_model

def parse_mlp_matrices(lines):
    matrices = dict()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('<MLMesh'):
            # Extract filename
            label_part = line.split('filename="')
            if len(label_part) < 2:
                i += 1
                continue
            filename = label_part[1].split('"')[0]

            # Look ahead for the corresponding matrix
            while i < len(lines) and '<MLMatrix44>' not in lines[i]:
                i += 1
            if i >= len(lines) - 5:
                break  # Not enough lines left for a matrix

            # Extract the 4 lines of the matrix
            matrix_lines = lines[i+1:i+5]
            matrix = np.array([[float(n) for n in l.strip().split()] for l in matrix_lines])
            matrices[filename] = matrix
            i += 5  # Skip past the matrix
        else:
            i += 1

    return matrices

if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler(sys.stdout))

    from config import (
        ETH3D_DATA_PATH, ETH3D_SCENES,
        MIP_NERF_360_DATA_PATH, MIP_NERF_360_SCENES,
        TANKS_AND_TEMPLES_DATA_PATH, TANKS_AND_TEMPLES_SCENES,
        GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH, COLMAP_FORMAT
    )

    # Run ETH3D evaluation
    for results in [GLOMAP_RESULTS_PATH, VGGSFM_RESULTS_PATH, FLOWMAP_RESULTS_PATH, ACEZERO_RESULTS_PATH]:
        print("Evaluating method: ", results)
        for scene in ETH3D_SCENES:
            scene_dir = ETH3D_DATA_PATH / scene
            est_path = results / "ETH3D" / scene
            if not est_path.exists() or not scene_dir.exists():
                continue

            print(scene)
            gt_pcd = o3d.io.read_point_cloud(str(scene_dir / "dslr_scan_eval/scan1.ply"))
            colmap_sparse_ref = read_model(scene_dir / "dslr_calibration_jpg")

            # We prefer the sparse reconstruction from the dense folder because it is aligned with the dense pcd.
            # During the colmap stereo_fusion step, the dense pcd might move from the original sparse coordinate system.
            # This is why stereo_fusion creates a new sparse reconstruction in the dense folder which is aligned with the dense pcd.
            est_sparse_path = est_path / "colmap/dense/sparse"
            est_dense_pcd = None
            if est_sparse_path.exists() and (est_path / "colmap/dense/fused.ply").exists():
                est_sparse_reconstruction = read_model(est_sparse_path)
                est_dense_pcd = o3d.io.read_point_cloud(str(est_path / "colmap/dense/fused.ply"))
            elif (est_path / "colmap/sparse/0").exists():
                print("Using sparse reconstruction from colmap/sparse/0")
                est_sparse_reconstruction = read_model(est_path / "colmap/sparse/0")
                if (est_path / "colmap/sparse/0/points3D.ply").exists():
                    est_dense_pcd = o3d.io.read_point_cloud(str(est_path / "colmap/sparse/0/points3D.ply"))
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
            gt_pcd.transform(init_alignment)

            run_triangulation_evaluation(
                scene_dir=scene_dir,
                est_path=est_path,
                gt_pcd=gt_pcd,
                colmap_sparse_ref=colmap_sparse_ref,
                est_sparse_reconstruction=est_sparse_reconstruction,
                est_dense_pcd=est_dense_pcd,
                cropfile=None,
                init_alignment=None,
                threshold=0.1,
            )

        # Run Tanks and Temples evaluation
        for scene in TANKS_AND_TEMPLES_SCENES:
            scene_dir = TANKS_AND_TEMPLES_DATA_PATH / scene
            est_path = results / "TanksAndTemples" / scene
            if not est_path.exists() or not scene_dir.exists():
                continue

            print(scene)
            gt_pcd = o3d.io.read_point_cloud(str(scene_dir / f"{scene_dir.name}.ply"))
            colmap_sparse_ref = read_model(scene_dir / "sparse/0")

            est_sparse_path = est_path / "colmap/dense/sparse"
            est_dense_pcd = None
            if est_sparse_path.exists() and (est_path / "colmap/dense/fused.ply").exists():
                est_sparse_reconstruction = read_model(est_sparse_path)
                est_dense_pcd = o3d.io.read_point_cloud(str(est_path / "colmap/dense/fused.ply"))
            elif (est_path / "colmap/sparse/0").exists():
                print("Using sparse reconstruction from colmap/sparse/0")
                est_sparse_reconstruction = read_model(est_path / "colmap/sparse/0")
                # Flowmap doesn't create a point3D.bin file but a points3D.ply file
                if (est_path / "colmap/sparse/0/points3D.ply").exists():
                    est_dense_pcd = o3d.io.read_point_cloud(str(est_path / "colmap/sparse/0/points3D.ply"))
            else:
                print("No sparse reconstruction found.")
                continue

            cropfile = scene_dir / f"{scene_dir.name}.json"
            # Tanks and Temples applies the alignment to the estimated reconstruction
            init_alignment = np.loadtxt(scene_dir / f"{scene_dir.name}_trans.txt")

            run_triangulation_evaluation(
                scene_dir=scene_dir,
                est_path=est_path,
                colmap_sparse_ref=colmap_sparse_ref,
                est_sparse_reconstruction=est_sparse_reconstruction,
                est_dense_pcd=est_dense_pcd,
                gt_pcd=gt_pcd,
                cropfile=cropfile,
                init_alignment=init_alignment,
                threshold=0.1,
            )


