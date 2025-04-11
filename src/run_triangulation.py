import os
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

    parser = argparse.ArgumentParser(description='Triangulation evaluation')
    parser.add_argument('--ref-colmap-path', type=str, required=True,
                        help='Path to the reference reconstruction from COLMAP')
    parser.add_argument('--est-colmap-path', type=str, required=True,
                        help='Path to the estimated reconstruction in COLMAP format')
    parser.add_argument('--gt-pcd-path', type=str, required=True,
                        help='Path to the ground truth point cloud')
    parser.add_argument('--est-pcd-path', type=str, required=False,
                        help='Path to the estimated point cloud')
    parser.add_argument('--cropfile', type=str,
                        help='Path to the crop file (Only for Tanks and Temples dataset)')
    parser.add_argument('--output', type=str, default="results",
                        help='Path to the output directory')
    parser.add_argument('--mlp', type=str,
                        help='Path to the MLP file (Only for ETH3D dataset)')
    parser.add_argument('--init-aligment', type=str,
                        help='Path to the json file with the initial alignment (Only for Tanks and Temples dataset)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for the F1 score')
    parser.add_argument('--stretch', type=float, default=5,
                        help='Stretch for the F1 score')
    args = parser.parse_args()

    assert args.mlp and args.init_aligment, "Either --mlp or --init-aligment should be provided, not both."

    if not Path(args.output).exists():
        os.makedirs(args.output)

    colmap_sparse_ref = Path(args.ref_colmap_path)
    est_sparse_model = Path(args.est_colmap_path)
    gt_pcd = o3d.io.read_point_cloud(args.gt_pcd_path)
    est_pcd = o3d.io.read_point_cloud(args.est_colmap_path) if args.est_pcd_path else None
    cropfile = args.cropfile if args.cropfile else None
    init_alignment = np.loadtxt(args.init_aligment) if args.init_aligment else None


    distance1, distance2 = run_triangulation_evaluation(
        colmap_sparse_ref=colmap_sparse_ref,
        est_sparse_reconstruction=est_sparse_model,
        gt_pcd=gt_pcd,
        est_pcd=est_pcd,
        cropfile=cropfile,
        init_alignment=init_alignment,
    )
    write_color_distances(est_model_path / f"{scene}_precision.ply", est_pcd, distance1, triangulation_threshold)
    write_color_distances(est_model_path / f"{scene}_recall.ply", gt_scan_pcd, distance2, triangulation_threshold)

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
