import sys
import argparse
import logging
from pathlib import Path

from evaluation.triangulation_evaluation import evaluate_multiview

if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply_path",
        type=str,
        required=False,
        default="../data/datasets/ETH3D/courtyard/model.ply",
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--mlp_path",
        type=str,
        required=False,
        default="../data/datasets/ETH3D/courtyard/dslr_scan_eval/scan_alignment.mlp",
        help="path to the dataset containing images"
    )

    args = parser.parse_args()
    ply_path = Path(args.ply_path)
    mlp_path = Path(args.mlp_path)
    _logger.info(f"Running evaluation on {ply_path} and {mlp_path}")
    evaluate_multiview(ply_path, mlp_path)
