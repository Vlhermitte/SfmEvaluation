# File for 3D triangulation evaluation using ETH3D Multi-view evaluation metrics.

import subprocess
from pathlib import Path
import time

from src.config import MULTIVIEW_EVALUATION_TOOL_PATH


def evaluate_multiview(ply_path: Path, mlp_path: Path):
    assert MULTIVIEW_EVALUATION_TOOL_PATH.exists(), f"{MULTIVIEW_EVALUATION_TOOL_PATH} executable not found. Install it at https://github.com/ETH3D/multi-view-evaluation"
    assert ply_path.exists(), f"{ply_path} not found"
    assert mlp_path.exists(), f"{mlp_path} not found"

    cmd = (
        f"{MULTIVIEW_EVALUATION_TOOL_PATH} "
        f"--reconstruction_ply_path {ply_path} "
        f"--ground_truth_mlp_path {mlp_path} "
        f"--tolerances 0.01,0.02,0.05,0.1,0.2,0.5 "
    )
    results_file = ply_path.parent / "multiview_results.txt"
    start = time.time()
    with open(results_file, "w") as results_file:
        subprocess.run(cmd, shell=True, stdout=results_file)
    end = time.time()
    print(f"Multi-view evaluation took {end-start} seconds")



