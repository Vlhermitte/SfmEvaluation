import subprocess
import os
import argparse
import logging

def run_nerfstudio(dataset_path, results_path):
    # First copy the images to the results directory if they are not already there
    print("Checking if images are in the results directory...")
    if not os.path.exists(results_path + "/images"):
        print("Copying images to the results directory...")
        cmd = f"cp -r {dataset_path} {results_path}"
        subprocess.run(cmd, shell=True)
    else:
        print("Images are already in the results directory.")

    # Train the NeRF model TODO: Investigate using Zip-NeRF for better quality
    print("Training the NeRF model...")
    cmd = f"ns-train nerfacto --data {results_path} --output-dir {results_path}/nerfacto colmap"
    subprocess.run(cmd, shell=True)

    # Evaluate the NeRF model
    print("Evaluating the NeRF model...")
    cmd = f"ns-eval --load-config {results_path}/nerfacto/config.yml --output-dir {results_path}/nerfacto/eval.json"
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        default="../datasets/ETH3D/courtyard",
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=False,
        default="../results/glomap/courtyard",
        help="path to the results directory containing the images and colmap model under 'colmap/sparse/0'"
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    results_path = args.results_path

    # check that results_path/colmap/sparse/0 exists
    if not os.path.exists(results_path + "/colmap/sparse/0"):
        _logger.error(f"Error: The path {results_path}/colmap/sparse/0 does not exist. Please check the results path and try again.")
        exit(1)

    run_nerfstudio(dataset_path, results_path)