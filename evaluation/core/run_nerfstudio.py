import subprocess
import os
import argparse
import logging
from tqdm import tqdm

def run_nerfstudio(dataset_path, results_path):
    # First copy the images to the results directory if they are not already there
    _logger.info("Checking if images are in the results directory...")
    if not os.path.exists(results_path + "/images"):
        _logger.info("Copying images to the results directory...")
        cp_cmd = f"cp -r {dataset_path}/images {results_path}"
        subprocess.run(cp_cmd, shell=True)
    else:
        _logger.info("Images are already in the results directory.")

    # Downscaling images by a factor of 4
    if not os.path.exists(results_path + "/images_4"):
        for image_name in tqdm(os.listdir(results_path + "/images"), desc="Downscaling images by a factor of 4"):
            image_path = os.path.join(results_path, "images", image_name)
            image_out = os.path.join(results_path, "images_4", image_name)
            if not os.path.exists(os.path.dirname(image_out)):
                os.makedirs(os.path.dirname(image_out), exist_ok=True)
            ffmpeg_cmd = (
                f'ffmpeg -y -noautorotate -i "{image_path}" '
                f'-q:v 2 -vf scale=iw/4:-1:flags=neighbor '
                f'-frames:v 1 -update 1 -f image2 "{image_out}" -loglevel quiet'
            )
            subprocess.run(ffmpeg_cmd, shell=True)
    else:
        _logger.info("Downscaled images are already in the results directory.")

    # Find how many CUDA GPUs are available
    print("Checking for available CUDA GPUs...")
    num_gpus = 2
    try:
        cmd = "nvidia-smi --query-gpu=count --format=csv,noheader"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        num_gpus = int(result.stdout.decode('utf-8').strip())
        _logger.info(f"Found {num_gpus} CUDA GPUs.")
    except Exception:
        _logger.error("CUDA not found. NerfStudio requires CUDA to run.")
        exit(1)

    # Set the number of GPUs to use for training
    CUDA_VISIBLE_DEVICES = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}"

    # Train the NeRF model TODO: Investigate using Zip-NeRF for better quality
    _logger.info("Training the NeRF model...")
    train_cmd = (f"{CUDA_VISIBLE_DEVICES} ns-train nerfacto "
           f"--machine.num-devices {num_gpus} --pipeline.datamanager.images-on-gpu True "
           f"--data {results_path} --output-dir {results_path}/nerfstudio colmap")
    subprocess.run(train_cmd, shell=True)

    # Evaluate the NeRF model
    _logger.info("Evaluating the NeRF model...")
    eval_cmd = (f"{CUDA_VISIBLE_DEVICES} ns-eval --load-config {results_path}/nerfacto/config.yml "
           f"--output-dir {results_path}/nerfacto/eval.json")
    subprocess.run(eval_cmd, shell=True)


if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        default="../../datasets/ETH3D/courtyard",
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=False,
        default="../../results/glomap/ETH3D/courtyard",
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