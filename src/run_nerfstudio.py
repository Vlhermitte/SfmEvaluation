import subprocess
import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from data.read_write_model import read_model, write_model


def compute_downscale_factor(dataset_path: Path, max_resolution: int=1600) -> int:
    # Find the max dimension of the images
    downscaling_factor = 1
    for image_name in os.listdir(os.path.join(dataset_path, "images")):
        image_path = os.path.join(dataset_path, "images", image_name)
        img = Image.open(image_path)
        height, width = img.size
        max_dim = max(height, width)
        if max_dim // downscaling_factor > max_resolution:
            downscaling_factor += 1
    return downscaling_factor

def downscale_images(dataset_path: Path, factor: int) -> None:
    if not os.path.exists(os.path.join(dataset_path, f"images_{factor}")):
        for image_name in tqdm(os.listdir(os.path.join(dataset_path, "images")), desc=f"Downscaling images by a factor of {factor}"):
            image_path = os.path.join(dataset_path, "images", image_name)
            image_out = os.path.join(dataset_path, f"images_{factor}", image_name)
            if not os.path.exists(os.path.dirname(image_out)):
                os.makedirs(os.path.dirname(image_out), exist_ok=True)
            ffmpeg_cmd = (
                f'ffmpeg -y -noautorotate -i "{image_path}" '
                f'-q:v 2 -vf scale=iw/{factor}:-1:flags=neighbor '
                f'-frames:v 1 -update 1 -f image2 "{image_out}" -loglevel quiet'
            )
            subprocess.run(ffmpeg_cmd, shell=True)
    else:
        _logger.info(f"Downscaled images_{factor} are already in {dataset_path}")

def run_nerfstudio(dataset_path: Path, results_path: Path, method: str ='nerfacto', viz: bool=False) -> None:
    _logger.info(f"Compute downscaling factor for {dataset_path} ...")
    # max resolution of 1600px, which is the default of nerfstudio
    downscale_factor = compute_downscale_factor(dataset_path, max_resolution=1600)
    _logger.info(f"Downscaling factor found : {downscale_factor}")
    downscale_images(dataset_path, downscale_factor)

    # Find how many CUDA GPUs are available
    _logger.info("Checking for available CUDA GPUs...")
    num_gpus = 0
    try:
        cmd = "nvidia-smi --query-gpu=count --format=csv,noheader"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        num_gpus = int(result.stdout.decode('utf-8').strip()[0])
        _logger.info(f"Found {num_gpus} CUDA GPUs.")
    except Exception:
        _logger.error("CUDA not found. NerfStudio requires CUDA to run.")
        exit(1)

    # Splatfacto does not support multi-gpus
    if method == 'splatfacto' and num_gpus > 1:
        _logger.warning("Splatfacto does not support multi-gpus. Using 1 GPU.")
        num_gpus = 1

    # Set the number of GPUs to use for training
    CUDA_VISIBLE_DEVICES = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}"

    # Detect the number of images in the dataset (if less than 1000, images will be loaded on GPU)
    num_images = len(os.listdir(os.path.join(dataset_path, "images")))

    # Train the NeRF model
    _logger.info(f"Training the model using : {method}")
    # The nerfstudio Args order is important: 1. Nerfstudio Args 2. DataParser Args
    train_cmd = (
        # Nerfstudio Args
        f"{CUDA_VISIBLE_DEVICES} ns-train {method} "
        f"--machine.num-devices {num_gpus} --pipeline.datamanager.images-on-gpu {'True' if num_images <= 1000 else 'False'} "
        f"--pipeline.model.camera-optimizer.mode off " # We do not want to optimize the camera parameters
        f"{'--viewer.make-share-url True' if viz else ''} "
        f"--viewer.quit-on-train-completion True "
        f"--experiment-name nerfstudio " # To store the results in a directory nerfstudio instead of the default name 'unnamed'
        f"--output-dir {results_path} "
        # DataParser Args
        f"colmap --images-path {dataset_path}/images --colmap-path {results_path} "
    )
    subprocess.run(train_cmd, shell=True)

    # # Evaluate the NeRF model
    _logger.info("Evaluating the NeRF model...")
    eval_cmd = (
        f"{CUDA_VISIBLE_DEVICES} ns-eval "
        f"--load-config {results_path}/nerfstudio/{method}/config.yml "
        f"--output-path {results_path}/nerfstudio/{method}/eval.json "
        f"--render-output-path {results_path}/nerfstudio/{method}/renders"
    )
    subprocess.run(eval_cmd, shell=True)

def sanity_check_colmap(path: Path) -> None:
    # read the colmap model
    cameras, images, points3D = read_model(path)

    # check that the model is not empty
    if len(cameras) == 0 or len(images) == 0 or len(points3D) == 0:
        _logger.error(f"Error: The colmap model at {path} is empty. Please check the results path and try again.")
        exit(1)

    # check that in images, each image has a valid path
    path_changed = False
    for image_id, image in images.items():
        img_path = Path(image.name)
        if len(img_path.parts) > 1:
            new_image = image._replace(name=img_path.parts[-1])
            images[image_id] = new_image
            path_changed = True
    if path_changed:
        _logger.info("Fixed image paths in the colmap model.")
        write_model(cameras=cameras, images=images, points3D=points3D, path=path, ext=".bin")


if __name__ == '__main__':
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        default="../data/datasets/MipNerf360/garden",
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=False,
        default="../data/results/glomap/MipNerf360/garden/colmap/sparse/0",
        help="path to the results directory containing colmap files."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="nerfacto",
        help="nerfacto, splatfacto"
    )
    parser.add_argument(
        "--viz",
        type=bool,
        required=False,
        default=False,
        help="Whether to visualize the results"
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    results_path = args.results_path
    method = args.method
    viz = args.viz

    # Check that the images are present in the dataset path
    if not os.path.exists(os.path.join(dataset_path, "images")):
        _logger.error(f"Error: The dataset at {dataset_path} does not contain images. Please check the dataset path and try again.")
        exit(1)

    # Check that colmap model exists (i.e .bin/.txt files)
    if not os.path.exists(os.path.join(results_path, "images.bin")):
        _logger.error(f"Error: The colmap model at {results_path} does not exist. Please check the results path and try again.")
        exit(1)

    # Sanity check on colmap model
    sanity_check_colmap(results_path)

    run_nerfstudio(dataset_path, results_path, method, viz)