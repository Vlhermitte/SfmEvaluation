import shutil
import subprocess
import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import math
import pycolmap

# check pycolmap version
if pycolmap.__version__ < "3.10.0":
    print(f"WARNING: pycolmap.__version__: {pycolmap.__version__}")

from utils.common import read_model

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(sys.stdout))

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


def downscale_single_image(image_path: Path, output_path: Path, factor: int) -> None:
    """Downscale a single image using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    w, h = img.size
    w_scaled, h_scaled = math.floor(w / factor), math.floor(h / factor)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-noautorotate",
        "-i", str(image_path),
        "-q:v", "2",
        "-vf", f"scale={w_scaled}:{h_scaled}",
        "-frames:v", "1", "-update", "1",
        "-f", "image2", str(output_path),
        "-loglevel", "quiet"
    ]
    subprocess.run(ffmpeg_cmd, check=True)


def downscale_images(dataset_path: Path, factor: int, viz: bool=True) -> None:
    """Downscale all images in the dataset by the given factor."""
    assert factor > 0, "Downscaling factor should be greater than 0"

    images_dir = Path(dataset_path) / "images"
    downscaled_dir = Path(dataset_path) / f"images_{factor}"

    assert images_dir.exists(), f"Images not found in {dataset_path}"
    assert shutil.which("ffmpeg"), "ffmpeg not found in PATH. Please install ffmpeg to downscale images."

    # If the downscaled directory doesn't exist, create it and process all images.
    if not downscaled_dir.exists():
        downscaled_dir.mkdir(parents=True, exist_ok=True)
        images_to_process = list(os.listdir(images_dir))
    else:
        # Compare filenames to determine which images need processing.
        original_images = set(os.listdir(images_dir))
        processed_images = set(os.listdir(downscaled_dir))
        if original_images == processed_images:
            _logger.info(f"Downscaled images_{factor} are already in {dataset_path}")
            return
        _logger.info(f"{len(original_images) - len(processed_images)} missing images in images_{factor}")
        images_to_process = list(original_images - processed_images)

    # Process images that need downscaling.
    for image_name in tqdm(images_to_process, desc=f"Downscaling images by a factor of {factor}", disable=not viz):
        image_path = images_dir / image_name
        image_out = downscaled_dir / image_name
        downscale_single_image(image_path, image_out, factor)

def run_nerfstudio(dataset_path: Path, results_path: Path, method: str ='nerfacto', viz: bool=True) -> None:
    _logger.info(f"Running NeRFStudio for {results_path} ...")

    _logger.info(f"Compute downscaling factor for {dataset_path} ...")
    # max resolution of 1600px, which is the default of nerfstudio
    downscale_factor = compute_downscale_factor(dataset_path, max_resolution=1600)
    _logger.info(f"Downscaling factor found : {downscale_factor}")
    if downscale_factor > 1:
        downscale_images(dataset_path, downscale_factor, viz=viz)

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
        return None

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
    start = time.time()
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
        f"--timestamp run "
        # DataParser Args
        f"colmap --images-path {dataset_path}/images --colmap-path {results_path} "
        f"--downscale-factor {downscale_factor} "
        f"--load-3D-points {'True' if method == 'splatfacto' else 'False'} "
    )
    if not viz:
        stdout = subprocess.DEVNULL
        subprocess.run(train_cmd, shell=True, stdout=stdout, stderr=stdout)
    else:
        subprocess.run(train_cmd, shell=True)
    end = time.time()
    _logger.info(f"Training completed in {end-start:.2f} seconds.")
    _logger.info(f"Results stored in {results_path}/nerfstudio/{method}/run/")

    # # Evaluate the NeRF model
    _logger.info("Evaluating the NeRF model...")
    start = time.time()
    eval_cmd = [
        f"{CUDA_VISIBLE_DEVICES} ns-eval "
        f"--load-config {results_path}/nerfstudio/{method}/run/config.yml "
        f"--output-path {results_path}/nerfstudio/{method}/run/eval.json "
        f"--render-output-path {results_path}/nerfstudio/{method}/run/renders"
    ]
    subprocess.run(eval_cmd, shell=True)

    end = time.time()
    # Check if evaluation was successful
    if not os.path.exists(f"{results_path}/nerfstudio/{method}/run/eval.json"):
        _logger.error(f"Error: {results_path}/nerfstudio/{method}/run/eval.json not found. Evaluation failed.")
    else:
        _logger.info(f"Evaluation completed in {end-start:.2f} seconds.")
        _logger.info(f"Results stored in {results_path}/nerfstudio/{method}/run/eval.json")

    export_cmd = [
        f"{CUDA_VISIBLE_DEVICES} ns-export {'pointcloud' if method == 'nerfacto' else 'gaussian-splat'} "
        f"--load-config {results_path}/nerfstudio/{method}/run/config.yml "
        f"--output-dir {results_path} "
        f"--save-world-frame True "
        f"--normal-method open3d "
    ]
    # subprocess.run(export_cmd, shell=True)

    _logger.info("#"*50)

def sanity_check_colmap(path: Path, images_path: Path) -> None:
    if not isinstance(path, Path):
        path = Path(path)
    # read the colmap model
    # cameras, images, points3D = read_model(path, detect_colmap_format(path))
    model = read_model(str(path))

    # check that the model is not empty
    if len(model.cameras) == 0 or len(model.images) == 0:
        _logger.error(f"Error: The colmap model at {path} is empty. Please check the results path and try again.")
        exit(1)

    # check that in images, each image has a valid path
    path_changed = False
    for image_id, image in model.images.items():
        path_changed = False
        img_path = Path(image.name)
        if len(img_path.parts) > 1:
            image.name = img_path.parts[-1]
            path_changed = True
        # Find corresponding image in images_path
        base_name = img_path.stem
        matching_files = list(images_path.glob(f"{base_name}.*"))
        if matching_files:
            image.name = matching_files[0].name
            path_changed = True

    # if points3D is empty, it could mean that no points3D.bin file was found. We need to create it.
    if len(model.points3D) == 0 and (path / "points3D.bin").exists():
        model.import_PLY(path / "points3D.ply")
        path_changed = True

    if path_changed:
        _logger.info("Fixed colmap model.")
        model.write(path)


if __name__ == '__main__':
    from distutils.util import strtobool
    def _strtobool(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
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
        type=_strtobool,
        required=False,
        default=True,
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
    if not (os.path.exists(os.path.join(results_path, "images.bin")) or os.path.exists(
            os.path.join(results_path, "images.txt"))):
        _logger.error(f"Error: The colmap model at {results_path} does not exist. Please check the results path and try again.")
        exit(1)

    # Sanity check on colmap model
    sanity_check_colmap(results_path, Path(dataset_path) / "images")

    run_nerfstudio(dataset_path, results_path, method, viz)