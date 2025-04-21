import shutil
import subprocess
import os
import sys
import argparse
import logging
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

GSPLAT_EXE = Path(__file__).resolve().parent.parent / "gsplat" / "examples" / "simple_trainer.py"

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


def downscale_images(dataset_path: Path, factor: int, viz: bool=True) -> Path:
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

    return downscaled_dir

def run_gsplat(dataset_path: Path, images_path: Path, result_path: Path, viz: bool=True) -> None:
    """
    Run the GSPLAT executable with the specified arguments.
    """
    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)
    if not isinstance(images_path, Path):
        images_path = Path(images_path)

    if not (dataset_path / "images").exists():
        shutil.copytree(images_path, dataset_path / "images", dirs_exist_ok=True)
        _logger.info(f"Copying images from {images_path} to {dataset_path / 'images'}")

    _logger.info(f"Compute downscaling factor for {dataset_path} ...")
    # max resolution of 1600px, which is the default of nerfstudio
    downscale_factor = compute_downscale_factor(dataset_path, max_resolution=1600)
    _logger.info(f"Downscaling factor found : {downscale_factor}")
    if downscale_factor > 1:
        downscaled_dir = downscale_images(dataset_path, downscale_factor, viz=viz)

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

    # Define the command to run the GSPLAT executable
    command = [
        sys.executable, str(GSPLAT_EXE), "default",
        "--data_dir", str(dataset_path),
        "--result_dir", str(dataset_path / result_path),
        "--data_factor", str(downscale_factor),
    ]

    # Execute the command
    subprocess.run(command, check=True)
    _logger.info(f"GSPLAT training completed. Results saved in {dataset_path / 'gsplat'}")

    shutil.rmtree(dataset_path / "images")
    # remove the downscaled images
    if downscale_factor > 1:
        shutil.rmtree(downscaled_dir)
        shutil.rmtree(f"{downscaled_dir}_png")
        _logger.info(f"Removed downscaled images_{downscale_factor} from {dataset_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GSPLAT on a dataset.")
    parser.add_argument("--dataset-path", type=str, help="Path to the colmap sparse reconstruction.", required=True)
    parser.add_argument("--images-path", type=str, help="Path to the images directory.", required=True)
    parser.add_argument("--results-path", type=str, help="Path to the results directory.", default="gsplat")
    parser.add_argument("--pose-opt", action="store_true", help="Enable pose optimization.")
    parser.add_argument("--viz", action="store_true", help="Enable visualization.")
    args = parser.parse_args()

    # Make sure that GSPLAT_EXE is present
    if not GSPLAT_EXE.exists():
        _logger.error(f"GSPLAT executable not found at {GSPLAT_EXE}. Please check clone repo or check path.")
        sys.exit(1)

    # Run the GSPLAT function with the provided arguments
    run_gsplat(args.dataset_path, args.images_path, args.results_path, args.viz)
