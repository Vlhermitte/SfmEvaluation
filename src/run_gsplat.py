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
from typing import Optional
from utils.common import read_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set working directory to the parent directory of the script
working_dir = Path(__file__).resolve().parent
if working_dir.name == "src":
    os.chdir(working_dir.parent)

GSPLAT_EXE = Path(__file__).resolve().parent / "evaluation" / "gsplat_trainer.py"

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


def check_cuda_gpus() -> Optional[int]:
    """
    Checks for available CUDA GPUs using nvidia-smi.

    Returns:
        The number of GPUs found, or None if an error occurs or no GPUs are found.
    """
    _logger.info("Checking for available CUDA GPUs...")
    try:
        cmd = "nvidia-smi --query-gpu=count --format=csv,noheader"
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,  # Raise CalledProcessError on non-zero exit code
            capture_output=True,
            text=True,   # Decode stdout/stderr as text
            timeout=10   # Add a timeout
        )
        output = result.stdout.strip()
        if not output:
             _logger.error("nvidia-smi returned empty output.")
             return None
        # Parse the first line of the output for the count
        num_gpus = int(output.splitlines()[0])
        if num_gpus > 0:
             _logger.info(f"Found {num_gpus} CUDA GPUs.")
             return num_gpus
        else:
             _logger.warning("nvidia-smi reported 0 GPUs. GSPLAT requires at least one.")
             return None
    except FileNotFoundError:
        _logger.error("'nvidia-smi' command not found. Is the NVIDIA driver installed and in PATH?")
        return None
    except subprocess.CalledProcessError as e:
        _logger.error(f"Failed to run 'nvidia-smi' (Exit code {e.returncode}). Stderr: {e.stderr}")
        return None
    except (ValueError, IndexError) as e:
         _logger.error(f"Failed to parse 'nvidia-smi' output '{output}': {e}")
         return None
    except subprocess.TimeoutExpired:
        _logger.error("'nvidia-smi' command timed out.")
        return None
    except Exception as e: # Catch unexpected errors
        _logger.error(f"An unexpected error occurred while checking GPUs: {e}")
        return None

def check_colmap_model(colmap_path: str, images_path: str) -> bool:
    """
    Check if the COLMAP model is valid by loading it with pycolmap.
    """
    if not isinstance(colmap_path, str):
        colmap_path = str(colmap_path)
    model = read_model(colmap_path)

    # Check images names
    is_overwrite = False
    for image in model.images.values():
        # keep only the image name
        image_name = os.path.basename(image.name)
        # make sure the image name is in the images path
        if not os.path.exists(os.path.join(images_path, image_name)):
            _logger.error(f"Image {image_name} not found in {images_path}")
            return False
        if image.name != image_name:
            # Overwrite the image name in the scene manager
            model.images[image.id].name = image_name
            is_overwrite = True
    if is_overwrite:
        # Save the updated scene manager
        model.save_images(colmap_path)

    return True


def run_gsplat(dataset_path: Path, images_path: Path, result_path: Path, pose_opt: bool = False, viz: bool=True) -> None:
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

    num_gpus = check_cuda_gpus()

    # Define the command to run the GSPLAT executable
    command = [
        sys.executable, str(GSPLAT_EXE), "default",
        "--data_dir", str(dataset_path),
        "--result_dir", str(dataset_path / result_path),
        "--data_factor", str(downscale_factor),
    ]
    if pose_opt:
        command.append("--pose_opt")

    if not viz:
        command.append("--disable-viewer")

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

    # Check that colmap model exists (i.e .bin/.txt files)
    paths = [
        Path(args.dataset_path) / "sparse/0/images.bin",
        Path(args.dataset_path) / "sparse/0/images.txt",
        Path(args.dataset_path) / "sparse/0/cameras.bin",
        Path(args.dataset_path) / "sparse/0/cameras.txt",
    ]
    if not any(path.exists() for path in paths):
        _logger.error(
            f"Error: The colmap model at {args.dataset_path}/sparse/0/ does not exist. Please check the results path and try again.")
        exit(1)

    is_ok = check_colmap_model(Path(args.dataset_path) / "sparse/0", args.images_path)
    if not is_ok:
        _logger.error(f"Error: The colmap model at {args.dataset_path}/sparse/0/ is not valid. Please check the results path and try again.")
        exit(1)

    # Run the GSPLAT function with the provided arguments
    run_gsplat(
        dataset_path=args.dataset_path,
        images_path=args.images_path,
        result_path=args.results_path,
        pose_opt=bool(args.pose_opt),
        viz=args.viz,
    )
