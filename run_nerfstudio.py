import subprocess
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from evaluation.utils.read_write_model import read_model, write_model

def run_nerfstudio(dataset_path, results_path, method='nerfacto', viz=False):
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

    # Train the NeRF model TODO: Investigate using Zip-NeRF for better quality
    _logger.info(f"Training the model using : {method}")
    train_cmd = (f"{CUDA_VISIBLE_DEVICES} ns-train {method} "
           f"--machine.num-devices {num_gpus} --pipeline.datamanager.images-on-gpu True "
           f"--pipeline.datamanager.dataloader-num-workers 8"
           f"{'--viewer.make-share-url True' if viz else ''} "
           f"--data {results_path} --output-dir {results_path}/nerfstudio colmap")
    subprocess.run(train_cmd, shell=True)

    # Move the trained model to the results directory
    # _logger.info("Moving the trained model to the results directory...")
    # if not os.path.exists(results_path + f"/{method}"):
    #     os.makedirs(results_path + f"/{method}", exist_ok=True)
    # mv_cmd = f"mv {results_path}/nerfstudio/{method}/* {results_path}/{method}"
    # subprocess.run(mv_cmd, shell=True)
    #
    # # Evaluate the NeRF model
    # _logger.info("Evaluating the NeRF model...")
    # eval_cmd = (f"{CUDA_VISIBLE_DEVICES} ns-eval --load-config {results_path}/{method}/config.yml "
    #        f"--output-dir {results_path}/{method}/eval.json")
    # subprocess.run(eval_cmd, shell=True)

def sanity_check_colmap(path):
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
        default="../../datasets/MipNerf360/garden",
        help="path to the dataset containing images"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=False,
        default="../../results/glomap/MipNerf360/garden",
        help="path to the results directory containing the images and colmap model under 'colmap/sparse/0'"
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

    # check that results_path/colmap/sparse/0 exists
    if not os.path.exists(results_path + "/colmap/sparse/0"):
        _logger.error(f"Error: The path {results_path}/colmap/sparse/0 does not exist. Please check the results path and try again.")
        exit(1)

    # Sanity check on colmap model
    sanity_check_colmap(results_path + "/colmap/sparse/0")

    run_nerfstudio(dataset_path, results_path, method, viz)