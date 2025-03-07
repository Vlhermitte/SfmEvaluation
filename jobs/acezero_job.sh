#!/bin/bash

#SBATCH --job-name=acezero_job
#SBATCH --output=acezero_job.out
#SBATCH --error=acezero_job.err
#SBATCH --time=08:00:00             # Request 8 hours of runtime
#SBATCH --partition=1day            # Use the '1day' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=32G                   # Request 32 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs


ETH3D_SCENES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
    "terrace"
    "terrains"
)

MIP_NERF_360_SCENE=(
  "bicycle"
  "bonsai"
  "counter"
  "garden"
  "kitchen"
  "room"
  "stump"
)

DATASETS_DIR="data/datasets"

# Base output directory
OUT_DIR="data/results/acezero"

# ETH3D
for SCENE in "${ETH3D_SCENES[@]}"; do
    echo "Processing scene: $SCENE"

    # Check if the output directory already exists
    if [ -d "${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    else
        mkdir -p ${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0
    fi

    # Check image format in the scene directory (png, jpg, JPG, etc.)
    image_format=$(ls $SCENE | head -n 1 | rev | cut -d'.' -f1 | rev)

    start_time=$(date +%s)

    python acezero/ace_zero.py "${DATASETS_DIR}/ETH3D/$SCENE/images/*.$image_format" ${OUT_DIR}/ETH3D/${SCENE}/acezero_format --export_point_cloud True

    end_time=$(date +%s)
    elapsed_time=$(( end_time - start_time ))
    echo "Elapsed time: $elapsed_time seconds" >> ${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0/time.txt

    python acezero/convert_to_colmap.py --src_dir ${OUT_DIR}/ETH3D/${SCENE}/acezero_format --dst_dir ${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done


# MIP_NERF_360
for SCENE in "${MIP_NERF_360_SCENE[@]}"; do
    echo "Processing scene: $SCENE"

    # Check if the output directory already exists
    if [ -d "${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    else
        mkdir -p ${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0
    fi

    # Check image format in the scene directory (png, jpg, JPG, etc.)
    image_format=$(ls $SCENE | head -n 1 | rev | cut -d'.' -f1 | rev)

    start_time=$(date +%s)

    python acezero/ace_zero.py "${DATASETS_DIR}/MipNerf360/$SCENE/images/*.$image_format" ${OUT_DIR}/MipNerf360/${SCENE}/acezero_format --export_point_cloud True

    end_time=$(date +%s)
    elapsed_time=$(( end_time - start_time ))
    echo "Elapsed time: $elapsed_time seconds" >> ${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0/time.txt

    python acezero/convert_to_colmap.py --src_dir ${OUT_DIR}/MipNerf360/${SCENE}/acezero_format --dst_dir ${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done


