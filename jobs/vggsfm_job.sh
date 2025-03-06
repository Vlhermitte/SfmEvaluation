#!/bin/bash

#SBATCH --job-name=vggsfm_job
#SBATCH --output=vggsfm_job.out
#SBATCH --error=vggsfm_job.err
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
OUT_DIR="data/results/vggsfm"

# Run the VGGSfm pipeline for each scene
for SCENE in "${ETH3D_SCENES[@]}"; do
    echo "Processing scene: $SCENE"

    # Check if the output directory already exists
    if [ -d "${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    else
        mkdir -p ${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0
    fi
    start_time=$(date +%s)

    python vggsfm/demo.py query_method=sp+aliked camera_type=SIMPLE_RADIAL SCENE_DIR=${DATASETS_DIR}/ETH3D/$SCENE/ OUTPUT_DIR=${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0

    end_time=$(date +%s)
    elapsed_time=$(( end_time - start_time ))

    echo "Elapsed time: $elapsed_time seconds" >> ${OUT_DIR}/ETH3D/${SCENE}/colmap/sparse/0/time.txt

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

echo "All scenes processed."

for SCENE in "${MIP_NERF_360_SCENE[@]}"; do
    echo "Processing scene: $SCENE"

    # Check if the output directory already exists
    if [ -d "${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    else
        mkdir -p ${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0
    fi
    start_time=$(date +%s)

    python vggsfm/demo.py query_method=sp+aliked camera_type=SIMPLE_RADIAL SCENE_DIR=${DATASETS_DIR}/MipNerf360/$SCENE/ OUTPUT_DIR=${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0

    end_time=$(date +%s)
    echo "Elapsed time: $elapsed_time seconds" >> ${OUT_DIR}/MipNerf360/${SCENE}/colmap/sparse/0/time.txt

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done


