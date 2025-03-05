#!/bin/bash

#SBATCH --job-name=all_sfm
#SBATCH --output=all_sfm.out
#SBATCH --error=all_sfm.err
#SBATCH --time=12:00:00             # Request 12 hours of runtime
#SBATCH --partition=1day            # Use the '1day' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=32G                   # Request 32 GB of RAM
#SBATCH --cpus-per-task=8          # Request 8 CPUs
#SBATCH --mail-user=lhermval@cvut.cz
#SBATCH --mail-type=ALL

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

SFM_METHODS=(
    "vggsfm"
    "flowmap"
    "acezero"
    "glomap"
)

if [ ! -z "$SLURM_JOB_ID" ]; then
  cd ~/SfmEvaluation
  base_data_path=~/SfmEvaluation/data
else
  base_data_path=../data
fi

for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${ETH3D_SCENES[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "${base_data_path}/datasets/ETH3D/${scene}" ]; then
      echo "Warning: No dataset found for ${base_data_path}/datasets/ETH3D/${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    bash ./run_${sfm_method}.sh ${base_data_path}/datasets/ETH3D/${scene}/images ${base_data_path}/results/${sfm_method}/ETH3D/${scene}/colmap/sparse/0
  done
done

# MipNerf360
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${MIP_NERF_360_SCENE[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "${base_data_path}/datasets/MipNerf360/${scene}" ]; then
      echo "Warning: No dataset found for ${base_data_path}/datasets/MipNerf360/${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    bash ./run_${sfm_method}.sh ${base_data_path}/datasets/MipNerf360/${scene}/images ${base_data_path}/results/${sfm_method}/MipNerf360/${scene}/colmap/sparse/0
  done
  echo ##############################################################################################################################
done
