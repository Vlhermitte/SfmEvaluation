#!/bin/bash

# Run the evaluation script evaluation/main.py
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

# Get arg or default is nerfacto
METHOD=${1:-"nerfacto"}

# Make sure we are in the root directory of the project
cd "$(dirname "$0")/.." || exit

# ETH3D
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${ETH3D_SCENES[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "data/results/${sfm_method}/ETH3D/${scene}/colmap/sparse/0" ]; then
      echo "Warning: ${sfm_method} model not found for ${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    python src/run_nerfstudio.py --dataset-path data/datasets/ETH3D/${scene} --results-path data/results/${sfm_method}/ETH3D/${scene}/colmap/sparse/0 --method ${METHOD}
  done
done

# MipNerf360
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${MIP_NERF_360_SCENE[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "data/results/${sfm_method}/MipNerf360/${scene}/colmap/sparse/0" ]; then
      echo "Warning: ${sfm_method} model not found for ${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    python src/run_nerfstudio.py --dataset-path data/datasets/MipNerf360/${scene} --results-path data/results/${sfm_method}/MipNerf360/${scene}/colmap/sparse/0 --method ${METHOD}
  done
done

