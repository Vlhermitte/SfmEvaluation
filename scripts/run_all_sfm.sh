#!/bin/bash


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


for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${ETH3D_SCENES[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "../data/datasets/ETH3D/${scene}" ]; then
      echo "Warning: No dataset found for ../data/datasets/ETH3D/${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    bash ./run_${sfm_method}.sh ../data/datasets/ETH3D/${scene}/images ../data/results/${sfm_method}/ETH3D/${scene}/colmap/sparse/0
  done
done

# MipNerf360
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${MIP_NERF_360_SCENE[@]}"; do
    # Check if the estimated model exists
    if [ ! -d "../data/datasets/MipNerf360/${scene}" ]; then
      echo "Warning: No dataset found for ../data/datasets/MipNerf360/${scene}. Skipping..."
      continue
    fi
    echo "${sfm_method}: Processing ${scene}"
    bash ./run_${sfm_method}.sh ../data/datasets/MipNerf360/${scene}/images ../data/results/${sfm_method}/MipNerf360/${scene}/colmap/sparse/0
  done
done
