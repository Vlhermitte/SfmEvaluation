#!/bin/bash

# Run the evaluation script evaluation/main.py
SCENES=(
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

# Make sure we are in the root directory of the project
cd "$(dirname "$0")/.." || exit

# VGGSfm
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "data/results/vggsfm/ETH3D/${scene}/colmap/sparse/0" ]; then
    echo "Warning: VGGSfm model not found for ${scene}. Skipping..."
    continue
  fi
  echo "VGGSfm: Processing ${scene}"
  python src/run_nerfstudio.py --dataset-path data/datasets/ETH3D/${scene} --results-path data/results/vggsfm/ETH3D/${scene}/colmap/sparse/0
done

# Flowmap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "data/results/flowmap/ETH3D/${scene}/colmap/sparse/0" ]; then
    echo "Warning: Flowmap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Flowmap: Processing ${scene}"
  python src/run_nerfstudio.py --dataset-path data/datasets/ETH3D/${scene} --results-path data/results/flowmap/ETH3D/${scene}/colmap/sparse/0
done

# AceZero
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "data/results/acezero/ETH3D/${scene}/colmap/sparse/0" ]; then
    echo "Warning: AceZero model not found for ${scene}. Skipping..."
    continue
  fi
  echo "AceZero: Processing ${scene}"
  python src/run_nerfstudio.py --dataset-path data/datasets/ETH3D/${scene} --results-path data/results/acezero/ETH3D/${scene}/colmap/sparse/0
done

# Glomap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "data/results/glomap/ETH3D/${scene}/colmap/sparse/0" ]; then
    echo "Warning: Glomap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Glomap: Processing ${scene}"
  python src/run_nerfstudio.py --dataset-path data/datasets/ETH3D/${scene} --results-path data/results/glomap/ETH3D/${scene}/colmap/sparse/0
done
