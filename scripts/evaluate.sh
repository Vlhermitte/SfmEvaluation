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
  if [ ! -d "results/vggsfm/ETH3D/${scene}/sparse" ]; then
    echo "VGGSfm model not found for ${scene}. Skipping..."
    continue
  fi
  echo "VGGSfm: Processing ${scene}"
  python main.py --est-model-path results/vggsfm/ETH3D/${scene}/sparse --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# Flowmap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/flowmap/ETH3D/${scene}/colmap/sparse/0" ]; then
    echo "Flowmap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Flowmap: Processing ${scene}"
  python main.py --est-model-path results/flowmap/ETH3D/${scene}/colmap/sparse/0 --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# AceZero
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/acezero/ETH3D/${scene}/colmap/sparse" ]; then
    echo "AceZero model not found for ${scene}. Skipping..."
    continue
  fi
  echo "AceZero: Processing ${scene}"
  python main.py --est-model-path results/acezero/ETH3D/${scene}/colmap/sparse --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# Glomap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/glomap/ETH3D/${scene}/sparse/0" ]; then
    echo "Glomap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Glomap: Processing ${scene}"
  python main.py --est-model-path results/glomap/ETH3D/${scene}/sparse/0 --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done
