#!/bin/bash

# Run the evaluation script Test/main.py
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

# VGGSfm
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/vggsfm/${scene}/sparse" ]; then
    echo "VGGSfm model not found for ${scene}. Skipping..."
    continue
  fi
  echo "VGGSfm: Processing ${scene}"
  python Tests/main.py --est-model-path results/vggsfm/${scene}/sparse --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# Flowmap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/flowmap/${scene}/colmap/sparse/0" ]; then
    echo "Flowmap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Flowmap: Processing ${scene}"
  python Tests/main.py --est-model-path results/flowmap/${scene}/colmap/sparse/0 --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# AceZero
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/flowmap/${scene}/colmap/sparse/0" ]; then
    echo "Flowmap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "AceZero: Processing ${scene}"
  python Tests/main.py --est-model-path results/acezero/${scene}/sparse/ --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done

# Glomap
for scene in "${SCENES[@]}"; do
  # Check if the estimated model exists
  if [ ! -d "results/glomap/${scene}/sparse/0" ]; then
    echo "Glomap model not found for ${scene}. Skipping..."
    continue
  fi
  echo "Glomap: Processing ${scene}"
  python Tests/main.py --est-model-path results/glomap/${scene}/sparse/0 --gt-model-path datasets/ETH3D/${scene}/dslr_calibration_jpg
done
