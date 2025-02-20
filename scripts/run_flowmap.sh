#!/bin/bash

scene=$1
out=$2

# Check if the scene and output directory are provided
if [ -z "$scene" ] || [ -z "$out" ]; then
    echo "Usage: ./run_flowmap.sh <scene_dir> <output_dir>"
    exit 1
fi

out_dir=$out
echo "Output directory: $out_dir"

# Check if the scene directory exists
if [ ! -d $scene ]; then
    echo "Scene directory does not exist"
    exit 1
fi

# Check if the output directory exists
if [ ! -d $out_dir ]; then
    mkdir -p $out_dir
fi

# Check if the output directory is empty ask to overwrite
if [ "$(ls -A $out_dir)" ]; then
    echo "Output directory is not empty. Do you want to overwrite? (y/n)"
    read answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
fi

# Check image format in the scene directory (png, jpg, JPG, etc.)
image_format=$(ls $scene | head -n 1 | rev | cut -d'.' -f1 | rev)

conda_env="flowmap"
echo "Activating conda environment: $conda_env"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh"; exit 1; }
conda activate $conda_env || { echo "Failed to activate conda environment: $conda_env"; exit 1; }

cd flowmap

# run the FlowMap pipeline
python3 -m flowmap.overfit dataset=images dataset.images.root=../$scene  output_dir=../$out_dir
