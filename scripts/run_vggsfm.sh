#!/bin/bash

scene=$1
out=$2

# Check if the scene and output directory are provided
if [ -z "$scene" ] || [ -z "$out" ]; then
    echo "Usage: ./run_vggsfm.sh <scene_dir> <output_dir>"
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

# check if scene basename is images
if [ $(basename $scene) != "images" ]; then
    echo "Scene directory should be named 'images'"
    exit 1
fi
# remove basename from scene (VGGSfm expects the scene directory to be named 'images')
scene=$(dirname $scene)

conda_env="vggsfm_tmp"
echo "Activating conda environment: $conda_env"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh"; exit 1; }
conda activate $conda_env || { echo "Failed to activate conda environment: $conda_env"; exit 1; }

# run the VGG-SfM pipeline
python ./vggsfm/demo.py query_method=sp+aliked camera_type=SIMPLE_RADIAL SCENE_DIR=$scene OUTPUT_DIR=$out_dir
