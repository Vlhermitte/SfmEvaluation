#!/bin/bash

scene=$1

# Check if the scene and output directory are provided
if [ -z "$scene" ]; then
    echo "Usage: ./run_vggsfm.sh <scene_dir>"
    exit 1
fi

# Get last part of the scene directory
scene_name=$(basename $scene)
echo "Scene name: $scene_name"

out_dir="results/vggsfm/$scene_name"
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

# run the VGG-SfM pipeline
python ./vggsfm/demo.py query_method=sp+aliked SCENE_DIR=$scene OUTPUT_DIR=$out_dir
