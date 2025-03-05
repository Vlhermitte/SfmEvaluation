#!/bin/bash

if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3
fi

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

# check if the scene directory has a subdirectory named 'images'
if [ -d "$scene/images" ]; then
    scene="$scene/images"
# else if the scene directory is already the 'images' directory
elif [ "$(basename $scene)" == "images" ]; then
    scene=$scene
else
    echo "Scene directory does not have a subdirectory named 'images'"
    exit 1
fi


conda_env="vggsfm_tmp"
echo "Activating conda environment: $conda_env"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh"; exit 1; }
conda activate $conda_env || { echo "Failed to activate conda environment: $conda_env"; exit 1; }

# run the VGG-SfM pipeline
start_time=$(date +%s)
python ./vggsfm/demo.py query_method=sp+aliked camera_type=SIMPLE_RADIAL SCENE_DIR=$scene OUTPUT_DIR=$out_dir

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> ${out_dir}/time.txt