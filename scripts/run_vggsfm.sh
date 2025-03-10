#!/bin/bash

scene=$1
out=$2

# Check if the scene and output directory are provided
if [ -z "$scene" ] || [ -z "$out" ]; then
    echo "Usage: ./run_vggsfm.sh <scene_dir> <output_dir>"
    exit 1
fi

if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda_env="vggsfm_tmp"
conda activate $conda_env || { echo "ERROR: Failed to activate Conda environment: $conda_env"; exit 1; }

# Check if the correct Conda environment is active
if [[ "$CONDA_PREFIX" != *"$conda_env"* ]]; then
    echo "ERROR: Conda environment $conda_env is not activated correctly!"
    echo "Current Conda environment: $CONDA_PREFIX"
    exit 1
fi

echo "Conda environment activated: $CONDA_PREFIX"

# Check Python path
python_path=$(which python)
echo "Using Python: $python_path"

python -c "import lightglue; print('LightGlue is working')"

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
if [ -z "$SLURM_JOB_ID" ]; then
  if [ "$(ls -A $out_dir)" ]; then
      echo "Output directory is not empty. Do you want to overwrite? (y/n)"
      read answer
      if [ "$answer" != "y" ]; then
          exit 1
      fi
  fi
fi

# check if the scene directory has a subdirectory named 'images'
if [ -d "$scene/images" ]; then # keep the scene directory as is
    scene="$scene"
# else if the scene directory is already the 'images' directory
elif [ "$(basename $scene)" == "images" ]; then # go up one level
    scene="$(dirname $scene)"
else
    echo "Scene directory does not have a subdirectory named 'images'"
    exit 1
fi

# run the VGG-SfM pipeline
start_time=$(date +%s)
python vggsfm/demo.py query_method=sp+aliked camera_type=SIMPLE_RADIAL SCENE_DIR=$scene OUTPUT_DIR=$out_dir

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> ${out_dir}/time.txt