#!/bin/bash

#SBATCH --job-name=acezero_job
#SBATCH --output=acezero_job.out
#SBATCH --error=acezero_job.err
#SBATCH --time=04:00:00
#SBATCH --partition=fast
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12

if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3
fi

scene=$1
out=$2

# Check if the scene and output directory are provided
if [ -z "$scene" ] || [ -z "$out" ]; then
    echo "Usage: ./run_acezero.sh <scene_dir> <output_dir>"
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

if [ -z "$SLURM_JOB_ID" ]; then
  # Check if the output directory is empty ask to overwrite
  if [ "$(ls -A $out_dir)" ]; then
      echo "Output directory is not empty. Do you want to overwrite? (y/n)"
      read answer
      if [ "$answer" != "y" ]; then
          exit 1
      fi
  fi
fi

# Check image format in the scene directory (png, jpg, JPG, etc.)
image_format=$(ls $scene | head -n 1 | rev | cut -d'.' -f1 | rev)

# Activate conda environment
conda_env="ace0"
echo "Activating conda environment: $conda_env"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh"; exit 1; }
conda activate $conda_env || { echo "Failed to activate conda environment: $conda_env"; exit 1; }

cd acezero || { echo "Failed to change directory to acezero"; exit 1; }

# Timing execution
parent_dir=$(dirname $out_dir)
mkdir -p ../$out_dir
mkdir -p ../$parent_dir/acezero_format

echo "Running ACE-Zero on $scene"
start_time=$(date +%s)
python ace_zero.py "../$scene/*.$image_format" ../$parent_dir/acezero_format --export_point_cloud True
end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> ../$out_dir/time.txt

python convert_to_colmap.py --src_dir ../$parent_dir/acezero_format --dst_dir ../$out_dir
