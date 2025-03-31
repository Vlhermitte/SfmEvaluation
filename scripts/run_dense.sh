#!/bin/bash

#SBATCH --job-name=dense_job
#SBATCH --output=dense_job.out
#SBATCH --error=dense_job.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# Check if the script is running on a Slurm device
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load COLMAP
fi

# Verify COLMAP and GLOMAP are executable
if ! command -v colmap &> /dev/null; then
    echo "COLMAP is not installed. Please install it from https://colmap.github.io/ or check your PATH"
    exit 1
fi

scene=$1
out=$2

# Check if the scene and output directory are provided
if [ -z "${scene}" ]; then
    echo "Usage: ./run_dense.sh <scene_dir> <output_dir>"
    exit 1
fi

out_dir=$out

# Check if the scene directory exists
if [ ! -d "${scene}" ]; then
    echo "Scene directory does not exist"
    exit 1
fi

# Check if the output directory exists
if [ ! -d "${out_dir}" ]; then
    mkdir -p "${out_dir}"
fi

if [ -z "$SLURM_JOB_ID" ]; then
  # Check if the output directory is not empty and prompt for overwrite
  if [ -d "${out_dir}" ] && [ "$(ls -A "${out_dir}")" ]; then
      echo "Output directory '$out_dir' is not empty. Do you want to overwrite? (y/n)"
      read -r answer
      case "$answer" in
          y|Y)
              echo "Overwriting '$out_dir'..."
              rm -rf "${out_dir:?}"/*
              ;;
          n|N)
              echo "Exiting without making changes."
              exit 1
              ;;
          *)
              echo "Invalid input. Exiting."
              exit 1
              ;;
      esac
  fi
fi

mkdir -p "${out_dir}"

echo "Running dense reconstruction on scene: ${scene}"
start_time=$(date +%s)

colmap image_undistorter \
    --image_path "${scene}" \
    --input_path "${scene}" \
    --output_path "${out_dir}" \
    --output_type COLMAP

colmap patch_match_stereo \
   --workspace_path "${out_dir}" \
   --workspace_format COLMAP \
   --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
   --workspace_path "${out_dir}" \
   --workspace_format COLMAP \
   --input_type geometric \
   --output_path "${out_dir}"/fused.ply

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> "${out_dir}"/dense_time.txt
