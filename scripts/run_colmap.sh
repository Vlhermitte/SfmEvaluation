#!/bin/bash

#SBATCH --job-name=colmap_job
#SBATCH --output=colmap_job.out
#SBATCH --error=colmap_job.err

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

image_path=$1
out_dir=$2
matcher=${3:-exhaustive_matcher}

# Check if the image_path and output directory are provided
if [ -z "$image_path" ]; then
    echo "Usage: ./run_glomap.sh <image_path> <output_path> [matcher]"
    exit 1
fi

# Check if the image_path directory exists
if [ ! -d "$image_path" ]; then
    echo "$image_path does not exist"
    exit 1
fi

# Check if the output directory exists
if [ ! -d "$out_dir" ]; then
    mkdir -p "$out_dir"
fi

if [ -z "$SLURM_JOB_ID" ]; then # if we are in a SLURM job we will automatically overwrite the output directory
  # Check if the output directory is not empty and prompt for overwrite
  if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir")" ]; then
      echo "Output directory '$out_dir' is not empty. Do you want to overwrite? (y/n)"
      read -r answer
      case "$answer" in
          y|Y)
              echo "Overwriting '$out_dir'..."
              rm -rf "${out_dir:?}/"*
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


start_time=$(date +%s)

# Init colmap database
DATABASE="${out_dir}/database.db"
# If database does not exist, create a new database
if [ ! -f "${DATABASE}" ]; then
  echo "Creating new database: ${DATABASE}"
  echo "COLMAP feature_extractor..."
  colmap feature_extractor \
    --database_path "${DATABASE}" \
    --image_path "${image_path}" \
	--ImageReader.camera_model RADIAL \
	--ImageReader.single_camera 1 \
	--SiftExtraction.use_gpu 1

  echo "COLMAP ${matcher}..."
  colmap "${matcher}" \
    --database_path "${DATABASE}" \
    --SiftMatching.use_gpu 1
fi

# GLOMAP execution
mkdir -p "${out_dir}/sparse"
echo "CLOMAP mapper..."
colmap mapper \
    --database_path "${DATABASE}" \
    --image_path "${image_path}" \
    --output_path "${out_dir}/sparse"

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> "${out_dir}/sparse/0/time.txt"

