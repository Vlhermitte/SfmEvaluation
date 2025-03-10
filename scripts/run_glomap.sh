#!/bin/bash

#SBATCH --job-name=glomap_job
#SBATCH --output=glomap_job.out
#SBATCH --error=glomap_job.err
#SBATCH --time=04:00:00
#SBATCH --partition=fast
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12

# Check if the script is running on a Slurm device
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load CMake
    module load GCC
    module load Ninja
    module load Eigen
    module load FLANN
    module load COLMAP
    # Module load GLOMAP if available
    if module avail GLOMAP &> /dev/null; then
        module load GLOMAP
    else
      export PATH=~/SfmEvaluation/glomap/build/glomap/:$PATH
    fi
fi

# Verify COLMAP and GLOMAP are executable
if ! command -v colmap &> /dev/null; then
    echo "COLMAP is not installed. Please install it from https://colmap.github.io/ or check your PATH"
    exit 1
fi

if ! command -v glomap &> /dev/null; then
    echo "GLOMAP is not installed. Please build from source : https://github.com/colmap/glomap or check your PATH"
    exit 1
fi

scene=$1
out=$2
matcher=${3:-exhaustive_matcher}

# Check if the scene and output directory are provided
if [ -z "$scene" ]; then
    echo "Usage: ./run_glomap.sh <scene_dir> <output_dir>"
    exit 1
fi

out_dir=$out

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
  # Check if the output directory is not empty and prompt for overwrite
  if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir")" ]; then
      echo "Output directory '$out_dir' is not empty. Do you want to overwrite? (y/n)"
      read -r answer
      case "$answer" in
          y|Y)
              echo "Overwriting '$out_dir'..."
              rm -rf "$out_dir"/*  # Clear the directory
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
DATABASE=${out_dir}/sample_reconstruction.db
# If database does not exist, create a new database
if [ ! -f ${DATABASE} ]; then
  echo "Creating new database: ${DATABASE}"
  echo "COLMAP feature_extractor..."
  colmap feature_extractor \
    --database_path ${DATABASE} \
    --image_path ${scene} \
	--ImageReader.camera_model RADIAL \
	--ImageReader.single_camera 1 \
	--SiftExtraction.use_gpu 1

  echo "COLMAP ${matcher}..."
  colmap "${matcher}" \
    --database_path ${DATABASE} \
    --SiftMatching.use_gpu 1
fi

# GLOMAP execution
mkdir -p ${out_dir}/sparse
echo "GLOMAP mapper..."
glomap mapper \
    --database_path ${DATABASE} \
    --image_path ${scene} \
    --output_path ${out_dir}/sparse

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo "Elapsed time: $elapsed_time seconds" >> ${out_dir}/sparse/0/time.txt

# Disable for now
#mkdir -p ${out_dir}/dense
#colmap image_undistorter \
#    --image_path ${scene} \
#    --input_path ${out_dir}/sparse/0 \
#    --output_path ${out_dir}/dense \
#    --output_type COLMAP
#
#colmap patch_match_stereo \
#   --workspace_path ${out_dir}/dense \
#   --workspace_format COLMAP \
#   --PatchMatchStereo.geom_consistency true
#
#colmap stereo_fusion \
#   --workspace_path ${out_dir}/dense \
#   --workspace_format COLMAP \
#   --input_type geometric \
#   --output_path ${out_dir}/dense/fused.ply
#
#colmap poisson_mesher \
#    --input_path ${out_dir}/dense/fused.ply \
#    --output_path ${out_dir}/dense/meshed-poisson.ply
#
#colmap delaunay_mesher \
#    --input_path ${out_dir}/dense \
#    --output_path ${out_dir}/dense/meshed-delaunay.ply
