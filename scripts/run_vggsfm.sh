#!/bin/bash

#SBATCH --job-name=vggsfm_job
#SBATCH --output=vggsfm_job.out
#SBATCH --error=vggsfm_job.err

scene=$1
out=$2

# Function to print messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting VGG-SfM pipeline"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
    source $(conda info --base)/etc/profile.d/conda.sh
    module unload SciPy-bundle
fi

# Validate input arguments
if [ -z "$scene" ] || [ -z "$out" ]; then
    echo "Usage: $0 <scene_dir> <output_dir>"
    exit 1
fi

# Ensure the output directory exists
mkdir -p "$out"
scene=$(realpath "$scene")  # Convert to absolute path
out=$(realpath "$out")

# Ensure the scene directory exists
if [ ! -d "$scene" ]; then
    log "ERROR: Scene directory does not exist: $scene"
    exit 1
fi



# Check if output directory is empty (skip for SLURM jobs)
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$(ls -A "$out")" ]; then
    echo "Output directory is not empty. Do you want to overwrite? (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        log "User chose not to overwrite. Exiting."
        exit 1
    fi
fi

# Ensure the scene directory contains an 'images' subdirectory
if [ -d "$scene/images" ]; then
    scene="$scene"
elif [ "$(basename "$scene")" == "images" ]; then
    scene="$(dirname "$scene")"
else
    log "ERROR: Scene directory does not have a subdirectory named 'images'"
    exit 1
fi

log "Scene directory verified: $scene"

# Verify Conda environment exists
conda_env="vggsfm_tmp"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

# Run Python script inside the correct Conda environment
log "Running VGG-SfM pipeline..."
start_time=$(date +%s)

#if ! conda run -n "$conda_env" python vggsfm/demo.py camera_type=SIMPLE_RADIAL SCENE_DIR="$scene" OUTPUT_DIR="$out"; then
#    log "ERROR: VGG-SfM pipeline execution failed"
#    exit 1
#fi

# conda activate "$conda_env"
conda run -n "$conda_env" python vggsfm/demo.py camera_type=SIMPLE_RADIAL SCENE_DIR="$scene" OUTPUT_DIR="$out"

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

log "Pipeline completed in $elapsed_time seconds"
echo "Elapsed time: $elapsed_time seconds" >> "$out/time.txt"

log "Process finished successfully."