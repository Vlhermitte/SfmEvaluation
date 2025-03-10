#!/bin/bash

#SBATCH --job-name=acezero_job
#SBATCH --output=acezero_job.out
#SBATCH --error=acezero_job.err
#SBATCH --time=04:00:00
#SBATCH --partition=fast
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12

# Function to print messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting ACE-Zero pipeline"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
fi

# Validate input arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <scene_dir> <output_dir>"
    exit 1
fi

scene=$(realpath "$1")  # Convert to absolute path
out=$(realpath "$2")    # Convert to absolute path

log "Scene directory: $scene"
log "Output directory: $out"

# Ensure the scene directory exists
if [ ! -d "$scene" ]; then
    log "ERROR: Scene directory does not exist: $scene"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$out"

# Check if output directory is empty (skip for SLURM jobs)
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$(ls -A "$out")" ]; then
    echo "Output directory is not empty. Do you want to overwrite? (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        log "User chose not to overwrite. Exiting."
        exit 1
    fi
fi

# Detect image format in the scene directory
image_format=$(find "$scene" -maxdepth 1 -type f | head -n 1 | rev | cut -d'.' -f1 | rev)

if [ -z "$image_format" ]; then
    log "ERROR: No images found in $scene"
    exit 1
fi

log "Detected image format: .$image_format"

# Verify Conda environment exists
conda_env="ace0"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

log "Running ACE-Zero using Conda environment: $conda_env"

# Change to the ACE-Zero directory
cd acezero || { log "ERROR: Failed to change directory to 'acezero'"; exit 1; }

# Prepare directories
parent_dir=$(dirname "$out")
mkdir -p "$parent_dir/acezero_format"

log "Running ACE-Zero on $scene"
start_time=$(date +%s)

if ! conda run -n "$conda_env" python ace_zero.py "$scene/*.$image_format" "$parent_dir/acezero_format" --export_point_cloud True; then
    log "ERROR: ACE-Zero execution failed"
    exit 1
fi

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

log "Pipeline completed in $elapsed_time seconds"
echo "Elapsed time: $elapsed_time seconds" >> "$out/time.txt"

log "Converting to COLMAP format..."
if ! conda run -n "$conda_env" python convert_to_colmap.py --src_dir "$parent_dir/acezero_format" --dst_dir "$out"; then
    log "ERROR: COLMAP conversion failed"
    exit 1
fi

log "Process finished successfully."
