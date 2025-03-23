#!/bin/bash

#SBATCH --job-name=flowmap_job
#SBATCH --output=flowmap_job.out
#SBATCH --error=flowmap_job.err
#SBATCH --ntasks-per-node=12

# Function to print messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting FlowMap pipeline"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
    source $(conda info --base)/etc/profile.d/conda.sh
    #module load Hydra || { log "ERROR: Failed to load Hydra module"; exit 1; }
fi

# Validate input arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <scene_dir> <output_dir>"
    exit 1
fi
# Ensure output directory exists
mkdir -p "$2"
scene=$(realpath "$1")  # Convert to absolute path
out=$(realpath "$2")

log "Scene directory: $scene"
log "Output directory: $out"

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

# Verify Conda environment exists
conda_env="flowmap"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

log "Running FlowMap pipeline using Conda environment: $conda_env"

# Change to the FlowMap directory
cd flowmap || { log "ERROR: Failed to change directory to 'flowmap'"; exit 1; }

# Run the FlowMap pipeline
start_time=$(date +%s)

if ! conda run -n "$conda_env" python3 -m flowmap.overfit dataset=images dataset.images.root="$scene" output_dir="$out"; then
    log "ERROR: FlowMap pipeline execution failed"
    exit 1
fi

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

log "Pipeline completed in $elapsed_time seconds"
echo "Elapsed time: $elapsed_time seconds" >> "$out/time.txt"

log "Process finished successfully."
