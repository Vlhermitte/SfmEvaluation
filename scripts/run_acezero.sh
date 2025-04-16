#!/bin/bash

#SBATCH --job-name=acezero_job
#SBATCH --output=acezero_job.out
#SBATCH --error=acezero_job.err
#SBATCH --time=12:00:00
#SBATCH --partition=1day
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=32G
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
    source $(conda info --base)/etc/profile.d/conda.sh
    module unload SciPy-bundle Pillow
fi

# Validate input arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <scene_dir> <output_dir>"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$2"
scene=$(realpath "$1")  # Convert to absolute path
out=$(realpath "$2")    # Convert to absolute path

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

PYTHON_BIN_EVAL="$(conda run -n "Evaluation" which python)"
PYTHON_BIN="$(conda run -n "$conda_env" which python)"
log "Using Python binary: $PYTHON_BIN"

log "Running ACE-Zero using Conda environment: $conda_env"

# Change to the ACE-Zero directory
cd acezero || { log "ERROR: Failed to change directory to 'acezero'"; exit 1; }

# Prepare directories
parent_dir=$(dirname "$out")
mkdir -p "$parent_dir/acezero_format"

vram_log="$out/vram_usage_${gpu_name}.log"

log "Starting VRAM monitoring for scene: $scene"
rm "$vram_log"
nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv -l 1 >> "$vram_log" &
vram_pid=$!

log "Running ACE-Zero on $scene"
start_time=$(date +%s)

if ! "$PYTHON_BIN" ace_zero.py "$scene/*.$image_format" "$parent_dir/acezero_format" --export_point_cloud True; then
    log "ERROR: ACE-Zero execution failed"
    exit 1
fi

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

log "Pipeline completed in $elapsed_time seconds"
echo "Elapsed time: $elapsed_time seconds" >> "$out/time.txt"

log "Converting to COLMAP format..."
if ! PYTHON_BIN_EVAL convert_to_colmap.py --src_dir "$parent_dir/acezero_format" --dst_dir "$out"; then
    log "ERROR: COLMAP conversion failed"
    exit 1
fi

log "Process finished successfully."
