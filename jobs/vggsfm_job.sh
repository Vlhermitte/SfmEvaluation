#!/bin/bash

#SBATCH --job-name=vggsfm_job
#SBATCH --output=vggsfm_job.out
#SBATCH --error=vggsfm_job.err

# Function to print messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting VGG-SfM batch processing"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
    source $(conda info --base)/etc/profile.d/conda.sh
    module unload SciPy-bundle
fi

# Define datasets
ETH3D_SCENES=(
    "courtyard" "delivery_area" "electro" "facade" "kicker" "meadow"
    "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains"
)

MIP_NERF_360_SCENES=(
    "bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump"
)

DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/vggsfm)"

# Verify Conda environment exists
conda_env="vggsfm_tmp"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}/colmap/sparse/0"

    log "Processing scene: $scene from $dataset"

    if [ ! -d "$scene_dir" ]; then
        log "ERROR: Scene directory does not exist: $scene_dir"
        return
    fi

    mkdir -p "$out_dir"

    start_time=$(date +%s)
    log "Running VGG-SfM pipeline on scene: $scene"
    if ! conda run -n "$conda_env" python vggsfm/demo.py camera_type=SIMPLE_RADIAL SCENE_DIR="$scene_dir" OUTPUT_DIR="$out_dir"; then
        log "ERROR: VGG-SfM pipeline execution failed for scene: $scene"
        return
    fi
    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))
    echo "Elapsed time: $elapsed_time seconds" >> "$out_dir/time.txt"
    log "Finished processing scene: $scene in $elapsed_time seconds"
}

# Process ETH3D scenes
for SCENE in "${ETH3D_SCENES[@]}"; do
    process_scene "ETH3D" "$SCENE"
done

# Process MipNeRF360 scenes
for SCENE in "${MIP_NERF_360_SCENES[@]}"; do
    process_scene "MipNerf360" "$SCENE"
done

log "All scenes processed."