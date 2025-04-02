#!/bin/bash

#SBATCH --job-name=vggsfm_job
#SBATCH --output=vggsfm_job.out
#SBATCH --error=vggsfm_job.err
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G

# Function to print messages with timestamps
log() {
    local msg="$(date +'%Y-%m-%d %H:%M:%S') - $1"
    echo "$msg"
    if [ -n "$LOG_FILE" ]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

mkdir -p data/results/vggsfm
DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/vggsfm)"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
log "Starting VGG-SfM batch processing GPU: $gpu_name"

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

TANKS_AND_TEMPLES=(
    "Barn" "Caterpillar" "Church" "Courthouse" "Ignatius" "Meetingroom" "Truck"
)

# Verify Conda environment exists
conda_env="vggsfm_tmp"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

PYTHON_BIN="$(conda run -n "$conda_env" which python)"
log "Using Python binary: $PYTHON_BIN"

# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local matcher=$3
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}/colmap/sparse/0"
    local vram_log="${OUT_DIR}/${dataset}/${scene}/vram_usage_${gpu_name}.log"
    LOG_FILE="${OUT_DIR}/${dataset}/${scene}/vggsfm.log"
    rm "$LOG_FILE"
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    echo "==============================================================================" >> "$LOG_FILE"
    log "Processing scene: $scene from $dataset on ${gpu_name}"
    echo "==============================================================================" >> "$LOG_FILE"

    if [ ! -d "$scene_dir" ]; then
        log "ERROR: Scene directory does not exist: $scene_dir"
        return
    fi

    mkdir -p "$out_dir"

    # Monitor VRAM usage during processing every seconds
    log "Starting VRAM monitoring for scene: $scene"
    rm "$vram_log"
    nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv -l 1 >> "$vram_log" &
    vram_pid=$!

    start_time=$(date +%s)
    if [ "${matcher}" = "sequential" ]; then
      log "Running VGG-SfM pipeline with matcher mode on scene: $scene"
        if ! "$PYTHON_BIN" vggsfm/video_demo.py camera_type=SIMPLE_RADIAL SCENE_DIR="$scene_dir" OUTPUT_DIR="$out_dir" 2>&1 | tee -a "$LOG_FILE"; then
            log "ERROR: VGG-SfM pipeline execution failed for scene: $scene"
        fi
    else
      log "Running VGG-SfM pipeline on scene: $scene"
        if ! "$PYTHON_BIN" vggsfm/demo.py camera_type=SIMPLE_RADIAL SCENE_DIR="$scene_dir" OUTPUT_DIR="$out_dir" 2>&1 | tee -a "$LOG_FILE"; then
            log "ERROR: VGG-SfM pipeline execution failed for scene: $scene"
        fi

    fi


    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    # Check if the reconstruction was successful (images.bin or images.txt should be present)
    if [ ! -f "${out_dir}/images.bin" ] && [ ! -f "${out_dir}/images.txt" ]; then
        log "ERROR: VGG-SFM pipeline execution failed for scene: $scene"
    fi

    # Stop VRAM monitoring
    log "Stopping VRAM monitoring for scene: $scene"
    kill $vram_pid

    log "Elapsed time: ${elapsed_time} seconds on ${gpu_name}" >> "${out_dir}/time.txt"
    log "Finished processing scene: $scene in $elapsed_time seconds"
}

dataset_choice="all"
# Parse command-line arguments for --dataset
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset)
            dataset_choice="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Process ETH3D scenes
if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "ETH3D" ] || [ "$dataset_choice" = "eth3d" ]; then
    for SCENE in "${ETH3D_SCENES[@]}"; do
        process_scene "ETH3D" "$SCENE"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "MipNeRF360" ] || [ "$dataset_choice" = "mipnerf360" ] || [ "$dataset_choice" = "mp360" ]; then
    for SCENE in "${MIP_NERF_360_SCENES[@]}"; do
        process_scene "MipNerf360" "$SCENE"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples" ] || [ "$dataset_choice" = "tanksandtemples" ] || [ "$dataset_choice" = "t2" ]; then
    for SCENE in "${TANKS_AND_TEMPLES[@]}"; do
        process_scene "TanksAndTemples" "$SCENE" "sequential"
    done
fi

log "All scenes processed."