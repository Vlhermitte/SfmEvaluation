#!/bin/bash

#SBATCH --job-name=flowmap_job
#SBATCH --output=flowmap_job.out
#SBATCH --error=flowmap_job.err
#SBATCH --partition=amdgpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --mail-user=lhermval@cvut.cz
#SBATCH --mail-type=ALL

# Function to print messages with timestamps
log() {
    local msg="$(date +'%Y-%m-%d %H:%M:%S') - $1"
    echo "$msg"
    if [ -n "$LOG_FILE" ]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

mkdir -p data/results/flowmap
DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/flowmap)"
LOG_FILE="${OUT_DIR}/${dataset}/${scene}/flowmap.log"

rm "$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
log "Starting FlowMap batch processing GPU: $gpu_name"

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
conda_env="flowmap"
if ! conda env list | grep -q "$conda_env"; then
    log "ERROR: Conda environment $conda_env not found."
    exit 1
fi

PYTHON_BIN="$(conda run -n "$conda_env" which python)"
log "Using Python binary: $PYTHON_BIN"

# Set PYTHONPATH so Python can find the flowmap module
FLOWMAP_DIR="$(realpath flowmap)"
log "Found FlowMap directory: $FLOWMAP_DIR"
export PYTHONPATH="$FLOWMAP_DIR:$PYTHONPATH"

# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}" # flowmap automatically outputs to colmap/sparse/0
    local vram_log="${OUT_DIR}/${dataset}/${scene}/vram_usage_${gpu_name}.log"

    echo "==============================================================================" >> "$LOG_FILE"
    log "Processing scene: $scene from $dataset"
    echo "==============================================================================" >> "$LOG_FILE"

    if [ ! -d "$scene_dir" ]; then
        log "ERROR: Scene directory does not exist: $scene_dir"
        return
    fi

    mkdir -p "$out_dir"

    # Count the number of images in the scene
    num_images=$(find "$scene_dir/images" -maxdepth 1 -type f | wc -l)
    log "Detected $num_images images in scene: $scene"
    image_format=$(find "$scene_dir/images" -maxdepth 1 -type f | head -n 1 | rev | cut -d'.' -f1 | rev)
    log "Detected image format: .$image_format"

    # Monitor VRAM usage during processing every seconds
    log "Starting VRAM monitoring for scene: $scene"
    rm "$vram_log"
    nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv -l 1 >> "$vram_log" &
    vram_pid=$!

    start_time=$(date +%s)
    log "Running FlowMap pipeline on scene: $scene"
    cd flowmap || { log "ERROR: Failed to change directory to 'flowmap'"; exit 1; }

    # If number of image is less than 150, use the default settings
    if [ "$num_images" -lt 150 ]; then
        if ! "$PYTHON_BIN" -m flowmap.overfit dataset=images dataset.images.root="$scene_dir/images" output_dir="$out_dir" 2>&1 | tee -a "$LOG_FILE"; then
        log "ERROR: FlowMap pipeline execution failed for scene: $scene"
    fi
    else
      log "Running FlowMap pipeline with low memory settings on scene: $scene"
        if ! "$PYTHON_BIN" -m flowmap.overfit dataset=images dataset.images.root="$scene_dir/images" output_dir="$out_dir" +experiment=low_memory 2>&1 | tee -a "$LOG_FILE"; then
            log "ERROR: FlowMap pipeline execution failed for scene: $scene"
        fi
    fi


    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    # Stop VRAM monitoring
    log "Stopping VRAM monitoring for scene: $scene"
    kill $vram_pid

    # Check if the reconstruction was successful (either images.bin or images.txt should be present)
    if [ ! -f "${out_dir}/images.bin" ] && [ ! -f "${out_dir}/images.txt" ]; then
        log "ERROR: FlowMap pipeline execution failed for scene: $scene"
    fi

    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    log "Elapsed time: ${elapsed_time} seconds on ${gpu_name}" >> "${out_dir}/time.txt"
    log "Finished processing scene: $scene in $elapsed_time seconds"
    cd ..
}

dataset_choice="all"
matcher="exhaustive" # Not in use for FlowMap
# Parse command-line arguments for --matcher and --dataset
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --matcher)
            matcher="$2"
            shift 2
            ;;
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

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "MipNeRF360" ] || [ "$dataset_choice" = "mipnerf360" ]; then
    for SCENE in "${MIP_NERF_360_SCENES[@]}"; do
        process_scene "MipNerf360" "$SCENE"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples" ] || [ "$dataset_choice" = "tanksandtemples" ] || [ "$dataset_choice" = "t2" ]; then
    for SCENE in "${TANKS_AND_TEMPLES[@]}"; do
        process_scene "TanksAndTemples" "$SCENE"
    done
fi

log "All scenes processed."