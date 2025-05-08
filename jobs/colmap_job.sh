#!/bin/bash

#SBATCH --job-name=colmap_job
#SBATCH --output=colmap_job.out
#SBATCH --error=colmap_job.err
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
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

mkdir -p data/results/colmap
DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/colmap)"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
log "Starting COLMAP batch processing GPU: $gpu_name"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load COLMAP
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

LAMAR_HGE_SCENES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    ios*) LAMAR_HGE_SCENES+=("$line") ;;
  esac
done < data/datasets/LaMAR/HGE/sessions/map/proc/subsessions.txt

LAMAR_CAB_SCENES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    ios*) LAMAR_CAB_SCENES+=("$line") ;;
  esac
done < data/datasets/LaMAR/CAB/sessions/map/proc/subsessions.txt

LAMAR_LIN_SCENES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    ios*) LAMAR_LIN_SCENES+=("$line") ;;
  esac
done < data/datasets/LaMAR/LIN/sessions/map/proc/subsessions.txt


# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local matcher=$3
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}/colmap/sparse"
    local database="${OUT_DIR}/${dataset}/${scene}/colmap/database.db"
    local vram_log="${OUT_DIR}/${dataset}/${scene}/vram_usage_${gpu_name}.log"
    LOG_FILE="${OUT_DIR}/${dataset}/${scene}/colmap.log"
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
    log "Running COLMAP pipeline on scene: $scene"

    if [ ! -f "${database}" ]; then
      log "Removing existing db"
      rm ${database}
    fi
    log "Creating new database: ${database}"
    rm "$database"
    log "COLMAP feature_extractor..."
    colmap feature_extractor \
      --database_path ${database} \
      --image_path ${scene_dir}/images \
      --ImageReader.camera_model SIMPLE_RADIAL \
      --SiftExtraction.use_gpu 1 2>&1 | tee -a "$LOG_FILE"

    if [ "${matcher}" = "sequential" ]; then
      log "COLMAP sequential_matcher..."
      colmap sequential_matcher \
        --database_path ${database} \
        --SiftMatching.use_gpu 1 2>&1 | tee -a "$LOG_FILE"
    else
      log "COLMAP exhaustive_matcher..."
      colmap exhaustive_matcher \
        --database_path ${database} \
        --SiftMatching.use_gpu 1 2>&1 | tee -a "$LOG_FILE"
    fi

    # COLMAP execution
    mkdir -p "${out_dir}"
    log "COLMAP mapper..."
    colmap mapper \
      --database_path ${database} \
      --image_path ${scene_dir}/images \
      --output_path ${out_dir} 2>&1 | tee -a "$LOG_FILE"
    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    # Check if the reconstruction was successful (cameras.bin, images.bin, points3D.bin)
    if [ ! -f "${out_dir}/0/cameras.bin" ] || [ ! -f "${out_dir}/0/images.bin" ] || [ ! -f "${out_dir}/0/points3D.bin" ]; then
        log "ERROR: COLMAP pipeline execution failed for scene: $scene"
    fi

    # Stop VRAM monitoring
    log "Stopping VRAM monitoring for scene: $scene"
    kill $vram_pid

    log "Elapsed time: ${elapsed_time} seconds on ${gpu_name}" >> "${out_dir}/time.txt"
    log "Finished processing scene: ${scene} in $elapsed_time seconds"
}

# Default parameter values
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

# Process scenes based on the dataset choice
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
        process_scene "TanksAndTemples" "$SCENE"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples_reduced" ] || [ "$dataset_choice" = "tanksandtemples_reduced" ] || [ "$dataset_choice" = "t2_r" ]; then
    for SCENE in "${TANKS_AND_TEMPLES[@]}"; do
        process_scene "TanksAndTemples_reduced" "$SCENE"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "LaMAR_HGE" ] || [ "$dataset_choice" = "lamar_hge" ]; then
    for SCENE in "${LAMAR_HGE_SCENES[@]}"; do
        process_scene "LaMAR/HGE/sessions/map/raw_data" "$SCENE" "sequential"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "LaMAR_CAB" ] || [ "$dataset_choice" = "lamar_cab" ]; then
    for SCENE in "${LAMAR_CAB_SCENES[@]}"; do
        process_scene "LaMAR/CAB/sessions/map/raw_data" "$SCENE" "sequential"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "LaMAR_LIN" ] || [ "$dataset_choice" = "lamar_lin" ]; then
    for SCENE in "${LAMAR_LIN_SCENES[@]}"; do
        process_scene "LaMAR/LIN/sessions/map/raw_data" "$SCENE" "sequential"
    done
fi

log "All scenes processed."