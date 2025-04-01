#!/bin/bash

#SBATCH --job-name=glomap_job
#SBATCH --output=glomap_job.out
#SBATCH --error=glomap_job.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
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

mkdir -p data/results/glomap
DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/glomap)"

LOG_FILE="${OUT_DIR}/${dataset}/${scene}/glomap.log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
log "Starting GLOMAP batch processing GPU: $gpu_name"

# Ensure SLURM environment loads required modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load GLOMAP
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
    "Barn" #"Caterpillar" "Church" "Courthouse" "Ignatius" "Meetingroom" "Truck"
)


#LAMAR_HGE_SCENES=(
#    "ios_2022-01-12_14.59.02_000" "ios_2022-01-12_15.15.53_000" "ios_2022-01-18_17.05.03_000"
#    "ios_2022-01-18_17.10.39_000" "ios_2022-01-20_16.52.33_001" "ios_2022-01-25_14.34.24_002"
#    "ios_2022-01-25_14.57.49_000" "ios_2022-01-25_15.13.54_000" "ios_2022-01-25_15.13.54_002"
#    "ios_2022-06-13_10.45.07_000" "ios_2022-06-13_15.59.36_000" "ios_2022-06-14_17.12.28_000"
#    "ios_2022-06-30_15.55.53_000" "ios_2022-07-01_15.18.09_000" "ios_2022-07-01_15.45.08_000"
#    "ios_2022-07-01_15.58.10_000" "ios_2022-07-03_16.00.37_000"
#)


# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}/colmap/sparse"
    local database="${OUT_DIR}/${dataset}/${scene}/colmap/sample_reconstruction.db"
    local vram_log="${OUT_DIR}/${dataset}/${scene}/vram_usage.log"

    echo "==============================================================================" >> "$LOG_FILE"
    log "Processing scene: $scene from $dataset"
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
    log "Running GLOMAP pipeline on scene: $scene"

    # If database does not exist, create a new database
    if [ ! -f "${database}" ]; then
      echo "Creating new database: ${database}"
      echo "COLMAP feature_extractor..."
      colmap feature_extractor \
        --database_path ${database} \
        --image_path ${scene_dir}/images \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --SiftExtraction.use_gpu 1 2>&1 | tee -a "$LOG_FILE"

      echo "COLMAP ${matcher}..."
      colmap ${matcher} \
        --database_path ${database} \
        --SiftMatching.use_gpu 1 2>&1 | tee -a "$LOG_FILE"
    fi

    # GLOMAP execution
    mkdir -p "${out_dir}"
    echo "GLOMAP mapper..."
    glomap mapper \
      --database_path ${database} \
      --image_path ${scene_dir}/images \
      --output_path ${out_dir} 2>&1 | tee -a "$LOG_FILE"
    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    # Check if the reconstruction was successful (cameras.bin, images.bin, points3D.bin)
    if [ ! -f "${out_dir}/0/cameras.bin" ] || [ ! -f "${out_dir}/0/images.bin" ] || [ ! -f "${out_dir}/0/points3D.bin" ]; then
        log "ERROR: GLOMAP pipeline execution failed for scene: $scene"
    fi

    # Stop VRAM monitoring
    log "Stopping VRAM monitoring for scene: $scene"
    kill $vram_pid

    log "Elapsed time: ${elapsed_time} seconds on ${gpu_name}" >> "${out_dir}/time.txt"
    log "Finished processing scene: ${scene} in $elapsed_time seconds"
}

# Default parameter values
matcher="exhaustive_matcher"
dataset_choice="all"

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

# Process scenes based on the dataset choice
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