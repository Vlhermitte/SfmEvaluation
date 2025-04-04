#!/bin/bash

#SBATCH --job-name=dense_job
#SBATCH --output=dense_job.out
#SBATCH --error=dense_job.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a16:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

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
RESULTS_DIR="$(realpath data/results/)"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "Starting COLMAP dense reconstruction on GPU: $gpu_name"

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
    local dataset="$1"
    local scene="$2"
    local method="$3"
    local image_dir="${DATASETS_DIR}/${dataset}/${scene}/images"
    local sparse_dir="${RESULTS_DIR}/${method}/${dataset}/${scene}"
    local out_dir="${RESULTS_DIR}/${method}/${dataset}/${scene}/dense"
    local vram_log="${out_dir}/vram_usage_dense.log"
    local LOG_FILE="${out_dir}/dense.log"
    rm "$LOG_FILE"
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$out_dir"
    touch "$LOG_FILE"


    echo "==============================================================================" >> "$LOG_FILE"
    log "Processing scene: $scene from $dataset on ${num_gpus} ${gpu_name}"
    echo "==============================================================================" >> "$LOG_FILE"

    if [ ! -d "$image_dir" ]; then
        log "ERROR: Image directory does not exist: $image_dir"
        return
    fi

    # Monitor VRAM usage during processing every seconds
    log "Starting VRAM monitoring for scene: $scene"
    rm "$vram_log"
    nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv -l 1 >> "$vram_log" &
    vram_pid=$!

    start_time=$(date +%s)
    log "Running COLMAP dense reconstruction for scene: $scene"

    log "Image undistortion..."
    colmap image_undistorter \
        --image_path "${image_dir}" \
        --input_path "${sparse_dir}/0" \
        --output_path "${out_dir}" \
        --output_type COLMAP

    log "PatchMatch Stereo..."
    colmap patch_match_stereo \
       --workspace_path "${out_dir}" \
       --workspace_format COLMAP \
       --PatchMatchStereo.geom_consistency true

    log "Stereo Fusion..."
    colmap stereo_fusion \
       --workspace_path "${out_dir}" \
       --workspace_format COLMAP \
       --input_type geometric \
       --output_path "${out_dir}"/fused.ply

    elapsed_time=$((end_time - start_time))

    # Check if fused.ply was created
    if [ ! -f "${out_dir}/fused.ply" ]; then
        log "ERROR: Failed to reconstruct scene: $scene. Fused file not found."
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
        --method)
            method="$2"
            shift 2
            ;;

        --dataset)
            dataset_choice="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--method <method>] [--dataset <dataset>]"
            echo "Options:"
            echo "  --method <method>       Specify the method to use (required)"
            echo "  --dataset <dataset>     Specify the dataset to process (default: all)"
            echo "                          Options: all, ETH3D, MipNeRF360, TanksAndTemples"
            echo "  --help                  Show this help message"
            echo "Example: $0 --method glomap --dataset eth3d"
            exit 0
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Process scenes based on the dataset choice
if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "ETH3D" ] || [ "$dataset_choice" = "eth3d" ]; then
    for SCENE in "${ETH3D_SCENES[@]}"; do
        process_scene "ETH3D" "$SCENE" "$method"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "MipNeRF360" ] || [ "$dataset_choice" = "mipnerf360" ] || [ "$dataset_choice" = "mp360" ]; then
    for SCENE in "${MIP_NERF_360_SCENES[@]}"; do
        process_scene "MipNerf360" "$SCENE" "$method"
    done
fi

if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples" ] || [ "$dataset_choice" = "tanksandtemples" ] || [ "$dataset_choice" = "t2" ]; then
    for SCENE in "${TANKS_AND_TEMPLES[@]}"; do
        process_scene "TanksAndTemples" "$SCENE" "$method"
    done
fi

log "All scenes processed."