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
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

matcher=${1:-exhaustive_matcher}

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

mkdir -p data/results/glomap
DATASETS_DIR="$(realpath data/datasets)"
OUT_DIR="$(realpath data/results/glomap)"

# Process each scene
process_scene() {
    local dataset=$1
    local scene=$2
    local scene_dir="${DATASETS_DIR}/${dataset}/${scene}"
    local out_dir="${OUT_DIR}/${dataset}/${scene}/colmap/sparse"
    local database="${OUT_DIR}/${dataset}/${scene}/colmap/sample_reconstruction.db"
    local vram_log="${OUT_DIR}/${dataset}/${scene}/vram_usage.log"

    log "Processing scene: $scene from $dataset"

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
        --SiftExtraction.use_gpu 1

      echo "COLMAP ${matcher}..."
      colmap ${matcher} \
        --database_path ${database} \
        --SiftMatching.use_gpu 1
    fi

    # GLOMAP execution
    mkdir -p "${out_dir}"
    echo "GLOMAP mapper..."
    glomap mapper \
      --database_path ${database} \
      --image_path ${scene_dir}/images \
      --output_path ${out_dir}
    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    # Check if the reconstruction was successful (cameras.bin, images.bin, points3D.bin)
    if [ ! -f "${out_dir}/0/cameras.bin" ] || [ ! -f "${out_dir}/0/images.bin" ] || [ ! -f "${out_dir}/0/points3D.bin" ]; then
        log "ERROR: GLOMAP pipeline execution failed for scene: $scene"
    fi

    # Stop VRAM monitoring
    log "Stopping VRAM monitoring for scene: $scene"
    kill $vram_pid

    echo "Elapsed time: ${elapsed_time} seconds on ${gpu_name}" >> "${out_dir}/time.txt"
    log "Finished processing scene: ${scene} in $elapsed_time seconds"
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