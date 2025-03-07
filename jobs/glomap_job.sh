#!/bin/bash

#SBATCH --job-name=glomap_job
#SBATCH --output=glomap_job.out
#SBATCH --error=glomap_job.err
#SBATCH --time=04:00:00             # Request 4 hours of runtime
#SBATCH --partition=fast            # Use the 'fast' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=48G                   # Request 48 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs

if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running on a Slurm-managed system. Loading required modules..."
    module load CMake
    module load GCC
    module load Ninja
    module load Eigen
    module load FLANN
    module load COLMAP
    export PATH=~/SfmEvaluation/glomap/build/glomap/:$PATH
fi

# Verify COLMAP and GLOMAP are executable
if ! command -v colmap &> /dev/null; then
    echo "COLMAP is not installed. Please install it from https://colmap.github.io/ or check your PATH"
    exit 1
fi

if ! command -v glomap &> /dev/null; then
    echo "GLOMAP is not installed. Please build from source : https://github.com/colmap/glomap or check your PATH"
    exit 1
fi

# Run Glomap on all scene.
ETH3D_SCENES=(
    "courtyard"
    "delivery_area"
    "electro"
    "facade"
    "kicker"
    "meadow"
    "office"
    "pipes"
    "playground"
    "relief"
    "relief_2"
    "terrace"
    "terrains"
)

MIP_NERF_360_SCENE=(
  "bicycle"
  "bonsai"
  "counter"
  "garden"
  "kitchen"
  "room"
  "stump"
)


DATASETS_DIR="data/datasets"
OUT_DIR="data/results/glomap"

# ETH3D
for SCENE in "${ETH3D_SCENES[@]}"; do
    echo "Processing scene: $SCENE"
    # Base scene directory
    SCENE_DIR="${DATASETS_DIR}/ETH3D/${SCENE}/images"
    OUTPUT_DIR="${OUT_DIR}/ETH3D/${SCENE}/colmap"

    # Check if scene_dir exists
    if [ ! -d "$SCENE_DIR" ]; then
        echo "Scene directory does not exist. Skipping scene: $SCENE"
    fi

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Overwriting scene: $SCENE"
        rm -rf ${OUTPUT_DIR}
    fi
    mkdir -p ${OUTPUT_DIR}

    start_time=$(date +%s)
    
    DATABASE=${OUTPUT_DIR}/sample_reconstruction.db
    # If database does not exist, create a new database
    if [ ! -f ${DATABASE} ]; then
    echo "Feature extraction"
    colmap feature_extractor \
        --database_path ${DATABASE} \
        --image_path ${SCENE_DIR} \
        --ImageReader.camera_model RADIAL \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1

    echo "Feature matching"
    colmap exhaustive_matcher \
        --database_path ${DATABASE} \
        --SiftMatching.use_gpu 1
    fi

    # GLOMAP execution
    mkdir -p ${OUTPUT_DIR}/sparse
    echo "GLOMAP mapper"
    glomap mapper \
        --database_path ${DATABASE} \
        --image_path ${SCENE_DIR} \
        --output_path ${OUTPUT_DIR}/sparse

    end_time=$(date +%s)
    elapsed_time=$(( end_time - start_time ))
    echo "Elapsed time: $elapsed_time seconds" >> ${OUTPUT_DIR}/sparse/0/time.txt
    
    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

# MIP_NERF_360
for SCENE in "${MIP_NERF_360_SCENE[@]}"; do
    echo "Processing scene: $SCENE"
    # Base scene directory
    SCENE_DIR="${DATASETS_DIR}/MipNerf360/${SCENE}/images"
    OUTPUT_DIR="${OUT_DIR}/MipNerf360/${SCENE}/colmap"

    # Check if scene_dir exists
    if [ ! -d "$SCENE_DIR" ]; then
        echo "Scene directory does not exist. Skipping scene: $SCENE"
    fi

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Overwriting scene: $SCENE"
        rm -rf ${OUTPUT_DIR}
    fi
    mkdir -p ${OUTPUT_DIR}

    start_time=$(date +%s)

    DATABASE=${OUTPUT_DIR}/sample_reconstruction.db
    # If database does not exist, create a new database
    if [ ! -f ${DATABASE} ]; then
    colmap feature_extractor \
        --database_path ${DATABASE} \
        --image_path ${SCENE_DIR} \
        --ImageReader.camera_model RADIAL \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1

    colmap exhaustive_matcher \
        --database_path ${DATABASE} \
        --SiftMatching.use_gpu 1
    fi

    # GLOMAP execution
    mkdir -p ${OUTPUT_DIR}/sparse
    glomap mapper \
        --database_path ${DATABASE} \
        --image_path ${SCENE_DIR} \
        --output_path ${OUTPUT_DIR}/sparse

    end_time=$(date +%s)
    elapsed_time=$(( end_time - start_time ))
    echo "Elapsed time: $elapsed_time seconds" >> ${OUTPUT_DIR}/sparse/0/time.txt

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done