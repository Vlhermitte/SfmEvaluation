#!/bin/bash

#SBATCH --job-name=flowmap_ETH3D
#SBATCH --output=flowmap_ETH3D.out
#SBATCH --error=flowmap_ETH3D.err
#SBATCH --time=04:00:00             # Request 4 hours of runtime
#SBATCH --partition=fast            # Use the 'fast' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=48G                   # Request 48 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs

export COLMAP_PATH="/usr/local/bin"
export GOLMAP_PATH="/usr/local/bin"

# Run Glomap on all scene.
SCENES=(
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


# Base output directory
OUT_DIR="data/results/glomap"

# Run the Glomap pipeline for each scene
for SCENE in "${SCENES[@]}"; do
    echo "Processing scene: $SCENE"
    # Base scene directory
    SCENE_DIR="datasets/ETH3D"
    OUTPUT_DIR="${OUT_DIR}/${SCENE}/colmap"

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Skipping scene: $SCENE"
        continue
    fi
    mkdir -p ${OUTPUT_DIR}
    
    DATABASE=${OUTPUT_DIR}/sample_reconstruction.db
    # If database does not exist, create a new database
    if [ ! -f ${DATABASE} ]; then
    ${COLMAP_PATH}/colmap feature_extractor \
        --database_path ${DATABASE} \
        --project_path ${OUTPUT_DIR} \
        --image_path ${SCENE_DIR}/${SCENE}/images \
        --ImageReader.camera_model RADIAL \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1

    ${COLMAP_PATH}/colmap exhaustive_matcher \
        --database_path ${DATABASE} \
        --SiftMatching.use_gpu 1
    fi

    # GLOMAP execution
    mkdir -p ${OUTPUT_DIR}/sparse
    ${GOLMAP_PATH}/glomap mapper \
        --database_path ${DATABASE} \
        --image_path ${SCENE_DIR}/${SCENE}/images \
        --output_path ${OUTPUT_DIR}/sparse
    
    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

echo "All scenes processed."