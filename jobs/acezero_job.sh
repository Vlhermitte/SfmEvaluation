#!/bin/bash

#SBATCH --job-name=acezero_ETH3D
#SBATCH --output=acezero_ETH3D.out
#SBATCH --error=acezero_ETH3D.err
#SBATCH --time=08:00:00             # Request 8 hours of runtime
#SBATCH --partition=1day            # Use the '1day' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=32G                   # Request 32 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs

# Go to SfmEvaluation directory
cd /home.nfs/lhermval/SfmEvaluation

source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }

# Check if the 'flowmap' conda environment exists
if conda env list | grep -q '^ace0'; then
    echo "Activating the existing 'ace0' environment..."
    conda activate ace0 || { echo "Failed to activate conda environment: ace0"; exit 1; }
else
    echo "Creating a new 'ace0' conda environment..."
    cd acezero
    conda create -n ace0 -f environment.yml
    conda activate ace0
    cd ..
fi

# Run flowmap on all scene.
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

cd /home.nfs/lhermval/SfmEvaluation/acezero

# Base output directory
OUT_DIR="../data/results/acezero"

# Run the FlowMap pipeline for each scene
for SCENE in "${SCENES[@]}"; do
    echo "Processing scene: $SCENE"
    OUTPUT_DIR="${OUT_DIR}/${SCENE}/acezero_format"

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    fi

    # Check image format in the scene directory (png, jpg, JPG, etc.)
    image_format=$(ls ../data/datasets/ETH3D/$SCENE/images | head -n 1 | rev | cut -d'.' -f1 | rev)

    python ace_zero.py "../data/datasets/ETH3D/$SCENE/images/*.$image_format" $OUTPUT_DIR --export_point_cloud True
    python convert_to_colmap.py --src_dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

echo "All scenes processed."


