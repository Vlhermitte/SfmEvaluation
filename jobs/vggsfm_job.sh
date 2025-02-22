#!/bin/bash

#SBATCH --job-name=vggsfm_ETH3D
#SBATCH --output=vggsfm_ETH3D.out
#SBATCH --error=vggsfm_ETH3D.err
#SBATCH --time=08:00:00             # Request 8 hours of runtime
#SBATCH --partition=1day            # Use the '1day' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=32G                   # Request 32 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs

# Go to SfmEvaluation directory
cd /home.nfs/lhermval/SfmEvaluation

source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }

# Check if the 'flowmap' conda environment exists
if conda env list | grep -q '^vggsfm_tmp'; then
    echo "Activating the existing 'vggsfm_tmp' environment..."
    conda activate vggsfm_tmp || { echo "Failed to activate conda environment: vggsfm_tmp"; exit 1; }
else
    echo "Creating a new 'vggsfm_tmp' conda environment..."
    cd /home.nfs/lhermval/SfmEvaluation/vggsfm
    source install.sh
    python -m pip install -e .
    cd /home.nfs/lhermval/SfmEvaluation/
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

cd /home.nfs/lhermval/SfmEvaluation

# Base output directory
OUT_DIR="data/results/vggsfm"

# Run the VGGSfm pipeline for each scene
for SCENE in "${SCENES[@]}"; do
    echo "Processing scene: $SCENE"
    OUTPUT_DIR="${OUT_DIR}/${SCENE}"

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Overwritting scene: $SCENE"
    fi

    python ./vggsfm/demo.py \
      query_method=sp+aliked camera_type=SIMPLE_RADIAL \
      SCENE_DIR=datasets/ETH3D/$SCENE/ \  # vggsfm assume image to be an images directory
      OUTPUT_DIR=$OUTPUT_DIR/colmap/

    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

echo "All scenes processed."


