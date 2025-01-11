#!/bin/bash

#SBATCH --job-name=flowmap_ETH3D
#SBATCH --output=flowmap_ETH3D.out
#SBATCH --error=flowmap_ETH3D.err
#SBATCH --time=04:00:00             # Request 4 hours of runtime
#SBATCH --partition=fast            # Use the 'fast' partition
#SBATCH --gres=gpu:a16:1            # Request 1 GPU (a16)
#SBATCH --mem=48G                   # Request 48 GB of RAM
#SBATCH --cpus-per-task=12          # Request 12 CPUs

# Go to SfmEvaluation directory
cd /home.nfs/lhermval/SfmEvaluation

source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }

# Check if the 'flowmap' conda environment exists
if conda env list | grep -q '^flowmap'; then
    echo "Activating the existing 'flowmap' environment..."
    conda activate flowmap || { echo "Failed to activate conda environment: flowmap"; exit 1; }
else
    echo "Creating a new 'flowmap' conda environment..."
    conda create -n flowmap python=3.11 -y
    conda activate flowmap

    # Install required dependencies for the flowmap project
    echo "Installing dependencies..."
    cd flowmap
    pip install -r requirements_exact.txt
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

cd /home.nfs/lhermval/SfmEvaluation/flowmap

# Base output directory
OUT_DIR="../results/flowmap"

# Run the FlowMap pipeline for each scene
for SCENE in "${SCENES[@]}"; do
    echo "Processing scene: $SCENE"
    OUTPUT_DIR="${OUT_DIR}/${SCENE}"

    # Check if the output directory already exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory exists. Skipping scene: $SCENE"
        continue
    fi
    
    python3 -m flowmap.overfit dataset=images dataset.images.root=../datasets/ETH3D/"$SCENE"/images output_dir="$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Finished processing scene: $SCENE"
    else
        echo "Error occurred while processing scene: $SCENE"
    fi
done

echo "All scenes processed."


