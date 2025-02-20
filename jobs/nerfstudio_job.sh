#!/bin/bash
#SBATCH --job-name=nerfstudio_job      # Job name
#SBATCH --output=nerfstudio_output.log # Standard output log
#SBATCH --error=nerfstudio_error.log   # Standard error log
#SBATCH --time=04:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=fast               # Use the 'fast' partition
#SBATCH --gres=gpu:a16:1               # Request 1 GPU (a16)
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --cpus-per-task=8              # Number of CPU cores per task

# Command-line arguments
DATASET_PATH=$1
RESULTS_PATH=$2
METHOD=${3:-nerfacto}     # Default method is 'nerfacto'

# Make sure we are in project's root directory
cd "$(dirname "$0")/.." || exit

# Run the Apptainer container with Nerfstudio
apptainer exec --nvccli nerstudio.sif python src/run_nerfstudio.py \
    --dataset-path "$DATASET_PATH" \
    --results-path "$RESULTS_PATH" \
    --method "$METHOD"

