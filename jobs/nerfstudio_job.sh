#!/bin/bash
#SBATCH --job-name=nerfstudio_job      # Job name
#SBATCH --output=nerfstudio_out.log    # Standard output log
#SBATCH --error=nerfstudio_err.log     # Standard error log
#SBATCH --time=12:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=1day               # Use the '1day' partition
#SBATCH --gres=gpu:a16:1               # Request 1 GPU (a16)
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --cpus-per-task=8              # Number of CPU cores per task

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

SFM_METHODS=(
    "vggsfm"
    "flowmap"
    "acezero"
    "glomap"
)

DATASET_PATH="data/datasets"
RESULTS_PATH="data/results"
METHOD="nerfacto"

# ETH3D scenes
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${ETH3D_SCENES[@]}"; do
    SCENE_PATH="${DATASET_PATH}/ETH3D/${scene}"
    COLMAP_PATH="${RESULTS_PATH}/${sfm_method}/ETH3D/${scene}/colmap/sparse/0"

    if [ ! -d ${COLMAP_PATH} ]; then
      echo "Warning: No dataset found for ${COLMAP_PATH}. Skipping..."
      continue
    fi

    if [ ! -z "$SLURM_JOB_ID" ]; then
      # Run the Apptainer container with Nerfstudio
      apptainer exec --nvccli nerstudio.sif python src/run_nerfstudio.py \
          --dataset-path ${SCENE_PATH} \
          --results-path ${COLMAP_PATH} \
          --method ${METHOD}
    else
      # Run the Nerfstudio pipeline directly
      python src/run_nerfstudio.py \
          --dataset-path ${SCENE_PATH} \
          --results-path ${COLMAP_PATH} \
          --method ${METHOD}
    fi
  done
done

# MipNerf360 scenes
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${MIP_NERF_360_SCENE[@]}"; do
    SCENE_PATH="${DATASET_PATH}/MipNerf360/${scene}"
    COLMAP_PATH="${RESULTS_PATH}/${sfm_method}/MipNerf360/${scene}/colmap/sparse/0"

    if [ ! -d ${COLMAP_PATH} ]; then
      echo "Warning: No dataset found for ${COLMAP_PATH}. Skipping..."
      continue
    fi

    if [ ! -z "$SLURM_JOB_ID" ]; then
        # Run the Apptainer container with Nerfstudio
        apptainer exec --nvccli nerstudio.sif python src/run_nerfstudio.py \
            --dataset-path ${SCENE_PATH} \
            --results-path ${COLMAP_PATH} \
            --method ${METHOD}
      else
        # Run the Nerfstudio pipeline directly
        python src/run_nerfstudio.py \
            --dataset-path ${SCENE_PATH} \
            --results-path ${COLMAP_PATH} \
            --method ${METHOD}
      fi
  done
done