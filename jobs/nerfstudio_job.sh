#!/bin/bash
#SBATCH --job-name=nerfstudio_job      # Job name
#SBATCH --output=nerfstudio_out.log    # Standard output log
#SBATCH --error=nerfstudio_err.log     # Standard error log
#SBATCH --time=16:00:00                # Time limit in the format HH:MM:SS
#SBATCH --partition=1day               # Use the '1day' partition
#SBATCH --gres=gpu:a16:1               # Request 1 GPU (a16)
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --cpus-per-task=8              # Number of CPU cores per task

log() {
    local msg="$(date +'%Y-%m-%d %H:%M:%S') - $1"
    echo "$msg"
    if [ -n "$LOG_FILE" ]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

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

MIP_NERF_360_SCENES=(
  "bicycle"
  "bonsai"
  "counter"
  "garden"
  "kitchen"
  "room"
  "stump"
)

TANKS_AND_TEMPLES_SCENES=(
    "Barn"
    "Caterpillar"
    "Church"
    "Courthouse"
    "Ignatius"
    "Meetingroom"
    "Truck"
)

SFM_METHODS=(
    "vggsfm"
    "flowmap"
    "acezero"
    "glomap"
)

METHOD=${1:-"nerfacto"}

# Determine paths based on SLURM context
if [ -n "$SLURM_JOB_ID" ]; then
  cd ~/SfmEvaluation
  DATASET_PATH=~/SfmEvaluation/data/datasets
  RESULTS_PATH=~/SfmEvaluation/data/results
else
  cd "$(dirname "$0")/.."
  DATASET_PATH=data/datasets
  RESULTS_PATH=data/results
fi

run_pipeline() {
  local dataset_name=$1
  local scene=$2
  local sfm_method=$3

  local SCENE_PATH="${DATASET_PATH}/${dataset_name}/${scene}"
  local COLMAP_PATH="${RESULTS_PATH}/${sfm_method}/${dataset_name}/${scene}/colmap/sparse/0"

  if [ ! -d "$COLMAP_PATH" ]; then
    log "Warning: No dataset found for ${COLMAP_PATH}. Skipping..."
    return
  fi

  log "Running pipeline for ${dataset_name}/${scene} using ${sfm_method} method"
  if [ -n "$SLURM_JOB_ID" ]; then
    apptainer exec --nvccli nerfstudio.sif python src/run_nerfstudio.py \
      --dataset-path "$SCENE_PATH" \
      --results-path "$COLMAP_PATH" \
      --method "$METHOD" \
      --viz False
  else
    python src/run_nerfstudio.py \
      --dataset-path "$SCENE_PATH" \
      --results-path "$COLMAP_PATH" \
      --method "$METHOD" \
      --viz False
  fi
}

# Process datasets
for sfm_method in "${SFM_METHODS[@]}"; do
  for scene in "${ETH3D_SCENES[@]}"; do
    run_pipeline "ETH3D" "$scene" "$sfm_method"
  done

  for scene in "${MIP_NERF_360_SCENES[@]}"; do
    run_pipeline "MipNerf360" "$scene" "$sfm_method"
  done

  for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
    run_pipeline "TanksAndTemples" "$scene" "$sfm_method"
  done

done