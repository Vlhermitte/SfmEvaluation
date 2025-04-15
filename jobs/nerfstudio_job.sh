#!/bin/bash
#SBATCH --job-name=nerfstudio_job
#SBATCH --output=nerfstudio_out.log
#SBATCH --error=nerfstudio_err.log
#SBATCH --time=3-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16

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

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

DATASET_PATH="$(realpath data/datasets)"
RESULTS_PATH="$(realpath data/results)"

run_pipeline() {
  local dataset_name=$1
  local scene=$2
  local sfm_method=$3

  local SCENE_PATH="${DATASET_PATH}/${dataset_name}/${scene}"
  local COLMAP_PATH="${RESULTS_PATH}/${sfm_method}/${dataset_name}/${scene}/colmap/sparse/0"

  LOG_FILE="${RESULTS_PATH}/${sfm_method}/${dataset_name}/${scene}/nerfstudio.log"
  rm "$LOG_FILE"
  mkdir -p "$(dirname "$LOG_FILE")"
  touch "$LOG_FILE"

  if [ ! -d "$COLMAP_PATH" ]; then
    log "Warning: No dataset found for ${COLMAP_PATH}. Skipping..."
    return
  fi

  log "Running pipeline for ${dataset_name}/${scene} using ${sfm_method} method on ${gpu_name}"
  if [ -n "$SLURM_JOB_ID" ]; then
    log "Slurm-managed system. Running with Apptainer..."
    apptainer exec --nv nerfstudio.sif python src/run_nerfstudio.py \
      --dataset-path "$SCENE_PATH" \
      --results-path "$COLMAP_PATH" \
      --method "$METHOD" \
      --viz False >> "$LOG_FILE" 2>&1
  else
    python src/run_nerfstudio.py \
      --dataset-path "$SCENE_PATH" \
      --results-path "$COLMAP_PATH" \
      --method "$METHOD" \
      --viz False >> "$LOG_FILE" 2>&1
  fi
}

sfm="all"
dataset_choice="all"
METHOD="nerfacto"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --sfm)
            sfm="$2"
            shift 2
            ;;
        --dataset)
            dataset_choice="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Process datasets
if [ "$sfm" = "all" ]; then
  for sfm_method in "${SFM_METHODS[@]}"; do
    if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "ETH3D" ] || [ "$dataset_choice" = "eth3d" ]; then
      for scene in "${ETH3D_SCENES[@]}"; do
        run_pipeline "ETH3D" "$scene" "$sfm_method"
      done
    fi

    if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "MipNeRF360" ] || [ "$dataset_choice" = "mipnerf360" ] || [ "$dataset_choice" = "mp360" ]; then
      for scene in "${MIP_NERF_360_SCENES[@]}"; do
        run_pipeline "MipNerf360" "$scene" "$sfm_method"
      done
    fi

    if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples" ] || [ "$dataset_choice" = "tanksandtemples" ] || [ "$dataset_choice" = "t2" ]; then
      for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
        run_pipeline "TanksAndTemples" "$scene" "$sfm_method"
      done
    fi
    if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples_reduced" ] || [ "$dataset_choice" = "tanksandtemples_reduced" ] || [ "$dataset_choice" = "t2_r" ]; then
    for SCENE in "${TANKS_AND_TEMPLES[@]}"; do
        process_scene "TanksAndTemples_reduced" "$SCENE"
    done
  fi
  done
else
  if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "ETH3D" ] || [ "$dataset_choice" = "eth3d" ]; then
    for scene in "${ETH3D_SCENES[@]}"; do
      run_pipeline "ETH3D" "$scene" "$sfm"
    done
  fi

  if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "MipNeRF360" ] || [ "$dataset_choice" = "mipnerf360" ] || [ "$dataset_choice" = "mp360" ]; then
    for scene in "${MIP_NERF_360_SCENES[@]}"; do
      run_pipeline "MipNerf360" "$scene" "$sfm"
    done
  fi

  if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples" ] || [ "$dataset_choice" = "tanksandtemples" ] || [ "$dataset_choice" = "t2" ]; then
    for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
      run_pipeline "TanksAndTemples" "$scene" "$sfm"
    done
  fi
  if [ "$dataset_choice" = "all" ] || [ "$dataset_choice" = "TanksAndTemples_reduced" ] || [ "$dataset_choice" = "tanksandtemples_reduced" ] || [ "$dataset_choice" = "t2_r" ]; then
    for SCENE in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
        run_pipeline "TanksAndTemples_reduced" "$scene" "$sfm"
    done
  fi
fi