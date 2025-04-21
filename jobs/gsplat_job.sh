#!/bin/bash
#SBATCH --job-name=gsplat_job
#SBATCH --output=gsplat_job_out.log
#SBATCH --error=gsplat_job_err.log
#SBATCH --time=3-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16

if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
    module load CUDA/12.4.0
    module load PyTorch/2.5.1-foss-2023b-CUDA-12.4.0
    module load torchvision/0.20.1-foss-2023b-CUDA-12.4.0
    module load torchaudio/2.5.1-foss-2023b-CUDA-12.4.0
    source $(conda info --base)/etc/profile.d/conda.sh
fi

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

    local IMAGES_PATH="${DATASET_PATH}/${dataset_name}/${scene}/images"
    local COLMAP_PATH="${RESULTS_PATH}/${sfm_method}/${dataset_name}/${scene}/colmap/"

    LOG_FILE="${RESULTS_PATH}/${sfm_method}/${dataset_name}/${scene}/gsplat.log"
    rm "$LOG_FILE"
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    if [ ! -d "$COLMAP_PATH/sparse/0" ]; then
      log "Warning: No dataset found for ${COLMAP_PATH}. Skipping..."
      return
    fi

    log "Running pipeline for ${dataset_name}/${scene} using ${sfm_method} method on ${gpu_name}"
    cmd_args=(
        "src/run_gsplat.py"
        "--dataset-path" "$COLMAP_PATH"
        "--images-path" "$IMAGES_PATH"
        "--results-path" "gsplat_pose_opt"
        "--pose-opt"
    )

    log "Executing: python ${cmd_args[*]}"
    python "${cmd_args[@]}"

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
      for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
          run_pipeline "TanksAndTemples_reduced" "$scene" "$sfm"
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
    for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do
        run_pipeline "TanksAndTemples_reduced" "$scene" "$sfm"
    done
  fi
fi