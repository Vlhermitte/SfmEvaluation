#!/bin/bash
#SBATCH --job-name=gsplat_script
#SBATCH --output=gsplat_script.log
#SBATCH --error=gsplat_script.log
#SBATCH --time=24:00:00
#SBATCH --partition=1day
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12

log() {
    local msg="$(date +'%Y-%m-%d %H:%M:%S') - $1"
    echo "$msg"
    if [ -n "$LOG_FILE" ]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "Running on a Slurm-managed system. Loading required modules..."
    module load Anaconda3 || { log "ERROR: Failed to load Anaconda3 module"; exit 1; }
    module load CUDA/12.4.0
    module load PyTorch/2.5.1-foss-2023b-CUDA-12.4.0
    module load torchvision/0.20.1-foss-2023b-CUDA-12.4.0
    module load torchaudio/2.5.1-foss-2023b-CUDA-12.4.0
    source $(conda info --base)/etc/profile.d/conda.sh
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --colmap-path)
            COLMAP_PATH="$2"
            shift 2
            ;;
        --images-path)
            IMAGES_PATH="$2"
            shift 2
            ;;
        --results-path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --pose-opt)
            POSE_OPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$COLMAP_PATH" ]; then
    log "ERROR: --colmap-path is required"
    exit 1
fi

if [ -z "$IMAGES_PATH" ]; then
    log "ERROR: --images-path is required"
    exit 1
fi

run_pipeline() {
  conda activate gsplat || { log "ERROR: Failed to activate conda environment"; exit 1; }

  local colmap_path=$1
  local images_path=$2
  local results_path=$3

  LOG_FILE="$(dirname ${colmap_path})/gsplat.log"
  rm "$LOG_FILE"
  mkdir -p "$(dirname "$LOG_FILE")"
  touch "$LOG_FILE"

  log "Running gsplat pipeline for ${colmap_path} and ${images_path}"
  # Base arguments for the python script
  cmd_args=(
      "src/run_gsplat.py"
      "--dataset-path" "$colmap_path"
      "--images-path" "$images_path"
  )

  # Conditionally add the pose optimization flag
  if [ -n "$POSE_OPT" ]; then
      log "Pose optimization is enabled"
      cmd_args+=("--pose-opt")
  else
      log "Pose optimization is disabled"
  fi

  # Add the results path if provided
  if [ -n "$results_path" ]; then
      log "Results will be saved to ${results_path}"
      cmd_args+=("--results-path" "$results_path")
  else
      log "No results path provided, using default"
  fi

  # Execute the command
  log "Executing: python ${cmd_args[*]}"
  python "${cmd_args[@]}" >> "$LOG_FILE" 2>&1

}

log "Starting gsplat pipeline script"
run_pipeline "$COLMAP_PATH" "$IMAGES_PATH" "$RESULTS_PATH"
log "Pipeline completed successfully"