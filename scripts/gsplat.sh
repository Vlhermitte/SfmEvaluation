#!/bin/bash
#SBATCH --job-name=gsplat_job
#SBATCH --output=gsplat_job.log
#SBATCH --error=gsplat_job.log
#SBATCH --time=24:00:00
#SBATCH --partition=1day
#SBATCH --gres=gpu:a16:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12

log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
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

  log "Running gsplat pipeline for ${COLMAP_PATH} and ${IMAGES_PATH}"
  # Base arguments for the python script
  cmd_args=(
      "src/run_gsplat.py"
      "--dataset-path" "$COLMAP_PATH"
      "--images-path" "$IMAGES_PATH"
  )

  # Conditionally add the pose optimization flag
  if [ -n "$POSE_OPT" ]; then
      log "Pose optimization is enabled"
      cmd_args+=("--pose-opt")
  else
      log "Pose optimization is disabled"
  fi

  # Execute the command
  log "Executing: python ${cmd_args[*]}"
  python "${cmd_args[@]}"

}

run_pipeline