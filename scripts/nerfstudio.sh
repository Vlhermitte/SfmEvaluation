#!/bin/bash
#SBATCH --job-name=nerfstudio_job      # Job name
#SBATCH --output=nerfstudio_out.log    # Standard output log
#SBATCH --error=nerfstudio_err.log     # Standard error log
#SBATCH --time=24:00:00                # Time limit in the format HH:MM:SS
#SBATCH --partition=1day               # Use the '1day' partition
#SBATCH --gres=gpu:a16:1               # Request 1 GPU (a16)
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --cpus-per-task=8              # Number of CPU cores per task

log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

run_pipeline() {
  local dataset_path=$1
  local sparse_path=$2
  local method=$3

  if [ ! -d "$COLMAP_PATH" ]; then
    log "Warning: No images found in ${dataset_path}. Exiting..."
    return
  fi

  log "Running pipeline for ${sparse_path} using ${method} method"
  if [ -n "$SLURM_JOB_ID" ]; then
    apptainer exec --nvccli nerfstudio.sif python src/run_nerfstudio.py \
      --dataset-path "$dataset_path" \
      --results-path "$sparse_path" \
      --method "$method" \
      --viz False
  else
    python src/run_nerfstudio.py \
      --dataset-path "$dataset_path" \
      --results-path "$sparse_path" \
      --method "$method" \
      --viz False
  fi
}

METHOD="nerfacto"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --sparse_path)
            SPARSE_PATH="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

run_pipeline "$DATASET_PATH" "$SPARSE_PATH" "$METHOD"

