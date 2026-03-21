#!/bin/bash
#SBATCH --job-name=gaussiancar-train
#SBATCH --partition=H200
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=96
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=gaussiancar
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh

if [ ! -f "$OUTPUT_SQSH" ]; then
  echo "Error: $OUTPUT_SQSH not found. Please create the container image first."
  exit 1
else
  echo "Found squashfile at $OUTPUT_SQSH."
fi

if [ ! -d "$PATH_TO_SOURCE_CODE/.venv" ]; then
  echo "Error: Virtual environment not found at $PATH_TO_SOURCE_CODE/.venv. Please run the installation script first."
  exit 1
else
  echo "Found virtual environment in source code directory."
fi


srun \
  --gpus=2 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/workspace,/raid/smontiel/Datasets/nuscenes:/data/nuscenes" \
  --container-env=WANDB_API_KEY \
  bash -c '
    cd /workspace
    uv run tools/train.py
    echo "Training completed"
  '