#!/bin/bash
#SBATCH --job-name=gaussiancar-install
#SBATCH --partition=H200
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_NAME=gaussiancar
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh

if [ ! -f "$OUTPUT_SQSH" ]; then
  enroot import \
    -o "$OUTPUT_SQSH" \
    docker://santimontiel/${REPO_NAME}:v1
else
  echo "Found $OUTPUT_SQSH, skipping import."
fi

echo "SBATCH cpus-per-task: $SLURM_CPUS_PER_TASK"


srun \
  --gpus=1 \
  --container-image="$OUTPUT_SQSH" \
  --container-mounts="$PATH_TO_SOURCE_CODE:/workspace,/raid/smontiel/Datasets/nuscenes:/data/nuscenes" \
  bash -c '
    ls -l /workspace
    cd /workspace
    echo "🔄 Checking virtual environment..."
    if [ ! -d ".venv" ]; then
      echo " .venv not found — updating virtual environment..."

      # Run uv sync and capture output
      if ! uv sync --link-mode=copy; then
        echo "❌ uv sync failed, cleaning up..."
        rm -rf .venv
        exit 1
      fi

      echo "✅ .venv updated"
    else
      echo "✅ .venv found"
    fi
    echo "Installation completed!"
  '
