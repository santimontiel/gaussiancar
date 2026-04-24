#!/bin/bash
echo "Running on $(hostname)"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
 
REPO_NAME=gaussiancar
PATH_TO_SOURCE_CODE=/raid/${USER}/Workspace/${REPO_NAME}
OUTPUT_SQSH=/raid/${USER}/enroot/sqsh/${REPO_NAME}_v1.sqsh
 
MOUNTS="${PATH_TO_SOURCE_CODE}:/workspace"
MOUNTS="${MOUNTS},/raid/smontiel/Datasets/nuscenes:/data/nuscenes"
MOUNTS="${MOUNTS},/dev/dri:/dev/dri"
 
srun \
    --partition=H200 \
    --gres=gpu:h200:1 \
    --mem=256G \
    --cpus-per-task=16 \
    --time=23:59:59 \
    --job-name=gaussiancar-terminal \
    --container-image="$OUTPUT_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --pty bash -c "/workspace/deploy/docker/entrypoint.sh && bash"