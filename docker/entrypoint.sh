#!/bin/bash
# WBC-AGILE Docker Entrypoint
# Usage:
#   No args          → 8-GPU distributed training (torchrun, gradient sync)
#   --task X ...     → single-GPU training with given args
#   bash             → drop into shell
set -e

# Source Isaac Sim python environment
source /isaac-sim/setup_python_env.sh

# Remove ml_archive and pip_prebundle paths that contain stale torch/torchvision
PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'ml_archive' | tr '\n' ':')
export PYTHONPATH

PYTHON=/isaac-sim/kit/python/bin/python3.11
WORKDIR=/workspace/WBC-AGILE
cd "$WORKDIR"

# If first arg is "bash" or "sh", drop into a shell
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

# If args are provided (e.g. --task ...), run single-process training
if [ $# -gt 0 ]; then
    echo "=========================================="
    echo " WBC-AGILE Single-Process Training"
    echo " Args: $@"
    echo "=========================================="
    exec $PYTHON scripts/train.py "$@" --headless
fi

# --- Default: Multi-GPU distributed training ---
TASK="${TASK:-Velocity-G1-History-v0}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
LOGGER="${LOGGER:-tensorboard}"

# Build extra CLI args
EXTRA_ARGS=""
if [ "$LOGGER" != "tensorboard" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --logger ${LOGGER}"
fi

# Detect available GPUs
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "[WARN] Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available. Using ${AVAILABLE_GPUS}."
    NUM_GPUS=$AVAILABLE_GPUS
fi

mkdir -p logs

echo "=========================================="
echo " WBC-AGILE ${NUM_GPUS}-GPU Distributed Training"
echo "=========================================="
echo " Task:           ${TASK}"
echo " Envs per GPU:   ${NUM_ENVS}"
echo " Total envs:     $((NUM_ENVS * NUM_GPUS))"
echo " Max iterations: ${MAX_ITERATIONS}"
echo " Seed:           ${SEED}"
echo " Logger:         ${LOGGER}"
echo " Mode:           torchrun (NCCL gradient sync)"
echo "=========================================="

# torchrun sets WORLD_SIZE, RANK, LOCAL_RANK automatically.
# The rsl_rl OnPolicyRunner reads these to enable:
#   - broadcast_parameters() at start (sync weights from rank 0)
#   - reduce_parameters() per mini-batch (all_reduce gradients)
#   - only rank 0 writes logs/wandb (disable_logs for rank != 0)
exec $PYTHON -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless \
    ${EXTRA_ARGS} \
    2>&1 | tee logs/train.log
