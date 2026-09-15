#!/usr/bin/env bash

set -euo pipefail

# Usage: ./pipeline/scripts/train_unsafe2safe.sh DIFFUSION_ROOT CONFIG LOG_DIR GPU_IDS
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DIFFUSION_ROOT=$(cd -- "$1" && pwd)
CONFIG_PATH="$2"
LOG_DIR="$3"
GPU_IDS="$4"

# The external checkout stays vanilla. Run its trainer from the project root
# so relative data, metadata, checkpoint, and log paths are predictable.
cd -- "$REPO_ROOT"
INSTRUCT_PIX2PIX_ROOT="$DIFFUSION_ROOT" \
PYTHONPATH="$REPO_ROOT:$DIFFUSION_ROOT/stable_diffusion:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$DIFFUSION_ROOT/main.py" \
  --name pix2pixSAFE \
  --base "$CONFIG_PATH" \
  --train \
  --gpus "$GPU_IDS" \
  --accelerator cuda \
  --strategy ddp \
  --logdir "$LOG_DIR"
