#!/usr/bin/env bash

set -euo pipefail

# Usage: ./pipeline/scripts/train_unsafe2safe.sh DIFFUSION_ROOT CONFIG LOG_DIR GPU_IDS
DIFFUSION_ROOT="$1"
CONFIG="$2"
LOG_DIR="$3"
GPU_IDS="$4"

CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$DIFFUSION_ROOT/main.py" \
  --name pix2pixSAFE \
  --base "$CONFIG" \
  --train \
  --gpus "$GPU_IDS" \
  --accelerator cuda \
  --strategy ddp \
  --logdir "$LOG_DIR"
