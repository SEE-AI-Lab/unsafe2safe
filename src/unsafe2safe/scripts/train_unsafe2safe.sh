#!/usr/bin/env bash

set -euo pipefail

# Usage: ./src/unsafe2safe/scripts/train_unsafe2safe.sh DIFFUSION_ROOT CONFIG LOG_DIR GPU_IDS
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DIFFUSION_ROOT=$(cd -- "$1" && pwd)
CONFIG_PATH="$2"
if [[ "$CONFIG_PATH" != /* && ! -f "$CONFIG_PATH" && -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi
CONFIG=$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)/$(basename -- "$CONFIG_PATH")
LOG_DIR=$(mkdir -p "$3" && cd -- "$3" && pwd)
GPU_IDS="$4"

# The external checkout stays vanilla. Run its trainer from the project root
# so relative data, metadata, checkpoint, and log paths are predictable.
cd -- "$REPO_ROOT"
INSTRUCT_PIX2PIX_ROOT="$DIFFUSION_ROOT" \
PYTHONPATH="$REPO_ROOT/src:$DIFFUSION_ROOT/stable_diffusion:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$DIFFUSION_ROOT/main.py" \
  --name pix2pixSAFE \
  --base "$CONFIG" \
  --train \
  --gpus "$GPU_IDS" \
  --accelerator cuda \
  --strategy ddp \
  --logdir "$LOG_DIR"
