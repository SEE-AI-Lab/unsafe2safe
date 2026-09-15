#!/usr/bin/env bash

# Usage: ./pipeline/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT
python pipeline/edit_unsafe2safe_df.py \
  --input "$1" \
  --output "$2" \
  --ckpt "$3" \
  --image-root "$4"
