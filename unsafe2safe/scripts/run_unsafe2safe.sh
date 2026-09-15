#!/usr/bin/env bash

# Usage: ./unsafe2safe/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT [FILE_COLUMN] [PUBLIC_CAPTION_COLUMN] [EDIT_CAPTION_COLUMN]
FILE_COLUMN="${5:-file}"
PUBLIC_CAPTION_COLUMN="${6:-caption_public}"
EDIT_CAPTION_COLUMN="${7:-caption_edit}"

exec python unsafe2safe/legacy/edit_unsafe2safe_df.py \
  --input "$1" \
  --output "$2" \
  --ckpt "$3" \
  --image-root "$4" \
  --file-column "$FILE_COLUMN" \
  --public-caption-column "$PUBLIC_CAPTION_COLUMN" \
  --edit-caption-column "$EDIT_CAPTION_COLUMN"
