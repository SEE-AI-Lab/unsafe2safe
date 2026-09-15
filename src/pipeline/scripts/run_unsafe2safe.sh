#!/usr/bin/env bash
set -euo pipefail

# Usage: ./src/pipeline/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT [FILE_COLUMN] [PUBLIC_CAPTION_COLUMN] [EDIT_CAPTION_COLUMN]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
FILE_COLUMN="${5:-file}"
PUBLIC_CAPTION_COLUMN="${6:-caption_public}"
EDIT_CAPTION_COLUMN="${7:-caption_edit}"

cd -- "$REPO_ROOT"
exec python src/pipeline/legacy/edit_unsafe2safe_df.py \
  --input "$1" \
  --output "$2" \
  --ckpt "$3" \
  --image-root "$4" \
  --file-column "$FILE_COLUMN" \
  --public-caption-column "$PUBLIC_CAPTION_COLUMN" \
  --edit-caption-column "$EDIT_CAPTION_COLUMN"
