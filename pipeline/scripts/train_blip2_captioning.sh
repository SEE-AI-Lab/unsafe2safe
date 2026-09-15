#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 LAVIS_ROOT CONFIG TRAIN_JSON VAL_JSON TEST_JSON IMAGE_ROOT" >&2
  echo "set NPROC_PER_NODE to choose the number of processes (default: 1)" >&2
  exit 2
fi

LAVIS_ROOT=$1
CONFIG=$2
TRAIN_JSON=$3
VAL_JSON=$4
TEST_JSON=$5
IMAGE_ROOT=$6
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

export PYTHONPATH="$LAVIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep the project adapter outside LAVIS: these overrides point its standard
# COCO builder at the locally prepared annotations and image root.
exec python -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$LAVIS_ROOT/train.py" \
  --cfg-path "$CONFIG" \
  --options \
  "datasets.coco_caption.build_info.annotations.train.url=$TRAIN_JSON" \
  "datasets.coco_caption.build_info.annotations.train.storage=$TRAIN_JSON" \
  "datasets.coco_caption.build_info.annotations.val.url=$VAL_JSON" \
  "datasets.coco_caption.build_info.annotations.val.storage=$VAL_JSON" \
  "datasets.coco_caption.build_info.annotations.test.url=$TEST_JSON" \
  "datasets.coco_caption.build_info.annotations.test.storage=$TEST_JSON" \
  "datasets.coco_caption.build_info.images.storage=$IMAGE_ROOT"
