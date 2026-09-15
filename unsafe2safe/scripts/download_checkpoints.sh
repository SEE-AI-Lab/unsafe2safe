#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT_DIR="$SCRIPT_DIR/../checkpoints"

mkdir -p "$CHECKPOINT_DIR"
curl -L http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt -o "$CHECKPOINT_DIR/instruct-pix2pix-00-22000.ckpt"
