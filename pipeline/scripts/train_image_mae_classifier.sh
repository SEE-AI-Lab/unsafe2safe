#!/usr/bin/env bash

set -euo pipefail

# Pass the same arguments as: python -m pipeline.train_image_mae_classifier
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 -m pipeline.train_image_mae_classifier "$@"
