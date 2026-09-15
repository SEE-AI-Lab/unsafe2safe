#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${OMINICONTROL_ROOT:?Set OMINICONTROL_ROOT to a local OminiControl checkout}"

export PYTHONPATH="${ROOT_DIR}/src:${OMINICONTROL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

CONFIG_PATH="${OMINI_CONFIG:-${ROOT_DIR}/src/pipeline/adapters/ominicontrol/config.yaml}"
cd -- "${ROOT_DIR}"
exec accelerate launch -m pipeline.adapters.ominicontrol.train_unsafe2safe --config "${CONFIG_PATH}"
