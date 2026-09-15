#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OMINICONTROL_ROOT:?Set OMINICONTROL_ROOT to a local OminiControl checkout}"

export PYTHONPATH="${ROOT_DIR}:${OMINICONTROL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

CONFIG_PATH="${OMINI_CONFIG:-${ROOT_DIR}/pipeline/adapters/ominicontrol/config.example.yaml}"
NUM_PROCESSES="${OMINI_NUM_PROCESSES:-1}"
cd -- "${ROOT_DIR}"
exec accelerate launch --num_processes "${NUM_PROCESSES}" -m pipeline.adapters.ominicontrol.train_unsafe2safe --config "${CONFIG_PATH}"
