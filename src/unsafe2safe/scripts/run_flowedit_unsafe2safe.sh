#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${FLOWEDIT_ROOT:?Set FLOWEDIT_ROOT to a clean FlowEdit checkout}"

export PYTHONPATH="${ROOT_DIR}/src:${FLOWEDIT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

exec python -m unsafe2safe.adapters.flowedit.generate_unsafe2safe "$@"
