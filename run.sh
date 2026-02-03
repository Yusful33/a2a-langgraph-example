#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv313"

# Make A2A SDK (from this repo) available to Python imports
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Error: venv not found at ${VENV_PATH}"
  exit 1
fi

source "${VENV_PATH}/bin/activate"

if [[ $# -eq 0 ]]; then
  echo "Usage: ./run.sh <python_file> [args...]"
  echo "Example: ./run.sh orchestrator_runner.py"
  exit 1
fi

python "$@"
