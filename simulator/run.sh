#!/usr/bin/env bash
# Launch evaluation with the active Python environment.
set -euo pipefail

simulator_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$simulator_dir")"

# Prefer the active Conda environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    simulator_python="$CONDA_PREFIX/bin/python"
else
    simulator_python="$project_dir/.venv-isaacsim/bin/python"
fi

if [[ ! -x "$simulator_python" ]]; then
    echo "Missing Python: $simulator_python" >&2
    exit 1
fi

echo "Using Python: $simulator_python"

cd -- "$project_dir"
exec "$simulator_python" "$simulator_dir/evaluate.py" "$@"

