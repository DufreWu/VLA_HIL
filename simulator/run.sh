#!/usr/bin/env bash
# Launch evaluation with the workspace's isolated Isaac Sim installation.
set -euo pipefail
simulator_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$simulator_dir")"
simulator_python="$project_dir/.venv-isaacsim/bin/python"
if [[ ! -x "$simulator_python" ]]; then
  echo "Missing $simulator_python; follow simulator/README.md installation instructions." >&2
  exit 1
fi
cd -- "$project_dir"
exec "$simulator_python" "$simulator_dir/evaluate.py" "$@"
