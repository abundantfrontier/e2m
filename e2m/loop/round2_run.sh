#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Launch the round-2 GPU pipeline under nohup. The Python process owns ~/gpu.lock
# and deletes it on every exit (success, error, or budget stop).
set -euo pipefail
ROOT=$E2M_ROOT
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$ROOT/runs/round2" "$ROOT/logs"
LOG=$ROOT/logs/build-round2.log
echo "# round2_run.sh $(date --iso-8601=seconds) cwd=$ROOT" >>"$LOG"
exec python -u -m e2m.loop round2
