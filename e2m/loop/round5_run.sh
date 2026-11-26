#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Launch the round-5 pipeline. The Python process owns ~/gpu.lock and deletes
# it on every exit. The shell execs Python so the nohup pid is the pipeline pid.
set -euo pipefail
ROOT=$E2M_ROOT
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export E2M_NO_DEADLINE=1
export E2M_HOLD_FILE=$ROOT/round5.hold
mkdir -p "$ROOT/runs/round5" "$ROOT/logs"
LOG=$ROOT/logs/round5-pipeline.log
echo "# round5_run.sh $(date --iso-8601=seconds) cwd=$ROOT" >>"$LOG"
exec python -u -m e2m.loop.round5 --resume >>"$LOG" 2>&1
