#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Run a command under nohup with pid/exit/log in a run directory.
# Usage: nohup_run.sh runs/<name> -- python -m e2m.loop run --out runs/<name> ...
set -euo pipefail
RUNDIR=${1:?run dir}
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
mkdir -p "$RUNDIR"
cd $E2M_ROOT
# shellcheck disable=SC1091
source $E2M_ROOT/.venv/bin/activate
export PYTHONUNBUFFERED=1
nohup "$@" >>"$RUNDIR/log" 2>&1 &
echo $! >"$RUNDIR/wrapper.pid"
echo "started pid=$(cat "$RUNDIR/wrapper.pid") dir=$RUNDIR"
