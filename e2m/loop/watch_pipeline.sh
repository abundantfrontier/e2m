#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Poll the loop pipeline. Print only on stage change, sparse progress, or terminal state.
set -euo pipefail
PIPE=${1:-$E2M_ROOT/runs/loop-pipeline}
prev_stage=""
declare -A prev_progress=()
prev_rw_h=0
while :; do
  if [[ -f "$PIPE/exit" ]]; then
    ec=$(tr -d '[:space:]' <"$PIPE/exit")
    stage=$(tr -d '[:space:]' <"$PIPE/stage.txt" 2>/dev/null || echo none)
    if [[ "$ec" == "0" ]]; then
      echo "DONE: pipeline exit 0 last_stage=$stage"
      exit 0
    fi
    echo "FAILED: pipeline exit $ec last_stage=$stage tail=$(tail -c 400 "$PIPE/log" | tr '\n' ' ')"
    exit 1
  fi
  pid=$(tr -d '[:space:]' <"$PIPE/pid" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
    sleep 2
    if [[ -f "$PIPE/exit" ]]; then
      continue
    fi
    echo "FAILED: pipeline pid $pid died without exit file tail=$(tail -c 400 "$PIPE/log" | tr '\n' ' ')"
    exit 1
  fi
  stage=$(tr -d '[:space:]' <"$PIPE/stage.txt" 2>/dev/null || echo "")
  if [[ -n "$stage" && "$stage" != "$prev_stage" ]]; then
    echo "ACTION_REQUIRED: stage ${prev_stage:-none} -> $stage"
    prev_stage=$stage
  fi
  for d in $E2M_ROOT/runs/loop-* $E2M_ROOT/runs/rewrite $E2M_ROOT/runs/lora2; do
    base=$(basename "$d")
    [[ "$base" == "loop-smoke" || "$base" == "loop-pipeline" ]] && continue
    [[ -f "$d/progress.txt" ]] || continue
    done_n=$(awk '{print $1}' "$d/progress.txt" 2>/dev/null || true)
    [[ "$done_n" == */* ]] || continue
    cur=${done_n%%/*}
    [[ "$cur" =~ ^[0-9]+$ ]] || continue
    last=${prev_progress[$base]:-0}
    if (( cur == last )); then
      continue
    fi
    if (( cur == 1 || cur % 10 == 0 || cur > last + 9 )); then
      echo "ACTION_REQUIRED: progress $base $done_n"
      prev_progress[$base]=$cur
    fi
  done
  cache=$E2M_ROOT/data/scale/rewrite-cache.jsonl
  if [[ -f "$cache" ]]; then
    n=$(wc -l <"$cache" | tr -d ' ')
    hundred=$((n / 100))
    if (( hundred > 0 && hundred != prev_rw_h && n < 3020 )); then
      echo "ACTION_REQUIRED: rewrite-cache $n/3020"
      prev_rw_h=$hundred
    elif (( n >= 3020 )); then
      prev_rw_h=$hundred
    fi
  fi
  sleep 30
done
