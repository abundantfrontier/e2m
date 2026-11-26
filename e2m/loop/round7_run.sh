#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# E2M round 7: beat the compiler on programs that already pass. No training. 4.0 GPU-hour hard cap.
#   round7_run.sh cpu   stages 0 (diff self-test), 1 (compiler portfolio), 3 (search control). CPU only:
#                       CUDA_VISIBLE_DEVICES is empty and the GPU lock is never read or written.
#   round7_run.sh gpu   waits for stage 1, then for ~/gpu.lock to be ABSENT (checked every 60 s; never
#                       takes over another holder's lock, live or not), creates it atomically (noclobber)
#                       as "e2m <pid> <iso> round 7", runs stages 2 (model edits, 2.5 h) and 4 (faster C,
#                       0.5 h), releases the lock (trap on EXIT/INT/TERM), waits for stage 3, then the
#                       CPU summary (stage 5) writes logs/round7.done.
# Pause: touch round7.hold -> the GPU stage stops at its next task boundary, the lock is released,
# and the pipeline waits until the file is removed, then re-acquires the lock the same way and resumes.
set -uo pipefail
ROOT=$E2M_ROOT
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
MODE=${1:?usage: round7_run.sh cpu|gpu}
export PYTHONUNBUFFERED=1 E2M_NO_DEADLINE=1 E2M_HOLD_FILE=$ROOT/round7.hold
PIPE=$ROOT/runs/round7
HOURS=$PIPE/gpu_hours.jsonl
LOCK=$HOME/gpu.lock
HOLD=$ROOT/round7.hold
BUDGET_H=4.0
mkdir -p "$PIPE/stages" "$ROOT/logs"
if [ "$MODE" = cpu ]; then LOG=$ROOT/logs/round7-cpu.log; else LOG=$ROOT/logs/round7-pipeline.log; fi
exec >>"$LOG" 2>&1
echo "$$" >"$PIPE/pipeline-$MODE.pid"
log(){ echo "$(date --iso-8601=seconds) [$MODE] $*"; }

# ------------------------------------------------------------------ CPU phase
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  trap 'log "signal; stopping"; [ -n "${CHILD:-}" ] && kill -TERM "$CHILD" 2>/dev/null; wait "${CHILD:-}" 2>/dev/null; exit 143' INT TERM
  cstage(){  # name cmd...
    local name=$1; shift; local sd=$PIPE/stages/$name; mkdir -p "$sd"
    if [ -f "$sd/exit" ] && [ "$(cat "$sd/exit")" = "0" ]; then log "stage $name already ok; skip"; return 0; fi
    log "stage start $name: $*"; local t0; t0=$(date +%s)
    "$@" >>"$sd/log" 2>&1 & CHILD=$!; echo $CHILD >"$sd/pid"; wait $CHILD; local rc=$?; CHILD=
    echo $rc >"$sd/exit"
    python -c "import json,sys;print(json.dumps({'name':sys.argv[1],'seconds':float(sys.argv[2]),'ok':sys.argv[3]=='0','code':int(sys.argv[3]),'gpu':False,'ts':sys.argv[4]}))" "$name" "$(( $(date +%s) - t0 ))" "$rc" "$(date --iso-8601=seconds)" >>"$HOURS"
    log "stage end $name rc=$rc elapsed_s=$(( $(date +%s) - t0 ))"
  }
  log "round7 CPU phase start pid=$$ (CUDA_VISIBLE_DEVICES='')"
  cstage s0-diff-selftest python -u -m e2m.loop.round7_cpu selftest --n 500
  cstage s1-portfolio python -u -m e2m.loop.round7_cpu portfolio --workers "${R7_WORKERS:-8}" --max-hours 0.5
  cstage s3-search python -u -m e2m.loop.round7_cpu search --workers "${R7_WORKERS:-8}" --max-hours 2.0
  log "round7 CPU phase done"
  exit 0
fi

# ------------------------------------------------------------------ GPU phase
export CUDA_VISIBLE_DEVICES=0
own_lock(){ [ -f "$LOCK" ] && [ "$(awk '{print $1" "$2}' "$LOCK")" = "e2m $$" ]; }
release_lock(){ if own_lock; then rm -f "$LOCK"; log "released gpu.lock"; fi; }
on_exit(){ rc=$?; release_lock; log "pipeline exit rc=$rc"; }
trap on_exit EXIT
trap 'log "signal; stopping"; [ -n "${CHILD:-}" ] && kill -TERM "$CHILD" 2>/dev/null; wait "${CHILD:-}" 2>/dev/null; exit 143' INT TERM

gpu_mem_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
take_lock(){  # wait for ABSENT lock, create atomically; never replace anyone's lock
  local last=0
  while true; do
    if own_lock; then return 0; fi
    if [ -f "$HOLD" ]; then sleep 30; continue; fi
    if [ -e "$LOCK" ]; then
      if [ $(( $(date +%s) - last )) -ge 900 ]; then
        holder=$(awk '{print $2}' "$LOCK" 2>/dev/null); alive=dead; ps -p "$holder" >/dev/null 2>&1 && alive=live
        log "gpu.lock present ($alive holder): $(head -c 140 "$LOCK"); waiting (re-check every 60 s)"; last=$(date +%s)
      fi
      sleep 60; continue
    fi
    if ( set -C; echo "e2m $$ $(date --iso-8601=seconds) round 7" >"$LOCK" ) 2>/dev/null; then
      log "took gpu.lock atomically: $(cat "$LOCK")"
      # the previous job removed its lock; make sure its process has also left the GPU
      local w=0 mem
      while mem=$(gpu_mem_mib); [ -n "$mem" ] && [ "$mem" -gt 3000 ]; do
        if [ $w -ge 600 ]; then log "GPU still has ${mem} MiB in use 10 min after the lock freed; releasing and waiting"; release_lock; sleep 60; continue 2; fi
        [ $(( w % 120 )) -eq 0 ] && log "GPU memory in use ${mem} MiB; waiting for it to drop below 3000"
        sleep 20; w=$(( w + 20 ))
      done
      return 0
    fi
    log "lost the race for gpu.lock; waiting"; sleep 60
  done
}
used_h(){ python - "$HOURS" <<'PY'
import json,sys,os
p=sys.argv[1]; t=0.0
if os.path.exists(p):
    for l in open(p):
        if l.strip():
            d=json.loads(l)
            if d.get("gpu"): t+=d["seconds"]
print(f"{t/3600:.4f}")
PY
}
cap_h(){ python -c "import sys;want,used,budget=map(float,sys.argv[1:]);print(f'{max(0.0,min(want,budget-used)):.3f}')" "$1" "$(used_h)" "$BUDGET_H"; }
wait_hold(){
  if [ -f "$HOLD" ]; then
    release_lock; log "hold file present; waiting"
    while [ -f "$HOLD" ]; do sleep 30; done
    log "hold cleared"
  fi
  take_lock
}
gstage(){  # name want_h cmd...   (re-runs after a hold; outputs are resume-safe)
  local name=$1 want=$2; shift 2
  local sd=$PIPE/stages/$name; mkdir -p "$sd"
  while true; do
    if [ -f "$sd/exit" ] && [ "$(cat "$sd/exit")" = "0" ]; then log "stage $name already ok; skip"; return 0; fi
    wait_hold
    local cap; cap=$(cap_h "$want")
    if ! python -c "import sys;sys.exit(0 if float(sys.argv[1])>=0.05 else 1)" "$cap"; then
      log "stage $name skipped: gpu-hour budget (cap $cap)"; echo skipped >"$sd/exit"; return 0; fi
    echo "$name" >"$PIPE/stage.txt"
    log "stage start $name cap_h=$cap: $* --max-hours $cap"
    local t0; t0=$(date +%s.%N)
    "$@" --max-hours "$cap" >>"$sd/log" 2>&1 & CHILD=$!; echo $CHILD >"$sd/pid"; wait $CHILD; local rc=$?; CHILD=
    local secs; secs=$(python -c "import sys,time;print(round(time.time()-float(sys.argv[1]),2))" "$t0")
    echo $rc >"$sd/exit"
    python -c "import json,sys;print(json.dumps({'name':sys.argv[1],'seconds':float(sys.argv[2]),'ok':sys.argv[3]=='0','code':int(sys.argv[3]),'gpu':True,'ts':sys.argv[4]}))" "$name" "$secs" "$rc" "$(date --iso-8601=seconds)" >>"$HOURS"
    log "stage end $name rc=$rc elapsed_s=$secs used_h=$(used_h)"
    if [ "$rc" = 3 ] && [ -f "$HOLD" ]; then continue; fi   # paused by the hold file: wait, retake, resume
    return 0
  done
}

log "round7 GPU waiter start pid=$$ budget_h=$BUDGET_H used_h=$(used_h)"
until [ -f "$PIPE/stages/s1-portfolio/exit" ]; do sleep 60; done
if [ "$(cat "$PIPE/stages/s1-portfolio/exit")" != 0 ]; then log "stage 1 failed; GPU stages need the portfolio. Not taking the GPU."; exit 1; fi
log "portfolio ready; waiting for ~/gpu.lock to be absent"
take_lock
gstage s2-model-edits 2.5 python -u -m e2m.loop.round7_gpu edits --k 8 --rounds 2
gstage s4-faster-c 0.5 python -u -m e2m.loop.round7_gpu fasterc --k 8
release_lock
echo summary >"$PIPE/stage.txt"
log "waiting for the CPU search (stage 3) to finish"
until [ -f "$PIPE/stages/s3-search/exit" ]; do sleep 60; done
log "stage 5: CPU summary"
CUDA_VISIBLE_DEVICES="" python -u -m e2m.loop.round7_summary --workers 8 >>"$PIPE/stages/summary.log" 2>&1 && log "round7 complete gpu_hours=$(used_h)" || log "summary failed rc=$?"
