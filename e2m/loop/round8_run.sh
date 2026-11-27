#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# E2M round 8: checker v3, re-validation of round 7, search, optional model stage. No training.
# CPU stages (A gate, B re-validate, C search, E audit/summary) run WITHOUT ~/gpu.lock, CPU only
# (CUDA_VISIBLE_DEVICES=''), at most 12 workers. Only stage D (model edits, 1.0 GPU-hour cap) takes the lock:
# it waits for ~/gpu.lock to be ABSENT (checked every 120 s; never takes over anyone's lock, live or not),
# creates it atomically (noclobber) as "e2m <pid> <iso> round 8", and releases it when D ends (trap on
# EXIT/INT/TERM too). D runs only if the checker gate passed and the search left >= 15 held-out near-misses.
# Gate rule: if v3 does not reject all 19 known-wrong round-7 wins (or rejects an equivalent pair), stop after B.
# Pause: touch round8.hold -> stages stop at their next boundary (D releases the lock); rm to resume.
# Ends with logs/round8.done (written by the summary), which the auto-sync watcher publishes.
set -uo pipefail
ROOT=${R8_ROOT:-$E2M_ROOT}
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1 E2M_NO_DEADLINE=1
export E2M_HOLD_FILE=${R8_HOLD:-$ROOT/round8.hold}
export E2M_R8_OUT=${R8_OUT:-$ROOT/runs/round8}
PIPE=$E2M_R8_OUT
HOURS=$PIPE/gpu_hours.jsonl
LOCK=${R8_LOCK:-$HOME/gpu.lock}
HOLD=$E2M_HOLD_FILE
POLL=${R8_POLL:-120}
W=${R8_WORKERS:-12}
SEARCH_H=${R8_SEARCH_H:-6.0}
GPU_CAP_H=1.0
LOG=${R8_LOG:-$ROOT/logs/round8-pipeline.log}
mkdir -p "$PIPE/stages" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "$$" >"$PIPE/pipeline.pid"
log(){ echo "$(date --iso-8601=seconds) [round8] $*"; }

own_lock(){ [ -f "$LOCK" ] && [ "$(awk '{print $1" "$2}' "$LOCK")" = "e2m $$" ]; }
release_lock(){ if own_lock; then rm -f "$LOCK"; log "released gpu.lock"; fi; }
on_exit(){ rc=$?; release_lock; log "pipeline exit rc=$rc"; }
trap on_exit EXIT
trap 'log "signal; stopping"; [ -n "${CHILD:-}" ] && kill -TERM "$CHILD" 2>/dev/null; wait "${CHILD:-}" 2>/dev/null; exit 143' INT TERM

rec(){  # name seconds rc gpu
  python -c "import json,sys;print(json.dumps({'name':sys.argv[1],'seconds':float(sys.argv[2]),'ok':sys.argv[3]=='0','code':int(sys.argv[3]),'gpu':sys.argv[4]=='1','ts':sys.argv[5]}))" "$1" "$2" "$3" "$4" "$(date --iso-8601=seconds)" >>"$HOURS"
}
wait_hold(){ if [ -f "$HOLD" ]; then release_lock; log "hold file present; waiting"; while [ -f "$HOLD" ]; do sleep 30; done; log "hold cleared"; fi; }

cstage(){  # name cap_h cmd...   CPU only, no lock; re-runs after a hold (resume-safe)
  local name=$1 cap=$2; shift 2; local sd=$PIPE/stages/$name; mkdir -p "$sd"
  while true; do
    if [ -f "$sd/exit" ] && [ "$(cat "$sd/exit")" = "0" ]; then log "stage $name already ok; skip"; return 0; fi
    wait_hold
    echo "$name" >"$PIPE/stage.txt"
    log "stage start $name (CPU, no lock, cap ${cap} h): $*"; local t0; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="" timeout --signal=TERM --kill-after=120 "$(python -c "import sys;print(int(float(sys.argv[1])*3600)+900)" "$cap")" "$@" >>"$sd/log" 2>&1 &
    CHILD=$!; echo $CHILD >"$sd/pid"; wait $CHILD; local rc=$?; CHILD=
    echo $rc >"$sd/exit"; rec "$name" "$(( $(date +%s) - t0 ))" "$rc" 0
    log "stage end $name rc=$rc elapsed_s=$(( $(date +%s) - t0 ))"
    if [ "$rc" = 3 ] && [ -f "$HOLD" ]; then continue; fi
    return $rc
  done
}

gpu_mem_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
take_lock(){  # wait for an ABSENT lock, create it atomically; never replace anyone's lock
  local last=0
  while true; do
    if own_lock; then return 0; fi
    if [ -f "$HOLD" ]; then sleep 30; continue; fi
    if [ -e "$LOCK" ]; then
      if [ $(( $(date +%s) - last )) -ge 1800 ]; then
        holder=$(awk '{print $2}' "$LOCK" 2>/dev/null); alive=dead; ps -p "$holder" >/dev/null 2>&1 && alive=live
        log "gpu.lock present ($alive holder): $(head -c 140 "$LOCK"); waiting (re-check every ${POLL} s)"; last=$(date +%s)
      fi
      sleep "$POLL"; continue
    fi
    if ( set -C; echo "e2m $$ $(date --iso-8601=seconds) round 8" >"$LOCK" ) 2>/dev/null; then
      log "took gpu.lock atomically: $(cat "$LOCK")"
      local w=0 mem
      while mem=$(gpu_mem_mib); [ -n "$mem" ] && [ "$mem" -gt 3000 ]; do
        if [ $w -ge 600 ]; then log "GPU still has ${mem} MiB in use 10 min after the lock freed; releasing and waiting"; release_lock; sleep "$POLL"; continue 2; fi
        [ $(( w % 120 )) -eq 0 ] && log "GPU memory in use ${mem} MiB; waiting for it to drop below 3000"
        sleep 20; w=$(( w + 20 ))
      done
      return 0
    fi
    log "lost the race for gpu.lock; waiting"; sleep "$POLL"
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
gstage(){  # name cmd...   takes the lock, runs with --max-hours (remaining of the 1.0 h cap), releases
  local name=$1; shift; local sd=$PIPE/stages/$name; mkdir -p "$sd"
  while true; do
    if [ -f "$sd/exit" ] && [ "$(cat "$sd/exit")" = "0" ]; then log "stage $name already ok; skip"; return 0; fi
    wait_hold
    log "stage $name: waiting for ~/gpu.lock to be absent (poll ${POLL} s)"
    take_lock
    local cap; cap=$(python -c "import sys;print(f'{max(0.0,float(sys.argv[1])-float(sys.argv[2])):.3f}')" "$GPU_CAP_H" "$(used_h)")
    if ! python -c "import sys;sys.exit(0 if float(sys.argv[1])>=0.05 else 1)" "$cap"; then
      log "stage $name skipped: gpu-hour cap (left $cap)"; echo skipped >"$sd/exit"; release_lock; return 0; fi
    echo "$name" >"$PIPE/stage.txt"
    log "stage start $name (GPU, cap_h=$cap): $* --max-hours $cap"
    local t0; t0=$(date +%s.%N)
    CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=120 "$(python -c "import sys;print(int(float(sys.argv[1])*3600)+600)" "$cap")" "$@" --max-hours "$cap" >>"$sd/log" 2>&1 &
    CHILD=$!; echo $CHILD >"$sd/pid"; wait $CHILD; local rc=$?; CHILD=
    local secs; secs=$(python -c "import sys,time;print(round(time.time()-float(sys.argv[1]),2))" "$t0")
    echo $rc >"$sd/exit"; rec "$name" "$secs" "$rc" 1
    release_lock
    log "stage end $name rc=$rc elapsed_s=$secs used_gpu_h=$(used_h)"
    if [ "$rc" = 3 ] && [ -f "$HOLD" ]; then continue; fi
    return 0
  done
}

log "round8 pipeline start pid=$$ out=$PIPE workers=$W search_h=$SEARCH_H gpu_cap_h=$GPU_CAP_H lock=$LOCK"
cstage A-gate 1.0 python -u -m e2m.loop.round8 gate --workers "$W" || { log "stage A failed; stopping"; exit 1; }
GATE=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['pass'])" "$PIPE/diff_selftest.json" 2>/dev/null || echo False)
log "checker gate pass=$GATE"
cstage B-revalidate 1.5 python -u -m e2m.loop.round8 revalidate --workers "$W" --max-hours 1.5 ${R8_LIMIT:+--limit $R8_LIMIT} || log "stage B rc!=0; continuing to the summary"
if [ "$GATE" != True ]; then
  log "gate failed: stopping after B (no search, no model stage)"
else
  cstage C-search "$SEARCH_H" python -u -m e2m.loop.round8 search --workers "$W" --max-hours "$SEARCH_H" --isa ${R8_LIMIT:+--limit $R8_LIMIT} || log "stage C rc!=0; continuing"
  NM=$(CUDA_VISIBLE_DEVICES="" python -m e2m.loop.round8_gpu nearmiss 2>/dev/null | tail -1)
  log "held-out near-misses after search: ${NM:-?} (stage D needs >= 15)"
  if [ -n "$NM" ] && [ "$NM" -ge "${R8_NM_MIN:-15}" ]; then
    gstage D-model python -u -m e2m.loop.round8_gpu edits --k 8 --rounds 1
  else
    log "stage D skipped: fewer than 15 held-out near-misses; the GPU lock is never taken"
    mkdir -p "$PIPE/stages/D-model"; echo skipped >"$PIPE/stages/D-model/exit"
  fi
fi
release_lock
cstage E-summary 1.5 python -u -m e2m.loop.round8 summary --workers "$W" ${R8_NO_DONE:+--no-done} && log "round8 complete gpu_hours=$(used_h)" || log "summary failed"
echo done >"$PIPE/stage.txt"
