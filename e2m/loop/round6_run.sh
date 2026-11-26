#!/usr/bin/env bash
E2M_ROOT="${E2M_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# E2M round 6 pipeline: C-path ceiling (c-only-7b C repair loop), untrained 7B in
# C mode, optional val C loop, then CPU summary. 5.0 GPU-hour hard cap.
# Owns ~/gpu.lock ("e2m <pid> <iso> round 6"); a trap releases it on every exit.
# No courtesy time gate. Pause: touch round6.hold (released at the next
# stage boundary, retaken when the file is removed). Resume-safe: loops and evals
# skip tasks already in their output files.
set -uo pipefail
ROOT=$E2M_ROOT
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 E2M_NO_DEADLINE=1 E2M_HOLD_FILE=$ROOT/round6.hold
PIPE=$ROOT/runs/round6
LOG=$ROOT/logs/round6-pipeline.log
HOURS=$PIPE/gpu_hours.jsonl
LOCK=$HOME/gpu.lock
BUDGET_H=5.0
MODEL=$ROOT/runs/models/Qwen2.5-Coder-7B-Instruct
mkdir -p "$PIPE/stages" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "$$" >"$PIPE/pipeline.pid"
log(){ echo "$(date --iso-8601=seconds) $*"; }

own_lock(){ [ -f "$LOCK" ] && [ "$(awk '{print $1" "$2}' "$LOCK")" = "e2m $$" ]; }
take_lock(){
  while true; do
    if [ -f "$LOCK" ]; then
      holder=$(awk '{print $2}' "$LOCK"); who=$(awk '{print $1}' "$LOCK")
      if [ "$who" = "e2m" ] && [ "$holder" = "$$" ]; then return 0; fi
      if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
        log "lock held by live pid $holder ($(head -c 120 "$LOCK")); waiting"; sleep 60; continue
      fi
      log "stale lock ($(head -c 120 "$LOCK")); replacing"
    fi
    echo "e2m $$ $(date --iso-8601=seconds) round 6" >"$LOCK"
    log "took gpu.lock: $(cat "$LOCK")"; return 0
  done
}
release_lock(){ if own_lock; then rm -f "$LOCK"; log "released gpu.lock"; fi; }
on_exit(){ rc=$?; release_lock; log "pipeline exit rc=$rc"; }
trap on_exit EXIT
trap 'log "signal; stopping"; [ -n "${CHILD:-}" ] && kill -TERM "$CHILD" 2>/dev/null; wait "${CHILD:-}" 2>/dev/null; exit 143' INT TERM

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
  if [ -f "$ROOT/round6.hold" ]; then
    release_lock; log "hold file present; waiting"
    while [ -f "$ROOT/round6.hold" ]; do sleep 30; done
    log "hold cleared"
  fi
  take_lock
}
stage(){  # name want_h cmd...
  local name=$1 want=$2; shift 2
  local sd=$PIPE/stages/$name; mkdir -p "$sd"
  if [ -f "$sd/exit" ] && [ "$(cat "$sd/exit")" = "0" ]; then log "stage $name already ok; skip"; return 0; fi
  wait_hold
  local cap; cap=$(cap_h "$want")
  if python -c "import sys;sys.exit(0 if float(sys.argv[1])>=0.05 else 1)" "$cap"; then :; else
    log "stage $name skipped: gpu-hour budget (cap $cap)"; echo skipped >"$sd/exit"; return 0; fi
  echo "$name" >"$PIPE/stage.txt"
  log "stage start $name cap_h=$cap: $* --max-hours $cap"
  local t0; t0=$(date +%s.%N)
  "$@" --max-hours "$cap" >>"$sd/log" 2>&1 &
  CHILD=$!; echo $CHILD >"$sd/pid"; wait $CHILD; local rc=$?; CHILD=
  local secs; secs=$(python -c "import sys,time;print(round(time.time()-float(sys.argv[1]),2))" "$t0")
  echo $rc >"$sd/exit"
  python -c "import json,sys;print(json.dumps({'name':sys.argv[1],'seconds':float(sys.argv[2]),'ok':sys.argv[3]=='0','code':int(sys.argv[3]),'gpu':True,'ts':sys.argv[4]}))" "$name" "$secs" "$rc" "$(date --iso-8601=seconds)" >>"$HOURS"
  log "stage end $name rc=$rc elapsed_s=$secs used_h=$(used_h)"
}

log "round6 start pid=$$ budget_h=$BUDGET_H used_h=$(used_h)"
take_lock

# Held-out task order: interleave MBPP and HumanEval so a capped partial loop is mixed.
python - <<'PY'
import json
src="$E2M_ROOT/eval/heldout.jsonl"; out="$E2M_ROOT/runs/round6/heldout_interleaved.jsonl"
rows=[json.loads(l) for l in open(src) if l.strip()]
mb=[r for r in rows if r["id"].startswith("mbpp")]; he=[r for r in rows if not r["id"].startswith("mbpp")]
order=[]; i=j=0
while i<len(mb) or j<len(he):          # ~1 MBPP per 2.4 HumanEval, spread evenly
    if i<len(mb) and (j>=len(he) or i*len(he) <= j*len(mb)): order.append(mb[i]); i+=1
    else: order.append(he[j]); j+=1
with open(out,"w") as f:
    for r in order: f.write(json.dumps(r)+"\n")
json.dump([r["id"] for r in order], open("$E2M_ROOT/runs/round6/heldout_order.json","w"), indent=0)
print("interleaved", len(order), "first 10:", [r["id"] for r in order[:10]])
PY

stage loop-c-only-7b-heldout 2.5 python -u -m e2m.loop run --model "$MODEL" --adapter "$ROOT/runs/c-only-7b/adapter" \
  --name c-only-7b --mode c --split heldout --tasks "$PIPE/heldout_interleaved.jsonl" --k 8 --rounds 3 --4bit \
  --out "$ROOT/runs/loop-c-only-7b-heldout"
stage base7b-c-eval-heldout 0.35 python -u -m e2m.train.eval_generate --model "$MODEL" --name base7b-c --mode c \
  --split heldout --4bit --out "$ROOT/runs/base7b-c/eval.jsonl"
stage base7b-c-eval-val 0.40 python -u -m e2m.train.eval_generate --model "$MODEL" --name base7b-c --mode c \
  --split val --4bit --out "$ROOT/runs/base7b-c/eval.jsonl"
stage loop-c-only-7b-val 1.25 python -u -m e2m.loop run --model "$MODEL" --adapter "$ROOT/runs/c-only-7b/adapter" \
  --name c-only-7b --mode c --split val --k 8 --rounds 3 --4bit --out "$ROOT/runs/loop-c-only-7b-val"

release_lock
echo summary >"$PIPE/stage.txt"
log "CPU summary"
python -u -m e2m.loop.round6_summary && log "round6 complete gpu_hours=$(used_h)"
