"""Round 5b: finish the round-5 eval, then retrain with distill in the majority.

Self-driving. GPU work waits on gate_wait() and stops at an 8-hour cap.
``--resume`` skips stages already recorded ok. CPU stages release ~/gpu.lock.
``--smoke`` and ``--dry-run`` do not take the lock and do not write logs/round5b.done.

Stage A scores runs/asm-distill-7b (the round-5 adapter) and does not train it again.
Stage B writes a new adapter at runs/asm-distill-7b-v2.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from e2m.loop.guard import LOCK_PATH, PT, pid_alive, release_gpu_lock
from e2m.loop.round5 import lock_is_ours, try_take_gpu_lock
from e2m.loop.round5_data import (
    SFT_LORA2,
    WORD5_RULE,
    attach_reference_c,
    distill_pairs,
    load_sft_rows,
    round4_comparison,
    screen_text,
    write_failure,
    write_mix,
    _read_jsonl,
    _task_index,
)
from e2m.loop.round5_sample import _latest_done_rows
from e2m.loop.summarize import summarize_loop
from e2m.train.format import pass_at_k

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL_7B = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"
TEACHER = ROOT / "runs" / "c-only-7b" / "adapter"
HELDOUT_TASKS = ROOT / "eval" / "heldout.jsonl"
SCALED_TASKS = ROOT / "tasks" / "scaled.jsonl"
VAL_SPLIT = ROOT / "eval" / "val_split.json"
ROUND5_DIR = ROOT / "runs" / "round5"
V1 = ROOT / "runs" / "asm-distill-7b"
V2 = ROOT / "runs" / "asm-distill-7b-v2"
V1_EVAL = V1 / "eval.jsonl"
V2_EVAL = V2 / "eval.jsonl"
V1_NAME = "asm-distill-7b"
V2_NAME = "asm-distill-7b-v2"
A_LOOP = ROOT / "runs" / "loop-asm-distill-heldout-c-then-asm"
A_LOOP_ASM = ROOT / "runs" / "loop-asm-distill-heldout-asm"
A_LOOP_VAL = ROOT / "runs" / "loop-asm-distill-val-c-then-asm"
B_LOOP = ROOT / "runs" / "loop-asm-distill-7b-v2-heldout-c-then-asm"
B_LOOP_VAL = ROOT / "runs" / "loop-asm-distill-7b-v2-val-c-then-asm"
PIPE_DIR = ROOT / "runs" / "round5b"
HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
STAGE_PATH = PIPE_DIR / "stage.txt"
STATE_PATH = PIPE_DIR / "state.json"
DONE_PATH = ROOT / "logs" / "round5b.done"
HOLD_PATH = Path.home() / "e2m" / "round5b.hold"
FULL_JSON = ROUND5_DIR / "RESULTS_full.json"
FULL_MD = ROUND5_DIR / "SUMMARY_full.md"
ROUND5B_GPU_HOURS = 8.0
EVAL_RATE_S = 48.73
EVAL_MARGIN_SMALL = 1.55
EVAL_MARGIN_LARGE = 1.31
LOOP_BASE_S = 18602.0
LOOP_TASKS = 78
LOOP_RATE_S = LOOP_BASE_S / LOOP_TASKS
LOOP_MARGIN = 1.10
HELDOUT_N = 78
VAL_N = 90
SAMPLE_K = 8
PRIOR_K = 4
SYNTH_CAP = 80
SYNTH_MAX_H = 0.35
SAMPLE_MAX_H = 1.50
TRAIN_MAX_H = 2.00
SFT_FRACTION = 0.25
LOCK_NOTE = "round 5b"
TERMINAL = {"ok", "stopped", "error", "skipped"}
HARD_STAGES = ("a-eval-heldout", "a-eval-val")
SET_ASIDE = ("b-train", "b-eval-heldout")
SOFT_STAGES = ("b-synth", "b-sample")
OPTIONAL_STAGES = (
    "a-loop-c-then-asm",
    "b-eval-val",
    "b-loop-c-then-asm",
    "a-loop-asm",
    "b-loop-val",
    "a-loop-val",
)
GPU_STAGES = set(HARD_STAGES) | set(SET_ASIDE) | set(SOFT_STAGES) | set(OPTIONAL_STAGES) | {"leftover-eval"}
STAGE_ORDER = [
    "a-eval-heldout",
    "a-eval-val",
    "a-results",
    "b-synth",
    "b-sample",
    "b-lower",
    "b-leakage-text",
    "b-leakage-embed",
    "b-mix",
    "b-train",
    "b-eval-heldout",
    "a-loop-c-then-asm",
    "b-eval-val",
    "b-loop-c-then-asm",
    "a-loop-asm",
    "b-loop-val",
    "a-loop-val",
    "leftover-eval",
    "failure-analysis",
]
# The brief's unfinished held-out ids. Smoke checks the eval file against this
# set. Scoring itself follows eval_generate._done_ids, not this list.
BRIEF_UNFINISHED = (
    "he82_prime_length",
    "he83_starts_one_ends",
    "he85_sum_even_odd_idx",
    "he89_encrypt",
    "he92_any_int",
    "he97_multiply_units",
    "he100_make_a_pile",
    "he102_choose_num",
    "he104_unique_digits",
    "he106_f_factorial_or_sum",
    "he114_min_subarray_sum",
    "he120_k_largest",
    "he121_sum_odd_even_idx",
    "he122_add_elements",
    "he131_odd_digit_product",
    "he138_sum_of_four_evens",
    "he139_special_factorial",
    "he142_indexed_sum_squares",
    "he146_special_filter",
    "he150_x_or_y",
    "he151_double_the_difference",
    "he152_compare_scores",
    "he155_even_odd_count",
    "he157_right_angle_triangle",
    "he159_eat",
    "he163_generate_integers",
)
CURRENT_PROC: dict[str, subprocess.Popen | None] = {"proc": None}


def configure_paths(*, smoke: bool, out: Path | None) -> Path:
    global PIPE_DIR, HOURS_PATH, STAGE_PATH, STATE_PATH, DONE_PATH
    if out is not None:
        PIPE_DIR = Path(out)
    else:
        PIPE_DIR = ROOT / "runs" / ("round5b-smoke" if smoke else "round5b")
    HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
    STAGE_PATH = PIPE_DIR / "stage.txt"
    STATE_PATH = PIPE_DIR / "state.json"
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    if smoke:
        DONE_PATH = PIPE_DIR / "round5b.done"
    else:
        DONE_PATH = ROOT / "logs" / "round5b.done"
    return PIPE_DIR


def eval_cap(n: int) -> float:
    """Hours for ``n`` direct-asm tasks at 48.73 s/task, with the brief's margin.

    Small remainders use 1.55x (26 tasks -> about 0.55 h). A full split uses
    1.31x (90 val tasks -> about 1.60 h, 78 held-out -> about 1.38 h). This is
    not the round-5 0.7 h eval cap.
    """
    if n <= 0:
        return 0.0
    raw = n * EVAL_RATE_S / 3600.0
    margin = EVAL_MARGIN_SMALL if n <= 40 else EVAL_MARGIN_LARGE
    return max(0.15, raw * margin)


def loop_want(n: int) -> float:
    """Hours for ``n`` C-then-asm loop tasks, from round 4's 18602 s / 78."""
    if n <= 0:
        return 0.0
    return max(0.15, n * LOOP_RATE_S * LOOP_MARGIN / 3600.0)


def _optional_take(want: float, left: float, later: float) -> float:
    if want <= 0 or left <= 0.05:
        return 0.0
    if later <= 0.05 or want <= left or left < 1.8:
        return min(want, left)
    return min(want, max(0.4, left * 0.60))


def allocate(remain_h: float, wants: dict[str, float]) -> dict[str, float]:
    """Give time in milestone order. Soft sampling is what is left after train and v2 eval are set aside."""
    alloc = {name: 0.0 for name in wants}
    left = max(0.0, float(remain_h))
    for name in HARD_STAGES:
        take = min(float(wants.get(name) or 0.0), left)
        alloc[name] = take
        left -= take
    aside = [name for name in SET_ASIDE if float(wants.get(name) or 0.0) > 0]
    need = sum(float(wants[name]) for name in aside)
    if need <= left + 1e-9:
        for name in aside:
            alloc[name] = float(wants[name])
            left -= float(wants[name])
    elif left > 0.15 and len(aside) == 2:
        train_w = float(wants["b-train"])
        eval_w = float(wants["b-eval-heldout"])
        eval_set = min(eval_w, max(0.40, left * 0.40))
        train_set = min(train_w, max(0.0, left - eval_set))
        if train_set < 0.20 and train_w > 0:
            train_set = min(train_w, left * 0.60)
            eval_set = min(eval_w, max(0.0, left - train_set))
        alloc["b-train"] = train_set
        alloc["b-eval-heldout"] = eval_set
        left -= train_set + eval_set
    elif left > 0.15 and len(aside) == 1:
        name = aside[0]
        alloc[name] = min(float(wants[name]), left)
        left -= alloc[name]
    left = max(0.0, left)
    for name in SOFT_STAGES:
        take = min(float(wants.get(name) or 0.0), left)
        alloc[name] = take
        left -= take
    for index, name in enumerate(OPTIONAL_STAGES):
        later = sum(float(wants.get(other) or 0.0) for other in OPTIONAL_STAGES[index + 1 :])
        take = _optional_take(float(wants.get(name) or 0.0), left, later)
        alloc[name] = take
        left = max(0.0, left - take)
    return {key: round(val, 4) for key, val in alloc.items()}


def compute_allocation(
    *,
    remain_h: float,
    counts: dict[str, int],
    hours: dict[str, float],
    settled: set[str],
    train_done: bool,
    sample_topped: bool,
    synth_done: bool,
) -> dict[str, Any]:
    def live(name: str, value: float) -> float:
        if name in settled:
            return 0.0
        return max(0.0, value)

    wants = {
        "a-eval-heldout": live("a-eval-heldout", eval_cap(max(0, HELDOUT_N - int(counts.get("a-heldout") or 0)))),
        "a-eval-val": live("a-eval-val", eval_cap(max(0, VAL_N - int(counts.get("a-val") or 0)))),
        "b-synth": 0.0
        if synth_done or "b-synth" in settled
        else max(0.0, SYNTH_MAX_H - float(hours.get("b-synth") or 0.0)),
        "b-sample": 0.0
        if sample_topped or "b-sample" in settled
        else max(0.0, SAMPLE_MAX_H - float(hours.get("b-sample") or 0.0)),
        "b-train": 0.0
        if train_done or "b-train" in settled
        else max(0.0, TRAIN_MAX_H - float(hours.get("b-train") or 0.0)),
        "b-eval-heldout": live("b-eval-heldout", eval_cap(max(0, HELDOUT_N - int(counts.get("b-heldout") or 0)))),
        "a-loop-c-then-asm": live("a-loop-c-then-asm", loop_want(max(0, HELDOUT_N - int(counts.get("a-loop") or 0)))),
        "b-eval-val": live("b-eval-val", eval_cap(max(0, VAL_N - int(counts.get("b-val") or 0)))),
        "b-loop-c-then-asm": live("b-loop-c-then-asm", loop_want(max(0, HELDOUT_N - int(counts.get("b-loop") or 0)))),
        "a-loop-asm": live("a-loop-asm", loop_want(max(0, HELDOUT_N - int(counts.get("a-loop-asm") or 0)))),
        "b-loop-val": live("b-loop-val", loop_want(max(0, VAL_N - int(counts.get("b-loop-val") or 0)))),
        "a-loop-val": live("a-loop-val", loop_want(max(0, VAL_N - int(counts.get("a-loop-val") or 0)))),
    }
    alloc = allocate(remain_h, wants)
    return {
        "remain_h": round(remain_h, 4),
        "budget_h": ROUND5B_GPU_HOURS,
        "wants": {key: round(val, 4) for key, val in wants.items()},
        "allocation": alloc,
        "rates": {
            "eval_s_per_task": EVAL_RATE_S,
            "eval_margin_small_n_le_40": EVAL_MARGIN_SMALL,
            "eval_margin_large": EVAL_MARGIN_LARGE,
            "loop_s_per_task": round(LOOP_RATE_S, 4),
            "loop_margin": LOOP_MARGIN,
            "loop_source": "round 4 C-then-asm held-out loop, 18602 s / 78 tasks",
            "not_the_round5_eval_cap_h": 0.7,
        },
        "notes": (
            "Stage A held-out remainder and val direct asm are funded first, then "
            "training and the v2 held-out eval are set aside, then synth and the "
            "k=8 teacher top-up, then the repair loops. Eval caps use 48.73 s/task "
            "with margin, not the round-5 0.7 h cap. The train pool stays at the "
            "round-5 screened 346 tasks so this budget still fits."
        ),
    }


def gpu_hours_used(path: Path | None = None) -> float:
    hours_path = path or HOURS_PATH
    total = 0.0
    if not hours_path.is_file():
        return 0.0
    with hours_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gpu") is False:
                continue
            total += float(row.get("seconds") or 0)
    return total / 3600.0


def hours_map(path: Path | None = None) -> dict[str, float]:
    hours_path = path or HOURS_PATH
    found: dict[str, float] = {}
    if not hours_path.is_file():
        return found
    with hours_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gpu") is False:
                continue
            name = str(row.get("name") or "")
            found[name] = found.get(name, 0.0) + float(row.get("seconds") or 0) / 3600.0
    return found


def log_hours(name: str, seconds: float, ok: bool, *, gpu: bool) -> None:
    HOURS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HOURS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "seconds": round(seconds, 2),
                    "ok": ok,
                    "gpu": gpu,
                    "ts": datetime.now(tz=PT).isoformat(),
                }
            )
            + "\n"
        )


def remaining_hours() -> float:
    return max(0.0, ROUND5B_GPU_HOURS - gpu_hours_used())


def scored_ids(path: Path, *, model: str, mode: str, split: str) -> set[str]:
    found: set[str] = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model") == model and row.get("mode") == mode and row.get("split") == split:
                tid = str(row.get("task_id") or "")
                if tid:
                    found.add(tid)
    return found


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _train_done(path: Path = V2) -> bool:
    cfg = path / "adapter" / "adapter_config.json"
    if not cfg.is_file():
        return False
    stop = path / "stop_reason"
    if stop.is_file() and stop.read_text(encoding="utf-8").strip() == "held":
        return False
    return True


def _sample_topped(sample_tasks: Path, sample_out: Path) -> bool:
    if not sample_tasks.is_file():
        return False
    want = []
    for row in _read_jsonl(sample_tasks):
        tid = str(row.get("id") or row.get("task_id") or "")
        if tid:
            want.append(tid)
    if not want:
        return False
    latest = _latest_done_rows(sample_out)
    for tid in want:
        row = latest.get(tid)
        if row is None:
            return False
        have = int(row["k"]) if "k" in row and row.get("k") is not None else PRIOR_K
        if have < SAMPLE_K:
            return False
    return True


def read_progress() -> dict[str, Any]:
    sample_tasks = PIPE_DIR / "sample_tasks.jsonl"
    sample_out = PIPE_DIR / "sample_c.jsonl"
    synth_path = PIPE_DIR / "synth_tasks.jsonl"
    counts = {
        "a-heldout": len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="heldout")),
        "a-val": len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="val")),
        "b-heldout": len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="heldout")),
        "b-val": len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="val")),
        "a-loop": len(scored_ids(A_LOOP / "loop.jsonl", model=V1_NAME, mode="c-then-asm", split="heldout")),
        "b-loop": len(scored_ids(B_LOOP / "loop.jsonl", model=V2_NAME, mode="c-then-asm", split="heldout")),
        "a-loop-asm": len(scored_ids(A_LOOP_ASM / "loop.jsonl", model=V1_NAME, mode="asm", split="heldout")),
        "b-loop-val": len(scored_ids(B_LOOP_VAL / "loop.jsonl", model=V2_NAME, mode="c-then-asm", split="val")),
        "a-loop-val": len(scored_ids(A_LOOP_VAL / "loop.jsonl", model=V1_NAME, mode="c-then-asm", split="val")),
    }
    synth_n = _count_lines(synth_path)
    return {
        "counts": counts,
        "hours": hours_map(),
        "train_done": _train_done(),
        "sample_topped": _sample_topped(sample_tasks, sample_out),
        "synth_done": synth_n >= SYNTH_CAP,
        "synth_n": synth_n,
    }


def current_plan(settled: set[str]) -> dict[str, Any]:
    progress = read_progress()
    plan = compute_allocation(
        remain_h=remaining_hours(),
        counts=progress["counts"],
        hours=progress["hours"],
        settled=settled,
        train_done=bool(progress["train_done"]),
        sample_topped=bool(progress["sample_topped"]),
        synth_done=bool(progress["synth_done"]),
    )
    plan["counts"] = progress["counts"]
    plan["synth_n"] = progress["synth_n"]
    return plan


class GateView:
    def __init__(self, action: str, reason: str | None, saw_live_non_e2m: bool):
        self.action = action
        self.reason = reason
        self.saw_live_non_e2m = saw_live_non_e2m


def gate_decision(
    *,
    now: datetime,
    hold_exists: bool,
    lock_text: str | None,
    self_pid: int,
    saw_live_non_e2m: bool,
    pid_alive_fn: Callable[[int], bool] = pid_alive,
) -> GateView:
    """Decide whether this poll may take the lock.

    Courtesy: if the lock is absent (or stale) and no live non-e2m lock has been
    seen yet, wait until one is seen and later clears, or until
    2026-11-25T18:45:00-08:00. A live lock always waits, whoever owns it.
    """
    courtesy_open = now >= datetime.fromisoformat("2026-11-25T18:45:00-08:00")
    text = (lock_text or "").strip()
    owner = ""
    holder: int | None = None
    live = False
    if text:
        parts = text.split()
        owner = parts[0] if parts else ""
        if len(parts) >= 2 and parts[1].isdigit():
            holder = int(parts[1])
            live = pid_alive_fn(holder)
        else:
            live = True
    saw = bool(saw_live_non_e2m or (live and owner != "e2m"))
    ours = live and owner == "e2m" and holder == int(self_pid)
    if ours:
        if hold_exists:
            return GateView("wait", "round5b.hold exists", saw)
        if not (saw or courtesy_open):
            return GateView(
                "wait",
                "courtesy: lock absent; waiting for a live non-e2m lock or 2026-11-25T18:45:00-08:00",
                saw,
            )
        return GateView("take", None, saw)
    if live:
        if holder is not None:
            reason = f"gpu.lock held by live pid {holder}"
        else:
            reason = f"gpu.lock held: {text[:120]}"
        return GateView("wait", reason, saw)
    if hold_exists:
        return GateView("wait", "round5b.hold exists", saw)
    if not (saw or courtesy_open):
        return GateView(
            "wait",
            "courtesy: lock absent; waiting for a live non-e2m lock or 2026-11-25T18:45:00-08:00",
            saw,
        )
    return GateView("take", None, saw)


def _load_gate_state(path: Path | None) -> tuple[bool, str]:
    """Return (saw a live non-e2m lock, the lock text we observed).

    ``courtesy_done`` alone does not open the gate. A restart may skip the
    18:45 wait only when this file still holds the non-e2m lock text we saw.
    """
    if path is None or not path.is_file():
        return False, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, ""
    observed = str(data.get("observed_lock") or "").strip()
    owner = observed.split(maxsplit=1)[0] if observed else ""
    saw = bool(data.get("saw_live_non_e2m") and owner and owner != "e2m")
    return saw, (observed if saw else "")


def _save_gate_state(
    path: Path | None,
    *,
    saw: bool,
    observed_lock: str,
    courtesy_done: bool,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "saw_live_non_e2m": bool(saw and observed_lock),
                "observed_lock": observed_lock if saw else "",
                "courtesy_done": courtesy_done,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def gate_wait(
    pid: int,
    *,
    sleep_s: float = 60.0,
    now_fn: Callable[[], datetime] | None = None,
    lock_path: Path | None = None,
    hold_path: Path | None = None,
    state_path: Path | None = None,
    max_loops: int | None = None,
) -> bool:
    """Block until round5b.hold is absent and the lock can be taken for round 5b.

    Start-up courtesy, one time: do not take a free GPU before other-job's
    Wren block. If the lock is absent, wait until a live non-e2m lock has been
    seen and later cleared, or until 2026-11-25T18:45:00-08:00. After that, the
    lock is taken only when it is absent or stale and round5b.hold does not exist.
    The lock text is "round 5b".
    """
    if os.environ.get("E2M_SMOKE") == "1":
        raise RuntimeError("smoke must not wait on the GPU gate or take ~/gpu.lock")
    lock = lock_path or LOCK_PATH
    hold = hold_path or HOLD_PATH
    state = state_path if state_path is not None else (PIPE_DIR / "gate_state.json")
    saw, observed = _load_gate_state(state)
    last_log = 0.0
    loops = 0
    while True:
        loops += 1
        now = now_fn() if now_fn is not None else datetime.now(tz=PT)
        lock_text = lock.read_text(encoding="utf-8") if lock.is_file() else None
        decision = gate_decision(
            now=now,
            hold_exists=hold.is_file(),
            lock_text=lock_text,
            self_pid=pid,
            saw_live_non_e2m=saw,
        )
        saw = decision.saw_live_non_e2m
        text = (lock_text or "").strip()
        owner = text.split(maxsplit=1)[0] if text else ""
        if saw and owner and owner != "e2m":
            observed = text[:200]
        if saw and observed:
            _save_gate_state(state, saw=True, observed_lock=observed, courtesy_done=False)
        if decision.action != "take" and lock_is_ours(pid, lock):
            release_gpu_lock(owner_pid=pid, path=lock)
        reason = decision.reason or "gate"
        if decision.action == "take":
            try:
                ok, detail = try_take_gpu_lock(pid, "round 5b", path=lock)
            except SystemExit as exc:
                ok, detail = False, str(exc)
            if ok and hold.is_file():
                release_gpu_lock(owner_pid=pid, path=lock)
                ok = False
                reason = "round5b.hold exists"
            elif ok and not lock_is_ours(pid, lock):
                ok = False
                reason = detail or "gpu.lock"
            elif ok:
                _save_gate_state(
                    state,
                    saw=bool(saw and observed),
                    observed_lock=observed,
                    courtesy_done=True,
                )
                print(
                    f"gate open saw={bool(saw and observed)} "
                    f"clock={now.isoformat()} observed={observed[:80]!r}",
                    flush=True,
                )
                return True
            else:
                reason = detail or "gpu.lock"
        elif lock_is_ours(pid, lock):
            release_gpu_lock(owner_pid=pid, path=lock)
        stamp = time.time()
        if last_log == 0.0 or stamp - last_log >= 600:
            print(f"gate wait: {reason}", flush=True)
            last_log = stamp
        if max_loops is not None and loops >= max_loops:
            return False
        time.sleep(sleep_s)


def _read_stage_map(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    stages = data.get("stages") or {}
    return {str(name): info for name, info in stages.items() if isinstance(info, dict)}


def load_prior_stages() -> dict[str, Any]:
    """Merge state.json and RESULTS.json without letting a later 'skipped' clobber 'ok'.

    A resume used to rewrite state.json with only the stages visited so far, and
    the next restart then re-ran finished work. An ok record in either file wins
    over a skipped placeholder. A newer stopped or error record in state.json
    still wins, so an incomplete stage is retried.
    """
    state = _read_stage_map(STATE_PATH)
    results = _read_stage_map(PIPE_DIR / "RESULTS.json")
    merged: dict[str, Any] = {}
    for name in set(state) | set(results):
        current = state.get(name)
        reported = results.get(name)
        current_status = (current or {}).get("status")
        reported_status = (reported or {}).get("status")
        if current_status == "ok":
            merged[name] = current
        elif reported_status == "ok" and current_status in {None, "skipped"}:
            merged[name] = reported
        else:
            merged[name] = current or reported
    return merged


def load_prior_ok() -> set[str]:
    return {name for name, info in load_prior_stages().items() if (info or {}).get("status") == "ok"}


def copy_if_missing(src: Path, dest: Path) -> bool:
    if dest.is_file() or not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return True


def _c_key(texts: list[str]) -> str:
    blob = "\n---\n".join(" ".join(str(text).split()) for text in texts)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class Pipeline:
    def __init__(self, *, smoke: bool, resume: bool, dry_run: bool, only: set[str] | None, env: dict[str, str]):
        self.smoke = smoke
        self.resume = resume
        self.dry_run = dry_run
        self.only = only
        self.env = env
        self.pid = os.getpid()
        self.halt = False
        self.halt_reason = ""
        self.block_train = False
        self.block_reason = ""
        if resume:
            self.stages = load_prior_stages()
            self.prior_ok = {name for name, info in self.stages.items() if (info or {}).get("status") == "ok"}
        else:
            self.stages = {}
            self.prior_ok = set()
        # Frozen at startup. An ok recorded later in this process can still be
        # corrected (an eval that exits 0 with tasks left is not finished).
        self.frozen_ok = set(self.prior_ok)
        self.own_lock = not smoke and not dry_run

    def settled(self) -> set[str]:
        names = set(self.prior_ok)
        for name, info in self.stages.items():
            if (info or {}).get("status") in TERMINAL:
                names.add(name)
        return names

    def plan(self) -> dict[str, Any]:
        plan = current_plan(self.settled())
        if not self.dry_run and not self.smoke:
            (PIPE_DIR / "budget.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        return plan

    def stage_cap(self, name: str) -> float:
        alloc = (self.plan().get("allocation") or {})
        return float(alloc.get(name) or 0.0)

    def save_state(self) -> None:
        if self.dry_run:
            return
        STATE_PATH.write_text(json.dumps({"stages": self.stages}, indent=2) + "\n", encoding="utf-8")

    def record(self, name: str, status: str, *, code: int | None = None, reason: str = "") -> None:
        if name in self.frozen_ok and status != "ok":
            print(f"stage {name} status=ok code=0 already recorded ok", flush=True)
            if (self.stages.get(name) or {}).get("status") != "ok":
                self.stages[name] = {"status": "ok", "code": 0, "reason": "already recorded ok"}
                self.save_state()
            return
        self.stages[name] = {"status": status, "code": code, "reason": reason}
        print(f"stage {name} status={status} code={code} {reason}", flush=True)
        self.save_state()

    def skip_reason(self, name: str) -> str | None:
        if self.only is not None and name not in self.only:
            return "not in --only"
        if self.resume and name in self.prior_ok:
            return "already recorded ok"
        if self.halt and name in GPU_STAGES:
            return self.halt_reason or "pipeline halted"
        return None

    def release(self) -> None:
        if not self.own_lock:
            return
        if lock_is_ours(self.pid):
            release_gpu_lock(owner_pid=self.pid)
            print("released gpu.lock", flush=True)

    def begin_cpu(self) -> None:
        self.release()

    def note_held(self) -> None:
        self.release()
        print("held", flush=True)

    def gpu_ready(self) -> bool:
        if self.smoke or self.dry_run:
            return False
        while True:
            if HOLD_PATH.is_file():
                self.note_held()
                gate_wait(self.pid)
                continue
            if lock_is_ours(self.pid):
                return True
            if not gate_wait(self.pid):
                continue
            if lock_is_ours(self.pid) and not HOLD_PATH.is_file():
                return True

    def run_cmd(self, name: str, argv: list[str], *, gpu: bool) -> int:
        rundir = PIPE_DIR / "stages" / name
        rundir.mkdir(parents=True, exist_ok=True)
        log = rundir / "log"
        if not self.dry_run:
            STAGE_PATH.write_text(name + "\n", encoding="utf-8")
        print(f"stage start {name}: {' '.join(argv)}", flush=True)
        t0 = time.time()
        code = 1
        try:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n# start {name} {datetime.now(tz=PT).isoformat()}\n")
                fh.flush()
                proc = subprocess.Popen(
                    argv,
                    cwd=str(ROOT),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=self.env,
                    start_new_session=True,
                )
                CURRENT_PROC["proc"] = proc
                (rundir / "wrapper.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
                try:
                    code = proc.wait()
                finally:
                    CURRENT_PROC["proc"] = None
        finally:
            elapsed = time.time() - t0
            (rundir / "exit").write_text(str(code) + "\n", encoding="utf-8")
            log_hours(name, elapsed, code == 0, gpu=gpu)
        print(f"stage end {name} code={code} elapsed_s={elapsed:.1f}", flush=True)
        return code

    def run_cpu(self, name: str, fn: Callable[[], None]) -> str:
        reason = self.skip_reason(name)
        if reason:
            self.record(name, "skipped", code=0 if reason.startswith("already") else None, reason=reason)
            return "skip"
        if self.dry_run:
            self.record(name, "planned", reason="dry-run cpu")
            return "plan"
        self.begin_cpu()
        if HOLD_PATH.is_file() and self.own_lock:
            self.note_held()
            gate_wait(self.pid)
            self.begin_cpu()
        if not self.dry_run:
            STAGE_PATH.write_text(name + "\n", encoding="utf-8")
        print(f"stage start {name}", flush=True)
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            log_hours(name, time.time() - t0, False, gpu=False)
            self.record(name, "error", code=1, reason=f"{type(exc).__name__}: {exc}"[:400])
            return "error"
        log_hours(name, time.time() - t0, True, gpu=False)
        print(f"stage end {name} elapsed_s={time.time() - t0:.1f}", flush=True)
        self.record(name, "ok", code=0)
        return "ok"

    def run_gpu(
        self,
        name: str,
        argv: list[str],
        *,
        accept_3: bool = False,
        accept_1: bool = False,
        preflight: Callable[[], str | None] | None = None,
    ) -> str:
        reason = self.skip_reason(name)
        if reason:
            self.record(name, "skipped", code=0 if reason.startswith("already") else None, reason=reason)
            return "skip"
        if self.smoke:
            self.record(name, "skipped", reason="smoke")
            return "skip"
        if self.dry_run:
            self.record(name, "planned", reason=f"dry-run cap_h={self.stage_cap(name):.3f}")
            return "plan"
        while True:
            if remaining_hours() <= 0.05:
                self.halt = True
                self.halt_reason = "gpu-hour budget"
                self.record(name, "skipped", code=3, reason="gpu-hour budget")
                return "budget"
            cap = self.stage_cap(name)
            if cap < 0.05:
                self.record(name, "skipped", code=3, reason="gpu-hour budget")
                return "budget"
            print(f"stage {name} cap_h={cap:.3f}", flush=True)
            if not self.gpu_ready():
                self.record(name, "error", reason="lock not acquired")
                return "error"
            if not lock_is_ours(self.pid):
                continue
            if preflight is not None:
                why = preflight()
                if why:
                    skipped = str(why).startswith("synth skipped")
                    self.record(name, "skipped" if skipped else "error", code=1, reason=why)
                    return "skip" if skipped else "error"
            cmd = list(argv)
            if "--max-hours" in cmd:
                cmd[cmd.index("--max-hours") + 1] = f"{cap:.3f}"
            code = self.run_cmd(name, cmd, gpu=True)
            if code == 0 or (code == 3 and accept_3):
                self.record(name, "ok", code=code, reason="max-hours" if code == 3 else "")
                return "ok"
            if code == 4:
                self.note_held()
                gate_wait(self.pid)
                continue
            if code == 2:
                print("gpu lock lost; not deleting a lock we no longer own", flush=True)
                gate_wait(self.pid)
                continue
            if code == 3:
                self.record(name, "stopped", code=3, reason="max-hours")
                return "budget"
            if code == 1 and accept_1:
                self.record(
                    name,
                    "skipped",
                    code=1,
                    reason="synth skipped: stage failed; kept the verified round-5 tasks",
                )
                return "skip"
            self.record(name, "error", code=code, reason=f"exit {code}")
            if name == "b-train":
                self.block_train = True
                self.block_reason = "train failed"
            return "error"


def _hold_pause(pipe: Pipeline) -> None:
    if HOLD_PATH.is_file() and pipe.own_lock:
        pipe.note_held()
        gate_wait(pipe.pid)
        pipe.begin_cpu()


def snapshot_v1_eval() -> None:
    dest = PIPE_DIR / "asm-distill-7b-eval.snapshot.jsonl"
    if dest.is_file() or not V1_EVAL.is_file():
        return
    copy_if_missing(V1_EVAL, dest)
    print(f"snapshotted {V1_EVAL} ({_count_lines(dest)} rows) to {dest}", flush=True)


def _eval_cmd(split: str, *, model_name: str, adapter: Path, out: Path) -> list[str]:
    tasks = HELDOUT_TASKS if split == "heldout" else SCALED_TASKS
    return [
        PY, "-m", "e2m.train.eval_generate",
        "--model", str(MODEL_7B),
        "--adapter", str(adapter),
        "--name", model_name,
        "--mode", "asm",
        "--split", split,
        "--tasks", str(tasks),
        "--val-split", str(VAL_SPLIT),
        "--out", str(out),
        "--4bit",
        "--max-hours", "1.000",
    ]


def _loop_cmd(split: str, *, model_name: str, adapter: Path, out: Path, mode: str) -> list[str]:
    tasks = HELDOUT_TASKS if split == "heldout" else SCALED_TASKS
    return [
        PY, "-m", "e2m.loop", "run",
        "--model", str(MODEL_7B),
        "--adapter", str(adapter),
        "--name", model_name,
        "--mode", mode,
        "--split", split,
        "--tasks", str(tasks),
        "--val-split", str(VAL_SPLIT),
        "--out", str(out),
        "--k", "8",
        "--rounds", "3",
        "--max-hours", "1.000",
        "--4bit",
    ]


def _v2_ready() -> bool:
    return (V2 / "adapter" / "adapter_config.json").is_file()


def ollama_qwen_reason() -> str | None:
    """Tags check only. Does not load qwen3:14b. Call after the lock is held."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return (
            "synth skipped: ollama tags failed "
            f"({exc}). Kept the round-5 synth tasks that already passed the harness."
        )
    names = [str(item.get("name") or "") for item in (body.get("models") or [])]
    if any(name == "qwen3:14b" or name.startswith("qwen3:14b") for name in names):
        return None
    return (
        "synth skipped: qwen3:14b is not in ollama tags "
        f"{names}. Kept the round-5 synth tasks that already passed the harness."
    )


def run_stages(pipe: Pipeline) -> None:
    synth_path = PIPE_DIR / "synth_tasks.jsonl"
    sample_tasks = PIPE_DIR / "sample_tasks.jsonl"
    sample_out = PIPE_DIR / "sample_c.jsonl"
    distill_path = PIPE_DIR / "distill.jsonl"
    lower_state_path = PIPE_DIR / "lower_state.json"
    text_kept = PIPE_DIR / "text_kept.jsonl"
    text_dropped = PIPE_DIR / "text_dropped.jsonl"
    embed_kept = PIPE_DIR / "embed_kept.jsonl"
    embed_dropped = PIPE_DIR / "embed_dropped.jsonl"
    embed_cache = PIPE_DIR / "embed_cache.jsonl"
    embed_report = PIPE_DIR / "embed_report.json"
    sft_out = PIPE_DIR / "sft.jsonl"
    stats: dict[str, Any] = {}

    def counted_eval(
        name: str,
        cmd: list[str],
        *,
        total: int,
        count_fn: Callable[[], int],
        retry: bool = False,
    ) -> None:
        """Score until ``total`` or the budget is gone.

        ``retry`` is for the non-negotiable direct evals. One max-hours stop
        recomputes a cap for the tasks still missing and continues in this
        process, so Stage B does not start while Stage A is short. An exit 0
        that did not score everyone stays ``stopped`` and is resumed later.
        """
        done = count_fn()
        reason = pipe.skip_reason(name)
        if reason and done < total and name in pipe.frozen_ok:
            # A previous process marked the stage ok before the file was complete.
            pipe.frozen_ok.discard(name)
            pipe.prior_ok.discard(name)
            reason = None
        if reason:
            pipe.record(name, "skipped", code=0 if reason.startswith("already") else None, reason=reason)
            return
        if done >= total:
            pipe.record(name, "ok", code=0, reason="already scored")
            return
        stalls = 0
        last = done
        while True:
            if retry:
                pipe.stages.pop(name, None)
            outcome = pipe.run_gpu(name, cmd, accept_3=False)
            done = count_fn()
            info = dict(pipe.stages.get(name) or {})
            if done >= total:
                pipe.record(
                    name,
                    "ok",
                    code=info.get("code") if info.get("code") is not None else 0,
                    reason=info.get("reason") or "",
                )
                return
            if info.get("status") == "ok":
                pipe.record(name, "stopped", code=info.get("code"), reason="incomplete after exit 0")
                info = dict(pipe.stages.get(name) or {})
            if not retry or info.get("status") == "error" or outcome == "skip":
                return
            if remaining_hours() <= 0.05 or (pipe.halt and outcome == "budget"):
                if (pipe.stages.get(name) or {}).get("status") not in TERMINAL:
                    pipe.record(name, "stopped", code=3, reason="incomplete")
                return
            if done <= last:
                stalls += 1
            else:
                stalls = 0
            last = done
            if stalls >= 2:
                pipe.record(
                    name,
                    "stopped",
                    code=info.get("code") if info.get("code") is not None else 3,
                    reason="no progress",
                )
                return
            pipe.stages.pop(name, None)
            if pipe.stage_cap(name) < 0.05:
                pipe.record(name, "stopped", code=3, reason="incomplete")
                return

    snapshot_v1_eval()
    counted_eval(
        "a-eval-heldout",
        _eval_cmd("heldout", model_name=V1_NAME, adapter=V1 / "adapter", out=V1_EVAL),
        total=HELDOUT_N,
        count_fn=lambda: len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="heldout")),
        retry=True,
    )
    counted_eval(
        "a-eval-val",
        _eval_cmd("val", model_name=V1_NAME, adapter=V1 / "adapter", out=V1_EVAL),
        total=VAL_N,
        count_fn=lambda: len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="val")),
        retry=True,
    )

    def write_a() -> None:
        write_stage_a_files(pipe.stages)

    pipe.run_cpu("a-results", write_a)

    def prep_synth_copy() -> None:
        copied = copy_if_missing(ROUND5_DIR / "synth_tasks.jsonl", synth_path)
        print(f"synth copy={copied} rows={_count_lines(synth_path)}", flush=True)

    prep_synth_copy()
    synth_cmd = [
        PY, "-m", "e2m.loop.round5_sample", "synth",
        "--out", str(synth_path),
        "--max-hours", f"{SYNTH_MAX_H:.3f}",
        "--cap", str(SYNTH_CAP),
        "--fail-streak", "40",
        "--ollama-giveup", "5",
    ]
    synth_skip = pipe.skip_reason("b-synth")
    if synth_skip:
        pipe.record("b-synth", "skipped", code=0 if synth_skip.startswith("already") else None, reason=synth_skip)
    elif _count_lines(synth_path) >= SYNTH_CAP:
        pipe.record("b-synth", "ok", code=0, reason="already at cap")
    else:
        pipe.run_gpu("b-synth", synth_cmd, accept_3=True, accept_1=True, preflight=ollama_qwen_reason)

    def write_sample_list() -> None:
        rows = []
        for path in (ROUND5_DIR / "screened_mbpp.jsonl", synth_path, ROUND5_DIR / "screened_train.jsonl"):
            rows.extend(_read_jsonl(path))
        sample_tasks.parent.mkdir(parents=True, exist_ok=True)
        with sample_tasks.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"sample list {len(rows)}", flush=True)

    copy_if_missing(ROUND5_DIR / "sample_c.jsonl", sample_out)
    write_sample_list()
    sample_cmd = [
        PY, "-m", "e2m.loop.round5_sample", "sample",
        "--tasks", str(sample_tasks),
        "--out", str(sample_out),
        "--model", str(MODEL_7B),
        "--adapter", str(TEACHER),
        "--4bit",
        "--k", str(SAMPLE_K),
        "--ensure-k", str(SAMPLE_K),
        "--prior-k", str(PRIOR_K),
        "--temperature", "0.8",
        "--max-new", "1024",
        "--max-hours", f"{SAMPLE_MAX_H:.3f}",
    ]
    sample_skip = pipe.skip_reason("b-sample")
    if sample_skip:
        pipe.record("b-sample", "skipped", code=0 if sample_skip.startswith("already") else None, reason=sample_skip)
    elif _sample_topped(sample_tasks, sample_out):
        pipe.record("b-sample", "ok", code=0, reason=f"already at k={SAMPLE_K}")
    else:
        pipe.run_gpu("b-sample", sample_cmd, accept_3=True)

    def lower() -> None:
        task_map = _task_index(
            [
                ROUND5_DIR / "screened_mbpp.jsonl",
                ROUND5_DIR / "screened_train.jsonl",
                synth_path,
                ROUND5_DIR / "mbpp_tasks.jsonl",
                ROUND5_DIR / "train_tasks.jsonl",
            ]
        )
        state: dict[str, Any] = {}
        if lower_state_path.is_file():
            try:
                state = json.loads(lower_state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        latest = _latest_done_rows(sample_out)
        distill_path.parent.mkdir(parents=True, exist_ok=True)
        new = 0
        skipped = 0
        with distill_path.open("a", encoding="utf-8") as fh:
            for index, (tid, row) in enumerate(latest.items(), 1):
                if index % 25 == 0:
                    _hold_pause(pipe)
                texts = [str(text) for text in (row.get("c_texts") or []) if str(text).strip()]
                key = _c_key(texts)
                prev = state.get(tid) or {}
                if prev.get("c_key") == key:
                    skipped += 1
                    continue
                task = task_map.get(tid)
                pairs: list[dict[str, Any]] = []
                if task is not None and texts:
                    pairs = distill_pairs(
                        task,
                        texts,
                        origin=str(row.get("origin") or task.get("origin") or ""),
                        max_distinct=0,
                    )
                for pair in pairs:
                    pair["c_key"] = key
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    new += 1
                fh.flush()
                state[tid] = {"c_key": key, "n_pairs": len(pairs)}
                if index % 20 == 0:
                    lower_state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
                    print(f"lower {index}/{len(latest)} new_rows={new}", flush=True)
        lower_state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
        stats["lower"] = {"tasks": len(latest), "new_rows": new, "skipped": skipped}
        print(f"lower tasks={len(latest)} new_rows={new} skipped={skipped}", flush=True)

    pipe.run_cpu("b-lower", lower)

    def latest_distill() -> list[dict[str, Any]]:
        state: dict[str, Any] = {}
        if lower_state_path.is_file():
            try:
                state = json.loads(lower_state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        rows = []
        for row in _read_jsonl(distill_path):
            tid = str(row.get("task_id") or "")
            info = state.get(tid) or {}
            if info and row.get("c_key") == info.get("c_key"):
                rows.append(row)
        return rows

    def leakage_text() -> None:
        index = _task_index([SCALED_TASKS, ROUND5_DIR / "mbpp_tasks.jsonl", synth_path, ROUND5_DIR / "train_tasks.jsonl"])
        rows = attach_reference_c(load_sft_rows(SFT_LORA2), index)
        rows.extend(latest_distill())
        kept, dropped, summary = screen_text(rows)
        _write_jsonl(text_kept, kept)
        _write_jsonl(text_dropped, dropped)
        summary["dropped_ids"] = [
            {"task_id": row.get("task_id"), "origin": row.get("origin"), "reason": row.get("reason")}
            for row in dropped
        ]
        summary["word5_rule"] = WORD5_RULE
        stats["leakage_text"] = {key: val for key, val in summary.items() if key != "dropped_ids"}
        (PIPE_DIR / "leakage_text.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"leakage text kept={summary.get('kept_text')} dropped={summary.get('dropped_text')}", flush=True)

    pipe.run_cpu("b-leakage-text", leakage_text)

    def embed_cpu() -> None:
        if not text_kept.is_file():
            raise RuntimeError("text_kept missing")
        copied = copy_if_missing(ROUND5_DIR / "embed_cache.jsonl", embed_cache)
        print(f"embed cache copy={copied}", flush=True)
        cmd = [
            PY, "-m", "e2m.loop.round5_sample", "embed",
            "--rows", str(text_kept),
            "--kept", str(embed_kept),
            "--dropped", str(embed_dropped),
            "--cache", str(embed_cache),
            "--report", str(embed_report),
            "--max-hours", "1.000",
            "--cpu",
        ]
        pipe.begin_cpu()
        code = pipe.run_cmd("b-leakage-embed", cmd, gpu=False)
        if code == 4:
            raise RuntimeError("held")
        if code not in {0, 3}:
            raise RuntimeError(f"cpu embed exit {code}")

    reason = pipe.skip_reason("b-leakage-embed")
    embed_done = embed_report.is_file() and '"embedded": true' in embed_report.read_text(encoding="utf-8")
    if reason:
        pipe.record("b-leakage-embed", "skipped", code=0 if reason.startswith("already") else None, reason=reason)
    elif pipe.dry_run:
        pipe.record("b-leakage-embed", "planned", reason="dry-run cpu embed")
    elif pipe.smoke:
        pipe.record("b-leakage-embed", "skipped", reason="smoke")
    elif embed_done and pipe.resume:
        pipe.record("b-leakage-embed", "skipped", code=0, reason="already embedded")
    else:
        pipe.begin_cpu()
        for _attempt in range(2):
            try:
                embed_cpu()
                break
            except RuntimeError as exc:
                if "held" in str(exc) and pipe.own_lock:
                    pipe.note_held()
                    gate_wait(pipe.pid)
                    pipe.begin_cpu()
                    continue
                print(f"cpu embed: {exc}", flush=True)
                break
        embedded = False
        if embed_report.is_file():
            try:
                embedded = bool(json.loads(embed_report.read_text(encoding="utf-8")).get("embedded"))
            except json.JSONDecodeError:
                embedded = False
        if embedded:
            pipe.record("b-leakage-embed", "ok", code=0)
        else:
            pipe.record("b-leakage-embed", "error", code=1, reason="embedding screen incomplete")
            pipe.block_train = True
            pipe.block_reason = "embedding leakage check failed"

    def mix() -> None:
        if not embed_kept.is_file():
            raise RuntimeError("embed_kept missing; refusing to train on unchecked rows")
        rows = _read_jsonl(embed_kept)
        stats["mix"] = write_mix(sft_out, kept_rows=rows, sft_fraction=SFT_FRACTION)
        stats["train_pool"] = {
            "enlarged": False,
            "reason": (
                "Reused runs/round5 screened_train.jsonl (346 tasks) and screened_mbpp.jsonl. "
                "The 8 h budget is reserved for Stage A eval, the k=8 top-up, training, and the v2 held-out eval."
            ),
        }
        (PIPE_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"mix {stats['mix']}", flush=True)
        if not stats["mix"].get("distill_majority"):
            raise RuntimeError(f"distill is not the majority: {stats['mix'].get('by_origin')}")

    if pipe.block_train and pipe.block_reason.startswith("embedding"):
        pipe.record("b-mix", "skipped", reason=pipe.block_reason)
    else:
        status = pipe.run_cpu("b-mix", mix)
        if status == "error":
            pipe.block_train = True
            pipe.block_reason = pipe.block_reason or "mix failed"

    train_cmd = [
        PY, "-m", "e2m.train.sft_lora",
        "--model", str(MODEL_7B),
        "--sft", str(sft_out),
        "--out", str(V2),
        "--val-split", str(VAL_SPLIT),
        "--epochs", "2",
        "--lr", "2e-4",
        "--rank", "32",
        "--alpha", "64",
        "--dropout", "0.05",
        "--batch-size", "1",
        "--grad-accum", "16",
        "--4bit",
        "--target", "asm",
        "--max-len", "2048",
        "--goals", "default,size",
        "--save-steps", "100",
        "--max-hours", f"{TRAIN_MAX_H:.3f}",
        "--hold-file", str(HOLD_PATH),
        "--resume",
    ]
    if pipe.block_train or not sft_out.is_file():
        pipe.record("b-train", "skipped", reason=pipe.block_reason or "mix missing")
    elif _train_done():
        pipe.record("b-train", "ok", code=0, reason="adapter already written")
    else:
        # An earlier ok without a saved adapter is not finished. Train again.
        pipe.frozen_ok.discard("b-train")
        pipe.prior_ok.discard("b-train")
        pipe.run_gpu("b-train", train_cmd, accept_3=True)

    if not _v2_ready():
        why = pipe.block_reason or "v2 adapter missing"
        pipe.record("b-eval-heldout", "skipped", reason=why)
    else:
        counted_eval(
            "b-eval-heldout",
            _eval_cmd("heldout", model_name=V2_NAME, adapter=V2 / "adapter", out=V2_EVAL),
            total=HELDOUT_N,
            count_fn=lambda: len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="heldout")),
            retry=True,
        )

    def run_loop(name: str, cmd: list[str], *, ready: bool, why: str) -> None:
        if not ready:
            pipe.record(name, "skipped", reason=why)
            return
        pipe.run_gpu(name, cmd, accept_3=True)

    run_loop(
        "a-loop-c-then-asm",
        _loop_cmd("heldout", model_name=V1_NAME, adapter=V1 / "adapter", out=A_LOOP, mode="c-then-asm"),
        ready=True,
        why="",
    )
    if not _v2_ready():
        pipe.record("b-eval-val", "skipped", reason=pipe.block_reason or "v2 adapter missing")
    else:
        counted_eval(
            "b-eval-val",
            _eval_cmd("val", model_name=V2_NAME, adapter=V2 / "adapter", out=V2_EVAL),
            total=VAL_N,
            count_fn=lambda: len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="val")),
        )
    run_loop(
        "b-loop-c-then-asm",
        _loop_cmd("heldout", model_name=V2_NAME, adapter=V2 / "adapter", out=B_LOOP, mode="c-then-asm"),
        ready=_v2_ready(),
        why=pipe.block_reason or "v2 adapter missing",
    )
    run_loop(
        "a-loop-asm",
        _loop_cmd("heldout", model_name=V1_NAME, adapter=V1 / "adapter", out=A_LOOP_ASM, mode="asm"),
        ready=True,
        why="",
    )
    run_loop(
        "b-loop-val",
        _loop_cmd("val", model_name=V2_NAME, adapter=V2 / "adapter", out=B_LOOP_VAL, mode="c-then-asm"),
        ready=_v2_ready(),
        why=pipe.block_reason or "v2 adapter missing",
    )
    run_loop(
        "a-loop-val",
        _loop_cmd("val", model_name=V1_NAME, adapter=V1 / "adapter", out=A_LOOP_VAL, mode="c-then-asm"),
        ready=True,
        why="",
    )

    def leftover() -> None:
        jobs = [
            ("a-heldout", HELDOUT_N, lambda: len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="heldout")),
             _eval_cmd("heldout", model_name=V1_NAME, adapter=V1 / "adapter", out=V1_EVAL)),
            ("a-val", VAL_N, lambda: len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="val")),
             _eval_cmd("val", model_name=V1_NAME, adapter=V1 / "adapter", out=V1_EVAL)),
        ]
        if _v2_ready():
            jobs.extend(
                [
                    ("b-heldout", HELDOUT_N, lambda: len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="heldout")),
                     _eval_cmd("heldout", model_name=V2_NAME, adapter=V2 / "adapter", out=V2_EVAL)),
                    ("b-val", VAL_N, lambda: len(scored_ids(V2_EVAL, model=V2_NAME, mode="asm", split="val")),
                     _eval_cmd("val", model_name=V2_NAME, adapter=V2 / "adapter", out=V2_EVAL)),
                ]
            )
        notes = []
        for label, total, count_fn, cmd in jobs:
            missing = total - count_fn()
            if missing <= 0:
                continue
            if remaining_hours() < 0.12:
                notes.append(f"{label} still missing {missing}; no budget")
                break
            cap = min(eval_cap(missing), remaining_hours())
            if cap < 0.08:
                notes.append(f"{label} still missing {missing}; cap {cap:.3f}")
                continue
            if not pipe.gpu_ready():
                notes.append(f"{label} lock not acquired")
                break
            cmd = list(cmd)
            cmd[cmd.index("--max-hours") + 1] = f"{cap:.3f}"
            code = pipe.run_cmd(f"leftover-{label}", cmd, gpu=True)
            notes.append(f"{label} exit {code} missing {total - count_fn()}")
            if code == 4:
                pipe.note_held()
                gate_wait(pipe.pid)
        if not notes:
            notes.append("direct eval already complete")
        stats["leftover"] = notes
        print("leftover " + "; ".join(notes), flush=True)

    # Always reconsider incomplete direct evals. This stage is not skipped just
    # because an earlier pass stopped on its own cap.
    if pipe.skip_reason("leftover-eval") and "already" in (pipe.skip_reason("leftover-eval") or ""):
        pipe.record("leftover-eval", "skipped", code=0, reason="already recorded ok")
    elif pipe.dry_run:
        pipe.record("leftover-eval", "planned", reason="dry-run")
    elif pipe.smoke:
        pipe.record("leftover-eval", "skipped", reason="smoke")
    else:
        pipe.run_cpu("leftover-eval", leftover)

    def failure() -> None:
        v1 = write_failure(V1_EVAL, PIPE_DIR / "failure_v1", model_name=V1_NAME, split="heldout")
        v2 = write_failure(V2_EVAL, PIPE_DIR / "failure_v2", model_name=V2_NAME, split="heldout")
        stats["failure_v1"] = {key: val for key, val in v1.items() if key != "failures"}
        stats["failure_v2"] = {key: val for key, val in v2.items() if key != "failures"}
        _write_failure_md(v1, v2)
        (PIPE_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")

    pipe.run_cpu("failure-analysis", failure)
    pipe.begin_cpu()
    _write_leakage(stats)
    write_stage_a_files(pipe.stages)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_failure_md(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    def _table(payload: dict[str, Any], title: str) -> list[str]:
        tally = payload.get("tally") or {}
        lines = [
            f"## {title}",
            "",
            f"Scored {payload.get('n')}. Passes {payload.get('pass_n')}. "
            f"Hybrid overlap (of {payload.get('hybrid_pass_n')} round-4 hybrid passes, not re-scored): "
            f"{payload.get('overlap_pass_n')}.",
            "",
            "| bucket | n |",
            "|---|---:|",
        ]
        for bucket in ("doesnt_assemble", "abi", "logic_mismatch", "crash", "timeout", "other"):
            lines.append(f"| {bucket} | {tally.get(bucket, 0)} |")
        lines.append("")
        return lines

    lines = [
        "# Round 5b failure analysis",
        "",
        "Buckets use `e2m.train.failure_analysis.classify_asm_fail` "
        "(`wrong_answer` is `logic_mismatch`). The 48 hybrid-pass tasks are copied from round 4.",
        "",
    ]
    lines.extend(_table(v1, "Stage A direct asm (asm-distill-7b, held-out)"))
    lines.extend(_table(v2, "Stage B direct asm (asm-distill-7b-v2, held-out)"))
    lines.append("")
    (PIPE_DIR / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leakage(stats: dict[str, Any]) -> None:
    text: dict[str, Any] = {}
    path = PIPE_DIR / "leakage_text.json"
    if path.is_file():
        try:
            text = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            text = {}
    embed: dict[str, Any] = {}
    report = PIPE_DIR / "embed_report.json"
    if report.is_file():
        try:
            embed = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            embed = {}
    dropped = text.get("dropped_ids") or []
    embed_dropped = _read_jsonl(PIPE_DIR / "embed_dropped.jsonl")
    payload = {
        "word5_rule": WORD5_RULE,
        "thresholds": {"spec_ratio": 0.85, "c_ratio": 0.85, "embed_cosine": 0.85, "word_n": 5},
        "text": {key: val for key, val in text.items() if key != "dropped_ids"},
        "embed": embed,
        "by_check": (text.get("by_check") or {}) | ({"embed": embed.get("dropped")} if embed.get("dropped") else {}),
        "by_source": text.get("by_source") or {},
        "dropped_ids": dropped + [
            {"task_id": row.get("task_id"), "origin": row.get("origin"), "reason": row.get("reason")}
            for row in embed_dropped
        ],
    }
    (PIPE_DIR / "leakage.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _eval_summary(path: Path, model: str, split: str) -> dict[str, Any]:
    rows = [
        row
        for row in _read_jsonl(path)
        if row.get("model") == model and row.get("mode") == "asm" and row.get("split") == split
    ]
    by_id = {str(row.get("task_id")): row for row in rows}
    rows = list(by_id.values())
    n = len(rows)
    pass1 = 0
    pass5 = 0.0
    sizes: list[float] = []
    insns: list[float] = []
    cats: dict[str, int] = {}
    for row in rows:
        greedy = row.get("greedy") or {}
        cat = str(greedy.get("category") or "other")
        cats[cat] = cats.get(cat, 0) + 1
        if greedy.get("ok"):
            pass1 += 1
            gcc = row.get("gcc") or {}
            if gcc.get("ok") and gcc.get("size") and greedy.get("size"):
                sizes.append(float(greedy["size"]) / float(gcc["size"]))
            if gcc.get("ok") and gcc.get("insn") and greedy.get("insn"):
                insns.append(float(greedy["insn"]) / float(gcc["insn"]))
        samples = row.get("samples") or []
        correct = sum(1 for sample in samples if sample.get("ok"))
        k = min(5, len(samples))
        if k:
            pass5 += pass_at_k(len(samples), correct, k)
    expect = HELDOUT_N if split == "heldout" else VAL_N

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "model": model,
        "split": split,
        "n": n,
        "expected": expect,
        "partial": n < expect,
        "pass_at_1_n": pass1,
        "pass_at_1": (pass1 / n) if n else None,
        "pass_at_1_frac": f"{pass1}/{n}" if n else "0/0",
        "pass_at_5": (pass5 / n) if n else None,
        "size_ratio_vs_gcc_o2": _mean(sizes),
        "insn_ratio_vs_gcc_o2": _mean(insns),
        "size_ratio_n": len(sizes),
        "insn_ratio_n": len(insns),
        "greedy_categories": cats,
    }


def _loop_report(path: Path, *, model: str, split: str, mode: str, expected: int) -> dict[str, Any]:
    loop_path = path / "loop.jsonl"
    if not loop_path.is_file():
        return {"n": 0, "pass_n": 0, "partial": True, "expected": expected, "mode": mode, "note": "not started"}
    summary = summarize_loop(loop_path, model=model, split=split)
    stopped = None
    summary_path = path / "summary.json"
    if summary_path.is_file():
        try:
            stopped = json.loads(summary_path.read_text(encoding="utf-8")).get("stopped")
        except json.JSONDecodeError:
            stopped = None
    n = int(summary.get("n") or 0)
    return {
        "model": model,
        "mode": mode,
        "split": split,
        "n": n,
        "expected": expected,
        "partial": n < expected,
        "pass_n": summary.get("passed_full"),
        "pass_rate": summary.get("pass_full"),
        "size_ratio_vs_gcc_o2": summary.get("size_ratio_vs_gcc_o2"),
        "insn_ratio_vs_gcc_o2": summary.get("insn_ratio_vs_gcc_o2"),
        "final_categories": summary.get("final_categories"),
        "stopped": stopped,
        "path": str(path),
    }


def _failure_public(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: val for key, val in payload.items() if key != "failures"}


def stage_a_payload(stages: dict[str, Any]) -> dict[str, Any]:
    held = _eval_summary(V1_EVAL, V1_NAME, "heldout")
    val = _eval_summary(V1_EVAL, V1_NAME, "val")
    failure = _failure_public(write_failure(V1_EVAL, PIPE_DIR / "failure_v1", model_name=V1_NAME, split="heldout"))
    loop_c = _loop_report(A_LOOP, model=V1_NAME, split="heldout", mode="c-then-asm", expected=HELDOUT_N)
    loop_asm = _loop_report(A_LOOP_ASM, model=V1_NAME, split="heldout", mode="asm", expected=HELDOUT_N)
    partial = bool(held["partial"] or val["partial"] or loop_c["partial"])
    return {
        "title": "E2M round 5 full eval",
        "model": V1_NAME,
        "adapter": str(V1 / "adapter"),
        "note": (
            "Direct asm resumes runs/asm-distill-7b/eval.jsonl (append; the first 52 rows "
            "are snapshotted under runs/round5b). The C-then-asm loop is a new directory. "
            "Round 4 numbers are copied, not re-scored."
        ),
        "partial": partial,
        "heldout_asm": held,
        "val_asm": val,
        "loop_heldout_c_then_asm": loop_c,
        "loop_heldout_asm": loop_asm,
        "failure": failure,
        "round4": round4_comparison(),
        "stages": {name: stages.get(name) for name in ("a-eval-heldout", "a-eval-val", "a-loop-c-then-asm", "a-loop-asm", "a-loop-val")},
    }


def write_stage_a_files(stages: dict[str, Any]) -> None:
    if os.environ.get("E2M_SMOKE") == "1":
        return
    payload = stage_a_payload(stages)
    FULL_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_stage_a_md(payload, FULL_MD)
    print("wrote", FULL_JSON, FULL_MD, flush=True)


def write_stage_a_md(results: dict[str, Any], path: Path) -> None:
    held = results.get("heldout_asm") or {}
    val = results.get("val_asm") or {}
    loop_c = results.get("loop_heldout_c_then_asm") or {}
    loop_a = results.get("loop_heldout_asm") or {}
    failure = results.get("failure") or {}
    prior = results.get("round4") or {}
    partial = "partial" if results.get("partial") else "complete"
    lines = [
        "# Round 5 full eval (stage A of round 5b)",
        "",
        f"Status: {partial}. Model `asm-distill-7b`. "
        "The original partial `runs/round5/RESULTS.json` and `SUMMARY.md` are unchanged.",
        "",
        results.get("note") or "",
        "",
        "## Held-out direct asm",
        "",
        f"Scored n={held.get('n')}/{held.get('expected')}. "
        f"pass@1 {held.get('pass_at_1_n')}/{held.get('n')} ({held.get('pass_at_1')}). "
        f"pass@5 {held.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {held.get('size_ratio_vs_gcc_o2')} (n={held.get('size_ratio_n')}). "
        f"Insn ratio: {held.get('insn_ratio_vs_gcc_o2')} (n={held.get('insn_ratio_n')}).",
        f"Greedy categories: `{held.get('greedy_categories')}`.",
        f"Failure buckets: `{failure.get('tally')}`.",
        f"Overlap with the {failure.get('hybrid_pass_n')} round-4 hybrid-pass tasks: "
        f"{failure.get('overlap_pass_n')} `{failure.get('overlap_pass_ids')}`.",
        "",
        f"C-then-asm repair loop (k=8, R=3): pass {loop_c.get('pass_n')} / n={loop_c.get('n')} "
        f"of {loop_c.get('expected')}. partial={loop_c.get('partial')}. stopped={loop_c.get('stopped')}.",
        f"Loop size ratio: {loop_c.get('size_ratio_vs_gcc_o2')}. Insn ratio: {loop_c.get('insn_ratio_vs_gcc_o2')}.",
        f"Loop categories: `{loop_c.get('final_categories')}`.",
        "",
        f"Direct-asm repair loop (only if budget remained): pass {loop_a.get('pass_n')} / n={loop_a.get('n')}. "
        f"note={loop_a.get('note') or loop_a.get('stopped')}.",
        "",
        "## Val direct asm",
        "",
        f"Scored n={val.get('n')}/{val.get('expected')}. "
        f"pass@1 {val.get('pass_at_1_n')}/{val.get('n')} ({val.get('pass_at_1')}). "
        f"pass@5 {val.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {val.get('size_ratio_vs_gcc_o2')}. "
        f"Insn ratio: {val.get('insn_ratio_vs_gcc_o2')}.",
        f"Greedy categories: `{val.get('greedy_categories')}`.",
        "",
        "## Round 4 comparison (copied, not re-scored)",
        "",
        f"Direct asm pass@1 {prior.get('heldout_direct_asm_pass_at_1')}, "
        f"pass@5 {prior.get('heldout_direct_asm_pass_at_5')}, "
        f"C-then-asm loop {prior.get('heldout_c_then_asm_loop')}, "
        f"hybrid gcc -O2 {prior.get('heldout_hybrid_o2')}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_results(stages: dict[str, Any]) -> dict[str, Any]:
    data_stats: dict[str, Any] = {}
    stats_path = PIPE_DIR / "data_stats.json"
    if stats_path.is_file():
        try:
            data_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data_stats = {}
    stage_a = {}
    if FULL_JSON.is_file():
        try:
            stage_a = json.loads(FULL_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stage_a = {}
    held_b = _eval_summary(V2_EVAL, V2_NAME, "heldout")
    val_b = _eval_summary(V2_EVAL, V2_NAME, "val")
    return {
        "title": "E2M round 5b",
        "target_heldout_direct_asm_pass_at_1": "20/78",
        "target_met": bool(held_b.get("pass_at_1_n") is not None and held_b.get("pass_at_1_n") >= 20 and not held_b.get("partial")),
        "teacher": "runs/c-only-7b/adapter",
        "stage_a_model": V1_NAME,
        "stage_b_model": V2_NAME,
        "train_out": str(V2),
        "epochs": 2,
        "hyperparameters": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "lr": 2e-4,
            "batch_size": 1,
            "grad_accum": 16,
            "max_len": 2048,
            "load_4bit": True,
            "goals": ["default", "size"],
            "save_steps": 100,
        },
        "sample_k": SAMPLE_K,
        "prior_k": PRIOR_K,
        "sft_fraction_cap": SFT_FRACTION,
        "max_distinct_asms": "unlimited",
        "auxiliary_c_to_asm": False,
        "round4": round4_comparison(),
        "stage_a": {
            "heldout_asm": stage_a.get("heldout_asm"),
            "val_asm": stage_a.get("val_asm"),
            "loop_heldout_c_then_asm": stage_a.get("loop_heldout_c_then_asm"),
            "loop_heldout_asm": stage_a.get("loop_heldout_asm"),
            "failure": stage_a.get("failure"),
            "partial": stage_a.get("partial"),
        },
        "heldout_asm": held_b,
        "val_asm": val_b,
        "loop_heldout": _loop_report(B_LOOP, model=V2_NAME, split="heldout", mode="c-then-asm", expected=HELDOUT_N),
        "loop_val": _loop_report(B_LOOP_VAL, model=V2_NAME, split="val", mode="c-then-asm", expected=VAL_N),
        "data_stats": data_stats,
        "stages": stages,
        "gpu_hours_budget": ROUND5B_GPU_HOURS,
        "gpu_hours_used": round(gpu_hours_used(), 4),
        "gpu_hours": _read_jsonl(HOURS_PATH),
    }


def write_summary_md(results: dict[str, Any], path: Path) -> None:
    held = results.get("heldout_asm") or {}
    val = results.get("val_asm") or {}
    loop_h = results.get("loop_heldout") or {}
    stage_a = (results.get("stage_a") or {}).get("heldout_asm") or {}
    stage_a_loop = (results.get("stage_a") or {}).get("loop_heldout_c_then_asm") or {}
    mix = ((results.get("data_stats") or {}).get("mix") or {})
    prior = results.get("round4") or {}
    lines = [
        "# Round 5b: finish round 5, then distill-heavy retrain",
        "",
        "Stage A finishes `runs/asm-distill-7b`. Stage B trains `runs/asm-distill-7b-v2` "
        "with distill rows in the majority and `sft_lora2` capped at about 25%. "
        "Teacher samples resume `runs/round5/sample_c.jsonl` (copied, not rewritten) and "
        "add temperature completions up to k=8. Every distinct passing asm is kept.",
        "",
        f"GPU hours: {results.get('gpu_hours_used')} / {results.get('gpu_hours_budget')}.",
        "",
        "## Stage A (round-5 adapter, completed here)",
        "",
        f"Held-out direct asm n={stage_a.get('n')}. pass@1 {stage_a.get('pass_at_1_frac')} "
        f"({stage_a.get('pass_at_1')}). pass@5 {stage_a.get('pass_at_5')}.",
        f"C-then-asm loop pass {stage_a_loop.get('pass_n')} / n={stage_a_loop.get('n')} "
        f"(partial={stage_a_loop.get('partial')}).",
        "Full writeup: `runs/round5/SUMMARY_full.md`.",
        "",
        "## Stage B held-out direct asm",
        "",
        f"Scored n={held.get('n')}/{held.get('expected')}. "
        f"pass@1 {held.get('pass_at_1_n')}/{held.get('n')} ({held.get('pass_at_1')}). "
        f"pass@5 {held.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {held.get('size_ratio_vs_gcc_o2')}. "
        f"Insn ratio: {held.get('insn_ratio_vs_gcc_o2')}.",
        f"Greedy categories: `{held.get('greedy_categories')}`.",
        f"Target >= 20/78: {results.get('target_met')}.",
        "",
        f"C-then-asm repair loop (k=8, R=3): pass {loop_h.get('pass_n')} / n={loop_h.get('n')} "
        f"partial={loop_h.get('partial')} stopped={loop_h.get('stopped')}.",
        f"Loop size ratio: {loop_h.get('size_ratio_vs_gcc_o2')}. Insn ratio: {loop_h.get('insn_ratio_vs_gcc_o2')}.",
        "",
        "## Stage B val direct asm",
        "",
        f"Scored n={val.get('n')}/{val.get('expected')}. "
        f"pass@1 {val.get('pass_at_1_n')}/{val.get('n')} ({val.get('pass_at_1')}). "
        f"pass@5 {val.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {val.get('size_ratio_vs_gcc_o2')}. "
        f"Insn ratio: {val.get('insn_ratio_vs_gcc_o2')}.",
        "",
        "## Comparison",
        "",
        f"Round 4 direct asm {prior.get('heldout_direct_asm_pass_at_1')}, "
        f"C-then-asm loop {prior.get('heldout_c_then_asm_loop')}, "
        f"hybrid {prior.get('heldout_hybrid_o2')} (copied, not re-scored).",
        f"Stage A direct asm {stage_a.get('pass_at_1_frac')} on n={stage_a.get('n')}.",
        f"Stage B direct asm {held.get('pass_at_1_frac')} on n={held.get('n')}.",
        "",
        "## Data mix",
        "",
        f"Rows {mix.get('rows')}. Proportions `{mix.get('proportions')}`. "
        f"By origin `{mix.get('by_origin')}`. Distill majority: {mix.get('distill_majority')}.",
        f"sft cap: `{mix.get('sft_cap')}`.",
        "",
        "Epochs 2, same hyperparameters as round 5 (r=32, alpha=64, dropout 0.05, "
        "lr 2e-4, batch 1, grad accum 16, 4-bit, max len 2048, goals default,size). "
        "The train pool was not enlarged past the round-5 screened 346 tasks.",
        "",
        "## Stages",
        "",
        "| stage | status | code | reason |",
        "|---|---|---:|---|",
    ]
    for name in STAGE_ORDER:
        info = (results.get("stages") or {}).get(name) or {}
        code = info.get("code")
        lines.append(
            f"| {name} | {info.get('status')} | {'' if code is None else code} | {info.get('reason') or ''} |"
        )
    if results.get("halt_reason"):
        lines += ["", f"Halt: {results['halt_reason']}."]
    synth_reason = ((results.get("stages") or {}).get("b-synth") or {}).get("reason") or ""
    if "synth skipped" in synth_reason or "ollama" in synth_reason:
        lines += ["", f"Synth: {synth_reason}"]
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish(pipe: Pipeline, *, code_hint: int | None = None) -> int:
    if pipe.dry_run or pipe.smoke:
        return 0 if pipe.dry_run or not any(v.get("status") == "error" for v in pipe.stages.values()) else 1
    pipe.release()
    try:
        write_stage_a_files(pipe.stages)
    except Exception:
        traceback.print_exc()
    results = collect_results(pipe.stages)
    if pipe.halt_reason:
        results["halt_reason"] = pipe.halt_reason
    if pipe.block_reason:
        results["block_reason"] = pipe.block_reason
    (PIPE_DIR / "RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    write_summary_md(results, PIPE_DIR / "SUMMARY.md")
    reason = pipe.halt_reason or pipe.block_reason
    if not reason and any(v.get("status") == "error" for v in pipe.stages.values()):
        reason = "stage error"
    payload = {
        "ts": datetime.now(tz=PT).isoformat(),
        "gpu_hours_used": results.get("gpu_hours_used"),
        "stages": {key: val.get("status") for key, val in pipe.stages.items()},
        "reason": reason or "finished",
    }
    DONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DONE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", PIPE_DIR / "RESULTS.json", PIPE_DIR / "SUMMARY.md", DONE_PATH, flush=True)
    if code_hint is not None:
        return code_hint
    if any(v.get("status") == "error" for v in pipe.stages.values()):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2M round-5b GPU pipeline")
    parser.add_argument("--smoke", action="store_true", help="CPU-only tiny run under runs/round5b-smoke")
    parser.add_argument("--resume", action="store_true", help="Skip stages already recorded ok")
    parser.add_argument("--only", nargs="+", metavar="STAGE")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan; no lock and no done file")
    parser.add_argument("--lock-note", default=None)
    return parser


def smoke_main() -> int:
    """CPU checks only. Must not take ~/gpu.lock or load a model on the 3090."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["E2M_SMOKE"] = "1"
    lock = LOCK_PATH
    before = lock.read_text(encoding="utf-8") if lock.is_file() else None
    report: dict[str, Any] = {"cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "ok": True, "notes": []}
    plan = compute_allocation(
        remain_h=ROUND5B_GPU_HOURS,
        counts={
            "a-heldout": len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="heldout")),
            "a-val": len(scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="val")),
            "b-heldout": 0,
            "b-val": 0,
            "a-loop": 0,
            "b-loop": 0,
            "a-loop-asm": 0,
            "b-loop-val": 0,
            "a-loop-val": 0,
        },
        hours={},
        settled=set(),
        train_done=False,
        sample_topped=False,
        synth_done=False,
    )
    (PIPE_DIR / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan["allocation"], indent=2), flush=True)
    held_ids = scored_ids(V1_EVAL, model=V1_NAME, mode="asm", split="heldout")
    all_ids = []
    for row in _read_jsonl(HELDOUT_TASKS):
        tid = str(row.get("id") or "")
        if tid:
            all_ids.append(tid)
    missing = [tid for tid in all_ids if tid not in held_ids]
    report["scored_heldout"] = len(held_ids)
    report["missing_heldout"] = missing
    if len(held_ids) != 52 or set(missing) != set(BRIEF_UNFINISHED):
        report["ok"] = False
        report["notes"].append("held-out resume set does not match the brief")
    copy_path = PIPE_DIR / "eval_copy.jsonl"
    with V1_EVAL.open(encoding="utf-8") as src, copy_path.open("w", encoding="utf-8") as dst:
        kept = 0
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            kept += 1
            if kept >= 2:
                break
    from e2m.train.eval_generate import _done_ids

    copied = _done_ids(copy_path, V1_NAME, "asm", "heldout")
    scanned = scored_ids(copy_path, model=V1_NAME, mode="asm", split="heldout")
    report["resume_ids"] = sorted(copied)
    if copied != scanned or len(copied) != 2:
        report["ok"] = False
        report["notes"].append("eval resume ids did not match the copied jsonl")
    if _count_lines(V1_EVAL) != 52:
        report["ok"] = False
        report["notes"].append("snapshot source eval.jsonl line count changed")
    task_map = _task_index([ROUND5_DIR / "screened_mbpp.jsonl", ROUND5_DIR / "mbpp_tasks.jsonl"])
    compiled = 0
    mix_rows: list[dict[str, Any]] = []
    for row in _read_jsonl(ROUND5_DIR / "sample_c.jsonl"):
        texts = [text for text in (row.get("c_texts") or []) if str(text).strip()]
        task = task_map.get(str(row.get("task_id") or ""))
        if task is None or len(texts) < 1:
            continue
        pairs = distill_pairs(task, texts[:2], origin=str(row.get("origin") or "mbpp"), max_distinct=0)
        if not pairs:
            continue
        capped = distill_pairs(task, texts[:2], origin=str(row.get("origin") or "mbpp"), max_distinct=2)
        if len(capped) > 2:
            report["ok"] = False
            report["notes"].append("default max_distinct exceeded 2")
        mix_rows.extend(pairs)
        compiled += 1
        if compiled >= 2:
            break
    report["compiled_tasks"] = compiled
    if compiled < 1:
        report["ok"] = False
        report["notes"].append("distill_pairs produced no rows from existing sample_c")
    for index in range(12):
        mix_rows.append(
            {
                "task_id": f"sft_smoke_{index}",
                "spec": "smoke sft row",
                "signature": "int64_t smoke(int64_t a)",
                "goal": "default",
                "asm": "ret\n",
                "origin": "sft_lora2",
            }
        )
    mix_stats = write_mix(PIPE_DIR / "sft.jsonl", kept_rows=mix_rows, sft_fraction=SFT_FRACTION)
    report["mix"] = mix_stats
    if not mix_stats.get("distill_majority"):
        report["ok"] = False
        report["notes"].append("smoke mix is not distill-majority")
    frac = (mix_stats.get("proportions") or {}).get("sft_lora2", 0)
    if frac > SFT_FRACTION + 1e-9:
        report["ok"] = False
        report["notes"].append("sft fraction exceeded 0.25")
    after = lock.read_text(encoding="utf-8") if lock.is_file() else None
    report["lock_before_is_e2m"] = bool(before and before.startswith("e2m"))
    report["lock_after_is_ours"] = bool(after and f"e2m {os.getpid()}" in after)
    if report["lock_after_is_ours"]:
        report["ok"] = False
        report["notes"].append("smoke took ~/gpu.lock")
    (PIPE_DIR / "SMOKE.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("ok", "scored_heldout", "compiled_tasks", "notes", "lock_after_is_ours")}, indent=2), flush=True)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    global LOCK_NOTE
    args = build_parser().parse_args(argv)
    if args.lock_note:
        LOCK_NOTE = args.lock_note
    if args.smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["E2M_SMOKE"] = "1"
        os.environ["E2M_NO_DEADLINE"] = "1"
        configure_paths(smoke=True, out=args.out)
        print(f"round5b smoke pid={os.getpid()} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
        return smoke_main()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["E2M_NO_DEADLINE"] = "1"
    env["E2M_HOLD_FILE"] = str(HOLD_PATH)
    env.pop("E2M_SMOKE", None)
    configure_paths(smoke=False, out=args.out)
    pipe = Pipeline(
        smoke=False,
        resume=args.resume,
        dry_run=args.dry_run,
        only=set(args.only) if args.only else None,
        env=env,
    )
    if args.dry_run:
        plan = compute_allocation(
            remain_h=remaining_hours(),
            counts=read_progress()["counts"],
            hours=hours_map(),
            settled=pipe.settled(),
            train_done=_train_done(),
            sample_topped=False,
            synth_done=False,
        )
        print(json.dumps(plan, indent=2), flush=True)
        for name in STAGE_ORDER:
            if name in GPU_STAGES:
                pipe.record(name, "planned", reason=f"dry-run cap_h={plan['allocation'].get(name, 0):.3f}")
            else:
                pipe.record(name, "planned", reason="dry-run cpu")
        return 0
    if pipe.own_lock:
        (PIPE_DIR / "pipeline.pid").write_text(str(pipe.pid) + "\n", encoding="utf-8")

    def _release() -> None:
        pipe.release()

    atexit.register(_release)

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"signal {signum}, releasing gpu.lock", flush=True)
        proc = CURRENT_PROC.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        pipe.halt_reason = f"signal {signum}"
        try:
            finish(pipe, code_hint=128 + signum)
        except Exception:
            traceback.print_exc()
        _release()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    plan = pipe.plan()
    print(
        f"round5b start pid={pipe.pid} resume={args.resume} budget_h={ROUND5B_GPU_HOURS} "
        f"used_h={gpu_hours_used():.2f} alloc={json.dumps(plan.get('allocation'))}",
        flush=True,
    )
    try:
        run_stages(pipe)
        if remaining_hours() <= 0.05 and not pipe.halt_reason:
            pipe.halt_reason = "gpu-hour budget"
        code = finish(pipe)
        print(f"round5b complete code={code} gpu_hours={gpu_hours_used():.3f}", flush=True)
        return code
    except SystemExit:
        _release()
        raise
    except Exception:
        traceback.print_exc()
        pipe.halt_reason = pipe.halt_reason or "fatal error"
        try:
            finish(pipe, code_hint=1)
        except Exception:
            traceback.print_exc()
        _release()
        return 1
    finally:
        _release()


if __name__ == "__main__":
    raise SystemExit(main())
