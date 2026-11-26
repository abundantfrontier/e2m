"""Round-5 pipeline: distill the C hybrid into direct English-to-asm.

Self-driving. GPU work waits on gate_wait() and stops at an 8-hour cap.
``--resume`` skips stages already recorded ok. CPU stages release ~/gpu.lock.
``--smoke`` and ``--dry-run`` do not take the lock and do not write logs/round5.done.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from e2m.loop.guard import LOCK_PATH, PT, pid_alive, release_gpu_lock
from e2m.loop.round5_data import (
    SFT_LORA2,
    WORD5_RULE,
    attach_reference_c,
    build_mbpp_pool,
    build_train_manifest,
    distill_pairs,
    load_sft_rows,
    round4_comparison,
    screen_text,
    write_failure,
    write_mix,
    _read_jsonl,
    _task_index,
)

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL_7B = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"
TEACHER = ROOT / "runs" / "c-only-7b" / "adapter"
HELDOUT_TASKS = ROOT / "eval" / "heldout.jsonl"
SCALED_TASKS = ROOT / "tasks" / "scaled.jsonl"
VAL_SPLIT = ROOT / "eval" / "val_split.json"
PIPE_DIR = ROOT / "runs" / "round5"
HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
STAGE_PATH = PIPE_DIR / "stage.txt"
DONE_PATH = ROOT / "logs" / "round5.done"
STATE_PATH = PIPE_DIR / "state.json"
HOLD_PATH = Path.home() / "e2m" / "round5.hold"
GO_PATH = Path.home() / "e2m" / "round5.go"
HOLD_NAME = "round5.hold"
GO_NAME = "round5.go"
GATE_OPEN = datetime(2026, 11, 25, 11, 15, tzinfo=ZoneInfo("America/Los_Angeles"))
ROUND5_GPU_HOURS = 8.0
SYNTH_CAP_H = 1.0
DATA_GROUP_H = 3.0
EMBED_CAP_H = 0.40
TRAIN_CAP_H = 2.5
EVAL_CAP_H = 0.70
LOOP_CAP_H = 1.8
# Keep train, eval, and loop available. Embed GPU is a 0.25 h reserve inside the tail.
TAIL_H = TRAIN_CAP_H + EVAL_CAP_H + LOOP_CAP_H + 0.25
LOCK_NOTE = "round 5"
TRAIN_OUT = ROOT / "runs" / "asm-distill-7b"
EVAL_PATH = TRAIN_OUT / "eval.jsonl"
LOOP_HELDOUT = ROOT / "runs" / "loop-asm-distill-heldout"
LOOP_VAL = ROOT / "runs" / "loop-asm-distill-val"
CURRENT_PROC: dict[str, subprocess.Popen | None] = {"proc": None}

STAGE_ORDER = [
    "prep-mbpp",
    "prep-train-manifest",
    "screen-tasks",
    "synth",
    "sample-c",
    "lower-asm",
    "leakage-text",
    "leakage-embed",
    "mix",
    "train",
    "eval-heldout",
    "eval-val",
    "loop-heldout",
    "loop-val",
    "failure-analysis",
]
GPU_STAGES = {
    "synth",
    "sample-c",
    "train",
    "eval-heldout",
    "eval-val",
    "loop-heldout",
    "loop-val",
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


def hours_of(name: str, path: Path | None = None) -> float:
    hours_path = path or HOURS_PATH
    total = 0.0
    if not hours_path.is_file():
        return 0.0
    with hours_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("name") != name or row.get("gpu") is False:
                continue
            total += float(row.get("seconds") or 0)
    return total / 3600.0


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
    return max(0.0, ROUND5_GPU_HOURS - gpu_hours_used())


def stage_cap(name: str) -> float:
    """Hours this GPU stage may still run, given the 8 h cap and the reserves."""
    remain = remaining_hours()
    synth_u = hours_of("synth")
    sample_u = hours_of("sample-c")
    data_left = DATA_GROUP_H - synth_u - sample_u
    eval_u = hours_of("eval-heldout") + hours_of("eval-val")
    loop_u = hours_of("loop-heldout") + hours_of("loop-val")
    if name == "synth":
        return max(0.0, min(SYNTH_CAP_H, data_left, remain - TAIL_H))
    if name == "sample-c":
        return max(0.0, min(data_left, remain - TAIL_H))
    if name == "leakage-embed":
        return max(0.0, min(EMBED_CAP_H - hours_of("leakage-embed"), remain - (TRAIN_CAP_H + EVAL_CAP_H + LOOP_CAP_H)))
    if name == "train":
        return max(0.0, min(TRAIN_CAP_H, remain - (EVAL_CAP_H + LOOP_CAP_H)))
    if name in {"eval-heldout", "eval-val"}:
        return max(0.0, min(EVAL_CAP_H - eval_u, remain - LOOP_CAP_H))
    if name in {"loop-heldout", "loop-val"}:
        return max(0.0, min(LOOP_CAP_H - loop_u, remain))
    return 0.0


def configure_paths(*, smoke: bool, out: Path | None) -> Path:
    global PIPE_DIR, HOURS_PATH, STAGE_PATH, DONE_PATH, STATE_PATH, EVAL_PATH
    if out is not None:
        PIPE_DIR = Path(out)
    else:
        PIPE_DIR = ROOT / "runs" / ("round5-smoke" if smoke else "round5")
    HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
    STAGE_PATH = PIPE_DIR / "stage.txt"
    STATE_PATH = PIPE_DIR / "state.json"
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    if smoke:
        DONE_PATH = PIPE_DIR / "round5.done"
        EVAL_PATH = PIPE_DIR / "eval.jsonl"
    return PIPE_DIR


def gate_block_reason(
    *,
    now: datetime,
    hold_path: Path,
    go_path: Path,
    lock_path: Path,
) -> str | None:
    """Why the gate is closed, or None when a lock attempt is allowed.

    The literals ``round5.hold``, ``round5.go``, and 11:15 PT are the audit points.
    """
    if hold_path.is_file():
        return "round5.hold exists"
    if not go_path.is_file() and now < GATE_OPEN:
        return "before 2026-11-25T11:15:00-08:00 and round5.go absent"
    if lock_path.is_file():
        text = lock_path.read_text(encoding="utf-8").strip()
        parts = text.split()
        if len(parts) >= 2 and parts[1].isdigit():
            holder = int(parts[1])
            if pid_alive(holder):
                return f"gpu.lock held by live pid {holder}"
            return None
        if text:
            return f"gpu.lock held: {text[:120]}"
    return None


def _lock_parts(path: Path) -> tuple[str, int | None, str] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    owner = parts[0] if parts else ""
    holder = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return owner, holder, text


def try_take_gpu_lock(pid: int, note: str, *, path: Path | None = None) -> tuple[bool, str]:
    """Take the lock without letting acquire_gpu_lock's SystemExit escape.

    A live holder is refused. A dead PID is stale. The create uses O_EXCL.
    """
    from e2m.loop.guard import acquire_gpu_lock

    lock = path or LOCK_PATH
    info = _lock_parts(lock)
    if info is not None:
        owner, holder, text = info
        if owner == "e2m" and holder == int(pid):
            return True, text
        if holder is not None and pid_alive(holder):
            try:
                acquire_gpu_lock(pid, note, path=lock)
            except SystemExit as exc:
                return False, str(exc)
            return False, text
        if holder is None:
            try:
                acquire_gpu_lock(pid, note, path=lock)
            except SystemExit as exc:
                return False, str(exc)
            return False, text
        again = _lock_parts(lock)
        if again is None:
            pass
        else:
            _owner, holder2, text2 = again
            if holder2 != holder or pid_alive(holder):
                return False, text2
            try:
                lock.unlink()
            except OSError as exc:
                return False, f"unlink failed: {exc}"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            taken = acquire_gpu_lock(pid, note, path=lock)
        except SystemExit as exc:
            return False, str(exc)
        return True, taken
    payload = f"e2m {pid} {datetime.now(tz=PT).isoformat()} {note}\n"
    os.write(fd, payload.encode())
    os.close(fd)
    return True, payload.strip()


def lock_is_ours(pid: int, path: Path | None = None) -> bool:
    info = _lock_parts(path or LOCK_PATH)
    if info is None:
        return False
    owner, holder, _text = info
    return owner == "e2m" and holder == int(pid)


def gate_wait(pid: int, *, sleep_s: float = 60.0, now_fn: Callable[[], datetime] | None = None) -> None:
    """Block until round5.hold is absent, the clock or round5.go opens the gate, and the lock is ours."""
    last_log = 0.0
    while True:
        if lock_is_ours(pid) and HOLD_PATH.is_file():
            release_gpu_lock(owner_pid=pid)
        now = now_fn() if now_fn is not None else datetime.now(tz=PT)
        reason = gate_block_reason(now=now, hold_path=HOLD_PATH, go_path=GO_PATH, lock_path=LOCK_PATH)
        if reason is None:
            ok, detail = try_take_gpu_lock(pid, LOCK_NOTE)
            if ok and not HOLD_PATH.is_file():
                print("gate open", flush=True)
                return
            if ok:
                release_gpu_lock(owner_pid=pid)
                reason = "round5.hold exists"
            else:
                reason = detail or "gpu.lock"
        elif lock_is_ours(pid):
            release_gpu_lock(owner_pid=pid)
        stamp = time.time()
        if last_log == 0.0 or stamp - last_log >= 600:
            print(f"gate wait: {reason}", flush=True)
            last_log = stamp
        time.sleep(sleep_s)


def load_prior_ok() -> set[str]:
    ok: set[str] = set()
    for path in (STATE_PATH, PIPE_DIR / "RESULTS.json"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, info in (data.get("stages") or {}).items():
            if (info or {}).get("status") == "ok":
                ok.add(name)
    return ok


class Pipeline:
    def __init__(self, *, smoke: bool, resume: bool, dry_run: bool, only: set[str] | None, env: dict[str, str]):
        self.smoke = smoke
        self.resume = resume
        self.dry_run = dry_run
        self.only = only
        self.env = env
        self.pid = os.getpid()
        self.stages: dict[str, Any] = {}
        self.halt = False
        self.halt_reason = ""
        self.prior_ok = load_prior_ok() if resume else set()
        self.own_lock = not smoke and not dry_run

    def save_state(self) -> None:
        if self.dry_run:
            return
        STATE_PATH.write_text(json.dumps({"stages": self.stages}, indent=2) + "\n", encoding="utf-8")

    def record(self, name: str, status: str, *, code: int | None = None, reason: str = "") -> None:
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
        """Hold the lock, or record that this smoke/dry run will not take it."""
        if self.smoke or self.dry_run:
            return False
        while True:
            if HOLD_PATH.is_file():
                self.note_held()
                gate_wait(self.pid)
                continue
            if lock_is_ours(self.pid):
                if HOLD_PATH.is_file():
                    continue
                return True
            gate_wait(self.pid)

    def run_cmd(self, name: str, argv: list[str], *, gpu: bool) -> int:
        rundir = PIPE_DIR / "stages" / name
        rundir.mkdir(parents=True, exist_ok=True)
        log = rundir / "log"
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
            self.record(name, "skipped", code=0 if "already" in reason else None, reason=reason)
            return "skip"
        if self.dry_run:
            self.record(name, "planned", reason="dry-run")
            return "plan"
        self.begin_cpu()
        if HOLD_PATH.is_file() and self.own_lock:
            self.note_held()
            gate_wait(self.pid)
            self.begin_cpu()
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
    ) -> str:
        reason = self.skip_reason(name)
        if reason:
            self.record(name, "skipped", code=0 if "already" in reason else None, reason=reason)
            return "skip"
        if self.smoke:
            self.record(name, "skipped", reason="smoke")
            return "skip"
        if self.dry_run:
            cap = stage_cap(name)
            self.record(name, "planned", reason=f"dry-run cap_h={cap:.3f}")
            return "plan"
        while True:
            cap = stage_cap(name)
            if cap < 0.05 or remaining_hours() <= 0:
                self.record(name, "skipped", code=3, reason="gpu-hour budget")
                self.halt = True
                self.halt_reason = "gpu-hour budget"
                return "budget"
            if not self.gpu_ready():
                self.record(name, "error", reason="lock not acquired")
                return "error"
            if not lock_is_ours(self.pid):
                continue
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
                self.record(name, "error", code=1, reason="stage failed; continuing")
                return "error"
            self.record(name, "error", code=code, reason=f"exit {code}")
            if name == "train":
                self.halt = True
                self.halt_reason = "train failed"
            return "error"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _concat(paths: list[Path], out: Path) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for path in paths:
            if not path.is_file():
                continue
            for row in _read_jsonl(path):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def run_stages(pipe: Pipeline) -> None:
    mbpp_path = PIPE_DIR / "mbpp_tasks.jsonl"
    train_path = PIPE_DIR / "train_tasks.jsonl"
    screened_mbpp = PIPE_DIR / "screened_mbpp.jsonl"
    screened_train = PIPE_DIR / "screened_train.jsonl"
    synth_path = PIPE_DIR / "synth_tasks.jsonl"
    sample_tasks = PIPE_DIR / "sample_tasks.jsonl"
    sample_out = PIPE_DIR / "sample_c.jsonl"
    distill_path = PIPE_DIR / "distill.jsonl"
    model_c_path = PIPE_DIR / "model_c.jsonl"
    text_kept = PIPE_DIR / "text_kept.jsonl"
    text_dropped = PIPE_DIR / "text_dropped.jsonl"
    embed_kept = PIPE_DIR / "embed_kept.jsonl"
    embed_dropped = PIPE_DIR / "embed_dropped.jsonl"
    embed_cache = PIPE_DIR / "embed_cache.jsonl"
    embed_report = PIPE_DIR / "embed_report.json"
    sft_out = PIPE_DIR / "sft.jsonl"
    stats: dict[str, Any] = {}

    def prep_mbpp() -> None:
        stats["mbpp"] = build_mbpp_pool(
            mbpp_path,
            limit_kept=2 if pipe.smoke else 0,
            jobs=2 if pipe.smoke else 8,
            stats_path=PIPE_DIR / "mbpp_stats.json",
        )

    def prep_train() -> None:
        stats["train_manifest"] = build_train_manifest(train_path, cap=3 if pipe.smoke else 400)

    def screen_tasks() -> None:
        pools = [("mbpp", mbpp_path, screened_mbpp), ("train", train_path, screened_train)]
        report: dict[str, Any] = {}
        for origin, src, dest in pools:
            rows = _read_jsonl(src)
            for row in rows:
                row["origin"] = row.get("origin") or origin
                row["task_id"] = row.get("task_id") or row.get("id")
            kept, dropped, summary = screen_text(rows)
            _write_jsonl(dest, kept)
            report[origin] = {**summary, "dropped_ids": [d.get("task_id") for d in dropped]}
            print(f"screen {origin} {summary}", flush=True)
        stats["screen_tasks"] = report

    def write_sample_list() -> None:
        _concat([screened_mbpp, synth_path, screened_train], sample_tasks)

    def lower() -> None:
        tasks: dict[str, dict[str, Any]] = {}
        for path in (screened_mbpp, synth_path, screened_train, mbpp_path):
            for row in _read_jsonl(path):
                tasks[str(row.get("id") or row.get("task_id"))] = row
        have = {str(row.get("task_id")) for row in _read_jsonl(distill_path)}
        samples = _read_jsonl(sample_out)
        if pipe.smoke and not samples:
            for row in _read_jsonl(screened_mbpp)[:2]:
                samples.append(
                    {
                        "task_id": row["id"],
                        "origin": "mbpp",
                        "done": True,
                        "c_texts": [row.get("reference_c") or ""],
                    }
                )
        distill_path.parent.mkdir(parents=True, exist_ok=True)
        new = 0
        with distill_path.open("a", encoding="utf-8") as dist, model_c_path.open("a", encoding="utf-8") as meta:
            for row in samples:
                tid = str(row.get("task_id") or "")
                if not tid or tid in have:
                    continue
                task = tasks.get(tid)
                texts = [t for t in (row.get("c_texts") or []) if str(t).strip()]
                if task is None or not texts:
                    have.add(tid)
                    continue
                pairs = distill_pairs(task, texts, origin=str(row.get("origin") or task.get("origin") or ""))
                for pair in pairs:
                    dist.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    meta.write(
                        json.dumps(
                            {
                                "task_id": pair["task_id"],
                                "origin": pair.get("origin"),
                                "goal": pair.get("goal"),
                                "model_c": pair.get("model_c"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    new += 1
                have.add(tid)
        stats["lower"] = {"new_rows": new, "tasks": len(have)}
        print(f"lower new_rows={new}", flush=True)

    def leakage_text() -> None:
        index = _task_index([SCALED_TASKS, mbpp_path, synth_path, train_path])
        rows = attach_reference_c(load_sft_rows(SFT_LORA2), index)
        if pipe.smoke:
            rows = rows[:30]
        rows.extend(_read_jsonl(distill_path))
        kept, dropped, summary = screen_text(rows)
        _write_jsonl(text_kept, kept)
        _write_jsonl(text_dropped, dropped)
        summary["dropped_ids"] = [
            {"task_id": d.get("task_id"), "origin": d.get("origin"), "reason": d.get("reason")} for d in dropped
        ]
        summary["word5_rule"] = WORD5_RULE
        stats["leakage_text"] = {k: v for k, v in summary.items() if k != "dropped_ids"}
        stats["leakage_text_dropped"] = len(dropped)
        (PIPE_DIR / "leakage_text.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"leakage text kept={summary.get('kept_text')} dropped={summary.get('dropped_text')}", flush=True)

    pipe.run_cpu("prep-mbpp", prep_mbpp)
    pipe.run_cpu("prep-train-manifest", prep_train)
    pipe.run_cpu("screen-tasks", screen_tasks)

    synth_cmd = [
        PY, "-m", "e2m.loop.round5_sample", "synth",
        "--out", str(synth_path),
        "--max-hours", "1.000",
        "--cap", "4" if pipe.smoke else "80",
        "--fail-streak", "40",
    ]
    pipe.run_gpu("synth", synth_cmd, accept_3=True, accept_1=True)
    if not pipe.dry_run and not pipe.smoke:
        write_sample_list()
    elif pipe.smoke:
        write_sample_list()

    sample_cmd = [
        PY, "-m", "e2m.loop.round5_sample", "sample",
        "--tasks", str(sample_tasks),
        "--out", str(sample_out),
        "--model", str(MODEL_7B),
        "--adapter", str(TEACHER),
        "--4bit",
        "--k", "4",
        "--temperature", "0.8",
        "--max-new", "1024",
        "--max-hours", "2.000",
    ]
    pipe.run_gpu("sample-c", sample_cmd, accept_3=True)

    pipe.run_cpu("lower-asm", lower)
    pipe.run_cpu("leakage-text", leakage_text)

    def embed_cpu() -> None:
        if not text_kept.is_file():
            raise RuntimeError("text_kept missing")
        cmd = [
            PY, "-m", "e2m.loop.round5_sample", "embed",
            "--rows", str(text_kept),
            "--kept", str(embed_kept),
            "--dropped", str(embed_dropped),
            "--cache", str(embed_cache),
            "--report", str(embed_report),
            "--max-hours", "0.050" if pipe.smoke else "2.000",
            "--cpu",
        ]
        code = pipe.run_cmd("leakage-embed-cpu", cmd, gpu=False)
        if code == 4:
            raise RuntimeError("held")
        if code not in {0, 3}:
            raise RuntimeError(f"cpu embed exit {code}")

    # Embeddings use Ollama with num_gpu=0 while the lock is released.
    # A GPU fallback runs only when the CPU pass does not finish.
    reason = pipe.skip_reason("leakage-embed")
    embed_done = embed_report.is_file() and '"embedded": true' in embed_report.read_text(encoding="utf-8")
    if reason:
        pipe.record("leakage-embed", "skipped", code=0 if "already" in reason else None, reason=reason)
    elif pipe.dry_run:
        pipe.record("leakage-embed", "planned", reason="dry-run cpu then gpu fallback")
    elif pipe.smoke:
        pipe.record("leakage-embed", "skipped", reason="smoke")
    elif embed_done and pipe.resume:
        pipe.record("leakage-embed", "skipped", code=0, reason="already embedded")
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
        if not embedded and pipe.own_lock:
            embed_gpu = [
                PY, "-m", "e2m.loop.round5_sample", "embed",
                "--rows", str(text_kept),
                "--kept", str(embed_kept),
                "--dropped", str(embed_dropped),
                "--cache", str(embed_cache),
                "--report", str(embed_report),
                "--max-hours", "0.400",
            ]
            pipe.run_gpu("leakage-embed", embed_gpu, accept_3=False)
            if embed_report.is_file():
                try:
                    embedded = bool(json.loads(embed_report.read_text(encoding="utf-8")).get("embedded"))
                except json.JSONDecodeError:
                    embedded = False
        if embedded:
            pipe.record("leakage-embed", "ok", code=0)
        elif pipe.stages.get("leakage-embed", {}).get("status") != "ok":
            pipe.record("leakage-embed", "error", code=1, reason="embedding screen incomplete")
            pipe.halt = True
            pipe.halt_reason = "embedding leakage check failed"

    def mix() -> None:
        src = embed_kept if embed_kept.is_file() else None
        if src is None and pipe.smoke and text_kept.is_file():
            src = text_kept
        if src is None:
            raise RuntimeError("embed_kept missing; refusing to train on unchecked rows")
        rows = _read_jsonl(src)
        stats["mix"] = write_mix(sft_out, kept_rows=rows)
        (PIPE_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"mix {stats['mix']}", flush=True)

    if pipe.halt and pipe.halt_reason.startswith("embedding"):
        pipe.record("mix", "skipped", reason=pipe.halt_reason)
    else:
        pipe.run_cpu("mix", mix)
    if pipe.stages.get("mix", {}).get("status") not in {"ok", "skipped", "planned"}:
        pipe.halt = True
        pipe.halt_reason = pipe.halt_reason or "mix failed"
    elif pipe.stages.get("mix", {}).get("status") == "skipped" and pipe.halt_reason.startswith("embedding"):
        pipe.halt = True

    train_cmd = [
        PY, "-m", "e2m.train.sft_lora",
        "--model", str(MODEL_7B),
        "--sft", str(sft_out),
        "--out", str(TRAIN_OUT),
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
        "--max-hours", "2.500",
        "--hold-file", str(HOLD_PATH),
        "--resume",
    ]
    pipe.run_gpu("train", train_cmd, accept_3=True)

    def eval_cmd(split: str) -> list[str]:
        tasks = HELDOUT_TASKS if split == "heldout" else SCALED_TASKS
        return [
            PY, "-m", "e2m.train.eval_generate",
            "--model", str(MODEL_7B),
            "--adapter", str(TRAIN_OUT / "adapter"),
            "--name", "asm-distill-7b",
            "--mode", "asm",
            "--split", split,
            "--tasks", str(tasks),
            "--val-split", str(VAL_SPLIT),
            "--out", str(EVAL_PATH),
            "--4bit",
            "--max-hours", "0.700",
        ]

    pipe.run_gpu("eval-heldout", eval_cmd("heldout"), accept_3=False)
    pipe.run_gpu("eval-val", eval_cmd("val"), accept_3=False)

    def loop_cmd(split: str, out: Path) -> list[str]:
        tasks = HELDOUT_TASKS if split == "heldout" else SCALED_TASKS
        return [
            PY, "-m", "e2m.loop", "run",
            "--model", str(MODEL_7B),
            "--adapter", str(TRAIN_OUT / "adapter"),
            "--name", "asm-distill-7b",
            "--mode", "asm",
            "--split", split,
            "--tasks", str(tasks),
            "--val-split", str(VAL_SPLIT),
            "--out", str(out),
            "--k", "8",
            "--rounds", "3",
            "--max-hours", "1.800",
            "--4bit",
        ]

    pipe.run_gpu("loop-heldout", loop_cmd("heldout", LOOP_HELDOUT), accept_3=False)
    # Val loop only when the loop group and the total budget still have time.
    if stage_cap("loop-val") >= 0.05:
        pipe.run_gpu("loop-val", loop_cmd("val", LOOP_VAL), accept_3=False)
    else:
        reason = pipe.skip_reason("loop-val") or "gpu-hour budget"
        pipe.record("loop-val", "skipped", code=3 if "budget" in reason else None, reason=reason)

    def failure() -> None:
        stats["failure"] = write_failure(EVAL_PATH, PIPE_DIR)
        (PIPE_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")

    pipe.run_cpu("failure-analysis", failure)
    pipe.begin_cpu()
    _write_leakage(stats)


def _write_leakage(stats: dict[str, Any]) -> None:
    text = {}
    path = PIPE_DIR / "leakage_text.json"
    if path.is_file():
        try:
            text = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            text = {}
    embed = {}
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
        "text": {k: v for k, v in text.items() if k != "dropped_ids"},
        "embed": embed,
        "by_check": (text.get("by_check") or {}) | ({"embed": embed.get("dropped")} if embed.get("dropped") else {}),
        "by_source": text.get("by_source") or {},
        "dropped_ids": dropped + [
            {"task_id": row.get("task_id"), "origin": row.get("origin"), "reason": row.get("reason")}
            for row in embed_dropped
        ],
        "screen_tasks": stats.get("screen_tasks"),
    }
    (PIPE_DIR / "leakage.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _eval_summary(path: Path, split: str) -> dict[str, Any]:
    from e2m.train.format import pass_at_k

    rows = []
    for row in _read_jsonl(path):
        if row.get("model") == "asm-distill-7b" and row.get("mode") == "asm" and row.get("split") == split:
            rows.append(row)
    by_id = {str(r.get("task_id")): r for r in rows}
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
    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n,
        "pass_at_1_n": pass1,
        "pass_at_1": (pass1 / n) if n else None,
        "pass_at_5": (pass5 / n) if n else None,
        "size_ratio_vs_gcc_o2": _mean(sizes),
        "insn_ratio_vs_gcc_o2": _mean(insns),
        "greedy_categories": cats,
    }


def _loop_summary(path: Path, split: str) -> dict[str, Any]:
    from e2m.loop.summarize import summarize_loop

    if not (path / "loop.jsonl").is_file():
        return {"n": 0, "pass_n": 0}
    summary = summarize_loop(path / "loop.jsonl", model="asm-distill-7b", split=split)
    return {
        "n": summary.get("n"),
        "pass_n": summary.get("passed_full"),
        "pass_rate": summary.get("pass_full"),
        "stopped": summary.get("stopped"),
    }


def collect_results(stages: dict[str, Any]) -> dict[str, Any]:
    held = _eval_summary(EVAL_PATH, "heldout")
    val = _eval_summary(EVAL_PATH, "val")
    data_stats: dict[str, Any] = {}
    stats_path = PIPE_DIR / "data_stats.json"
    if stats_path.is_file():
        try:
            data_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data_stats = {}
    hours = _read_jsonl(HOURS_PATH)
    return {
        "title": "E2M round 5",
        "target_heldout_direct_asm_pass_at_1": "20/78",
        "teacher": "runs/c-only-7b/adapter",
        "train_out": str(TRAIN_OUT),
        "epochs": 2,
        "goals": ["default", "size"],
        "auxiliary_c_to_asm": False,
        "round4": round4_comparison(),
        "heldout_asm": held,
        "val_asm": val,
        "loop_heldout": _loop_summary(LOOP_HELDOUT, "heldout"),
        "loop_val": _loop_summary(LOOP_VAL, "val"),
        "data_stats": data_stats,
        "stages": stages,
        "gpu_hours_budget": ROUND5_GPU_HOURS,
        "gpu_hours_used": round(gpu_hours_used(), 4),
        "gpu_hours": hours,
    }


def write_summary_md(results: dict[str, Any], path: Path) -> None:
    held = results.get("heldout_asm") or {}
    val = results.get("val_asm") or {}
    loop_h = results.get("loop_heldout") or {}
    loop_v = results.get("loop_val") or {}
    mix = ((results.get("data_stats") or {}).get("mix") or {})
    prior = results.get("round4") or {}
    lines = [
        "# Round 5: direct English to asm",
        "",
        "Teacher is `runs/c-only-7b` (held-out C pass@1 46/78, pass@5 56/78 in round 4). "
        "Training targets are gcc `-O2` / `-Os` assembly of C that the teacher passed, "
        "mixed with the existing `sft_lora2` English-to-asm pairs. "
        "No C-to-asm auxiliary: the eval prompt is English-to-asm, and a row that contains "
        "both `c` and `asm` is trained as C-then-asm text.",
        "",
        f"GPU hours: {results.get('gpu_hours_used')} / {results.get('gpu_hours_budget')}.",
        "",
        "## Held-out direct asm",
        "",
        f"Scored n={held.get('n')}. pass@1 {held.get('pass_at_1_n')}/{held.get('n')} "
        f"({held.get('pass_at_1')}). pass@5 {held.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {held.get('size_ratio_vs_gcc_o2')}. "
        f"Insn ratio: {held.get('insn_ratio_vs_gcc_o2')}.",
        f"Greedy categories: `{held.get('greedy_categories')}`.",
        "",
        f"Repair loop (k=8, R=3): pass {loop_h.get('pass_n')} / n={loop_h.get('n')}.",
        "",
        "## Val direct asm",
        "",
        f"Scored n={val.get('n')}. pass@1 {val.get('pass_at_1_n')}/{val.get('n')} "
        f"({val.get('pass_at_1')}). pass@5 {val.get('pass_at_5')}.",
        f"Size ratio vs gcc -O2: {val.get('size_ratio_vs_gcc_o2')}. "
        f"Insn ratio: {val.get('insn_ratio_vs_gcc_o2')}.",
        f"Val loop: pass {loop_v.get('pass_n')} / n={loop_v.get('n')}.",
        "",
        "## Round 4 comparison (copied, not re-scored)",
        "",
        f"Direct asm pass@1 {prior.get('heldout_direct_asm_pass_at_1')}, "
        f"pass@5 {prior.get('heldout_direct_asm_pass_at_5')}, "
        f"C-then-asm loop {prior.get('heldout_c_then_asm_loop')}, "
        f"hybrid gcc -O2 {prior.get('heldout_hybrid_o2')}. "
        f"C-only pass@1 {prior.get('heldout_c_only_pass_at_1')}, "
        f"pass@5 {prior.get('heldout_c_only_pass_at_5')}.",
        "",
        "Of the 42 round-4 C-pass/asm-fail tasks: logic_mismatch 23, crash 9, timeout 9, "
        "doesnt_assemble 1, abi 0.",
        "",
        "## Data mix",
        "",
        f"Rows {mix.get('rows')}. Proportions `{mix.get('proportions')}`. "
        f"By origin `{mix.get('by_origin')}`. By goal `{mix.get('by_goal')}`.",
        "",
        f"Auxiliary C-to-asm: {mix.get('auxiliary_c_to_asm')}. {mix.get('auxiliary_reason') or ''}",
        "",
        "## Stages",
        "",
        "| stage | status | code | reason |",
        "|---|---|---:|---|",
    ]
    for name in STAGE_ORDER:
        info = (results.get("stages") or {}).get(name) or {}
        lines.append(
            f"| {name} | {info.get('status')} | {info.get('code') if info.get('code') is not None else ''} | {info.get('reason') or ''} |"
        )
    if results.get("halt_reason"):
        lines += ["", f"Halt: {results['halt_reason']}."]
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish(pipe: Pipeline, *, code_hint: int | None = None) -> int:
    if pipe.dry_run or pipe.smoke:
        return 0 if pipe.dry_run or not any(v.get("status") == "error" for v in pipe.stages.values()) else 1
    pipe.release()
    results = collect_results(pipe.stages)
    if pipe.halt_reason:
        results["halt_reason"] = pipe.halt_reason
    (PIPE_DIR / "RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    write_summary_md(results, PIPE_DIR / "SUMMARY.md")
    reason = pipe.halt_reason
    if not reason and any(v.get("status") == "error" for v in pipe.stages.values()):
        reason = "stage error"
    payload = {
        "ts": datetime.now(tz=PT).isoformat(),
        "gpu_hours_used": results.get("gpu_hours_used"),
        "stages": {k: v.get("status") for k, v in pipe.stages.items()},
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
    parser = argparse.ArgumentParser(description="E2M round-5 GPU pipeline")
    parser.add_argument("--smoke", action="store_true", help="CPU-only tiny run under runs/round5-smoke")
    parser.add_argument("--resume", action="store_true", help="Skip stages already recorded ok")
    parser.add_argument("--only", nargs="+", metavar="STAGE")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan; no lock and no done file")
    parser.add_argument("--lock-note", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    global LOCK_NOTE
    args = build_parser().parse_args(argv)
    if args.lock_note:
        LOCK_NOTE = args.lock_note
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["E2M_NO_DEADLINE"] = "1"
    env["E2M_HOLD_FILE"] = str(HOLD_PATH)
    configure_paths(smoke=args.smoke, out=args.out)
    pipe = Pipeline(
        smoke=args.smoke,
        resume=args.resume,
        dry_run=args.dry_run,
        only=set(args.only) if args.only else None,
        env=env,
    )
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

    if pipe.own_lock:
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

    print(
        f"round5 start pid={pipe.pid} smoke={args.smoke} resume={args.resume} "
        f"dry_run={args.dry_run} budget_h={ROUND5_GPU_HOURS} "
        f"used_h={gpu_hours_used():.2f} gate_open={GATE_OPEN.isoformat()}",
        flush=True,
    )
    try:
        if args.dry_run:
            for name in STAGE_ORDER:
                if name in GPU_STAGES or name == "leakage-embed":
                    pipe.record(name, "planned", reason=f"dry-run cap_h={stage_cap(name):.3f}")
                else:
                    pipe.record(name, "planned", reason="dry-run cpu")
            print(json.dumps(pipe.stages, indent=2), flush=True)
            return 0
        run_stages(pipe)
        code = finish(pipe)
        print(f"round5 complete code={code} gpu_hours={gpu_hours_used():.3f}", flush=True)
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
