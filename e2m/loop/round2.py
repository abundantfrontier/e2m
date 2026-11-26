"""Round-2 GPU pipeline: lora2-val loop, DPO, 7B baseline, optional QLoRA v2.

Self-driving. Stops at a stage boundary on deadline or the 60-hour cap.
A stage that errors is logged and skipped; later independent stages still run.
``--resume`` skips stages already recorded ok and never reruns loop-dpo-heldout.
``--round2b`` writes runs/round2b and finishes the 7B stages.
The process writes ~/gpu.lock on start and deletes it on every exit.
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
from typing import Any

from e2m.loop.guard import (
    DEADLINE,
    LOCK_PATH,
    PT,
    ROUND2_GPU_HOURS,
    check_gpu_lock,
    install_run_files,
    past_deadline,
    release_gpu_lock,
    write_gpu_lock,
)
from e2m.loop.pipeline import _exit_ok, wait_if_running
from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import summarize as summarize_eval

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL_15 = ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct"
MODEL_7B = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"
ADAPTER_LORA2 = ROOT / "runs" / "lora2" / "adapter"
PIPE_DIR = ROOT / "runs" / "round2"
HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
STAGE_PATH = PIPE_DIR / "stage.txt"
DONE_PATH = ROOT / "logs" / "round2.done"
LOG_PATH = ROOT / "logs" / "build-round2.log"
SFT_LORA2 = ROOT / "data" / "scaled" / "sft_lora2.jsonl"
PAIRS = ROOT / "data" / "scaled" / "pairs.jsonl"
VAL_SPLIT = ROOT / "eval" / "val_split.json"
HELDOUT_TASKS = ROOT / "eval" / "heldout.jsonl"
SCALED_TASKS = ROOT / "tasks" / "scaled.jsonl"
CURRENT_PROC: dict[str, subprocess.Popen | None] = {"proc": None}
LOCK_NOTE = "round 2 pipeline"
RESULTS_TITLE = "E2M round 2"

STAGE_ORDER = [
    "loop-lora2-val",
    "dpo-train",
    "dpo-eval-val",
    "dpo-eval-heldout",
    "loop-dpo-heldout",
    "7b-base-asm",
    "qlora7b-v2-train",
    "qlora7b-v2-eval-val",
    "qlora7b-v2-eval-heldout",
    "loop-qlora7b-v2-heldout",
]
SKIP_ON_RESUME = frozenset({"loop-dpo-heldout"})
ROUND2B_STAGES = [
    "7b-base-asm",
    "qlora7b-v2-train",
    "qlora7b-v2-eval-val",
    "qlora7b-v2-eval-heldout",
    "loop-qlora7b-v2-heldout",
]


def gpu_hours_used(path: Path | None = None) -> float:
    hours_path = path or HOURS_PATH
    if not hours_path.is_file():
        return 0.0
    total = 0.0
    with hours_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                total += float(json.loads(line).get("seconds") or 0)
    return total / 3600.0


def log_hours(name: str, seconds: float, ok: bool, *, path: Path | None = None) -> None:
    hours_path = path or HOURS_PATH
    hours_path.parent.mkdir(parents=True, exist_ok=True)
    with hours_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "seconds": round(seconds, 2),
                    "ok": ok,
                    "ts": datetime.now(tz=PT).isoformat(),
                }
            )
            + "\n"
        )


def remaining_ok(*, need_h: float = 0.25, hours_path: Path | None = None, check_lock: bool = True) -> bool:
    if past_deadline():
        print("deadline reached", flush=True)
        return False
    used = gpu_hours_used(hours_path)
    if used >= ROUND2_GPU_HOURS:
        print(f"gpu-hour budget used: {used:.2f}/{ROUND2_GPU_HOURS}", flush=True)
        return False
    if used + need_h > ROUND2_GPU_HOURS + 0.25:
        print(
            f"gpu-hour budget would exceed {ROUND2_GPU_HOURS}h (used {used:.2f}, need {need_h})",
            flush=True,
        )
        return False
    if check_lock:
        check_gpu_lock()
    return True


def remaining_hours(hours_path: Path | None = None) -> float:
    until_deadline = max(0.0, (DEADLINE - datetime.now(tz=PT)).total_seconds() / 3600.0)
    until_cap = max(0.0, ROUND2_GPU_HOURS - gpu_hours_used(hours_path))
    return min(until_deadline, until_cap)


def eval_rows(path: Path, model_name: str, split: str, mode: str = "asm") -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") == model_name and row.get("mode") == mode and row.get("split") == split:
                n += 1
    return n


def py_mod(mod: str, *args: str) -> list[str]:
    return [PY, "-m", mod, *args]


def load_ok_stages(*paths: Path | None) -> set[str]:
    ok: set[str] = set()
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, info in (data.get("stages") or {}).items():
            if (info or {}).get("status") == "ok":
                ok.add(name)
    return ok


def skip_reason(
    name: str,
    *,
    resume: bool = False,
    only: set[str] | None = None,
    prior_ok: set[str] | None = None,
    halt: bool = False,
    already: bool = False,
) -> str | None:
    """Return why a stage should be skipped, or None if it should run."""
    if only is not None and name not in only:
        return "not in --only"
    if resume and name in SKIP_ON_RESUME:
        return "resume does not rerun or continue loop-dpo-heldout"
    if resume and name in (prior_ok or set()):
        return "already recorded ok"
    if already:
        return "already complete"
    if halt:
        return "pipeline already halted"
    return None


def valid_base7b_row(row: dict[str, Any]) -> bool:
    if row.get("model") != "base7b" or row.get("mode") != "asm" or row.get("split") != "heldout":
        return False
    if not row.get("task_id"):
        return False
    greedy = row.get("greedy")
    if not isinstance(greedy, dict) or "category" not in greedy:
        return False
    samples = row.get("samples")
    if not isinstance(samples, list) or len(samples) < 5:
        return False
    return True


def seed_carried_hours(src: Path, dest: Path, marker: Path) -> float:
    """Copy prior gpu_hours.jsonl into dest once. Returns carried hours from src."""
    if marker.is_file():
        try:
            return float(json.loads(marker.read_text(encoding="utf-8")).get("hours") or 0)
        except json.JSONDecodeError:
            pass
    hours = gpu_hours_used(src) if src.is_file() else 0.0
    if src.is_file() and not (dest.is_file() and dest.stat().st_size > 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"hours": hours, "seconds": round(hours * 3600, 2)}) + "\n",
        encoding="utf-8",
    )
    return hours


def configure_paths(
    *,
    smoke: bool,
    out: Path | None = None,
    round2b: bool = False,
    lock_note: str | None = None,
) -> Path:
    """Point hour/stage files at the pipeline dir. Smoke tests do not count against the 60h cap."""
    global PIPE_DIR, HOURS_PATH, STAGE_PATH, DONE_PATH, LOG_PATH, LOCK_NOTE, RESULTS_TITLE
    if out is not None:
        PIPE_DIR = Path(out)
    elif round2b:
        PIPE_DIR = ROOT / "runs" / ("round2b-smoke" if smoke else "round2b")
    else:
        PIPE_DIR = ROOT / "runs" / ("round2-smoke" if smoke else "round2")
    HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
    STAGE_PATH = PIPE_DIR / "stage.txt"
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    if round2b:
        LOCK_NOTE = "round 2b pipeline"
        RESULTS_TITLE = "E2M round 2b"
        if not smoke:
            DONE_PATH = ROOT / "logs" / "round2b.done"
            LOG_PATH = ROOT / "logs" / "build-round2b.log"
    if lock_note:
        LOCK_NOTE = lock_note
    return PIPE_DIR


def run_cmd(name: str, argv: list[str], *, rundir: Path, env: dict | None = None) -> int:
    rundir.mkdir(parents=True, exist_ok=True)
    log = rundir / "log"
    STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_PATH.write_text(name + "\n", encoding="utf-8")
    print(f"stage start {name}: {' '.join(argv)}", flush=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# start {name} {datetime.now(tz=PT).isoformat()}\n")
        fh.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        CURRENT_PROC["proc"] = proc
        (rundir / "wrapper.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        code = 1
        try:
            code = proc.wait()
        finally:
            CURRENT_PROC["proc"] = None
    elapsed = time.time() - t0
    if not (rundir / "exit").is_file():
        (rundir / "exit").write_text(str(code) + "\n", encoding="utf-8")
    log_hours(name, elapsed, code == 0)
    print(f"stage end {name} code={code} elapsed_s={elapsed:.1f}", flush=True)
    return code


def seed_base7b_eval(dst: Path) -> int:
    """Copy the interrupted 7B base-asm rows so eval_generate can resume."""
    if dst.is_file():
        n_valid = 0
        with dst.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip() and valid_base7b_row(json.loads(line)):
                    n_valid += 1
        if n_valid > 0:
            return n_valid
    src = ROOT / "runs" / "qlora7b" / "eval.jsonl"
    if not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if valid_base7b_row(row):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def frac(hits: int | None, n: int | None) -> str:
    if hits is None or n is None or n == 0:
        return "—"
    return f"{hits}/{n}"


def greedy_hits(summary: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not summary:
        return None, None
    n = summary.get("n")
    cats = summary.get("greedy_categories") or {}
    hits = cats.get("pass")
    if hits is None and summary.get("pass_at_1") is not None and n:
        hits = round(summary["pass_at_1"] * n)
    return hits, n


def loop_line(summary: dict[str, Any] | None) -> str:
    if not summary or not summary.get("n"):
        return "—"
    n = summary["n"]
    full = summary.get("passed_full")
    vis = summary.get("passed_visible")
    return f"full {frac(full, n)}, visible {frac(vis, n)}"


def collect_results(stages: dict[str, Any], *, carried_h: float = 0.0) -> dict[str, Any]:
    greedy: dict[str, Any] = {}
    for path, name, split, key in [
        (ROOT / "runs" / "lora1" / "eval.jsonl", "lora", "val", "lora1/asm/val"),
        (ROOT / "runs" / "lora1" / "eval.jsonl", "lora", "heldout", "lora1/asm/heldout"),
        (ROOT / "runs" / "qlora7b" / "eval.jsonl", "qlora7b", "val", "qlora7b/asm/val"),
        (ROOT / "runs" / "qlora7b" / "eval.jsonl", "qlora7b", "heldout", "qlora7b/asm/heldout"),
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "val", "lora2/asm/val"),
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "heldout", "lora2/asm/heldout"),
        (ROOT / "runs" / "dpo" / "eval.jsonl", "dpo", "val", "dpo/asm/val"),
        (ROOT / "runs" / "dpo" / "eval.jsonl", "dpo", "heldout", "dpo/asm/heldout"),
        (ROOT / "runs" / "base7b" / "eval.jsonl", "base7b", "heldout", "base7b/asm/heldout"),
        (ROOT / "runs" / "qlora7b-v2" / "eval.jsonl", "qlora7b-v2", "val", "qlora7b-v2/asm/val"),
        (ROOT / "runs" / "qlora7b-v2" / "eval.jsonl", "qlora7b-v2", "heldout", "qlora7b-v2/asm/heldout"),
    ]:
        if path.is_file():
            s = summarize_eval(path, name, "asm", split)
            if s.get("n"):
                greedy[key] = s
    loops: dict[str, Any] = {}
    for path in [
        ROOT / "runs" / "loop-lora1-val" / "loop.jsonl",
        ROOT / "runs" / "loop-lora1-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-lora2-val" / "loop.jsonl",
        ROOT / "runs" / "loop-lora2-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-dpo-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-qlora7b-v2-heldout" / "loop.jsonl",
    ]:
        if path.is_file():
            s = summarize_loop(path)
            if s.get("n"):
                loops[path.parent.name] = s
    dpo_summary = None
    dpo_path = ROOT / "runs" / "dpo" / "summary.json"
    if dpo_path.is_file():
        dpo_summary = json.loads(dpo_path.read_text(encoding="utf-8"))
    qlora_summary = None
    qlora_path = ROOT / "runs" / "qlora7b-v2" / "summary.json"
    if qlora_path.is_file():
        qlora_summary = json.loads(qlora_path.read_text(encoding="utf-8"))
    hours_jobs = []
    if HOURS_PATH.is_file():
        hours_jobs = [json.loads(l) for l in HOURS_PATH.read_text().splitlines() if l.strip()]
    used = gpu_hours_used()
    return {
        "title": RESULTS_TITLE,
        "deadline": DEADLINE.isoformat(),
        "gpu_hours_budget": ROUND2_GPU_HOURS,
        "gpu_hours_used": round(used, 4),
        "gpu_hours_carried": round(carried_h, 4),
        "gpu_hours_this_round": round(max(0.0, used - carried_h), 4),
        "gpu_hours": hours_jobs,
        "stages": stages,
        "dpo_train": dpo_summary,
        "qlora7b_v2_train": qlora_summary,
        "greedy": greedy,
        "loops": loops,
        "finished_at": datetime.now(tz=PT).isoformat(),
    }


def write_summary_md(results: dict[str, Any], path: Path) -> None:
    used = results.get("gpu_hours_used") or 0.0
    carried = results.get("gpu_hours_carried") or 0.0
    this_round = results.get("gpu_hours_this_round")
    if this_round is None:
        this_round = max(0.0, used - carried)
    title = results.get("title") or RESULTS_TITLE
    lines = [
        f"# {title}",
        "",
        f"Deadline: {DEADLINE.isoformat()} (America/Los_Angeles). "
        f"GPU-hour budget: {ROUND2_GPU_HOURS:.0f} additional hours.",
        f"GPU hours used: {used:.2f}/{ROUND2_GPU_HOURS:.0f} "
        f"({this_round:.2f} this round, {carried:.2f} carried).",
        "",
        "## Stages",
        "",
        "| stage | status | code | reason |",
        "|---|---|---:|---|",
    ]
    for name, info in (results.get("stages") or {}).items():
        status = info.get("status") or "—"
        code = info.get("code")
        reason = (info.get("reason") or "").replace("|", "/")
        code_s = "" if code is None else str(code)
        lines.append(f"| {name} | {status} | {code_s} | {reason} |")
    lines += [
        "",
        "## Greedy pass@1 / pass@5",
        "",
        "Compared with lora1, lora2, qlora7b, and dpo.",
        "",
        "| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |",
        "|---|---|---:|---|---|---|---|",
    ]
    greedy = results.get("greedy") or {}
    for key in [
        "lora1/asm/val",
        "lora1/asm/heldout",
        "qlora7b/asm/val",
        "qlora7b/asm/heldout",
        "lora2/asm/val",
        "lora2/asm/heldout",
        "dpo/asm/val",
        "dpo/asm/heldout",
        "base7b/asm/heldout",
        "qlora7b-v2/asm/val",
        "qlora7b-v2/asm/heldout",
    ]:
        s = greedy.get(key)
        model, _, split = key.split("/")
        if not s:
            lines.append(f"| {model} | {split} | — | — | — | — | — |")
            continue
        hits, n = greedy_hits(s)
        p1 = s.get("pass_at_1")
        p5 = s.get("pass_at_5")
        p1_s = f"{p1:.3f} ({frac(hits, n)})" if p1 is not None else frac(hits, n)
        p5_hits = None
        if p5 is not None and n:
            p5_hits = round(p5 * n)
        p5_s = f"{p5:.3f} ({frac(p5_hits, n)})" if p5 is not None else "—"
        sr = s.get("size_ratio_vs_gcc_o2")
        ir = s.get("insn_ratio_vs_gcc_o2")
        sr_s = f"{sr:.3f} (n={s.get('size_ratio_n')})" if sr is not None else "—"
        ir_s = f"{ir:.3f} (n={s.get('insn_ratio_n')})" if ir is not None else "—"
        lines.append(f"| {model} | {split} | {n} | {p1_s} | {p5_s} | {sr_s} | {ir_s} |")
    lines += ["", "## Repair loops", "", "| run | n | pass_full | pass_visible |", "|---|---:|---|---|"]
    loops = results.get("loops") or {}
    for name in [
        "loop-lora1-val",
        "loop-lora1-heldout",
        "loop-lora2-val",
        "loop-lora2-heldout",
        "loop-dpo-heldout",
        "loop-qlora7b-v2-heldout",
    ]:
        s = loops.get(name)
        if not s:
            lines.append(f"| {name} | — | — | — |")
            continue
        n = s.get("n")
        lines.append(
            f"| {name} | {n} | {frac(s.get('passed_full'), n)} | {frac(s.get('passed_visible'), n)} |"
        )
    dpo = results.get("dpo_train")
    if dpo:
        lines += [
            "",
            "## DPO train",
            "",
            f"pairs loaded {dpo.get('n_pairs_loaded')}, train {dpo.get('n_train')}, "
            f"eval-holdout pairs {dpo.get('n_eval')}, beta {dpo.get('beta')}, "
            f"steps {dpo.get('global_steps')}, train_loss {dpo.get('train_loss')}.",
        ]
        ev = dpo.get("eval") or {}
        if ev.get("eval_loss") is not None:
            lines.append(
                f"pair eval_loss {ev['eval_loss']:.4f} on {ev.get('eval_batches')} batches "
                f"(accuracy {ev.get('eval_accuracy')})."
            )
    qlora = results.get("qlora7b_v2_train")
    if qlora:
        lines += [
            "",
            "## QLoRA 7B v2 train",
            "",
            f"train examples {qlora.get('train_examples')}/{qlora.get('train_tasks')} tasks, "
            f"steps {qlora.get('global_steps')}, train_loss {qlora.get('train_loss')}.",
        ]
    lines += ["", "## GPU hours", "", "| stage | seconds | hours | ok |", "|---|---:|---:|---|"]
    for row in results.get("gpu_hours") or []:
        sec = float(row.get("seconds") or 0)
        lines.append(f"| {row.get('name')} | {sec:.2f} | {sec/3600:.2f} | {row.get('ok')} |")
    lines.append(f"| **total** | | **{used:.2f}** | |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _loop_done(rundir: Path, min_n: int) -> bool:
    summary = rundir / "summary.json"
    if not (_exit_ok(rundir) and summary.is_file()):
        return False
    n = json.loads(summary.read_text(encoding="utf-8")).get("n") or 0
    return n >= min_n


def _train_done(rundir: Path) -> bool:
    return (rundir / "summary.json").is_file() and (
        (rundir / "adapter" / "adapter_config.json").is_file()
        or (rundir / "merged" / "config.json").is_file()
    )


class StageRunner:
    def __init__(
        self,
        env: dict[str, str],
        *,
        smoke: bool = False,
        resume: bool = False,
        only: set[str] | None = None,
        prior_ok: set[str] | None = None,
        dry_run: bool = False,
    ):
        self.env = env
        self.smoke = smoke
        self.resume = resume
        self.only = only
        self.prior_ok = prior_ok or set()
        self.dry_run = dry_run
        self.stages: dict[str, Any] = {}
        self.halt = False

    def record(self, name: str, status: str, *, code: int | None = None, reason: str = "") -> None:
        self.stages[name] = {"status": status, "code": code, "reason": reason}
        print(f"stage {name} status={status} code={code} {reason}", flush=True)

    def max_hours_arg(self, need_h: float) -> str:
        remain = remaining_hours()
        cap = max(0.05, min(need_h, remain - 0.05) if remain > 0 else 0.05)
        return f"{cap:.3f}"

    def run_or_skip(
        self,
        name: str,
        *,
        need_h: float,
        already: bool,
        argv: list[str],
        rundir: Path,
    ) -> str:
        reason = skip_reason(
            name,
            resume=self.resume,
            only=self.only,
            prior_ok=self.prior_ok,
            halt=self.halt,
            already=already,
        )
        if reason:
            code = 0 if reason in {"already recorded ok", "already complete"} else None
            self.record(name, "skipped", code=code, reason=reason)
            return "skip"
        if not remaining_ok(need_h=need_h):
            self.record(name, "skipped", code=3, reason="deadline or gpu-hour budget")
            self.halt = True
            return "budget"
        if self.dry_run:
            self.record(name, "planned", reason="dry-run")
            return "plan"
        wait_if_running(rundir)
        try:
            code = run_cmd(name, argv, rundir=rundir, env=self.env)
        except Exception as exc:  # noqa: BLE001
            self.record(name, "error", reason=f"{type(exc).__name__}: {exc}"[:400])
            traceback.print_exc()
            return "error"
        if code == 0:
            self.record(name, "ok", code=0)
            return "ok"
        if code == 2:
            self.record(name, "error", code=2, reason="gpu lock lost")
            self.halt = True
            return "lock"
        if code == 3:
            self.record(name, "stopped", code=3, reason="deadline or max-hours")
            if not remaining_ok(need_h=0.25):
                self.halt = True
            return "budget"
        self.record(name, "error", code=code, reason=f"exit {code}")
        return "error"


def loop_argv(
    *,
    name: str,
    adapter: Path | None,
    model: Path,
    split: str,
    k: int,
    rounds: int,
    out: Path,
    load_4bit: bool,
    limit: int,
    max_hours: str,
    max_new: int | None = None,
) -> list[str]:
    argv = py_mod(
        "e2m.loop",
        "run",
        "--name",
        name,
        "--model",
        str(model),
        "--split",
        split,
        "--k",
        str(k),
        "--rounds",
        str(rounds),
        "--out",
        str(out),
        "--max-hours",
        max_hours,
    )
    if adapter is not None:
        argv += ["--adapter", str(adapter)]
    if load_4bit:
        argv += ["--4bit"]
    if limit > 0:
        argv += ["--limit", str(limit)]
    if max_new is not None:
        argv += ["--max-new", str(max_new)]
    return argv


def eval_argv(
    *,
    name: str,
    model: Path,
    adapter: Path | None,
    split: str,
    out: Path,
    load_4bit: bool,
    limit: int,
    max_new: int | None = None,
    max_hours: str | None = None,
) -> list[str]:
    tasks = HELDOUT_TASKS if split == "heldout" else SCALED_TASKS
    argv = py_mod(
        "e2m.train.eval_generate",
        "--model",
        str(model),
        "--name",
        name,
        "--mode",
        "asm",
        "--split",
        split,
        "--tasks",
        str(tasks),
        "--out",
        str(out),
    )
    if adapter is not None:
        argv += ["--adapter", str(adapter)]
    if load_4bit:
        argv += ["--4bit"]
    if limit > 0:
        argv += ["--limit", str(limit)]
    if max_new is not None:
        argv += ["--max-new", str(max_new)]
    if max_hours is not None:
        argv += ["--max-hours", max_hours]
    return argv


def run_stages(runner: StageRunner) -> None:
    smoke = runner.smoke
    limit = 1 if smoke else 0
    k = 2 if smoke else 8
    rounds = 1 if smoke else 3
    max_new = 256 if smoke else None
    loop_need = 0.4 if smoke else 6.0
    dpo_need = 0.5 if smoke else 8.0
    eval_need = 0.4 if smoke else 1.0
    dpo_loop_need = 0.4 if smoke else 10.0
    base7_need = 0.5 if smoke else 4.0
    qlora_need = 0.5 if smoke else 5.0
    qlora_eval_need = 0.5 if smoke else 2.0

    # 1. loop-lora2-val
    rundir = ROOT / "runs" / ("round2-smoke/loop-lora2-val" if smoke else "loop-lora2-val")
    runner.run_or_skip(
        "loop-lora2-val",
        need_h=loop_need,
        already=(not smoke) and _loop_done(rundir, 90),
        argv=loop_argv(
            name="lora2",
            adapter=ADAPTER_LORA2,
            model=MODEL_15,
            split="val",
            k=k,
            rounds=rounds,
            out=rundir,
            load_4bit=False,
            limit=limit,
            max_hours=runner.max_hours_arg(loop_need),
            max_new=max_new,
        ),
        rundir=rundir,
    )

    # 2. dpo-train
    dpo_dir = ROOT / "runs" / ("round2-smoke/dpo" if smoke else "dpo")
    dpo_argv = py_mod(
        "e2m.train.dpo",
        "--model",
        str(MODEL_15),
        "--init-adapter",
        str(ADAPTER_LORA2),
        "--pairs",
        str(PAIRS),
        "--out",
        str(dpo_dir),
        "--beta",
        "0.1",
        "--epochs",
        "1",
        "--lr",
        "5e-5",
        "--batch-size",
        "1",
        "--grad-accum",
        "1" if smoke else "8",
        "--eval-n",
        "2" if smoke else "512",
    )
    if smoke:
        dpo_argv += ["--limit", "12", "--max-steps", "2", "--max-len", "768"]
    runner.run_or_skip(
        "dpo-train",
        need_h=dpo_need,
        already=(not smoke) and _train_done(dpo_dir) and (dpo_dir / "merged" / "config.json").is_file(),
        argv=dpo_argv,
        rundir=dpo_dir,
    )

    dpo_model = dpo_dir / "merged"
    dpo_ready = (dpo_model / "config.json").is_file()
    dpo_eval_path = dpo_dir / "eval.jsonl"

    # 3. dpo-eval (val then heldout)
    for split, min_n, tag in (("val", 90, "dpo-eval-val"), ("heldout", 78, "dpo-eval-heldout")):
        eval_dir = ROOT / "runs" / (f"round2-smoke/{tag}" if smoke else tag)
        if not dpo_ready and not runner.dry_run and not (runner.resume and tag in runner.prior_ok):
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=runner.halt,
                already=False,
            ):
                runner.run_or_skip(
                    tag,
                    need_h=eval_need,
                    already=False,
                    argv=["true"],
                    rundir=eval_dir,
                )
            else:
                runner.record(tag, "skipped", reason="dpo merged model missing")
            continue
        already = (not smoke) and eval_rows(dpo_eval_path, "dpo", split) >= min_n
        runner.run_or_skip(
            tag,
            need_h=eval_need,
            already=already,
            argv=eval_argv(
                name="dpo",
                model=dpo_model,
                adapter=None,
                split=split,
                out=dpo_eval_path,
                load_4bit=False,
                limit=limit,
                max_new=max_new,
                max_hours=runner.max_hours_arg(eval_need),
            ),
            rundir=eval_dir,
        )

    # 4. loop-dpo-heldout — --resume never reruns or continues this stage
    loop_dpo = ROOT / "runs" / ("round2-smoke/loop-dpo-heldout" if smoke else "loop-dpo-heldout")
    if runner.resume and "loop-dpo-heldout" in SKIP_ON_RESUME:
        runner.run_or_skip(
            "loop-dpo-heldout",
            need_h=dpo_loop_need,
            already=False,
            argv=["true"],
            rundir=loop_dpo,
        )
    elif not dpo_ready and not runner.dry_run:
        if skip_reason(
            "loop-dpo-heldout",
            resume=runner.resume,
            only=runner.only,
            prior_ok=runner.prior_ok,
            halt=runner.halt,
            already=False,
        ):
            runner.run_or_skip(
                "loop-dpo-heldout",
                need_h=dpo_loop_need,
                already=False,
                argv=["true"],
                rundir=loop_dpo,
            )
        else:
            runner.record("loop-dpo-heldout", "skipped", reason="dpo merged model missing")
    else:
        runner.run_or_skip(
            "loop-dpo-heldout",
            need_h=dpo_loop_need,
            already=(not smoke) and _loop_done(loop_dpo, 78),
            argv=loop_argv(
                name="dpo",
                adapter=None,
                model=dpo_model,
                split="heldout",
                k=k,
                rounds=rounds,
                out=loop_dpo,
                load_4bit=False,
                limit=limit,
                max_hours=runner.max_hours_arg(dpo_loop_need),
                max_new=max_new,
            ),
            rundir=loop_dpo,
        )

    # 5. 7b-base-asm (resume the interrupted held-out baseline)
    base_dir = ROOT / "runs" / ("round2-smoke/base7b" if smoke else "base7b")
    base_eval = base_dir / "eval.jsonl"
    if not smoke:
        seeded = seed_base7b_eval(base_eval)
        print(f"base7b seeded {seeded} interrupted rows", flush=True)
    runner.run_or_skip(
        "7b-base-asm",
        need_h=base7_need,
        already=(not smoke) and eval_rows(base_eval, "base7b", "heldout") >= 78,
        argv=eval_argv(
            name="base7b",
            model=MODEL_7B,
            adapter=None,
            split="heldout",
            out=base_eval,
            load_4bit=True,
            limit=limit,
            max_new=max_new,
            max_hours=runner.max_hours_arg(base7_need),
        ),
        rundir=base_dir,
    )

    # 6. qlora7b-v2 — a failure of stage 5 does not skip this block; only overall budget/lock does.
    if runner.halt:
        for tag in (
            "qlora7b-v2-train",
            "qlora7b-v2-eval-val",
            "qlora7b-v2-eval-heldout",
            "loop-qlora7b-v2-heldout",
        ):
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=True,
                already=False,
            ) == "not in --only":
                runner.record(tag, "skipped", reason="not in --only")
            else:
                runner.record(tag, "skipped", reason="pipeline already halted")
        return
    if not remaining_ok(need_h=qlora_need):
        for tag in (
            "qlora7b-v2-train",
            "qlora7b-v2-eval-val",
            "qlora7b-v2-eval-heldout",
            "loop-qlora7b-v2-heldout",
        ):
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=False,
                already=False,
            ) == "not in --only":
                runner.record(tag, "skipped", reason="not in --only")
            else:
                runner.record(tag, "skipped", code=3, reason="no budget remaining after stage 5")
        return

    qlora_dir = ROOT / "runs" / ("round2-smoke/qlora7b-v2" if smoke else "qlora7b-v2")
    qlora_argv = py_mod(
        "e2m.train.sft_lora",
        "--model",
        str(MODEL_7B),
        "--sft",
        str(SFT_LORA2),
        "--out",
        str(qlora_dir),
        "--val-split",
        str(VAL_SPLIT),
        "--epochs",
        "1" if smoke else "3",
        "--lr",
        "2e-4",
        "--rank",
        "32",
        "--alpha",
        "64",
        "--dropout",
        "0.05",
        "--batch-size",
        "1",
        "--grad-accum",
        "1" if smoke else "16",
        "--4bit",
    )
    if smoke:
        qlora_argv += ["--limit", "8", "--max-steps", "2"]
    runner.run_or_skip(
        "qlora7b-v2-train",
        need_h=qlora_need,
        already=(not smoke) and _train_done(qlora_dir),
        argv=qlora_argv,
        rundir=qlora_dir,
    )
    qlora_adapter = qlora_dir / "adapter"
    qlora_ready = (qlora_adapter / "adapter_config.json").is_file() or runner.dry_run
    qlora_eval = qlora_dir / "eval.jsonl"
    for split, min_n, tag in (("val", 90, "qlora7b-v2-eval-val"), ("heldout", 78, "qlora7b-v2-eval-heldout")):
        eval_dir = ROOT / "runs" / (f"round2-smoke/{tag}" if smoke else tag)
        if not qlora_ready:
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=runner.halt,
                already=False,
            ):
                runner.run_or_skip(
                    tag,
                    need_h=qlora_eval_need,
                    already=False,
                    argv=["true"],
                    rundir=eval_dir,
                )
            else:
                runner.record(tag, "skipped", reason="qlora7b-v2 adapter missing")
            continue
        already = (not smoke) and eval_rows(qlora_eval, "qlora7b-v2", split) >= min_n
        runner.run_or_skip(
            tag,
            need_h=qlora_eval_need,
            already=already,
            argv=eval_argv(
                name="qlora7b-v2",
                model=MODEL_7B,
                adapter=qlora_adapter,
                split=split,
                out=qlora_eval,
                load_4bit=True,
                limit=limit,
                max_new=max_new,
                max_hours=runner.max_hours_arg(qlora_eval_need),
            ),
            rundir=eval_dir,
        )
    loop_q = ROOT / "runs" / ("round2-smoke/loop-qlora7b-v2-heldout" if smoke else "loop-qlora7b-v2-heldout")
    if not qlora_ready:
        if skip_reason(
            "loop-qlora7b-v2-heldout",
            resume=runner.resume,
            only=runner.only,
            prior_ok=runner.prior_ok,
            halt=runner.halt,
            already=False,
        ):
            runner.run_or_skip(
                "loop-qlora7b-v2-heldout",
                need_h=0.4,
                already=False,
                argv=["true"],
                rundir=loop_q,
            )
        else:
            runner.record("loop-qlora7b-v2-heldout", "skipped", reason="qlora7b-v2 adapter missing")
        return
    remain = remaining_hours()
    qlora_loop_need = 0.4 if smoke else min(18.0, max(0.25, remain - 0.25))
    if not remaining_ok(need_h=min(qlora_loop_need, 0.4 if smoke else 1.0)):
        runner.record("loop-qlora7b-v2-heldout", "skipped", code=3, reason="no budget remaining for 7B loop")
        return
    runner.run_or_skip(
        "loop-qlora7b-v2-heldout",
        need_h=qlora_loop_need,
        already=(not smoke) and _loop_done(loop_q, 78),
        argv=loop_argv(
            name="qlora7b-v2",
            adapter=qlora_adapter,
            model=MODEL_7B,
            split="heldout",
            k=k,
            rounds=rounds,
            out=loop_q,
            load_4bit=True,
            limit=limit,
            max_hours=runner.max_hours_arg(qlora_loop_need),
            max_new=max_new,
        ),
        rundir=loop_q,
    )


def finish(state: dict[str, Any], stages: dict[str, Any], *, smoke: bool, carried_h: float = 0.0) -> int:
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_results(stages, carried_h=carried_h)
    results["smoke"] = smoke
    (PIPE_DIR / "RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    write_summary_md(results, PIPE_DIR / "SUMMARY.md")
    if not smoke:
        DONE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DONE_PATH.write_text(
            json.dumps(
                {
                    "ts": datetime.now(tz=PT).isoformat(),
                    "gpu_hours_used": results.get("gpu_hours_used"),
                    "gpu_hours_this_round": results.get("gpu_hours_this_round"),
                    "stages": {k: v.get("status") for k, v in stages.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("wrote", PIPE_DIR / "RESULTS.json", flush=True)
    print("wrote", PIPE_DIR / "SUMMARY.md", flush=True)
    codes = [v.get("code") for v in stages.values() if isinstance(v.get("code"), int)]
    if any(c == 2 for c in codes):
        state["code"] = 2
    elif any(c == 3 for c in codes) and not any(v.get("status") == "ok" for v in stages.values()):
        state["code"] = 3
    elif any(v.get("status") == "error" for v in stages.values()):
        state["code"] = 1
    else:
        state["code"] = 0
    return int(state["code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2M round-2 GPU pipeline")
    parser.add_argument("--smoke", action="store_true", help="1-2 tasks per stage, then exit")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages already recorded ok; do not rerun or continue loop-dpo-heldout",
    )
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="Run only these stages")
    parser.add_argument(
        "--prior",
        type=Path,
        action="append",
        default=None,
        help="RESULTS.json whose ok stages are skipped under --resume (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Pipeline directory for RESULTS/hours")
    parser.add_argument(
        "--round2b",
        action="store_true",
        help="Finish the 7B stages into runs/round2b (implies --resume --carry-hours)",
    )
    parser.add_argument(
        "--carry-hours",
        action="store_true",
        help="Count round2 GPU hours against the 60h cap",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan without GPU work")
    parser.add_argument("--lock-note", default=None, help="Text after pid/time in ~/gpu.lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.round2b:
        args.resume = True
        args.carry_hours = True
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    pid = os.getpid()
    run_dir = configure_paths(
        smoke=args.smoke, out=args.out, round2b=args.round2b, lock_note=args.lock_note
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    own_lock = not args.smoke and not args.dry_run
    if not args.dry_run:
        # A prior dry-run or interrupted launch must not freeze the exit code.
        (run_dir / "exit").unlink(missing_ok=True)
    state = install_run_files(run_dir)
    check_gpu_lock()
    released = {"done": False}

    def _release() -> None:
        if not own_lock or released["done"]:
            return
        released["done"] = True
        release_gpu_lock(owner_pid=pid)

    carried_h = 0.0
    if args.carry_hours and not args.smoke:
        carried_h = seed_carried_hours(
            ROOT / "runs" / "round2" / "gpu_hours.jsonl",
            HOURS_PATH,
            run_dir / "carried_hours.json",
        )
        print(f"carried {carried_h:.2f} GPU h from round2", flush=True)

    prior_paths: list[Path] = []
    if args.prior:
        prior_paths.extend(args.prior)
    if args.resume:
        prior_paths.append(ROOT / "runs" / "round2" / "RESULTS.json")
        prior_paths.append(run_dir / "RESULTS.json")
    prior_ok = load_ok_stages(*prior_paths) if args.resume else set()
    only = set(args.only) if args.only else None

    if own_lock:
        write_gpu_lock(pid, LOCK_NOTE)
        atexit.register(_release)

        def _on_signal(signum: int, _frame: Any) -> None:
            print(f"signal {signum}, releasing gpu.lock", flush=True)
            proc = CURRENT_PROC.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    proc.terminate()
            _release()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

    runner = StageRunner(
        env,
        smoke=args.smoke,
        resume=args.resume,
        only=only,
        prior_ok=prior_ok,
        dry_run=args.dry_run,
    )
    try:
        print(
            f"round2 start pid={pid} smoke={args.smoke} resume={args.resume} round2b={args.round2b} "
            f"dry_run={args.dry_run} only={sorted(only) if only else None} "
            f"deadline={DEADLINE.isoformat()} budget_h={ROUND2_GPU_HOURS} "
            f"used_h={gpu_hours_used():.2f} remain_h={remaining_hours():.2f} "
            f"lock={LOCK_PATH.read_text(encoding='utf-8').strip()}",
            flush=True,
        )
        run_stages(runner)
        if args.dry_run:
            print("dry-run stages " + json.dumps(runner.stages, indent=2), flush=True)
            state["code"] = 0
            return 0
        code = finish(state, runner.stages, smoke=args.smoke, carried_h=carried_h)
        print(f"round2 complete code={code} gpu_hours={gpu_hours_used():.3f}", flush=True)
        return code
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        state["code"] = code
        if not args.dry_run:
            try:
                finish(state, runner.stages, smoke=args.smoke, carried_h=carried_h)
            except Exception:
                traceback.print_exc()
        raise
    except Exception:
        traceback.print_exc()
        state["code"] = 1
        if not args.dry_run:
            try:
                finish(state, runner.stages, smoke=args.smoke, carried_h=carried_h)
            except Exception:
                traceback.print_exc()
        return 1
    finally:
        _release()
        if own_lock:
            print("gpu.lock released", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
