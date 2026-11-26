"""Round-3 GPU pipeline: DPO repair, then C-then-assembly QLoRA.

Self-driving. Stops at a stage boundary on the 30-hour cap. No calendar deadline.
A stage that errors is logged and skipped; later independent stages still run.
``--resume`` skips stages already recorded ok.
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
    LOCK_PATH,
    PT,
    ROUND3_GPU_HOURS,
    check_gpu_lock,
    install_run_files,
    release_gpu_lock,
    write_gpu_lock,
)
from e2m.loop.pipeline import _exit_ok, wait_if_running
from e2m.loop.round2 import (
    ADAPTER_LORA2,
    HELDOUT_TASKS,
    MODEL_15,
    MODEL_7B,
    SCALED_TASKS,
    SFT_LORA2,
    VAL_SPLIT,
    _loop_done,
    _train_done,
    eval_rows,
    frac,
    greedy_hits,
    load_ok_stages,
    py_mod,
    skip_reason,
)
from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import summarize as summarize_eval

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PAIRS = ROOT / "data" / "scaled" / "pairs.jsonl"
PIPE_DIR = ROOT / "runs" / "round3"
HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
STAGE_PATH = PIPE_DIR / "stage.txt"
DONE_PATH = ROOT / "logs" / "round3.done"
LOG_PATH = ROOT / "logs" / "build-round3.log"
LOCK_NOTE = "round 3"
RESULTS_TITLE = "E2M round 3"
CURRENT_PROC: dict[str, subprocess.Popen | None] = {"proc": None}

# A then B: collapse guard should fail-fast on DPO and leave budget for C-then-asm.
ORDER_CHOICE = "A then B"
ORDER_REASON = (
    "DPO collapse guard should stop within a few hundred steps if generation breaks again; "
    "C-then-asm is the held-out-gap experiment and gets the remaining budget."
)
LORA2_VAL_PASS = 38
LORA2_VAL_N = 90

STAGE_ORDER = [
    "dpo-pairs",
    "dpo2-train",
    "dpo2-eval-val",
    "dpo2-eval-heldout",
    "loop-dpo2-heldout",
    "c-then-asm-data",
    "c-then-asm-train",
    "c-then-asm-eval-val",
    "c-then-asm-eval-heldout",
    "loop-c-then-asm-heldout",
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
    used = gpu_hours_used(hours_path)
    if used >= ROUND3_GPU_HOURS:
        print(f"gpu-hour budget used: {used:.2f}/{ROUND3_GPU_HOURS}", flush=True)
        return False
    if used + need_h > ROUND3_GPU_HOURS + 0.25:
        print(
            f"gpu-hour budget would exceed {ROUND3_GPU_HOURS}h (used {used:.2f}, need {need_h})",
            flush=True,
        )
        return False
    if check_lock:
        check_gpu_lock()
    return True


def remaining_hours(hours_path: Path | None = None) -> float:
    return max(0.0, ROUND3_GPU_HOURS - gpu_hours_used(hours_path))


def configure_paths(*, smoke: bool, out: Path | None = None, lock_note: str | None = None) -> Path:
    global PIPE_DIR, HOURS_PATH, STAGE_PATH, DONE_PATH, LOG_PATH, LOCK_NOTE
    if out is not None:
        PIPE_DIR = Path(out)
    else:
        PIPE_DIR = ROOT / "runs" / ("round3-smoke" if smoke else "round3")
    HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
    STAGE_PATH = PIPE_DIR / "stage.txt"
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    if smoke:
        DONE_PATH = PIPE_DIR / "round3.done"
        LOG_PATH = PIPE_DIR / "build-round3.log"
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
        if not remaining_ok(need_h=need_h, check_lock=not self.dry_run):
            self.record(name, "skipped", code=3, reason="gpu-hour budget")
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
            self.record(name, "stopped", code=3, reason="max-hours")
            if not remaining_ok(need_h=0.25):
                self.halt = True
            return "budget"
        self.record(name, "error", code=code, reason=f"exit {code}")
        return "error"


def eval_argv(
    *,
    name: str,
    model: Path,
    adapter: Path | None,
    split: str,
    out: Path,
    load_4bit: bool,
    limit: int,
    mode: str = "asm",
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
        mode,
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
    mode: str = "asm",
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
        "--mode",
        mode,
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


def dpo2_val_hits(eval_path: Path) -> tuple[int | None, int | None]:
    if not eval_path.is_file():
        return None, None
    summary = summarize_eval(eval_path, "dpo2", "asm", "val")
    return greedy_hits(summary)


def dpo2_last_good_step(summary_path: Path) -> int | None:
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    step = data.get("last_good_step")
    return int(step) if step is not None else None


def collect_results(stages: dict[str, Any]) -> dict[str, Any]:
    greedy: dict[str, Any] = {}
    for path, name, mode, split, key in [
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "asm", "val", "lora2/asm/val"),
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "asm", "heldout", "lora2/asm/heldout"),
        (ROOT / "runs" / "qlora7b-v2" / "eval.jsonl", "qlora7b-v2", "asm", "val", "qlora7b-v2/asm/val"),
        (ROOT / "runs" / "qlora7b-v2" / "eval.jsonl", "qlora7b-v2", "asm", "heldout", "qlora7b-v2/asm/heldout"),
        (ROOT / "runs" / "dpo" / "eval.jsonl", "dpo", "asm", "val", "dpo/asm/val"),
        (ROOT / "runs" / "dpo" / "eval.jsonl", "dpo", "asm", "heldout", "dpo/asm/heldout"),
        (ROOT / "runs" / "lora1" / "eval.jsonl", "base", "c", "heldout", "base-c/c/heldout"),
        (ROOT / "runs" / "dpo2" / "eval.jsonl", "dpo2", "asm", "val", "dpo2/asm/val"),
        (ROOT / "runs" / "dpo2" / "eval.jsonl", "dpo2", "asm", "heldout", "dpo2/asm/heldout"),
        (ROOT / "runs" / "c-then-asm" / "eval.jsonl", "c-then-asm", "c-then-asm", "val", "c-then-asm/c-then-asm/val"),
        (
            ROOT / "runs" / "c-then-asm" / "eval.jsonl",
            "c-then-asm",
            "c-then-asm",
            "heldout",
            "c-then-asm/c-then-asm/heldout",
        ),
    ]:
        if path.is_file():
            s = summarize_eval(path, name, mode, split)
            if s.get("n"):
                greedy[key] = s
    loops: dict[str, Any] = {}
    for path in [
        ROOT / "runs" / "loop-lora2-val" / "loop.jsonl",
        ROOT / "runs" / "loop-lora2-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-qlora7b-v2-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-base-c-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-dpo2-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-c-then-asm-heldout" / "loop.jsonl",
    ]:
        if path.is_file():
            s = summarize_loop(path)
            if s.get("n"):
                loops[path.parent.name] = s
    dpo2_summary = None
    dpo2_path = ROOT / "runs" / "dpo2" / "summary.json"
    if dpo2_path.is_file():
        dpo2_summary = json.loads(dpo2_path.read_text(encoding="utf-8"))
    pairs_meta = None
    meta_path = ROOT / "runs" / "dpo2" / "pairs_meta.json"
    if meta_path.is_file():
        pairs_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cta_summary = None
    cta_path = ROOT / "runs" / "c-then-asm" / "summary.json"
    if cta_path.is_file():
        cta_summary = json.loads(cta_path.read_text(encoding="utf-8"))
    cta_data = None
    data_stats = ROOT / "data" / "scaled" / "c_then_asm_stats.json"
    if data_stats.is_file():
        cta_data = json.loads(data_stats.read_text(encoding="utf-8"))
    hours_jobs = []
    if HOURS_PATH.is_file():
        hours_jobs = [json.loads(l) for l in HOURS_PATH.read_text().splitlines() if l.strip()]
    used = gpu_hours_used()
    return {
        "title": RESULTS_TITLE,
        "deadline": None,
        "gpu_hours_budget": ROUND3_GPU_HOURS,
        "gpu_hours_used": round(used, 4),
        "order": ORDER_CHOICE,
        "order_reason": ORDER_REASON,
        "gpu_hours": hours_jobs,
        "stages": stages,
        "dpo2_pairs": pairs_meta,
        "dpo2_train": dpo2_summary,
        "c_then_asm_data": cta_data,
        "c_then_asm_train": cta_summary,
        "greedy": greedy,
        "loops": loops,
        "finished_at": datetime.now(tz=PT).isoformat(),
    }


def _greedy_line(s: dict[str, Any] | None, model: str, split: str) -> str:
    if not s:
        return f"| {model} | {split} | — | — | — | — | — |"
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
    return f"| {model} | {split} | {n} | {p1_s} | {p5_s} | {sr_s} | {ir_s} |"


def write_summary_md(results: dict[str, Any], path: Path) -> None:
    used = results.get("gpu_hours_used") or 0.0
    lines = [
        f"# {results.get('title') or RESULTS_TITLE}",
        "",
        "No calendar deadline. "
        f"GPU-hour budget: {ROUND3_GPU_HOURS:.0f} hours.",
        f"GPU hours used: {used:.2f}/{ROUND3_GPU_HOURS:.0f}.",
        f"Order: {results.get('order') or ORDER_CHOICE}. {results.get('order_reason') or ORDER_REASON}",
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
    meta = results.get("dpo2_pairs") or {}
    lines += [
        "",
        "## A1 pair-goal choice",
        "",
        f"Choice: `{meta.get('choice') or 'pair_goal_in_prompt'}`. "
        f"{meta.get('reason') or 'user_asm already supports Goal: default|size|speed.'}",
        f"Pairs loaded: {meta.get('n_pairs_loaded', '—')}. Goal counts: {meta.get('goal_counts') or '—'}.",
        "",
        "## Greedy pass@1 / pass@5",
        "",
        "Compared with lora2, qlora7b-v2, round-2 dpo, and the C control. "
        "c-then-asm pass@k scores the assembly half only.",
        "",
        "| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |",
        "|---|---|---:|---|---|---|---|",
    ]
    greedy = results.get("greedy") or {}
    for key, model, split in [
        ("lora2/asm/val", "lora2", "val"),
        ("lora2/asm/heldout", "lora2", "heldout"),
        ("qlora7b-v2/asm/val", "qlora7b-v2", "val"),
        ("qlora7b-v2/asm/heldout", "qlora7b-v2", "heldout"),
        ("dpo/asm/val", "dpo", "val"),
        ("dpo/asm/heldout", "dpo", "heldout"),
        ("dpo2/asm/val", "dpo2", "val"),
        ("dpo2/asm/heldout", "dpo2", "heldout"),
        ("c-then-asm/c-then-asm/val", "c-then-asm", "val"),
        ("c-then-asm/c-then-asm/heldout", "c-then-asm", "heldout"),
        ("base-c/c/heldout", "base-c", "heldout"),
    ]:
        lines.append(_greedy_line(greedy.get(key), model, split))
    lines += ["", "## C-then-asm C half and agreement", ""]
    for key, split in [
        ("c-then-asm/c-then-asm/val", "val"),
        ("c-then-asm/c-then-asm/heldout", "heldout"),
    ]:
        s = greedy.get(key)
        if not s:
            lines.append(f"- {split}: —")
            continue
        n = s.get("n")
        c_n = s.get("c_pass_n")
        agree_n = s.get("c_asm_agree_n")
        lines.append(
            f"- {split}: C pass {frac(c_n, n)}, C/asm agree {frac(agree_n, n)}, "
            f"both {s.get('c_and_asm_pass')}, C-only {s.get('c_only_pass')}, "
            f"asm-only {s.get('asm_only_pass')}"
        )
    lines += ["", "## Repair loops", "", "| run | n | pass_full | pass_visible |", "|---|---:|---|---|"]
    loops = results.get("loops") or {}
    for name in [
        "loop-lora2-val",
        "loop-lora2-heldout",
        "loop-qlora7b-v2-heldout",
        "loop-base-c-heldout",
        "loop-dpo2-heldout",
        "loop-c-then-asm-heldout",
    ]:
        s = loops.get(name)
        if not s:
            lines.append(f"| {name} | — | — | — |")
            continue
        n = s.get("n")
        lines.append(
            f"| {name} | {n} | {frac(s.get('passed_full'), n)} | {frac(s.get('passed_visible'), n)} |"
        )
    dpo = results.get("dpo2_train")
    if dpo:
        lines += [
            "",
            "## DPO2 train",
            "",
            f"pairs loaded {dpo.get('n_pairs_loaded')}, train {dpo.get('n_train')}, "
            f"eval-holdout pairs {dpo.get('n_eval')}, beta {dpo.get('beta')}, "
            f"lr {dpo.get('lr')}, rpo_weight {dpo.get('rpo_weight')}, "
            f"prompt_goal {dpo.get('prompt_goal')}, steps {dpo.get('global_steps')}, "
            f"last_good_step {dpo.get('last_good_step')}, collapsed {dpo.get('collapsed')}, "
            f"train_loss {dpo.get('train_loss')}.",
            f"stop_reason: {dpo.get('stop_reason')}",
        ]
        curve = dpo.get("collapse_curve") or []
        if curve:
            lines += ["", "Collapse assemble-rate curve:", ""]
            for row in curve:
                lines.append(
                    f"- step {row.get('step')}: {row.get('assembles')}/{row.get('n')} "
                    f"= {row.get('rate')} (threshold {row.get('threshold')}, keep {row.get('keep')})"
                )
    qlora = results.get("c_then_asm_train")
    if qlora:
        lines += [
            "",
            "## C-then-asm QLoRA train",
            "",
            f"train examples {qlora.get('train_examples')}/{qlora.get('train_tasks')} tasks, "
            f"steps {qlora.get('global_steps')}, train_loss {qlora.get('train_loss')}, "
            f"max_len {qlora.get('max_len')}, target {qlora.get('target')}.",
        ]
    data = results.get("c_then_asm_data")
    if data:
        lines += [
            "",
            f"C-then-asm rows {data.get('n_written')}/{data.get('n_sft')} "
            f"(missing C {data.get('n_missing_c')}).",
        ]
    lines += ["", "## GPU hours", "", "| stage | seconds | hours | ok |", "|---|---:|---:|---|"]
    for row in results.get("gpu_hours") or []:
        sec = float(row.get("seconds") or 0)
        lines.append(f"| {row.get('name')} | {sec:.2f} | {sec/3600:.2f} | {row.get('ok')} |")
    lines.append(f"| **total** | | **{used:.2f}** | |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish(state: dict[str, Any], stages: dict[str, Any], *, smoke: bool, halt_reason: str = "") -> int:
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_results(stages)
    results["smoke"] = smoke
    if halt_reason:
        results["halt_reason"] = halt_reason
    (PIPE_DIR / "RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    write_summary_md(results, PIPE_DIR / "SUMMARY.md")
    DONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(tz=PT).isoformat(),
        "gpu_hours_used": results.get("gpu_hours_used"),
        "stages": {k: v.get("status") for k, v in stages.items()},
        "order": ORDER_CHOICE,
    }
    if halt_reason:
        payload["reason"] = halt_reason
    elif any(v.get("status") == "skipped" and v.get("code") == 3 for v in stages.values()):
        payload["reason"] = "gpu-hour budget"
    DONE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", PIPE_DIR / "RESULTS.json", flush=True)
    print("wrote", PIPE_DIR / "SUMMARY.md", flush=True)
    print("wrote", DONE_PATH, flush=True)
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


def run_stages(runner: StageRunner) -> None:
    smoke = runner.smoke
    limit = 1 if smoke else 0
    k = 2 if smoke else 8
    rounds = 1 if smoke else 3
    max_new_asm = 256 if smoke else None
    max_new_cta = 256 if smoke else 2048
    dpo_need = 0.4 if smoke else 4.0
    eval_need = 0.3 if smoke else 1.5
    loop_need = 0.4 if smoke else 10.0
    data_need = 0.05
    cta_need = 0.5 if smoke else 6.0
    cta_eval_need = 0.5 if smoke else 2.5

    dpo_dir = ROOT / "runs" / ("round3-smoke/dpo2" if smoke else "dpo2")
    runner.run_or_skip(
        "dpo-pairs",
        need_h=data_need,
        already=(not smoke) and (dpo_dir / "pairs_meta.json").is_file(),
        argv=py_mod(
            "e2m.train.dpo",
            "--prepare-only",
            "--pairs",
            str(PAIRS),
            "--out",
            str(dpo_dir),
            "--collapse-n",
            "2" if smoke else "30",
        ),
        rundir=dpo_dir / "prepare",
    )

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
        "0.2",
        "--epochs",
        "0.5",
        "--lr",
        "5e-6",
        "--batch-size",
        "1",
        "--grad-accum",
        "1" if smoke else "8",
        "--eval-n",
        "2" if smoke else "256",
        "--rpo-weight",
        "1.0",
        "--pair-goal",
        "--collapse-every",
        "1" if smoke else "250",
        "--collapse-n",
        "2" if smoke else "30",
        "--max-steps",
        "2" if smoke else "1500",
    )
    if smoke:
        dpo_argv += ["--limit", "8", "--max-len", "768", "--collapse-max-new", "64"]
    runner.run_or_skip(
        "dpo2-train",
        need_h=dpo_need,
        already=(not smoke) and _train_done(dpo_dir) and (dpo_dir / "merged" / "config.json").is_file(),
        argv=dpo_argv,
        rundir=dpo_dir,
    )

    dpo_model = dpo_dir / "merged"
    dpo_ready = (dpo_model / "config.json").is_file() or runner.dry_run
    dpo_eval_path = dpo_dir / "eval.jsonl"
    for split, min_n, tag, need in (
        ("val", 90, "dpo2-eval-val", eval_need),
        ("heldout", 78, "dpo2-eval-heldout", eval_need),
    ):
        eval_dir = ROOT / "runs" / (f"round3-smoke/{tag}" if smoke else tag)
        if not dpo_ready and not runner.dry_run:
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=runner.halt,
            ):
                runner.run_or_skip(tag, need_h=need, already=False, argv=["true"], rundir=eval_dir)
            else:
                runner.record(tag, "skipped", reason="dpo2 merged model missing")
            continue
        already = (not smoke) and eval_rows(dpo_eval_path, "dpo2", split) >= min_n
        runner.run_or_skip(
            tag,
            need_h=need,
            already=already,
            argv=eval_argv(
                name="dpo2",
                model=dpo_model,
                adapter=None,
                split=split,
                out=dpo_eval_path,
                load_4bit=False,
                limit=limit,
                max_new=max_new_asm,
                max_hours=runner.max_hours_arg(need),
            ),
            rundir=eval_dir,
        )

    loop_dpo = ROOT / "runs" / ("round3-smoke/loop-dpo2-heldout" if smoke else "loop-dpo2-heldout")
    hits, n_val = dpo2_val_hits(dpo_eval_path)
    last_good = dpo2_last_good_step(dpo_dir / "summary.json")
    loop_reason = None
    if not dpo_ready and not runner.dry_run:
        loop_reason = "dpo2 merged model missing"
    elif last_good == 0:
        loop_reason = "last good checkpoint is step 0 (no DPO progress); skip rerunning lora2"
    elif hits is None or n_val is None:
        loop_reason = "dpo2 val eval missing"
    elif hits < LORA2_VAL_PASS:
        loop_reason = f"val pass@1 {hits}/{n_val} < lora2 {LORA2_VAL_PASS}/{LORA2_VAL_N}"
    if loop_reason and not runner.dry_run:
        if skip_reason(
            "loop-dpo2-heldout",
            resume=runner.resume,
            only=runner.only,
            prior_ok=runner.prior_ok,
            halt=runner.halt,
        ):
            runner.run_or_skip(
                "loop-dpo2-heldout",
                need_h=loop_need,
                already=False,
                argv=["true"],
                rundir=loop_dpo,
            )
        else:
            runner.record("loop-dpo2-heldout", "skipped", reason=loop_reason)
    else:
        runner.run_or_skip(
            "loop-dpo2-heldout",
            need_h=loop_need,
            already=(not smoke) and _loop_done(loop_dpo, 78),
            argv=loop_argv(
                name="dpo2",
                adapter=None,
                model=dpo_model,
                split="heldout",
                k=k,
                rounds=rounds,
                out=loop_dpo,
                load_4bit=False,
                limit=limit,
                max_hours=runner.max_hours_arg(loop_need),
                max_new=max_new_asm,
            ),
            rundir=loop_dpo,
        )

    cta_sft = ROOT / "data" / "scaled" / ("sft_c_then_asm_smoke.jsonl" if smoke else "sft_c_then_asm.jsonl")
    cta_dir = ROOT / "runs" / ("round3-smoke/c-then-asm" if smoke else "c-then-asm")
    data_argv = py_mod(
        "e2m.train.c_then_asm",
        "--sft",
        str(SFT_LORA2),
        "--out",
        str(cta_sft),
    )
    if smoke:
        data_argv += ["--limit", "8"]
    runner.run_or_skip(
        "c-then-asm-data",
        need_h=data_need,
        already=(not smoke) and cta_sft.is_file() and cta_sft.stat().st_size > 0,
        argv=data_argv,
        rundir=cta_dir / "data",
    )

    remain = remaining_hours()
    epochs = "1" if smoke else ("2" if remain < 8 else "3")
    cta_argv = py_mod(
        "e2m.train.sft_lora",
        "--model",
        str(MODEL_7B),
        "--sft",
        str(cta_sft),
        "--out",
        str(cta_dir),
        "--val-split",
        str(VAL_SPLIT),
        "--epochs",
        epochs,
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
        "--target",
        "c-then-asm",
        "--max-len",
        "768" if smoke else "2048",
    )
    if smoke:
        cta_argv += ["--limit", "8", "--max-steps", "2"]
    runner.run_or_skip(
        "c-then-asm-train",
        need_h=cta_need,
        already=(not smoke) and _train_done(cta_dir),
        argv=cta_argv,
        rundir=cta_dir,
    )
    cta_adapter = cta_dir / "adapter"
    cta_ready = (cta_adapter / "adapter_config.json").is_file() or runner.dry_run
    cta_eval = cta_dir / "eval.jsonl"
    for split, min_n, tag in (
        ("val", 90, "c-then-asm-eval-val"),
        ("heldout", 78, "c-then-asm-eval-heldout"),
    ):
        eval_dir = ROOT / "runs" / (f"round3-smoke/{tag}" if smoke else tag)
        if not cta_ready:
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=runner.halt,
            ):
                runner.run_or_skip(tag, need_h=cta_eval_need, already=False, argv=["true"], rundir=eval_dir)
            else:
                runner.record(tag, "skipped", reason="c-then-asm adapter missing")
            continue
        already = (not smoke) and eval_rows(cta_eval, "c-then-asm", split, mode="c-then-asm") >= min_n
        runner.run_or_skip(
            tag,
            need_h=cta_eval_need,
            already=already,
            argv=eval_argv(
                name="c-then-asm",
                model=MODEL_7B,
                adapter=cta_adapter,
                split=split,
                out=cta_eval,
                load_4bit=True,
                limit=limit,
                mode="c-then-asm",
                max_new=max_new_cta,
                max_hours=runner.max_hours_arg(cta_eval_need),
            ),
            rundir=eval_dir,
        )

    loop_cta = ROOT / "runs" / ("round3-smoke/loop-c-then-asm-heldout" if smoke else "loop-c-then-asm-heldout")
    if not cta_ready:
        if skip_reason(
            "loop-c-then-asm-heldout",
            resume=runner.resume,
            only=runner.only,
            prior_ok=runner.prior_ok,
            halt=runner.halt,
        ):
            runner.run_or_skip(
                "loop-c-then-asm-heldout",
                need_h=loop_need,
                already=False,
                argv=["true"],
                rundir=loop_cta,
            )
        else:
            runner.record("loop-c-then-asm-heldout", "skipped", reason="c-then-asm adapter missing")
        return
    if not remaining_ok(need_h=min(loop_need, 0.4 if smoke else 1.0)):
        runner.record("loop-c-then-asm-heldout", "skipped", code=3, reason="no budget remaining for C-then-asm loop")
        return
    runner.run_or_skip(
        "loop-c-then-asm-heldout",
        need_h=loop_need,
        already=(not smoke) and _loop_done(loop_cta, 78),
        argv=loop_argv(
            name="c-then-asm",
            adapter=cta_adapter,
            model=MODEL_7B,
            split="heldout",
            k=k,
            rounds=rounds,
            out=loop_cta,
            load_4bit=True,
            limit=limit,
            max_hours=runner.max_hours_arg(loop_need),
            mode="c-then-asm",
            max_new=max_new_cta,
        ),
        rundir=loop_cta,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2M round-3 GPU pipeline")
    parser.add_argument("--smoke", action="store_true", help="1-2 tasks per stage, then exit")
    parser.add_argument("--resume", action="store_true", help="Skip stages already recorded ok")
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="Run only these stages")
    parser.add_argument("--out", type=Path, default=None, help="Pipeline directory for RESULTS/hours")
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan without GPU work")
    parser.add_argument("--lock-note", default=None, help="Text after pid/time in ~/gpu.lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["E2M_NO_DEADLINE"] = "1"
    pid = os.getpid()
    run_dir = configure_paths(smoke=args.smoke, out=args.out, lock_note=args.lock_note)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    own_lock = not args.smoke and not args.dry_run
    if not args.dry_run:
        (run_dir / "exit").unlink(missing_ok=True)
    state = install_run_files(run_dir)
    if not args.dry_run:
        check_gpu_lock()
    released = {"done": False}

    def _release() -> None:
        if not own_lock or released["done"]:
            return
        released["done"] = True
        release_gpu_lock(owner_pid=pid)

    prior_ok = set()
    if args.resume:
        prior_ok = load_ok_stages(run_dir / "RESULTS.json")
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
    halt_reason = ""
    try:
        print(
            f"round3 start pid={pid} smoke={args.smoke} resume={args.resume} "
            f"dry_run={args.dry_run} only={sorted(only) if only else None} "
            f"order={ORDER_CHOICE} budget_h={ROUND3_GPU_HOURS} "
            f"used_h={gpu_hours_used():.2f} remain_h={remaining_hours():.2f} "
            f"lock={LOCK_PATH.read_text(encoding='utf-8').strip()}",
            flush=True,
        )
        run_stages(runner)
        if args.dry_run:
            print("dry-run stages " + json.dumps(runner.stages, indent=2), flush=True)
            state["code"] = 0
            return 0
        if runner.halt:
            halt_reason = "gpu-hour budget or lock"
        code = finish(state, runner.stages, smoke=args.smoke, halt_reason=halt_reason)
        print(f"round3 complete code={code} gpu_hours={gpu_hours_used():.3f}", flush=True)
        return code
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        state["code"] = code
        if not args.dry_run:
            try:
                finish(state, runner.stages, smoke=args.smoke, halt_reason="system exit")
            except Exception:
                traceback.print_exc()
        raise
    except Exception:
        traceback.print_exc()
        state["code"] = 1
        if not args.dry_run:
            try:
                finish(state, runner.stages, smoke=args.smoke, halt_reason="error")
            except Exception:
                traceback.print_exc()
        return 1
    finally:
        _release()
        if own_lock:
            print("gpu.lock released", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
