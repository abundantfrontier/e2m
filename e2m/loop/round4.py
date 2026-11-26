"""Round-4 GPU pipeline: 7B C-only control, hybrid gcc lowering, loop resume, failure analysis.

Self-driving. Stops GPU stages at a 30-hour cap (10 h per stage). No calendar deadline.
CPU stages (hybrid, failure analysis) still run if the GPU budget is exhausted.
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
    ROUND4_GPU_HOURS,
    acquire_gpu_lock,
    check_gpu_lock,
    install_run_files,
    release_gpu_lock,
)
from e2m.loop.pipeline import wait_if_running
from e2m.loop.round2 import (
    HELDOUT_TASKS,
    MODEL_7B,
    SCALED_TASKS,
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
from e2m.loop.round3 import eval_argv, loop_argv
from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import summarize as summarize_eval

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SFT_C_THEN_ASM = ROOT / "data" / "scaled" / "sft_c_then_asm.jsonl"
CTA_DIR = ROOT / "runs" / "c-then-asm"
CTA_EVAL = CTA_DIR / "eval.jsonl"
CTA_ADAPTER = CTA_DIR / "adapter"
ROUND3_RESULTS = ROOT / "runs" / "round3" / "RESULTS.json"
PIPE_DIR = ROOT / "runs" / "round4"
HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
STAGE_PATH = PIPE_DIR / "stage.txt"
DONE_PATH = ROOT / "logs" / "round4.done"
LOG_PATH = ROOT / "logs" / "build-round4.log"
LOCK_NOTE = "round 4"
RESULTS_TITLE = "E2M round 4"
CURRENT_PROC: dict[str, subprocess.Popen | None] = {"proc": None}

ORDER_CHOICE = "A then B then C then D"
ORDER_REASON = (
    "Matched 7B C control first, then CPU hybrid gcc lowering, then resume the "
    "incomplete C-then-asm held-out loop, then CPU failure analysis."
)

STAGE_ORDER = [
    "c-only-train",
    "c-only-eval-val",
    "c-only-eval-heldout",
    "hybrid-score",
    "loop-c-then-asm-heldout",
    "failure-analysis",
]

PRIOR_GREEDY_KEYS = [
    ("lora2/asm/val", "lora2", "asm", "val"),
    ("lora2/asm/heldout", "lora2", "asm", "heldout"),
    ("qlora7b-v2/asm/val", "qlora7b-v2", "asm", "val"),
    ("qlora7b-v2/asm/heldout", "qlora7b-v2", "asm", "heldout"),
    ("c-then-asm/c-then-asm/val", "c-then-asm", "c-then-asm", "val"),
    ("c-then-asm/c-then-asm/heldout", "c-then-asm", "c-then-asm", "heldout"),
    ("base-c/c/heldout", "base", "c", "heldout"),
]
PRIOR_LOOP_NAMES = [
    "loop-lora2-val",
    "loop-lora2-heldout",
    "loop-qlora7b-v2-heldout",
    "loop-base-c-heldout",
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
                row = json.loads(line)
                if row.get("gpu") is False:
                    continue
                total += float(row.get("seconds") or 0)
    return total / 3600.0


def log_hours(name: str, seconds: float, ok: bool, *, path: Path | None = None, gpu: bool = True) -> None:
    hours_path = path or HOURS_PATH
    hours_path.parent.mkdir(parents=True, exist_ok=True)
    with hours_path.open("a", encoding="utf-8") as fh:
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


def remaining_ok(*, need_h: float = 0.25, hours_path: Path | None = None, check_lock: bool = True) -> bool:
    used = gpu_hours_used(hours_path)
    if used >= ROUND4_GPU_HOURS:
        print(f"gpu-hour budget used: {used:.2f}/{ROUND4_GPU_HOURS}", flush=True)
        return False
    if used + need_h > ROUND4_GPU_HOURS + 0.25:
        print(
            f"gpu-hour budget would exceed {ROUND4_GPU_HOURS}h (used {used:.2f}, need {need_h})",
            flush=True,
        )
        return False
    if check_lock:
        check_gpu_lock()
    return True


def remaining_hours(hours_path: Path | None = None) -> float:
    return max(0.0, ROUND4_GPU_HOURS - gpu_hours_used(hours_path))


def configure_paths(*, smoke: bool, out: Path | None = None, lock_note: str | None = None) -> Path:
    global PIPE_DIR, HOURS_PATH, STAGE_PATH, DONE_PATH, LOG_PATH, LOCK_NOTE
    if out is not None:
        PIPE_DIR = Path(out)
    else:
        PIPE_DIR = ROOT / "runs" / ("round4-smoke" if smoke else "round4")
    HOURS_PATH = PIPE_DIR / "gpu_hours.jsonl"
    STAGE_PATH = PIPE_DIR / "stage.txt"
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    if smoke:
        DONE_PATH = PIPE_DIR / "round4.done"
        LOG_PATH = PIPE_DIR / "build-round4.log"
    if lock_note:
        LOCK_NOTE = lock_note
    return PIPE_DIR


def run_cmd(
    name: str,
    argv: list[str],
    *,
    rundir: Path,
    env: dict | None = None,
    log_gpu: bool = True,
) -> int:
    rundir.mkdir(parents=True, exist_ok=True)
    log = rundir / "log"
    STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_PATH.write_text(name + "\n", encoding="utf-8")
    (rundir / "exit").unlink(missing_ok=True)
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
    (rundir / "exit").write_text(str(code) + "\n", encoding="utf-8")
    if log_gpu:
        log_hours(name, elapsed, code == 0, gpu=True)
    else:
        log_hours(name, elapsed, code == 0, gpu=False)
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
        self.halt_reason = ""

    def record(self, name: str, status: str, *, code: int | None = None, reason: str = "") -> None:
        self.stages[name] = {"status": status, "code": code, "reason": reason}
        print(f"stage {name} status={status} code={code} {reason}", flush=True)

    def max_hours_arg(self, need_h: float) -> str:
        remain = remaining_hours()
        cap = max(0.05, min(need_h, remain - 0.05) if remain > 0 else 0.05)
        cap = min(cap, 10.0)
        return f"{cap:.3f}"

    def run_or_skip(
        self,
        name: str,
        *,
        need_h: float,
        already: bool,
        argv: list[str],
        rundir: Path,
        cpu: bool = False,
    ) -> str:
        halt_blocks = self.halt and (not cpu or self.halt_reason == "lock")
        reason = skip_reason(
            name,
            resume=self.resume,
            only=self.only,
            prior_ok=self.prior_ok,
            halt=halt_blocks,
            already=already,
        )
        if reason:
            code = 0 if reason in {"already recorded ok", "already complete"} else None
            self.record(name, "skipped", code=code, reason=reason)
            return "skip"
        if not cpu and not remaining_ok(need_h=need_h, check_lock=not self.dry_run):
            self.record(name, "skipped", code=3, reason="gpu-hour budget")
            self.halt = True
            self.halt_reason = "budget"
            return "budget"
        if cpu and not self.dry_run:
            try:
                check_gpu_lock()
            except SystemExit as exc:
                self.record(name, "error", code=2, reason=str(exc)[:400])
                self.halt = True
                self.halt_reason = "lock"
                return "lock"
        if self.dry_run:
            self.record(name, "planned", reason="dry-run")
            return "plan"
        wait_if_running(rundir)
        try:
            code = run_cmd(name, argv, rundir=rundir, env=self.env, log_gpu=not cpu)
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
            self.halt_reason = "lock"
            return "lock"
        if code == 3:
            self.record(name, "stopped", code=3, reason="max-hours")
            if not cpu and not remaining_ok(need_h=0.25):
                self.halt = True
                self.halt_reason = "budget"
            return "budget"
        self.record(name, "error", code=code, reason=f"exit {code}")
        return "error"


def _copy_prior_results() -> dict[str, Any]:
    if not ROUND3_RESULTS.is_file():
        return {}
    try:
        return json.loads(ROUND3_RESULTS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def loop_unfinished(loop_path: Path, heldout_path: Path) -> list[str]:
    done: set[str] = set()
    if loop_path.is_file():
        with loop_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = str(row.get("task_id") or "")
                if tid:
                    done.add(tid)
    wanted: list[str] = []
    if heldout_path.is_file():
        with heldout_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                tid = str(json.loads(line).get("id") or "")
                if tid and tid not in done:
                    wanted.append(tid)
    return wanted


def hybrid_complete(path: Path, model: str, split: str, min_n: int) -> bool:
    return eval_rows(path, model, split, mode="hybrid-O2") >= min_n and eval_rows(
        path, model, split, mode="hybrid-Os"
    ) >= min_n


def collect_results(stages: dict[str, Any]) -> dict[str, Any]:
    prior = _copy_prior_results()
    greedy: dict[str, Any] = {}
    for key, _name, _mode, _split in PRIOR_GREEDY_KEYS:
        row = (prior.get("greedy") or {}).get(key)
        if row:
            greedy[key] = row
    conly_eval = ROOT / "runs" / "c-only-7b" / "eval.jsonl"
    if PIPE_DIR.name.endswith("smoke"):
        conly_eval = PIPE_DIR / "c-only-7b" / "eval.jsonl"
    for split, key in (("val", "c-only-7b/c/val"), ("heldout", "c-only-7b/c/heldout")):
        if conly_eval.is_file():
            s = summarize_eval(conly_eval, "c-only-7b", "c", split)
            if s.get("n"):
                greedy[key] = s
    hybrid_path = PIPE_DIR / "hybrid.jsonl"
    for model, split, opt in (
        ("c-then-asm", "val", "-O2"),
        ("c-then-asm", "val", "-Os"),
        ("c-then-asm", "heldout", "-O2"),
        ("c-then-asm", "heldout", "-Os"),
        ("c-only-7b", "val", "-O2"),
        ("c-only-7b", "val", "-Os"),
        ("c-only-7b", "heldout", "-O2"),
        ("c-only-7b", "heldout", "-Os"),
    ):
        mode = f"hybrid{opt}"
        key = f"{model}/{mode}/{split}"
        if hybrid_path.is_file():
            s = summarize_eval(hybrid_path, model, mode, split)
            if s.get("n"):
                greedy[key] = s
    loops: dict[str, Any] = {}
    for name in PRIOR_LOOP_NAMES:
        row = (prior.get("loops") or {}).get(name)
        if row:
            loops[name] = row
    loop_cta = ROOT / "runs" / "loop-c-then-asm-heldout" / "loop.jsonl"
    if PIPE_DIR.name.endswith("smoke"):
        loop_cta = PIPE_DIR / "loop-c-then-asm-heldout" / "loop.jsonl"
    if loop_cta.is_file():
        s = summarize_loop(loop_cta)
        if s.get("n"):
            loops["loop-c-then-asm-heldout"] = s
    unfinished = loop_unfinished(loop_cta, HELDOUT_TASKS)
    train_summary = None
    train_path = (PIPE_DIR / "c-only-7b" / "summary.json") if PIPE_DIR.name.endswith("smoke") else (
        ROOT / "runs" / "c-only-7b" / "summary.json"
    )
    if train_path.is_file():
        train_summary = json.loads(train_path.read_text(encoding="utf-8"))
    failure = None
    fail_path = PIPE_DIR / "failure_tally.json"
    if fail_path.is_file():
        raw_fail = json.loads(fail_path.read_text(encoding="utf-8"))
        failure = {k: v for k, v in raw_fail.items() if k != "failures"}
        failure["n_failures"] = raw_fail.get("c_only_pass")
    hours_jobs = []
    if HOURS_PATH.is_file():
        hours_jobs = [json.loads(l) for l in HOURS_PATH.read_text().splitlines() if l.strip()]
    used = gpu_hours_used()
    return {
        "title": RESULTS_TITLE,
        "deadline": None,
        "gpu_hours_budget": ROUND4_GPU_HOURS,
        "gpu_hours_used": round(used, 4),
        "order": ORDER_CHOICE,
        "order_reason": ORDER_REASON,
        "gpu_hours": hours_jobs,
        "stages": stages,
        "c_only_train": train_summary,
        "greedy": greedy,
        "loops": loops,
        "loop_unfinished": unfinished,
        "failure_analysis": failure,
        "prior_copied_from": str(ROUND3_RESULTS) if prior else None,
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
        f"GPU-hour budget: {ROUND4_GPU_HOURS:.0f} hours. Each GPU stage capped at 10 h.",
        f"GPU hours used: {used:.2f}/{ROUND4_GPU_HOURS:.0f}.",
        f"Order: {results.get('order') or ORDER_CHOICE}. {results.get('order_reason') or ORDER_REASON}",
        "",
        "Prior numbers for lora2, qlora7b-v2, c-then-asm, and the 1.5B C control are copied "
        "from `runs/round3/RESULTS.json`. They were not re-scored.",
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
        "c-then-asm pass@k scores the assembly half only. c-only-7b has no direct assembly.",
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
        ("c-then-asm/c-then-asm/val", "c-then-asm asm", "val"),
        ("c-then-asm/c-then-asm/heldout", "c-then-asm asm", "heldout"),
        ("c-only-7b/c/val", "c-only-7b", "val"),
        ("c-only-7b/c/heldout", "c-only-7b", "heldout"),
        ("base-c/c/heldout", "base-c 1.5B", "heldout"),
    ]:
        lines.append(_greedy_line(greedy.get(key), model, split))
    lines += [
        "",
        "## C half and hybrid (model C, gcc lowers it)",
        "",
        "Hybrid compiles the model's C with gcc and scores the image in the same harness. "
        "c-only-7b has no direct assembly.",
        "",
        "| model | set | path | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |",
        "|---|---|---|---:|---|---|---|---|",
    ]

    def _hybrid_line(key: str, model: str, split: str, path_label: str) -> str:
        s = greedy.get(key)
        if not s:
            return f"| {model} | {split} | {path_label} | — | — | — | — | — |"
        hits, n = greedy_hits(s)
        p1 = s.get("pass_at_1")
        p5 = s.get("pass_at_5")
        p1_s = f"{p1:.3f} ({frac(hits, n)})" if p1 is not None else frac(hits, n)
        p5_hits = round(p5 * n) if p5 is not None and n else None
        p5_s = f"{p5:.3f} ({frac(p5_hits, n)})" if p5 is not None else "—"
        sr = s.get("size_ratio_vs_gcc_o2")
        ir = s.get("insn_ratio_vs_gcc_o2")
        sr_s = f"{sr:.3f} (n={s.get('size_ratio_n')})" if sr is not None else "—"
        ir_s = f"{ir:.3f} (n={s.get('insn_ratio_n')})" if ir is not None else "—"
        return f"| {model} | {split} | {path_label} | {n} | {p1_s} | {p5_s} | {sr_s} | {ir_s} |"

    for split in ("val", "heldout"):
        cta = greedy.get(f"c-then-asm/c-then-asm/{split}")
        if cta:
            n = cta.get("n")
            lines.append(
                f"| c-then-asm | {split} | C half gcc -O2 (copied) | {n} | "
                f"{frac(cta.get('c_pass_n'), n)} | — | — | — |"
            )
        lines.append(_hybrid_line(f"c-then-asm/c-then-asm/{split}", "c-then-asm", split, "direct asm"))
        lines.append(_hybrid_line(f"c-then-asm/hybrid-O2/{split}", "c-then-asm", split, "C gcc -O2"))
        lines.append(_hybrid_line(f"c-then-asm/hybrid-Os/{split}", "c-then-asm", split, "C gcc -Os"))
        conly = greedy.get(f"c-only-7b/c/{split}")
        n_conly = conly.get("n") if conly else "—"
        lines.append(_hybrid_line(f"c-only-7b/hybrid-O2/{split}", "c-only-7b", split, "C gcc -O2"))
        lines.append(_hybrid_line(f"c-only-7b/hybrid-Os/{split}", "c-only-7b", split, "C gcc -Os"))
        lines.append(f"| c-only-7b | {split} | direct asm | {n_conly} | no direct asm | — | — | — |")
    lines += ["", "## C-then-asm C half and agreement (copied)", ""]
    for key, split in [
        ("c-then-asm/c-then-asm/val", "val"),
        ("c-then-asm/c-then-asm/heldout", "heldout"),
    ]:
        s = greedy.get(key)
        if not s:
            lines.append(f"- {split}: —")
            continue
        n = s.get("n")
        lines.append(
            f"- {split}: C pass {frac(s.get('c_pass_n'), n)}, C/asm agree {frac(s.get('c_asm_agree_n'), n)}, "
            f"both {s.get('c_and_asm_pass')}, C-only {s.get('c_only_pass')}, "
            f"asm-only {s.get('asm_only_pass')}"
        )
    lines += ["", "## Repair loops", "", "| run | n | pass_full | pass_visible |", "|---|---:|---|---|"]
    loops = results.get("loops") or {}
    for name in PRIOR_LOOP_NAMES:
        s = loops.get(name)
        if not s:
            lines.append(f"| {name} | — | — | — |")
            continue
        n = s.get("n")
        lines.append(
            f"| {name} | {n} | {frac(s.get('passed_full'), n)} | {frac(s.get('passed_visible'), n)} |"
        )
    unfinished = results.get("loop_unfinished") or []
    loop_s = loops.get("loop-c-then-asm-heldout") or {}
    n_loop = loop_s.get("n")
    lines += [
        "",
        f"loop-c-then-asm-heldout combined coverage: {frac(n_loop, 78)} of held-out. "
        f"Unfinished ({len(unfinished)}): {', '.join(unfinished) if unfinished else 'none'}.",
    ]
    train = results.get("c_only_train")
    if train:
        lines += [
            "",
            "## C-only 7B QLoRA train",
            "",
            f"train examples {train.get('train_examples')}/{train.get('train_tasks')} tasks, "
            f"dropped_over_p99 {train.get('dropped_over_p99')}, "
            f"token_p50 {train.get('token_p50')}, token_p99 {train.get('token_p99')}, "
            f"max_len {train.get('max_len')}, steps {train.get('global_steps')}, "
            f"train_loss {train.get('train_loss')}, target {train.get('target')}.",
        ]
    fail = results.get("failure_analysis")
    if fail:
        tally = fail.get("tally") or {}
        lines += [
            "",
            "## Part D: C-pass / asm-fail held-out",
            "",
            f"n={fail.get('n')}, C-only {fail.get('c_only_pass')}, both {fail.get('c_and_asm_pass')}, "
            f"asm-only {fail.get('asm_only_pass')}, agree {fail.get('agree_n')}/{fail.get('n')}.",
            "",
            "| bucket | n |",
            "|---|---:|",
        ]
        for bucket in ("doesnt_assemble", "abi", "logic_mismatch", "crash", "timeout", "other"):
            lines.append(f"| {bucket} | {tally.get(bucket, 0)} |")
        lines += ["", "See `runs/round4/failure_analysis.md` for truncated C vs asm examples."]
    lines += ["", "## GPU hours", "", "| stage | seconds | hours | gpu | ok |", "|---|---:|---:|---|---|"]
    gpu_total = 0.0
    for row in results.get("gpu_hours") or []:
        sec = float(row.get("seconds") or 0)
        is_gpu = row.get("gpu", True)
        if is_gpu:
            gpu_total += sec
        lines.append(
            f"| {row.get('name')} | {sec:.2f} | {sec/3600:.2f} | {is_gpu} | {row.get('ok')} |"
        )
    lines.append(f"| **GPU total** | | **{used:.2f}** | | |")
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
    elif any(v.get("status") == "stopped" and v.get("code") == 3 for v in stages.values()):
        payload["reason"] = "max-hours"
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
    max_new_c = 256 if smoke else 1536
    max_new_cta = 256 if smoke else 2048
    train_need = 0.5 if smoke else 6.0
    eval_need = 0.3 if smoke else 2.5
    loop_need = 0.4 if smoke else 10.0
    cpu_need = 0.05

    conly_dir = PIPE_DIR / "c-only-7b" if smoke else ROOT / "runs" / "c-only-7b"
    cta_eval = PIPE_DIR / "c-then-asm" / "eval.jsonl" if smoke else CTA_EVAL
    cta_adapter = PIPE_DIR / "c-then-asm" / "adapter" if smoke else CTA_ADAPTER
    if not smoke:
        cta_adapter = CTA_ADAPTER
        cta_eval = CTA_EVAL

    train_argv = py_mod(
        "e2m.train.sft_lora",
        "--model",
        str(MODEL_7B),
        "--sft",
        str(SFT_C_THEN_ASM),
        "--out",
        str(conly_dir),
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
        "--target",
        "c",
        "--max-len",
        "768" if smoke else "2048",
    )
    if smoke:
        train_argv += ["--limit", "8", "--max-steps", "2"]
    runner.run_or_skip(
        "c-only-train",
        need_h=train_need,
        already=(not smoke) and _train_done(conly_dir),
        argv=train_argv,
        rundir=conly_dir,
    )

    adapter = conly_dir / "adapter"
    adapter_ready = (adapter / "adapter_config.json").is_file() or runner.dry_run
    conly_eval = conly_dir / "eval.jsonl"
    for split, min_n, tag in (("val", 90, "c-only-eval-val"), ("heldout", 78, "c-only-eval-heldout")):
        eval_dir = PIPE_DIR / tag if smoke else ROOT / "runs" / tag
        if not adapter_ready:
            if skip_reason(
                tag,
                resume=runner.resume,
                only=runner.only,
                prior_ok=runner.prior_ok,
                halt=runner.halt and runner.halt_reason == "lock",
            ):
                runner.run_or_skip(tag, need_h=eval_need, already=False, argv=["true"], rundir=eval_dir)
            else:
                runner.record(tag, "skipped", reason="c-only-7b adapter missing")
            continue
        already = (not smoke) and eval_rows(conly_eval, "c-only-7b", split, mode="c") >= min_n
        runner.run_or_skip(
            tag,
            need_h=eval_need,
            already=already,
            argv=eval_argv(
                name="c-only-7b",
                model=MODEL_7B,
                adapter=adapter,
                split=split,
                out=conly_eval,
                load_4bit=True,
                limit=limit,
                mode="c",
                max_new=max_new_c,
                max_hours=runner.max_hours_arg(eval_need),
            ),
            rundir=eval_dir,
        )

    hybrid_out = PIPE_DIR / "hybrid.jsonl"
    hybrid_dir = PIPE_DIR / "hybrid"
    cta_min_val = 1 if smoke else 90
    cta_min_ho = 1 if smoke else 78
    conly_needed = (not smoke) and eval_rows(conly_eval, "c-only-7b", "heldout", mode="c") >= 78
    already_hybrid = (
        (not smoke)
        and hybrid_complete(hybrid_out, "c-then-asm", "val", cta_min_val)
        and hybrid_complete(hybrid_out, "c-then-asm", "heldout", cta_min_ho)
        and (not conly_needed or (
            hybrid_complete(hybrid_out, "c-only-7b", "val", 90)
            and hybrid_complete(hybrid_out, "c-only-7b", "heldout", 78)
        ))
    )
    hybrid_argv = py_mod(
        "e2m.train.hybrid",
        "--cta-eval",
        str(cta_eval),
        "--conly-eval",
        str(conly_eval),
        "--out",
        str(hybrid_out),
        "--opts",
        "O2",
        "Os",
    )
    if smoke:
        hybrid_argv += ["--limit", "1"]
    runner.run_or_skip(
        "hybrid-score",
        need_h=cpu_need,
        already=already_hybrid,
        argv=hybrid_argv,
        rundir=hybrid_dir,
        cpu=True,
    )

    loop_cta = PIPE_DIR / "loop-c-then-asm-heldout" if smoke else ROOT / "runs" / "loop-c-then-asm-heldout"
    cta_ready = (cta_adapter / "adapter_config.json").is_file() or runner.dry_run
    if not cta_ready:
        runner.record("loop-c-then-asm-heldout", "skipped", reason="c-then-asm adapter missing")
    else:
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

    fail_dir = PIPE_DIR
    already_fail = (fail_dir / "failure_tally.json").is_file() and (fail_dir / "failure_analysis.md").is_file()
    runner.run_or_skip(
        "failure-analysis",
        need_h=cpu_need,
        already=(not smoke) and already_fail,
        argv=py_mod(
            "e2m.train.failure_analysis",
            "--eval",
            str(CTA_EVAL if not smoke else cta_eval),
            "--out-dir",
            str(fail_dir),
        ),
        rundir=fail_dir / "failure",
        cpu=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2M round-4 GPU pipeline")
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
        acquire_gpu_lock(pid, LOCK_NOTE)
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
        lock_text = ""
        if LOCK_PATH.is_file():
            lock_text = LOCK_PATH.read_text(encoding="utf-8").strip()
        print(
            f"round4 start pid={pid} smoke={args.smoke} resume={args.resume} "
            f"dry_run={args.dry_run} only={sorted(only) if only else None} "
            f"order={ORDER_CHOICE} budget_h={ROUND4_GPU_HOURS} "
            f"used_h={gpu_hours_used():.2f} remain_h={remaining_hours():.2f} "
            f"lock={lock_text}",
            flush=True,
        )
        run_stages(runner)
        if args.dry_run:
            print("dry-run stages " + json.dumps(runner.stages, indent=2), flush=True)
            state["code"] = 0
            return 0
        if runner.halt:
            halt_reason = "gpu-hour budget or lock" if runner.halt_reason != "lock" else "gpu lock lost"
        code = finish(state, runner.stages, smoke=args.smoke, halt_reason=halt_reason)
        print(f"round4 complete code={code} gpu_hours={gpu_hours_used():.3f}", flush=True)
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
