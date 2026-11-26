"""Round 7 shared helpers: pass set, compiler portfolio, acceptance check.

Oracle: gcc -O2 of the task's reference C. A candidate (assembly) is accepted
only when it passes every test (visible and hidden) AND agrees with the oracle
on N generated inputs (e2m.harness.diff). Size = bytes of .text; speed =
instructions executed summed over the task's tests.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from e2m.harness.assemble import Assembled, AssembleError, assemble
from e2m.harness.diff import diff_check
from e2m.harness.tasks import GCC_COMMON, C_PREAMBLE, Task, load_jsonl, run_task
from e2m.train.format import extract_c

ROOT = Path(".")
OUT = ROOT / "runs" / "round7"
CLANG = ROOT / "tools" / "LLVM-23.1.5-Linux-X64" / "bin" / "clang"
COMPILERS = [("gcc", "-O2"), ("gcc", "-O3"), ("gcc", "-Os"), ("clang", "-O2"), ("clang", "-Oz")]
SPEED_COMPILERS = {("gcc", "-O2"), ("gcc", "-O3"), ("clang", "-O2")}
N_ACCEPT = 500


def load_split(split: str) -> list[Task]:
    if split == "heldout":
        order = ROOT / "runs" / "round6" / "heldout_interleaved.jsonl"
        return load_jsonl(order if order.is_file() else ROOT / "eval" / "heldout.jsonl")
    wanted = set(json.loads((ROOT / "eval" / "val_split.json").read_text())["task_ids"])
    return [t for t in load_jsonl(ROOT / "tasks" / "scaled.jsonl") if t.id in wanted]


def interleaved_tasks() -> list[tuple[str, Task]]:
    """Held-out first (round-6 interleaved MBPP/HumanEval order), then val."""
    return [("heldout", t) for t in load_split("heldout")] + [("val", t) for t in load_split("val")]


def _clean_clang(asm: str) -> str:
    out = []
    for line in asm.splitlines():
        s = line.strip()
        if s.startswith((".addrsig", ".prefalign")):
            if s.startswith(".prefalign"):
                out.append("\t.p2align\t4")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def compile_c(c_source: str, compiler: str = "gcc", opt: str = "-O2") -> str:
    body = c_source if "#include" in c_source else C_PREAMBLE + c_source
    with tempfile.TemporaryDirectory(prefix="e2m-r7-") as td:
        src = Path(td) / "f.c"
        asm = Path(td) / "f.s"
        src.write_text(body, encoding="utf-8")
        exe = "gcc" if compiler == "gcc" else str(CLANG)
        flags = list(GCC_COMMON)
        cmd = [exe, opt, "-S", *flags, "-o", str(asm), str(src)]
        if compiler == "clang":
            cmd[3:3] = ["-w"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0 or not asm.is_file():
            raise AssembleError(f"{compiler} {opt} -S failed", details=(proc.stderr or proc.stdout)[-800:])
        text = asm.read_text(encoding="utf-8", errors="replace")
    return _clean_clang(text) if compiler == "clang" else text


def oracle_image(task: Task) -> Assembled:
    return assemble(compile_c(task.reference_c, "gcc", "-O2"), syntax="att")


def measure(task: Task, asm: str | Assembled, *, timeout_s: float = 0.5) -> dict[str, Any]:
    run = run_task(task, asm, timeout_s=timeout_s)
    bad = next((c for c in run.cases if not c.passed), None)
    return {"passed": bool(run.passed), "size": run.code_size, "insn": run.insn_count,
            "assemble_error": (run.assemble_error or "")[:400],
            "fail": (f"{bad.case_id}: {bad.reason}"[:300] if bad else "")}


def accept(task: Task, asm: str, oracle: Assembled, *, n: int = N_ACCEPT, seed: int = 0) -> dict[str, Any]:
    """Full tests, then differential check. Returns a JSON-able verdict."""
    m = measure(task, asm)
    if not m["passed"]:
        m.update(accepted=False, stage="tests")
        return m
    d = diff_check(task, asm, oracle, n=n, seed=seed)
    m.update(accepted=bool(d.ok), stage="diff" if not d.ok else "ok", diff_cases=d.n_cases,
             diff_skipped=d.n_skipped, diff_detail=d.detail[:300], counterexample=d.counterexample)
    return m


def pass_set_candidates() -> dict[str, list[dict[str, str]]]:
    """task_id -> list of {'src', 'c'} from round 6 loop best pass, base7b greedy/samples, round 4 hybrid."""
    cands: dict[str, list[dict[str, str]]] = {}

    def add(tid: str, src: str, text: str | None) -> None:
        if not text:
            return
        c = extract_c(text).strip()
        if not c:
            return
        lst = cands.setdefault(tid, [])
        if all(x["c"] != c for x in lst):
            lst.append({"src": src, "c": c})

    for p in ("runs/loop-c-only-7b-heldout/loop.jsonl", "runs/loop-c-only-7b-val/loop.jsonl"):
        f = ROOT / p
        if f.is_file():
            for line in f.open():
                r = json.loads(line)
                if r.get("passed_full") and r.get("best_pass"):
                    add(r["task_id"], "r6-loop", r["best_pass"].get("text"))
    f = ROOT / "runs/base7b-c/eval.jsonl"
    if f.is_file():
        for line in f.open():
            r = json.loads(line)
            if (r.get("greedy") or {}).get("ok"):
                add(r["task_id"], "base7b-greedy", r["greedy"].get("text"))
            for s in r.get("samples") or []:
                if s.get("ok"):
                    add(r["task_id"], "base7b-sample", s.get("text"))
    f = ROOT / "runs/round4/hybrid.jsonl"
    if f.is_file():
        for line in f.open():
            r = json.loads(line)
            if r.get("mode") == "hybrid-O2" and (r.get("greedy") or {}).get("ok"):
                add(r["task_id"], "r4-hybrid", r["greedy"].get("text"))
    # cap per task to keep stage 1 bounded; keep source priority order
    return {k: v[:6] for k, v in cands.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


ASM_LINE = re.compile(r"^\s+[a-z]")


def strip_asm(asm: str) -> str:
    """Drop comments and non-essential directives so prompts and search work on the body."""
    keep = []
    for line in asm.splitlines():
        s = line.split("#", 1)[0].rstrip()
        if not s.strip():
            continue
        st = s.strip()
        if st.startswith((".file", ".ident", ".section\t.note", ".section .note", '.section\t".note', ".addrsig",
                          ".size", ".type")):
            continue
        keep.append(s)
    return "\n".join(keep) + "\n"
