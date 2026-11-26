"""pass@k scoring of assembly candidates against JSONL tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from e2m.harness.tasks import Task, TaskRun, load_jsonl, run_gcc_baseline, run_task


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., Codex / HumanEval, 2021)."""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        k = n
    if c < 0:
        c = 0
    if c > n:
        c = n
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


@dataclass
class Candidate:
    task_id: str
    code: str | bytes
    syntax: str = "auto"
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def load_candidates(path: str | Path) -> list[Candidate]:
    out: list[Candidate] = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            code: str | bytes
            if "bytes" in obj:
                code = bytes.fromhex(obj["bytes"]) if isinstance(obj["bytes"], str) else bytes(obj["bytes"])
            elif "hex" in obj:
                code = str(obj["hex"])
                syntax = obj.get("syntax", "hex")
                out.append(
                    Candidate(
                        task_id=str(obj.get("task_id", obj.get("id"))),
                        code=code,
                        syntax=syntax,
                        index=i,
                        meta=obj,
                    )
                )
                continue
            else:
                code = str(obj.get("code", obj.get("asm", obj.get("text", ""))))
            out.append(
                Candidate(
                    task_id=str(obj.get("task_id", obj.get("id"))),
                    code=code,
                    syntax=str(obj.get("syntax", "auto")),
                    index=i,
                    meta=obj,
                )
            )
    return out


@dataclass
class TaskScore:
    task_id: str
    n: int
    c: int
    candidates: list[dict[str, Any]]
    baseline: dict[str, Any] = field(default_factory=dict)

    def pass_at(self, k: int) -> float:
        return pass_at_k(self.n, self.c, k)


def score_tasks(
    tasks: list[Task],
    candidates: list[Candidate],
    *,
    ks: Iterable[int] = (1, 10),
    baseline_opt: str | None = "-O2",
) -> dict[str, Any]:
    grouped: dict[str, list[Candidate]] = {t.id: [] for t in tasks}
    for cand in candidates:
        grouped.setdefault(cand.task_id, []).append(cand)

    baselines: dict[str, TaskRun] = {}
    if baseline_opt:
        for task in tasks:
            baselines[task.id] = run_gcc_baseline(task, opt=baseline_opt)

    per_task: list[TaskScore] = []
    for task in tasks:
        runs_out: list[dict[str, Any]] = []
        c = 0
        for cand in grouped.get(task.id, []):
            run = run_task(task, cand.code, syntax=cand.syntax)
            rec = {
                "index": cand.index,
                "ok": run.passed,
                "code_size": run.code_size,
                "insn_count": run.insn_count,
                "status": "ok" if run.passed else (run.fail_reason or "fail"),
                "cases": [
                    {"id": case.case_id, "ok": case.passed, "status": case.emu.status, "reason": case.reason}
                    for case in run.cases
                ],
            }
            if run.passed:
                c += 1
            runs_out.append(rec)
        base = baselines.get(task.id)
        base_info: dict[str, Any] = {}
        if base is not None:
            base_info = {
                "ok": base.passed,
                "code_size": base.code_size,
                "insn_count": base.insn_count,
                "opt": baseline_opt,
                "fail_reason": base.fail_reason if not base.passed else "",
            }
        per_task.append(
            TaskScore(
                task_id=task.id,
                n=len(grouped.get(task.id, [])),
                c=c,
                candidates=runs_out,
                baseline=base_info,
            )
        )

    ks_list = list(ks)
    # Micro-average pass@k across tasks that have at least one candidate.
    scored = [t for t in per_task if t.n > 0]
    overall = {}
    for k in ks_list:
        if not scored:
            overall[str(k)] = 0.0
        else:
            overall[str(k)] = sum(t.pass_at(k) for t in scored) / len(scored)

    return {
        "pass_at_k": overall,
        "n_tasks": len(tasks),
        "n_tasks_with_candidates": len(scored),
        "ks": ks_list,
        "tasks": [
            {
                "id": t.task_id,
                "n": t.n,
                "c": t.c,
                "pass_at_k": {str(k): t.pass_at(k) for k in ks_list},
                "baseline": t.baseline,
                "candidates": t.candidates,
            }
            for t in per_task
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# E2M harness score", ""]
    pak = report.get("pass_at_k", {})
    bits = ", ".join(f"pass@{k} = {v:.3f}" for k, v in pak.items())
    lines.append(f"Overall ({report.get('n_tasks_with_candidates', 0)} tasks with candidates): {bits}")
    lines.append("")
    lines.append("| task | n | c | pass@1 | size vs O2 | insns vs O2 | fail reasons |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for t in report.get("tasks", []):
        base = t.get("baseline") or {}
        bsz = base.get("code_size") or 0
        binsn = base.get("insn_count") or 0
        sizes = [c["code_size"] for c in t["candidates"] if c.get("ok")]
        insns = [c["insn_count"] for c in t["candidates"] if c.get("ok")]
        size_s = _ratio(min(sizes) if sizes else None, bsz)
        insn_s = _ratio(min(insns) if insns else None, binsn)
        fails = sorted(
            {
                c["status"]
                for c in t["candidates"]
                if not c.get("ok")
            }
        )
        fail_s = "; ".join(fails) if fails else ""
        p1 = t.get("pass_at_k", {}).get("1", 0.0)
        lines.append(
            f"| {t['id']} | {t['n']} | {t['c']} | {p1:.2f} | {size_s} | {insn_s} | {fail_s} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _ratio(num: int | None, den: int) -> str:
    if num is None or not den:
        return "—"
    return f"{num}/{den} ({num / den:.2f}x)"


def write_report(report: dict[str, Any], json_path: str | Path | None, md_path: str | Path | None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if md_path:
        Path(md_path).write_text(render_markdown(report), encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))
