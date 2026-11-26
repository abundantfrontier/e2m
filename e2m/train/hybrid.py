"""Score already-generated C with gcc -O2 / -Os through the harness.

CPU-only. Reuses model C from eval JSONL (c-then-asm or c mode) and compiles
it with ``assemble_reference`` / ``_score_c`` rather than going around the harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e2m.harness.tasks import Task, load_jsonl
from e2m.train.eval_generate import _done_ids, _gcc_reference, _score_c, summarize
from e2m.train.format import extract_c, extract_c_from_c_then_asm

ROOT = Path(__file__).resolve().parents[2]


def extract_model_c(text: str, source_mode: str) -> str:
    if source_mode == "c-then-asm":
        return extract_c_from_c_then_asm(text or "")
    return extract_c(text or "")


def norm_opt(opt: str) -> str:
    text = str(opt).strip()
    return text if text.startswith("-") else f"-{text}"


def hybrid_mode(opt: str) -> str:
    return f"hybrid{norm_opt(opt)}"


def _tasks_by_id(path: Path) -> dict[str, Task]:
    if not path.is_file():
        return {}
    return {t.id: t for t in load_jsonl(path)}


def _load_eval_rows(path: Path, model_name: str, source_mode: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("model") == model_name
                and row.get("mode") == source_mode
                and row.get("split") == split
            ):
                rows.append(row)
    by_id = {str(r.get("task_id")): r for r in rows}
    return list(by_id.values())


def score_source(
    *,
    eval_path: Path,
    model_name: str,
    source_mode: str,
    split: str,
    tasks_path: Path,
    out_path: Path,
    opts: list[str],
    limit: int = 0,
) -> dict[str, Any]:
    tasks = _tasks_by_id(tasks_path)
    rows = _load_eval_rows(eval_path, model_name, source_mode, split)
    if limit > 0:
        rows = rows[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_missing_task = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for opt in opts:
            opt_flag = norm_opt(opt)
            mode = hybrid_mode(opt_flag)
            done = _done_ids(out_path, model_name, mode, split)
            for row in rows:
                tid = str(row.get("task_id") or "")
                if tid in done:
                    n_skipped += 1
                    continue
                task = tasks.get(tid)
                if task is None:
                    n_missing_task += 1
                    continue
                greedy_text = str((row.get("greedy") or {}).get("text") or "")
                greedy = _score_c(task, extract_model_c(greedy_text, source_mode), opt=opt_flag)
                greedy["source_text"] = greedy_text[:1500]
                samples_out: list[dict[str, Any]] = []
                for sample in row.get("samples") or []:
                    sample_text = str(sample.get("text") or "")
                    scored = _score_c(task, extract_model_c(sample_text, source_mode), opt=opt_flag)
                    scored["source_text"] = sample_text[:1500]
                    samples_out.append(scored)
                gcc = row.get("gcc") if isinstance(row.get("gcc"), dict) else None
                if not gcc or not gcc.get("ok"):
                    gcc = _gcc_reference(task)
                out_row = {
                    "model": model_name,
                    "mode": mode,
                    "source_mode": source_mode,
                    "split": split,
                    "task_id": tid,
                    "opt": opt_flag,
                    "greedy": greedy,
                    "samples": samples_out,
                    "gcc": gcc,
                }
                fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fh.flush()
                n_written += 1
    summaries = {}
    for opt in opts:
        mode = hybrid_mode(opt)
        summaries[mode] = summarize(out_path, model_name, mode, split)
    stats = {
        "eval": str(eval_path),
        "model": model_name,
        "source_mode": source_mode,
        "split": split,
        "n_eval_rows": len(rows),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_missing_task": n_missing_task,
        "summaries": summaries,
    }
    print("hybrid", json.dumps(stats, default=str), flush=True)
    return stats


def run_hybrid(
    *,
    cta_eval: Path,
    conly_eval: Path,
    out_path: Path,
    opts: list[str],
    splits: list[str],
    limit: int = 0,
    scaled_tasks: Path | None = None,
    heldout_tasks: Path | None = None,
) -> dict[str, Any]:
    scaled = scaled_tasks or (ROOT / "tasks" / "scaled.jsonl")
    heldout = heldout_tasks or (ROOT / "eval" / "heldout.jsonl")
    jobs: list[dict[str, Any]] = []
    sources = []
    if cta_eval.is_file():
        sources.append((cta_eval, "c-then-asm", "c-then-asm"))
    if conly_eval.is_file():
        sources.append((conly_eval, "c-only-7b", "c"))
    for eval_path, name, mode in sources:
        for split in splits:
            tasks_path = heldout if split == "heldout" else scaled
            jobs.append(
                score_source(
                    eval_path=eval_path,
                    model_name=name,
                    source_mode=mode,
                    split=split,
                    tasks_path=tasks_path,
                    out_path=out_path,
                    opts=opts,
                    limit=limit,
                )
            )
    payload = {"out": str(out_path), "n_jobs": len(jobs), "jobs": jobs}
    print("hybrid done", json.dumps({"out": str(out_path), "n_jobs": len(jobs)}), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score model C via gcc -O2/-Os")
    parser.add_argument("--cta-eval", type=Path, default=ROOT / "runs" / "c-then-asm" / "eval.jsonl")
    parser.add_argument("--conly-eval", type=Path, default=ROOT / "runs" / "c-only-7b" / "eval.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "round4" / "hybrid.jsonl")
    parser.add_argument("--opts", nargs="+", default=["O2", "Os"])
    parser.add_argument("--splits", nargs="+", default=["val", "heldout"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scaled-tasks", type=Path, default=ROOT / "tasks" / "scaled.jsonl")
    parser.add_argument("--heldout-tasks", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    args = parser.parse_args(argv)
    run_hybrid(
        cta_eval=args.cta_eval,
        conly_eval=args.conly_eval,
        out_path=args.out,
        opts=list(args.opts),
        splits=list(args.splits),
        limit=args.limit,
        scaled_tasks=args.scaled_tasks,
        heldout_tasks=args.heldout_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
