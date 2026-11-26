"""Classify C-pass / asm-fail tasks from a c-then-asm eval JSONL. CPU only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from e2m.train.format import extract_asm_from_c_then_asm, extract_c_from_c_then_asm

ROOT = Path(__file__).resolve().parents[2]


def classify_asm_fail(category: str | None) -> str:
    cat = (category or "other").strip() or "other"
    if cat == "doesnt_assemble":
        return "doesnt_assemble"
    if cat == "abi":
        return "abi"
    if cat == "wrong_answer":
        return "logic_mismatch"
    if cat in {"crash", "timeout"}:
        return cat
    return "other"


def _clip(text: str, n_lines: int = 16, n_chars: int = 700) -> str:
    body = (text or "").strip()
    lines = body.splitlines()
    if len(lines) > n_lines:
        body = "\n".join(lines[:n_lines]) + "\n…"
    if len(body) > n_chars:
        body = body[: n_chars - 1] + "…"
    return body


def analyze_eval(
    eval_path: Path,
    *,
    model_name: str = "c-then-asm",
    mode: str = "c-then-asm",
    split: str = "heldout",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if eval_path.is_file():
        with eval_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("model") == model_name
                    and row.get("mode") == mode
                    and row.get("split") == split
                ):
                    rows.append(row)
    by_id = {str(r.get("task_id")): r for r in rows}
    rows = list(by_id.values())
    both = 0
    c_only = 0
    asm_only = 0
    agree = 0
    neither = 0
    failures: list[dict[str, Any]] = []
    tally: Counter[str] = Counter()
    for row in rows:
        greedy = row.get("greedy") or {}
        c_ok = bool(greedy.get("c_ok"))
        asm_ok = bool(greedy.get("ok"))
        if c_ok == asm_ok:
            agree += 1
        if c_ok and asm_ok:
            both += 1
        elif c_ok:
            c_only += 1
            bucket = classify_asm_fail(str(greedy.get("category") or ""))
            tally[bucket] += 1
            text = str(greedy.get("text") or "")
            failures.append(
                {
                    "task_id": row.get("task_id"),
                    "category": greedy.get("category"),
                    "bucket": bucket,
                    "detail": str(greedy.get("detail") or "")[:400],
                    "c": _clip(extract_c_from_c_then_asm(text)),
                    "asm": _clip(extract_asm_from_c_then_asm(text)),
                    "c_category": greedy.get("c_category"),
                }
            )
        elif asm_ok:
            asm_only += 1
        else:
            neither += 1
    failures.sort(key=lambda r: (str(r.get("bucket")), str(r.get("task_id"))))
    return {
        "eval": str(eval_path),
        "model": model_name,
        "mode": mode,
        "split": split,
        "n": len(rows),
        "c_and_asm_pass": both,
        "c_only_pass": c_only,
        "asm_only_pass": asm_only,
        "neither": neither,
        "agree_n": agree,
        "tally": dict(tally),
        "failures": failures,
    }


def pick_examples(failures: list[dict[str, Any]], per_bucket: int = 1) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    extra: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    for row in failures:
        bucket = str(row.get("bucket") or "other")
        if seen[bucket] < per_bucket:
            chosen.append(row)
            seen[bucket] += 1
        elif len(extra) < 2:
            extra.append(row)
    return chosen + extra


def write_analysis(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tally_path = out_dir / "failure_tally.json"
    md_path = out_dir / "failure_analysis.md"
    tally_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    n = payload.get("n") or 0
    tally = payload.get("tally") or {}
    lines = [
        "# C-then-asm held-out failure analysis",
        "",
        "Source: `runs/c-then-asm/eval.jsonl`, greedy held-out, C half vs assembly half.",
        "Agreement and C-only counts are copied from the eval rows; categories use the "
        "harness labels already stored on the assembly half.",
        "",
        f"Tasks: {n}. C and asm both pass: {payload.get('c_and_asm_pass')}. "
        f"C-only: {payload.get('c_only_pass')}. Asm-only: {payload.get('asm_only_pass')}. "
        f"Neither: {payload.get('neither')}. Agree: {payload.get('agree_n')}/{n}.",
        "",
        "## Tally of C-pass / asm-fail",
        "",
        "| bucket | n | harness category |",
        "|---|---:|---|",
        f"| doesnt_assemble | {tally.get('doesnt_assemble', 0)} | doesnt_assemble |",
        f"| abi | {tally.get('abi', 0)} | abi |",
        f"| logic_mismatch | {tally.get('logic_mismatch', 0)} | wrong_answer vs the model's own C |",
        f"| crash | {tally.get('crash', 0)} | crash |",
        f"| timeout | {tally.get('timeout', 0)} | timeout / insn_limit |",
        f"| other | {tally.get('other', 0)} | other |",
        f"| **total** | **{payload.get('c_only_pass')}** | |",
        "",
        "`logic_mismatch` means the C half passes the tests after gcc -O2, and the "
        "assembly half assembles and runs but returns the wrong value. That is a "
        "semantic gap between the two halves, not a compile failure.",
        "",
        "## Examples",
        "",
    ]
    examples = pick_examples(payload.get("failures") or [])
    if not examples:
        lines.append("No C-pass / asm-fail rows.")
    for row in examples:
        lines += [
            f"### `{row.get('task_id')}` — {row.get('bucket')} ({row.get('category')})",
            "",
            f"Detail: `{row.get('detail')}`",
            "",
            "C (truncated):",
            "",
            "```c",
            str(row.get("c") or "").rstrip(),
            "```",
            "",
            "Asm (truncated):",
            "",
            "```asm",
            str(row.get("asm") or "").rstrip(),
            "```",
            "",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", tally_path, flush=True)
    print("wrote", md_path, flush=True)
    return {"tally": tally_path, "md": md_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-pass / asm-fail analysis")
    parser.add_argument("--eval", type=Path, default=ROOT / "runs" / "c-then-asm" / "eval.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "round4")
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args(argv)
    payload = analyze_eval(args.eval, split=args.split)
    write_analysis(payload, args.out_dir)
    print(json.dumps({k: payload[k] for k in ("n", "c_only_pass", "tally") if k in payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
