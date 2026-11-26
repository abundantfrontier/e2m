"""Budget tables and failure breakdowns from a loop JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_BUDGETS = (1, 4, 8, 16, 32)


def load_rows(path: Path, *, model: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model is not None and row.get("model") != model:
                continue
            if split is not None and row.get("split") != split:
                continue
            rows.append(row)
    by_id = {r["task_id"]: r for r in rows}
    return list(by_id.values())


def _rate(rows: list[dict[str, Any]], field: str, budget: int) -> float | None:
    if not rows:
        return None
    hits = 0
    for row in rows:
        call = row.get(field)
        if call is not None and int(call) <= budget:
            hits += 1
    return hits / len(rows)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize_loop(
    path: Path,
    *,
    model: str | None = None,
    split: str | None = None,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    rows = load_rows(path, model=model, split=split)
    n = len(rows)
    cats: dict[str, int] = {}
    size_ratios: list[float] = []
    insn_ratios: list[float] = []
    full_n = 0
    vis_n = 0
    hid_n = 0
    calls_sum = 0
    for row in rows:
        calls_sum += int(row.get("calls") or 0)
        if row.get("passed_full"):
            full_n += 1
            gcc = row.get("gcc") or {}
            best = row.get("best_pass") or {}
            if gcc.get("ok") and gcc.get("size") and best.get("size"):
                size_ratios.append(best["size"] / gcc["size"])
            if gcc.get("ok") and gcc.get("insn") and best.get("insn"):
                insn_ratios.append(best["insn"] / gcc["insn"])
        if row.get("passed_visible"):
            vis_n += 1
        if row.get("passed_hidden"):
            hid_n += 1
        last = (row.get("attempts") or [{}])[-1]
        cat = last.get("category_full") or last.get("category_visible") or "crash"
        if row.get("passed_full"):
            cat = "pass"
        cats[cat] = cats.get(cat, 0) + 1
    budget_full = {str(b): _rate(rows, "first_full_call", b) for b in budgets}
    budget_visible = {str(b): _rate(rows, "first_visible_call", b) for b in budgets}
    budget_hidden = {str(b): _rate(rows, "first_hidden_call", b) for b in budgets}
    return {
        "model": model or (rows[0].get("model") if rows else None),
        "split": split or (rows[0].get("split") if rows else None),
        "n": n,
        "passed_full": full_n,
        "passed_visible": vis_n,
        "passed_hidden": hid_n,
        "pass_full": (full_n / n) if n else None,
        "pass_visible": (vis_n / n) if n else None,
        "pass_hidden": (hid_n / n) if n else None,
        "mean_calls": (calls_sum / n) if n else None,
        "budget_full": budget_full,
        "budget_visible": budget_visible,
        "budget_hidden": budget_hidden,
        "size_ratio_vs_gcc_o2": _mean(size_ratios),
        "insn_ratio_vs_gcc_o2": _mean(insn_ratios),
        "size_ratio_n": len(size_ratios),
        "insn_ratio_n": len(insn_ratios),
        "final_categories": cats,
    }
