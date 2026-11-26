"""Build the lora2 SFT mix: gcc defaults + rewritten-spec gcc + reject-loop passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def mix_sft(
    *,
    gcc_sft: Path,
    rewritten: Path,
    reject: Path,
    out_path: Path,
    val_ids: set[str],
    heldout_ids: set[str],
    starter_sft: Path | None = None,
) -> dict[str, Any]:
    blocked = set(val_ids) | set(heldout_ids)
    gcc_by_id: dict[str, dict[str, Any]] = {}
    n_gcc = 0
    sources = [gcc_sft]
    if starter_sft is not None and starter_sft.is_file():
        sources.append(starter_sft)
    gcc_rows: list[dict[str, Any]] = []
    for path in sources:
        for row in _load_jsonl(path):
            if row.get("goal") != "default":
                continue
            tid = str(row.get("task_id") or "")
            if not tid or tid in blocked:
                continue
            gcc_by_id[tid] = row
            gcc_rows.append(row)
            n_gcc += 1

    rewritten_rows: list[dict[str, Any]] = []
    n_rewritten_skip = 0
    for row in _load_jsonl(rewritten):
        tid = str(row.get("task_id") or "")
        if not tid or tid in blocked or row.get("dropped"):
            n_rewritten_skip += 1
            continue
        gcc = gcc_by_id.get(tid)
        if gcc is None or not gcc.get("asm"):
            n_rewritten_skip += 1
            continue
        rewritten_rows.append(
            {
                "task_id": tid,
                "spec": row["spec"],
                "signature": row.get("signature") or gcc["signature"],
                "goal": "default",
                "asm": gcc["asm"],
                "sha256": gcc.get("sha256"),
                "size": gcc.get("size"),
                "insn_count": gcc.get("insn_count"),
                "compiler": gcc.get("compiler"),
                "flags": gcc.get("flags"),
                "source": "rewritten_gcc",
            }
        )

    reject_rows: list[dict[str, Any]] = []
    n_reject_skip = 0
    for row in _load_jsonl(reject):
        tid = str(row.get("task_id") or "")
        if not tid or tid in blocked:
            n_reject_skip += 1
            continue
        if not row.get("asm"):
            n_reject_skip += 1
            continue
        reject_rows.append(
            {
                "task_id": tid,
                "spec": row["spec"],
                "signature": row["signature"],
                "goal": "default",
                "asm": row["asm"],
                "size": row.get("size"),
                "insn_count": row.get("insn"),
                "source": row.get("source") or "reject_loop",
                "spec_kind": row.get("spec_kind") or "original",
            }
        )

    mixed = gcc_rows + rewritten_rows + reject_rows
    _write_jsonl(out_path, mixed)
    stats = {
        "n_gcc_default": n_gcc,
        "n_rewritten_gcc": len(rewritten_rows),
        "n_reject": len(reject_rows),
        "n_mixed": len(mixed),
        "n_blocked": len(blocked),
        "n_rewritten_skip": n_rewritten_skip,
        "n_reject_skip": n_reject_skip,
        "out": str(out_path),
        "unique_task_ids": len({r["task_id"] for r in mixed}),
    }
    return stats
