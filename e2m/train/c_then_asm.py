"""Build the C-then-assembly SFT set from the lora2 mix plus task C sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e2m.train.format import format_c_then_asm_response

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_c_by_id(*paths: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = str(row.get("id") or row.get("task_id") or "")
                c_src = str(row.get("reference_c") or row.get("c") or "")
                if tid and c_src.strip():
                    out[tid] = c_src
    return out


def build_c_then_asm(
    *,
    sft_path: Path,
    scaled_path: Path,
    starter_path: Path,
    out_path: Path,
    limit: int = 0,
) -> dict[str, Any]:
    c_by_id = load_c_by_id(scaled_path, starter_path)
    rows = _load_jsonl(sft_path)
    if limit > 0:
        rows = rows[:limit]
    out_rows: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        tid = str(row.get("task_id") or "")
        asm = str(row.get("asm") or "")
        c_src = c_by_id.get(tid, "")
        if not tid or not asm.strip() or not c_src.strip():
            missing += 1
            continue
        response = format_c_then_asm_response(c_src, asm)
        out_rows.append(
            {
                "task_id": tid,
                "spec": row.get("spec"),
                "signature": row.get("signature"),
                "goal": row.get("goal") or "default",
                "c": c_src,
                "asm": asm,
                "response": response,
                "compiler": row.get("compiler"),
                "flags": row.get("flags"),
                "source": row.get("source"),
                "sha256": row.get("sha256"),
                "size": row.get("size"),
                "insn_count": row.get("insn_count"),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "sft": str(sft_path),
        "out": str(out_path),
        "n_sft": len(rows),
        "n_written": len(out_rows),
        "n_missing_c": missing,
        "n_c_sources": len(c_by_id),
        "unique_task_ids": len({r["task_id"] for r in out_rows}),
        "delimiter": "===C=== then ===ASM===",
    }
    (out_path.parent / "c_then_asm_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print("c-then-asm data", json.dumps(stats), flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join sft_lora2 with C sources for C-then-asm SFT")
    parser.add_argument("--sft", type=Path, default=ROOT / "data" / "scaled" / "sft_lora2.jsonl")
    parser.add_argument("--scaled", type=Path, default=ROOT / "tasks" / "scaled.jsonl")
    parser.add_argument("--starter", type=Path, default=ROOT / "tasks" / "starter.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "scaled" / "sft_c_then_asm.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    stats = build_c_then_asm(
        sft_path=args.sft,
        scaled_path=args.scaled,
        starter_path=args.starter,
        out_path=args.out,
        limit=args.limit,
    )
    if stats["n_written"] <= 0:
        raise SystemExit("c-then-asm dataset empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
