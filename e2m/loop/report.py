"""Assemble numbers for STATUS.md from files under runs/ and data/. Never invents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import summarize

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def greedy_row(eval_path: Path, name: str, mode: str, split: str) -> dict[str, Any] | None:
    if not eval_path.is_file():
        return None
    return summarize(eval_path, name, mode, split)


def fmt_pass(summary: dict[str, Any] | None, key: str = "pass_at_1") -> str:
    if not summary or summary.get(key) is None or not summary.get("n"):
        return "—"
    n = summary["n"]
    hits = None
    if key == "pass_at_1":
        cats = summary.get("greedy_categories") or {}
        hits = cats.get("pass")
        if hits is None:
            hits = round(summary[key] * n)
    val = summary[key]
    if hits is not None and key == "pass_at_1":
        return f"{val:.3f} ({hits}/{n})"
    return f"{val:.3f}"


def collect() -> dict[str, Any]:
    out: dict[str, Any] = {
        "rewrite": _load(ROOT / "data" / "scaled" / "rewrite-leakage.json"),
        "mix": None,
        "lora1_train": _load(ROOT / "runs" / "lora1" / "summary.json"),
        "qlora7b_train": _load(ROOT / "runs" / "qlora7b" / "summary.json"),
        "lora2_train": _load(ROOT / "runs" / "lora2" / "summary.json"),
        "gpu_hours": None,
        "greedy": {},
        "loops": {},
    }
    mix_txt = ROOT / "runs" / "mix-sft" / "mix.json"
    if mix_txt.is_file():
        try:
            out["mix"] = json.loads(mix_txt.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out["mix"] = {"raw": mix_txt.read_text(encoding="utf-8")[:2000]}
    hours_path = ROOT / "runs" / "loop-pipeline" / "gpu_hours.jsonl"
    if hours_path.is_file():
        rows = [json.loads(l) for l in hours_path.read_text().splitlines() if l.strip()]
        out["gpu_hours"] = {
            "seconds": sum(r.get("seconds") or 0 for r in rows),
            "jobs": rows,
        }
    greedy_specs = [
        (ROOT / "runs" / "lora1" / "eval.jsonl", "lora", "asm", "val"),
        (ROOT / "runs" / "lora1" / "eval.jsonl", "lora", "asm", "heldout"),
        (ROOT / "runs" / "lora1" / "eval.jsonl", "base", "asm", "heldout"),
        (ROOT / "runs" / "lora1" / "eval.jsonl", "base", "c", "heldout"),
        (ROOT / "runs" / "qlora7b" / "eval.jsonl", "qlora7b", "asm", "val"),
        (ROOT / "runs" / "qlora7b" / "eval.jsonl", "qlora7b", "asm", "heldout"),
        (ROOT / "runs" / "qlora7b" / "eval.jsonl", "base7b", "asm", "heldout"),
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "asm", "val"),
        (ROOT / "runs" / "lora2" / "eval.jsonl", "lora2", "asm", "heldout"),
    ]
    for path, name, mode, split in greedy_specs:
        s = greedy_row(path, name, mode, split)
        if s and s.get("n"):
            out["greedy"][f"{name}/{mode}/{split}"] = s
    loop_specs = [
        ROOT / "runs" / "loop-lora1-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-lora1-val" / "loop.jsonl",
        ROOT / "runs" / "loop-base-c-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-lora2-heldout" / "loop.jsonl",
        ROOT / "runs" / "loop-lora2-val" / "loop.jsonl",
        ROOT / "runs" / "loop-reject-train" / "loop.jsonl",
    ]
    for path in loop_specs:
        if not path.is_file():
            continue
        s = summarize_loop(path)
        if s.get("n"):
            out["loops"][path.parent.name] = s
    reject = ROOT / "data" / "scaled" / "reject.jsonl"
    if reject.is_file():
        n = sum(1 for line in reject.read_text().splitlines() if line.strip())
        out["n_reject"] = n
    rewritten = ROOT / "data" / "scaled" / "rewritten.jsonl"
    if rewritten.is_file():
        out["n_rewritten"] = sum(1 for line in rewritten.read_text().splitlines() if line.strip())
    return out


def main() -> int:
    print(json.dumps(collect(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
