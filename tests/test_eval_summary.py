"""Eval-jsonl aggregation. No model and no GPU."""

import json
from pathlib import Path

from e2m.train.eval_generate import summarize


def _row(task_id: str, greedy_ok: bool, sample_ok: int, *, size: int = 10, insn: int = 20) -> dict:
    samples = []
    for i in range(5):
        ok = i < sample_ok
        samples.append(
            {
                "ok": ok,
                "category": "pass" if ok else "wrong_answer",
                "size": size,
                "insn": insn,
                "text": "x",
            }
        )
    return {
        "model": "lora",
        "mode": "asm",
        "split": "heldout",
        "task_id": task_id,
        "greedy": {
            "ok": greedy_ok,
            "category": "pass" if greedy_ok else "doesnt_assemble",
            "size": size if greedy_ok else 0,
            "insn": insn if greedy_ok else 0,
        },
        "samples": samples,
        "gcc": {"ok": True, "size": 10, "insn": 10},
    }


def test_summarize_last_write_wins_and_ratios(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    first = _row("a", False, 0)
    second = _row("a", True, 1)
    other = _row("b", False, 5)
    other["split"] = "val"
    lines = [first, second, _row("c", False, 0)]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines + [other]), encoding="utf-8")
    summary = summarize(path, "lora", "asm", "heldout")
    assert summary["n"] == 2
    assert summary["pass_at_1"] == 0.5
    # a has 1/5 samples correct -> pass@5 = 1; c has 0 -> 0. Mean 0.5.
    assert summary["pass_at_5"] == 0.5
    assert summary["size_ratio_vs_gcc_o2"] == 1.0
    assert summary["insn_ratio_vs_gcc_o2"] == 2.0
    assert summary["size_ratio_n"] == 1
    assert summary["greedy_categories"]["pass"] == 1
    assert summary["greedy_categories"]["doesnt_assemble"] == 1
    assert summary["sample_categories"]["pass"] == 1
    assert summary["sample_categories"]["wrong_answer"] == 9
