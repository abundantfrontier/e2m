"""Held-out set: schema, provenance, and separation from the training tasks."""

import json
from pathlib import Path

from e2m.harness.tasks import load_jsonl, run_gcc_baseline
from e2m.heldout.near_dup import collision, load_scaled_index

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "eval" / "heldout.jsonl"
SCALED = ROOT / "tasks" / "scaled.jsonl"


def test_heldout_covers_both_benchmarks_and_stays_out_of_train():
    tasks = load_jsonl(HELDOUT)
    assert len(tasks) >= 60
    raw = [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = {row["source"].split("/")[0] for row in raw}
    assert "HumanEval" in sources and "MBPP" in sources
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    for task in tasks:
        assert task.spec.strip()
        assert task.signature.strip()
        assert task.reference_c.strip()
        assert task.tests
    names, bodies = load_scaled_index(SCALED)
    hits = [row["id"] for row in raw if collision(row, names, bodies)]
    assert hits == []


def test_heldout_sample_passes_gcc_o2():
    tasks = {t.id: t for t in load_jsonl(HELDOUT)}
    for tid in ("he13_gcd", "mbpp8_square_nums", "he45_triangle_area"):
        run = run_gcc_baseline(tasks[tid], opt="-O2")
        assert run.passed, f"{tid}: {run.fail_reason}"
