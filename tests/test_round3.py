"""Round-3 budget helpers, pair-goal / C-then-asm data, and SUMMARY formatting. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

from e2m.loop.guard import ROUND3_GPU_HOURS, past_deadline
from e2m.loop.round3 import (
    ORDER_CHOICE,
    STAGE_ORDER,
    build_parser,
    collect_results,
    frac,
    gpu_hours_used,
    remaining_ok,
    write_summary_md,
)
from e2m.train.c_then_asm import build_c_then_asm
from e2m.train.dpo import collapse_task_ids, pair_goal, prepare_dpo2
from e2m.train.format import format_c_then_asm_response, split_c_then_asm


def test_round3_budget_is_30h_and_ignores_deadline(tmp_path: Path, monkeypatch):
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(json.dumps({"name": "x", "seconds": 10 * 3600, "ok": True}) + "\n")
    assert ROUND3_GPU_HOURS == 30.0
    assert remaining_ok(need_h=1.0, hours_path=path, check_lock=False)
    assert remaining_ok(need_h=19.0, hours_path=path, check_lock=False)
    assert not remaining_ok(need_h=21.0, hours_path=path, check_lock=False)
    path.write_text(json.dumps({"name": "x", "seconds": 30 * 3600, "ok": True}) + "\n")
    assert abs(gpu_hours_used(path) - 30.0) < 1e-6
    assert not remaining_ok(need_h=0.5, hours_path=path, check_lock=False)
    monkeypatch.setenv("E2M_NO_DEADLINE", "1")
    assert past_deadline() is False


def test_stage_order_is_a_then_b():
    assert ORDER_CHOICE == "A then B"
    assert STAGE_ORDER[0] == "dpo-pairs"
    assert STAGE_ORDER[1] == "dpo2-train"
    assert "c-then-asm-train" in STAGE_ORDER
    assert STAGE_ORDER.index("dpo2-train") < STAGE_ORDER.index("c-then-asm-train")


def test_pair_goal_uses_the_row_goal():
    assert pair_goal({"goal": "size"}) == "size"
    assert pair_goal({"goal": "speed"}) == "speed"
    assert pair_goal({"goal": "default"}) == "default"
    assert pair_goal({"goal": "size"}, use_pair_goal=False) == "default"
    assert pair_goal({}) == "default"


def test_collapse_subset_is_stable_and_size_30():
    ids = [f"t{i}" for i in range(90)]
    first = collapse_task_ids(ids, n=30, seed=0)
    second = collapse_task_ids(ids, n=30, seed=0)
    assert first == second
    assert len(first) == 30
    assert len(set(first)) == 30
    assert collapse_task_ids(ids, n=30, seed=1) != first


def test_prepare_dpo2_records_pair_goal(tmp_path: Path):
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        {
            "task_id": "a",
            "goal": "size",
            "spec": "s",
            "signature": "int64_t a(void)",
            "chosen": {"asm": "ok", "sha256": "c1"},
            "rejected": {"asm": "bad", "sha256": "r1"},
        },
        {
            "task_id": "b",
            "goal": "default",
            "spec": "s",
            "signature": "int64_t b(void)",
            "chosen": {"asm": "ok", "sha256": "c2"},
            "rejected": {"asm": "bad", "sha256": "r2"},
        },
    ]
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    val = tmp_path / "val.json"
    val.write_text(json.dumps({"task_ids": ["v1", "v2", "v3"]}), encoding="utf-8")
    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps({"id": "h1"}) + "\n", encoding="utf-8")
    out = tmp_path / "dpo2"
    meta = prepare_dpo2(
        pairs_path=pairs,
        val_path=val,
        heldout_path=held,
        out_dir=out,
        lora2_eval=tmp_path / "missing.jsonl",
        tasks_path=tmp_path / "tasks.jsonl",
        collapse_n=2,
    )
    assert meta["choice"] == "pair_goal_in_prompt"
    assert meta["prompt_goal"] == "pair"
    assert meta["goal_counts"]["size"] == 1
    assert (out / "pairs_meta.json").is_file()
    assert json.loads((out / "collapse_subset.json").read_text())["n"] == 2


def test_c_then_asm_split_and_build(tmp_path: Path):
    c_src = "int64_t add(int64_t a, int64_t b) { return a + b; }"
    asm = ".text\n.globl add\nadd:\n\tret"
    blob = format_c_then_asm_response(c_src, asm)
    got_c, got_a = split_c_then_asm(blob)
    assert "return a + b" in got_c
    assert ".text" in got_a
    prefix = "Sure.\n" + blob
    got_c, got_a = split_c_then_asm(prefix)
    assert "return a + b" in got_c
    assert "add:" in got_a

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "add_i64", "reference_c": c_src, "signature": "int64_t add(int64_t a, int64_t b)"})
        + "\n",
        encoding="utf-8",
    )
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "task_id": "add_i64",
                "spec": "add two integers",
                "signature": "int64_t add(int64_t a, int64_t b)",
                "goal": "default",
                "asm": asm,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "cta.jsonl"
    stats = build_c_then_asm(
        sft_path=sft,
        scaled_path=tasks,
        starter_path=tmp_path / "starter.jsonl",
        out_path=out,
    )
    assert stats["n_written"] == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert "===C===" in row["response"]
    assert "===ASM===" in row["response"]
    assert row["c"] == c_src


def test_summary_md_has_denominators_and_comparisons(tmp_path: Path):
    results = {
        "title": "E2M round 3",
        "gpu_hours_used": 4.5,
        "order": "A then B",
        "stages": {"dpo-pairs": {"status": "ok", "code": 0, "reason": ""}},
        "dpo2_pairs": {"choice": "pair_goal_in_prompt", "n_pairs_loaded": 46472, "goal_counts": {"default": 1}},
        "greedy": {
            "lora2/asm/val": {
                "n": 90,
                "pass_at_1": 38 / 90,
                "pass_at_5": 45 / 90,
                "greedy_categories": {"pass": 38},
            },
            "qlora7b-v2/asm/heldout": {
                "n": 78,
                "pass_at_1": 5 / 78,
                "pass_at_5": 12 / 78,
                "greedy_categories": {"pass": 5},
            },
            "base-c/c/heldout": {
                "n": 78,
                "pass_at_1": 31 / 78,
                "pass_at_5": 49 / 78,
                "greedy_categories": {"pass": 31},
            },
            "c-then-asm/c-then-asm/heldout": {
                "n": 78,
                "pass_at_1": 10 / 78,
                "pass_at_5": 20 / 78,
                "greedy_categories": {"pass": 10},
                "c_pass_n": 25,
                "c_asm_agree_n": 40,
                "c_and_asm_pass": 8,
                "c_only_pass": 17,
                "asm_only_pass": 2,
            },
        },
        "loops": {"loop-lora2-heldout": {"n": 78, "passed_full": 10, "passed_visible": 15}},
        "gpu_hours": [{"name": "dpo2-train", "seconds": 3600, "ok": True}],
    }
    path = tmp_path / "SUMMARY.md"
    write_summary_md(results, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# E2M round 3")
    assert "38/90" in text
    assert "5/78" in text
    assert "31/78" in text
    assert "10/78" in text
    assert "25/78" in text
    assert "lora2" in text
    assert "qlora7b-v2" in text
    assert "base-c" in text
    assert "pair_goal_in_prompt" in text
    assert "4.50/30" in text
    assert frac(10, 78) == "10/78"


def test_collect_results_shape():
    payload = collect_results({"dpo-pairs": {"status": "skipped", "code": 0, "reason": "test"}})
    assert payload["gpu_hours_budget"] == 30.0
    assert payload["deadline"] is None
    assert payload["order"] == "A then B"
    assert payload["stages"]["dpo-pairs"]["status"] == "skipped"


def test_parser_round3():
    args = build_parser().parse_args(["--dry-run", "--only", "dpo-pairs", "dpo2-train"])
    assert args.dry_run
    assert args.only == ["dpo-pairs", "dpo2-train"]


def test_dry_run_plans_a_then_b(tmp_path: Path):
    from e2m.loop.round3 import main as round3_main

    code = round3_main(["--dry-run", "--out", str(tmp_path / "pipe")])
    assert code == 0
