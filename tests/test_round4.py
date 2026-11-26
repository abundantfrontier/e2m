"""Round-4 budget helpers, C-only target, hybrid opt, and SUMMARY formatting. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

from e2m.loop.guard import ROUND4_GPU_HOURS, acquire_gpu_lock, pid_alive
from e2m.loop.round4 import (
    ORDER_CHOICE,
    STAGE_ORDER,
    build_parser,
    collect_results,
    frac,
    gpu_hours_used,
    remaining_ok,
    write_summary_md,
)
from e2m.train.failure_analysis import classify_asm_fail
from e2m.train.hybrid import hybrid_mode, norm_opt
from e2m.train.sft_lora import _target_text


def test_round4_budget_is_30h(tmp_path: Path):
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(json.dumps({"name": "x", "seconds": 10 * 3600, "ok": True}) + "\n")
    assert ROUND4_GPU_HOURS == 30.0
    assert remaining_ok(need_h=1.0, hours_path=path, check_lock=False)
    assert remaining_ok(need_h=19.0, hours_path=path, check_lock=False)
    assert not remaining_ok(need_h=21.0, hours_path=path, check_lock=False)
    path.write_text(json.dumps({"name": "x", "seconds": 30 * 3600, "ok": True}) + "\n")
    assert abs(gpu_hours_used(path) - 30.0) < 1e-6
    assert not remaining_ok(need_h=0.5, hours_path=path, check_lock=False)


def test_cpu_hours_do_not_count_against_gpu_budget(tmp_path: Path):
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(
        json.dumps({"name": "hybrid-score", "seconds": 40 * 3600, "ok": True, "gpu": False}) + "\n"
        + json.dumps({"name": "c-only-train", "seconds": 3600, "ok": True, "gpu": True}) + "\n"
    )
    assert abs(gpu_hours_used(path) - 1.0) < 1e-6
    assert remaining_ok(need_h=10.0, hours_path=path, check_lock=False)


def test_stage_order_is_a_b_c_d():
    assert ORDER_CHOICE == "A then B then C then D"
    assert STAGE_ORDER == [
        "c-only-train",
        "c-only-eval-val",
        "c-only-eval-heldout",
        "hybrid-score",
        "loop-c-then-asm-heldout",
        "failure-analysis",
    ]
    assert STAGE_ORDER.index("c-only-train") < STAGE_ORDER.index("hybrid-score")
    assert STAGE_ORDER.index("hybrid-score") < STAGE_ORDER.index("loop-c-then-asm-heldout")
    assert STAGE_ORDER.index("loop-c-then-asm-heldout") < STAGE_ORDER.index("failure-analysis")


def test_c_target_uses_c_field_not_c_then_asm_response():
    row = {
        "c": "int64_t add(int64_t a, int64_t b) { return a + b; }",
        "asm": ".text\nret",
        "response": "===C===\nint64_t add(int64_t a, int64_t b) { return a + b; }\n===ASM===\n.text\nret\n",
    }
    assert _target_text(row, kind="c") == row["c"]
    assert "===ASM===" in _target_text(row, kind="c-then-asm")
    assert _target_text(row, kind="asm").startswith("===C===")


def test_hybrid_mode_and_opt_normalization():
    assert norm_opt("O2") == "-O2"
    assert norm_opt("-Os") == "-Os"
    assert hybrid_mode("O2") == "hybrid-O2"
    assert hybrid_mode("-Os") == "hybrid-Os"


def test_classify_c_pass_asm_fail_buckets():
    assert classify_asm_fail("doesnt_assemble") == "doesnt_assemble"
    assert classify_asm_fail("abi") == "abi"
    assert classify_asm_fail("wrong_answer") == "logic_mismatch"
    assert classify_asm_fail("crash") == "crash"
    assert classify_asm_fail("timeout") == "timeout"
    assert classify_asm_fail("weird") == "other"


def test_summary_md_has_denominators_and_copied_comparisons(tmp_path: Path):
    results = {
        "title": "E2M round 4",
        "gpu_hours_used": 4.5,
        "order": "A then B then C then D",
        "stages": {"c-only-train": {"status": "ok", "code": 0, "reason": ""}},
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
                "pass_at_1": 7 / 78,
                "pass_at_5": 9 / 78,
                "greedy_categories": {"pass": 7},
                "c_pass_n": 48,
                "c_asm_agree_n": 35,
                "c_and_asm_pass": 6,
                "c_only_pass": 42,
                "asm_only_pass": 1,
            },
            "c-only-7b/c/heldout": {
                "n": 78,
                "pass_at_1": 40 / 78,
                "pass_at_5": 50 / 78,
                "greedy_categories": {"pass": 40},
            },
        },
        "loops": {"loop-c-then-asm-heldout": {"n": 78, "passed_full": 20, "passed_visible": 25}},
        "loop_unfinished": [],
        "failure_analysis": {
            "n": 78,
            "c_only_pass": 42,
            "c_and_asm_pass": 6,
            "asm_only_pass": 1,
            "agree_n": 35,
            "tally": {"logic_mismatch": 23, "crash": 9, "timeout": 9, "doesnt_assemble": 1},
        },
        "gpu_hours": [{"name": "c-only-train", "seconds": 3600, "ok": True, "gpu": True}],
    }
    path = tmp_path / "SUMMARY.md"
    write_summary_md(results, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# E2M round 4")
    assert "38/90" in text
    assert "5/78" in text
    assert "31/78" in text
    assert "7/78" in text
    assert "48/78" in text
    assert "40/78" in text
    assert "no direct asm" in text
    assert "lora2" in text
    assert "qlora7b-v2" in text
    assert "c-only-7b" in text
    assert "logic_mismatch" in text
    assert "4.50/30" in text
    assert frac(42, 78) == "42/78"


def test_collect_results_shape():
    payload = collect_results({"c-only-train": {"status": "skipped", "code": 0, "reason": "test"}})
    assert payload["gpu_hours_budget"] == 30.0
    assert payload["deadline"] is None
    assert payload["order"] == "A then B then C then D"
    assert payload["stages"]["c-only-train"]["status"] == "skipped"


def test_parser_round4():
    args = build_parser().parse_args(["--dry-run", "--only", "c-only-train", "hybrid-score"])
    assert args.dry_run
    assert args.only == ["c-only-train", "hybrid-score"]


def test_dry_run_plans_a_b_c_d(tmp_path: Path):
    from e2m.loop.round4 import main as round4_main

    code = round4_main(["--dry-run", "--out", str(tmp_path / "pipe")])
    assert code == 0


def test_acquire_gpu_lock_overwrites_stale(tmp_path: Path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("e2m 1 2020-01-01 stale\n", encoding="utf-8")
    assert not pid_alive(1)
    text = acquire_gpu_lock(999, "round 4", path=lock)
    assert text.startswith("e2m 999 ")
    assert "round 4" in text


def test_score_c_opt_o2_and_os():
    from e2m.harness.tasks import load_jsonl
    from e2m.train.eval_generate import _score_c

    tasks = load_jsonl((Path(__file__).resolve().parents[1] / "tasks/starter.jsonl"))
    add = next(t for t in tasks if t.id == "add_i64")
    r_o2 = _score_c(add, add.reference_c, opt="-O2")
    r_os = _score_c(add, add.reference_c, opt="-Os")
    assert r_o2["ok"] is True
    assert r_os["ok"] is True
    assert r_o2["opt"] == "-O2"
    assert r_os["opt"] == "-Os"
