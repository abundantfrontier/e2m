"""Round-2 budget helpers, resume/only skip logic, and SUMMARY formatting. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

from e2m.loop.round2 import (
    ROUND2_GPU_HOURS,
    SKIP_ON_RESUME,
    build_parser,
    collect_results,
    frac,
    gpu_hours_used,
    load_ok_stages,
    remaining_ok,
    seed_carried_hours,
    skip_reason,
    valid_base7b_row,
    write_summary_md,
)


def test_frac_prints_denominator():
    assert frac(10, 78) == "10/78"
    assert frac(0, 90) == "0/90"
    assert frac(None, 90) == "—"


def test_gpu_hours_and_budget_gate(tmp_path: Path):
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(
        json.dumps({"name": "x", "seconds": 60 * 3600, "ok": True}) + "\n",
        encoding="utf-8",
    )
    assert abs(gpu_hours_used(path) - 60.0) < 1e-6
    assert not remaining_ok(need_h=0.5, hours_path=path, check_lock=False)
    path.write_text(
        json.dumps({"name": "x", "seconds": 10 * 3600, "ok": True}) + "\n",
        encoding="utf-8",
    )
    assert remaining_ok(need_h=1.0, hours_path=path, check_lock=False)
    assert not remaining_ok(need_h=55.0, hours_path=path, check_lock=False)
    assert ROUND2_GPU_HOURS == 60.0


def test_summary_md_uses_denominators(tmp_path: Path):
    results = {
        "gpu_hours_used": 1.5,
        "stages": {"loop-lora2-val": {"status": "ok", "code": 0, "reason": ""}},
        "greedy": {
            "dpo/asm/heldout": {
                "n": 78,
                "pass_at_1": 5 / 78,
                "pass_at_5": 8 / 78,
                "greedy_categories": {"pass": 5},
                "size_ratio_vs_gcc_o2": 1.1,
                "insn_ratio_vs_gcc_o2": 1.2,
                "size_ratio_n": 5,
                "insn_ratio_n": 5,
            }
        },
        "loops": {
            "loop-lora2-val": {"n": 90, "passed_full": 49, "passed_visible": 51},
        },
        "gpu_hours": [{"name": "loop-lora2-val", "seconds": 5400, "ok": True}],
    }
    path = tmp_path / "SUMMARY.md"
    write_summary_md(results, path)
    text = path.read_text(encoding="utf-8")
    assert "5/78" in text
    assert "49/90" in text
    assert "51/90" in text
    assert "1.50/60" in text or "1.50/60." in text
    assert "loop-lora2-val" in text


def test_collect_results_shape():
    payload = collect_results({"loop-lora2-val": {"status": "skipped", "code": 0, "reason": "test"}})
    assert payload["gpu_hours_budget"] == 60.0
    assert "deadline" in payload
    assert payload["stages"]["loop-lora2-val"]["status"] == "skipped"
    assert "greedy" in payload
    assert "loops" in payload


def test_skip_reason_resume_does_not_continue_dpo():
    assert "loop-dpo-heldout" in SKIP_ON_RESUME
    assert (
        skip_reason("loop-dpo-heldout", resume=True, already=False)
        == "resume does not rerun or continue loop-dpo-heldout"
    )
    assert skip_reason("7b-base-asm", resume=True, already=False) is None
    assert skip_reason("7b-base-asm", resume=True, prior_ok={"7b-base-asm"}) == "already recorded ok"
    assert skip_reason("dpo-train", resume=True, prior_ok={"dpo-train"}) == "already recorded ok"
    assert skip_reason("qlora7b-v2-train", resume=True, already=True) == "already complete"


def test_skip_reason_only_and_halt():
    only = {"7b-base-asm", "qlora7b-v2-train"}
    assert skip_reason("loop-lora2-val", only=only) == "not in --only"
    assert skip_reason("7b-base-asm", only=only) is None
    assert skip_reason("qlora7b-v2-train", only=only, halt=True) == "pipeline already halted"
    assert skip_reason("loop-dpo-heldout", resume=True, only={"loop-dpo-heldout"}) == (
        "resume does not rerun or continue loop-dpo-heldout"
    )


def test_load_ok_stages(tmp_path: Path):
    prior = tmp_path / "RESULTS.json"
    prior.write_text(
        json.dumps(
            {
                "stages": {
                    "dpo-train": {"status": "ok", "code": 0},
                    "loop-dpo-heldout": {"status": "stopped", "code": 3},
                    "7b-base-asm": {"status": "skipped"},
                }
            }
        ),
        encoding="utf-8",
    )
    later = tmp_path / "round2b.json"
    later.write_text(
        json.dumps({"stages": {"7b-base-asm": {"status": "ok", "code": 0}}}),
        encoding="utf-8",
    )
    ok = load_ok_stages(prior, later)
    assert ok == {"dpo-train", "7b-base-asm"}
    assert "loop-dpo-heldout" not in ok


def test_remaining_after_round2_hours(tmp_path: Path):
    src = (Path(__file__).resolve().parents[1] / "results/round2/gpu_hours.jsonl")
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    used = gpu_hours_used(path)
    assert abs(used - 23.3488) < 0.01
    assert remaining_ok(need_h=10.0, hours_path=path, check_lock=False)
    assert remaining_ok(need_h=36.0, hours_path=path, check_lock=False)
    assert not remaining_ok(need_h=40.0, hours_path=path, check_lock=False)


def test_seed_carried_hours_copies_once(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"name": "dpo-train", "seconds": 3600, "ok": True}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "dest.jsonl"
    marker = tmp_path / "carried.json"
    hours = seed_carried_hours(src, dest, marker)
    assert abs(hours - 1.0) < 1e-6
    assert dest.is_file()
    dest.write_text(dest.read_text(encoding="utf-8") + json.dumps({"name": "7b-base-asm", "seconds": 100}) + "\n")
    hours2 = seed_carried_hours(src, dest, marker)
    assert abs(hours2 - 1.0) < 1e-6
    rows = [json.loads(l) for l in dest.read_text().splitlines() if l.strip()]
    assert len(rows) == 2


def test_valid_base7b_row_requires_samples():
    row = {
        "model": "base7b",
        "mode": "asm",
        "split": "heldout",
        "task_id": "mbpp3_is_not_prime",
        "greedy": {"ok": False, "category": "doesnt_assemble"},
        "samples": [{"ok": False, "category": "doesnt_assemble"}] * 5,
    }
    assert valid_base7b_row(row)
    bad = dict(row)
    bad["samples"] = []
    assert not valid_base7b_row(bad)
    missing = dict(row)
    del missing["task_id"]
    assert not valid_base7b_row(missing)


def test_parser_round2b_implies_flags():
    args = build_parser().parse_args(["--round2b", "--dry-run"])
    assert args.round2b and args.dry_run
    args = build_parser().parse_args(["--resume", "--only", "7b-base-asm", "qlora7b-v2-train"])
    assert args.resume
    assert args.only == ["7b-base-asm", "qlora7b-v2-train"]


def test_summary_md_compares_lora1_qlora7b(tmp_path: Path):
    results = {
        "title": "E2M round 2b",
        "gpu_hours_used": 25.0,
        "gpu_hours_carried": 23.35,
        "gpu_hours_this_round": 1.65,
        "stages": {"7b-base-asm": {"status": "ok", "code": 0, "reason": ""}},
        "greedy": {
            "lora1/asm/heldout": {
                "n": 78,
                "pass_at_1": 2 / 78,
                "pass_at_5": 5 / 78,
                "greedy_categories": {"pass": 2},
            },
            "qlora7b/asm/heldout": {
                "n": 78,
                "pass_at_1": 5 / 78,
                "pass_at_5": 8 / 78,
                "greedy_categories": {"pass": 5},
            },
            "dpo/asm/heldout": {
                "n": 78,
                "pass_at_1": 0.0,
                "pass_at_5": 0.0,
                "greedy_categories": {},
            },
        },
        "loops": {},
        "gpu_hours": [],
    }
    path = tmp_path / "SUMMARY.md"
    write_summary_md(results, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# E2M round 2b")
    assert "2/78" in text
    assert "5/78" in text
    assert "lora1" in text
    assert "qlora7b" in text
    assert "dpo" in text
    assert "1.65 this round, 23.35 carried" in text
    assert "Compared with lora1, lora2, qlora7b, and dpo." in text
