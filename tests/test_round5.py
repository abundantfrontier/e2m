"""Round-5 gate, leakage, MBPP leaf conversion, and training-row shape. No GPU."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from e2m.loop.round5 import (
    GATE_OPEN,
    ROUND5_GPU_HOURS,
    gate_block_reason,
    gpu_hours_used,
    try_take_gpu_lock,
)
from e2m.loop.round5_data import (
    check_names,
    convert_mbpp_record,
    leakage_reason,
    screen_text,
    sft_row,
    verify_reference,
    write_mix,
    _indexes,
    _Ref,
)
from e2m.train.sft_lora import _target_text

PT = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parents[1]


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "hold": tmp_path / "round5.hold",
        "go": tmp_path / "round5.go",
        "lock": tmp_path / "gpu.lock",
    }


def test_gate_closed_before_1115_without_go_file(tmp_path: Path):
    paths = _paths(tmp_path)
    now = datetime(2026, 11, 25, 11, 14, tzinfo=PT)
    reason = gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"])
    assert reason is not None
    assert "round5.go" in reason


def test_gate_open_when_go_file_exists_before_the_clock(tmp_path: Path):
    paths = _paths(tmp_path)
    paths["go"].write_text("go\n", encoding="utf-8")
    now = datetime(2026, 11, 25, 11, 0, tzinfo=PT)
    assert gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"]) is None


def test_gate_open_at_1115_without_go_file(tmp_path: Path):
    paths = _paths(tmp_path)
    now = datetime(2026, 11, 25, 11, 15, tzinfo=PT)
    assert gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"]) is None
    assert GATE_OPEN == datetime(2026, 11, 25, 11, 15, tzinfo=PT)


def test_hold_blocks_even_after_the_gate_opens(tmp_path: Path):
    paths = _paths(tmp_path)
    paths["hold"].write_text("pause\n", encoding="utf-8")
    paths["go"].write_text("go\n", encoding="utf-8")
    now = datetime(2026, 11, 25, 12, 0, tzinfo=PT)
    reason = gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"])
    assert reason is not None
    assert "round5.hold" in reason


def test_live_lock_blocks_and_dead_pid_does_not(tmp_path: Path):
    paths = _paths(tmp_path)
    now = datetime(2026, 11, 25, 12, 0, tzinfo=PT)
    paths["lock"].write_text(f"other-job {os.getpid()} 2026-11-25T12:00:00-08:00 train\n", encoding="utf-8")
    reason = gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"])
    assert reason is not None
    assert "live" in reason
    paths["lock"].write_text("other-job 99999999 2026-11-25T12:00:00-08:00 stale\n", encoding="utf-8")
    assert gate_block_reason(now=now, hold_path=paths["hold"], go_path=paths["go"], lock_path=paths["lock"]) is None


def test_try_take_replaces_stale_lock_and_refuses_a_live_one(tmp_path: Path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("other-job 99999999 2020-01-01T00:00:00-08:00 stale\n", encoding="utf-8")
    ok, text = try_take_gpu_lock(os.getpid(), "round 5", path=lock)
    assert ok
    assert text.startswith(f"e2m {os.getpid()} ")
    assert text.endswith("round 5")
    lock.write_text(f"other-job {os.getpid()} 2026-11-25T12:00:00-08:00 busy\n", encoding="utf-8")
    ok, _detail = try_take_gpu_lock(os.getpid() + 1, "round 5", path=lock)
    assert ok is False
    assert "other-job" in lock.read_text(encoding="utf-8")


def test_budget_is_8h_and_cpu_time_does_not_count(tmp_path: Path):
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(
        json.dumps({"name": "train", "seconds": 2 * 3600, "gpu": True})
        + "\n"
        + json.dumps({"name": "lower-asm", "seconds": 40 * 3600, "gpu": False})
        + "\n",
        encoding="utf-8",
    )
    assert ROUND5_GPU_HOURS == 8.0
    assert abs(gpu_hours_used(path) - 2.0) < 1e-6


def test_source_has_the_gate_literals_and_does_not_open_humaneval():
    pipe = (ROOT / "e2m" / "loop" / "round5.py").read_text(encoding="utf-8")
    assert "round5.hold" in pipe
    assert "round5.go" in pipe
    assert "2026, 11, 25, 11, 15" in pipe
    data = (ROOT / "e2m" / "loop" / "round5_data.py").read_text(encoding="utf-8")
    comment = "Do not load datasets/heldout/HumanEval.jsonl.gz. The whole benchmark stays out."
    assert comment in data
    assert "HumanEval.jsonl" not in data.replace(comment, "")
    assert "HumanEval" not in pipe


def test_mbpp_leaf_converts_and_passes_the_harness():
    row = {
        "task_id": 900001,
        "text": "Add two integers and return the sum.",
        "code": "def add(a, b):\n    return a + b\n",
        "test_setup_code": "",
        "test_list": ["assert add(1, 2) == 3", "assert add(-4, 4) == 0"],
        "challenge_test_list": ["assert add(0, 0) == 0"],
    }
    task = convert_mbpp_record(row)
    assert task["signature"].startswith("int64_t mbpp900001_add(")
    assert verify_reference(task) is None
    scoped = {
        "task_id": 900002,
        "text": "Return the last decimal digit, or zero when the input is zero.",
        "code": (
            "def last_digit(n):\n"
            "    res = 0\n"
            "    while n > 0:\n"
            "        res = n % 10\n"
            "        n = n // 10\n"
            "        break\n"
            "    return res\n"
        ),
        "test_setup_code": "",
        "test_list": ["assert last_digit(5) == 5", "assert last_digit(10) == 0", "assert last_digit(0) == 0"],
        "challenge_test_list": [],
    }
    scoped_task = convert_mbpp_record(scoped)
    assert "int64_t res = 0;" in scoped_task["reference_c"]
    assert verify_reference(scoped_task) is None


def test_leakage_drops_id_near_copy_and_word5():
    protected = [
        {
            "id": "held_sum",
            "spec": "count the odd values in the array and return that count exactly once",
            "signature": "int64_t count_odd(const int64_t *a, int64_t n)",
            "reference_c": "int64_t count_odd(const int64_t *a, int64_t n) { return n; }\n",
            "source": "MBPP/9",
        }
    ]
    refs = [_Ref(row) for row in protected]
    indexes = _indexes(refs)
    indexes["sources"].add("MBPP/9")
    same_id = {
        "task_id": "held_sum",
        "spec": "something else entirely about widgets and gears in a warehouse",
        "signature": "int64_t other_name(int64_t x)",
        "reference_c": "int64_t other_name(int64_t x) { return x + 3; }\n",
        "origin": "mbpp",
    }
    assert leakage_reason(same_id, refs, indexes=indexes).startswith("id:")
    copied = {
        "task_id": "fresh_task",
        "spec": "count the odd values in the array and return that count exactly once",
        "signature": "int64_t fresh_task(int64_t x)",
        "reference_c": "int64_t fresh_task(int64_t x) { return x + 9; }\n",
        "origin": "synth",
    }
    reason = leakage_reason(copied, refs, indexes=indexes)
    assert reason is not None
    assert reason.split(":", 1)[0] in {"spec_ratio", "word5", "body", "c_ratio"}
    kept, dropped, _summary = screen_text([same_id, copied], protected=protected)
    assert kept == []
    assert len(dropped) == 2
    corpus = {
        "task_id": "exebench_other_deadbeef",
        "id": "exebench_other_deadbeef",
        "spec": "widget gear warehouse packing order for the night shift crew",
        "signature": "int64_t totally_unrelated(int64_t x)",
        "reference_c": "int64_t totally_unrelated(int64_t x) { return x + 4; }\n",
        "source": "exebench",
        "origin": "train",
    }
    protected_ex = protected + [
        {
            "id": "exebench_val_task",
            "spec": "a different specification about primes and triangles in the north field",
            "signature": "int64_t val_only(int64_t x)",
            "reference_c": "int64_t val_only(int64_t x) { return 1; }\n",
            "source": "exebench",
        }
    ]
    kept2, dropped2, _summary2 = screen_text([corpus], protected=protected_ex)
    assert dropped2 == []
    assert len(kept2) == 1


def test_training_rows_omit_c_so_the_target_stays_asm(tmp_path: Path):
    pair = {
        "task_id": "mbpp1_add",
        "spec": "Add two integers.",
        "signature": "int64_t mbpp1_add(int64_t a, int64_t b)",
        "goal": "default",
        "asm": ".text\n.globl mbpp1_add\nmbpp1_add:\n  ret\n",
        "model_c": "int64_t mbpp1_add(int64_t a, int64_t b) { return a + b; }\n",
        "origin": "mbpp",
        "sha256": "abc",
    }
    row = sft_row(pair)
    assert "c" not in row
    assert "model_c" not in row
    assert check_names(row) is None
    assert _target_text(row, kind="asm") == row["asm"]
    both = dict(row)
    both["c"] = pair["model_c"]
    assert check_names(both) == "both c and asm"
    assert _target_text(both, kind="asm").startswith("===C===")
    stats = write_mix(tmp_path / "sft.jsonl", kept_rows=[pair])
    saved = json.loads((tmp_path / "sft.jsonl").read_text(encoding="utf-8"))
    assert "c" not in saved
    assert saved["asm"].startswith(".text")
    assert stats["auxiliary_c_to_asm"] is False
