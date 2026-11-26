"""Pure helpers for the QEMU unsupported_insn overlay. No emulator."""

from e2m.data.qemu_recover import (
    INSN_COUNT_UNKNOWN,
    apply_insn_counts,
    job_from_cache_row,
    overlay_rows,
)


def test_job_from_unsupported_row_keeps_identity():
    row = {
        "config_id": "abc",
        "task_id": "exebench_foo",
        "compiler": "gcc",
        "flags": ["-O3", "-march=x86-64-v3", "-mavx2"],
        "seed": None,
        "passes": None,
        "knobs": {"opt": "-O3", "cpu": "x86-64-v3", "feat": "-mavx2", "toggle": ""},
        "status": "fail",
        "reason": "unsupported_insn",
        "detail": "vfmadd",
    }
    job = job_from_cache_row(row)
    assert job["config_id"] == "abc"
    assert job["flags"] == ["-O3", "-march=x86-64-v3", "-mavx2"]
    assert "status" not in job


def test_overlay_replaces_matching_config_and_does_not_append():
    cache = [
        {"config_id": "a", "status": "fail", "reason": "unsupported_insn"},
        {"config_id": "b", "status": "pass", "sha256": "keep"},
    ]
    recovered = {
        "a": {"config_id": "a", "status": "pass", "sha256": "new", "insn_count": 10},
        "missing": {"config_id": "missing", "status": "pass"},
    }
    out = overlay_rows(cache, recovered)
    assert len(out) == 2
    assert out[0]["status"] == "pass" and out[0]["sha256"] == "new"
    assert out[1]["sha256"] == "keep"


def test_unknown_insn_count_does_not_rank_as_fastest():
    rows = [
        {"status": "pass", "task_id": "t", "sha256": "s", "insn_count": 0},
        {"status": "pass", "task_id": "t", "sha256": "known", "insn_count": 0},
        {"status": "fail", "task_id": "t", "sha256": "x", "insn_count": 0},
    ]
    unknown = apply_insn_counts(rows, {("t", "known"): 40})
    assert rows[0]["insn_count"] == INSN_COUNT_UNKNOWN
    assert rows[1]["insn_count"] == 40
    assert rows[2]["insn_count"] == 0
    assert unknown == 1
    assert INSN_COUNT_UNKNOWN > 10**9
