"""Extractor, dedupe, and a tiny resumable sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from e2m.data.aggregate import pair_records, sft_records, unique_variants, write_outputs
from e2m.data.compile import find_tool
from e2m.data.extract import ExtractError, extract_function_asm, unsupported_instruction
from e2m.data.grid import build_jobs, config_id
from e2m.data.sweep import _init, _llc_opt_flag, evaluate_job, run_sweep
from e2m.harness.assemble import assemble
from e2m.harness.starter import STARTER_TASKS
from e2m.harness.tasks import compile_reference_asm, run_task

CALL_ASM = """
	.globl	add_i64
	.type	add_i64, @function
add_i64:
	call	puts
	ret
	.size	add_i64, .-add_i64
"""

RODATA_ASM = """
	.section	.rodata
	.align	8
.LC0:
	.quad	42
	.text
	.globl	add_i64
	.type	add_i64, @function
add_i64:
	movq	.LC0(%rip), %rax
	ret
	.size	add_i64, .-add_i64
"""


def test_extract_rejects_call():
    with pytest.raises(ExtractError, match="call"):
        extract_function_asm(CALL_ASM, "add_i64")


def test_extract_keeps_rodata_and_assembles():
    asm = extract_function_asm(RODATA_ASM, "add_i64")
    assert ".LC0:" in asm
    image = assemble(asm, syntax="att")
    assert image.rodata
    assert image.entry("add_i64") >= image.text_addr


def test_extract_label_with_comment():
    asm = """
	.globl	add_i64                         # -- Begin function add_i64
	.prefalign	4, .Lfunc_end0, nop
	.type	add_i64,@function
add_i64:                                # @add_i64
	leaq	(%rdi,%rsi), %rax
	retq
.Lfunc_end0:
	.size	add_i64, .Lfunc_end0-add_i64
"""
    out = extract_function_asm(asm, "add_i64")
    image = assemble(out, syntax="att")
    assert image.entry("add_i64") == image.text_addr
    assert image.code_size > 0


def test_gcc_extract_matches_harness():
    task = next(t for t in STARTER_TASKS if t.id == "add_i64")
    raw = compile_reference_asm(task.reference_c, opt="-O2")
    asm = extract_function_asm(raw, "add_i64")
    image = assemble(asm, syntax="att")
    run = run_task(task, image)
    assert run.passed, run.fail_reason
    assert unsupported_instruction(image.text, image.text_addr) is None
    assert image.entry("add_i64") == image.text_addr


def test_ymm_detected():
    # vpxor ymm0, ymm0, ymm0
    blob = bytes.fromhex("c5fdefc0")
    reason = unsupported_instruction(blob, 0x400000)
    assert reason and "ymm" in reason


def test_vex_vvvv_must_alias_dest():
    # vpaddq xmm2, xmm1, xmm0 — VEX.vvvv is xmm1, dest is xmm2. Unicorn
    # ignores vvvv and returns the rm operand.
    bad = bytes.fromhex("c5f1d4d0c3")
    reason = unsupported_instruction(bad, 0)
    assert reason and "vex.vvvv" in reason
    # vpaddq xmm1, xmm1, xmm0 — destructive form, which Unicorn gets right.
    good = bytes.fromhex("c5f1d4c8c3")
    assert unsupported_instruction(good, 0) is None
    # andn rdx, rcx, rax is VEX too, but Unicorn's BMI2 path honors vvvv.
    assert unsupported_instruction(bytes.fromhex("c4e2f0f2d0c3"), 0) is None
    # vpbroadcastd xmm0, xmm0 is UC_ERR_INSN_INVALID on Unicorn 2.1.4.
    reason = unsupported_instruction(bytes.fromhex("c4e27958c0c3"), 0)
    assert reason and "vpbroadcast" in reason


def test_grid_is_cartesian_and_stable():
    jobs = build_jobs(["add_i64"], ["gcc"], llvm_seeds=2)
    assert len(jobs) == 6 * 4 * 5 * 8
    assert len({j["config_id"] for j in jobs}) == len(jobs)
    again = build_jobs(["add_i64"], ["gcc"], llvm_seeds=2)
    assert [j["config_id"] for j in jobs] == [j["config_id"] for j in again]
    assert config_id(jobs[0]) == jobs[0]["config_id"]


def test_reduced_grid_is_smaller():
    full = build_jobs(["add_i64"], ["gcc", "clang"], llvm_seeds=0)
    red = build_jobs(["add_i64"], ["gcc", "clang"], llvm_seeds=0, grid="reduced")
    assert len(red) == 2 * 4 * 2 * 3 * 3  # compilers × opts × cpus × feats × toggles
    assert len(red) < len(full)
    assert len({j["config_id"] for j in red}) == len(red)


def test_llc_opt_flag_ignores_pipeline_name():
    jobs = build_jobs(["add_i64"], ["llvm-opt"], llvm_seeds=2)
    defaults = [j for j in jobs if str(j["knobs"]["opt"]).startswith("default")]
    assert len(defaults) == 6
    # Pipeline names are not llc flags. Os/Oz become -O2 plus size attributes.
    assert [_llc_opt_flag(j, j["knobs"]) for j in defaults] == [
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-O2",
        "-O2",
    ]
    assert defaults[4]["knobs"]["ir_attrs"] == "optsize"
    assert defaults[4]["passes"] == "default<O2>"
    assert defaults[5]["knobs"]["ir_attrs"] == "minsize optsize"
    randoms = [j for j in jobs if j["knobs"]["toggle"] == "random-pipeline"]
    assert [j["knobs"]["opt"] for j in randoms] == ["-O0", "-O1"]
    assert all(_llc_opt_flag(j, j["knobs"]) == j["knobs"]["opt"] for j in randoms)


def test_dedupe_rank_and_pairs():
    tasks = {"add_i64": {"spec": "add", "signature": "int64_t add_i64(int64_t a, int64_t b)"}}
    rows = [
        _row("a" * 64, 4, 10, "gcc", "-O2"),
        _row("a" * 64, 4, 10, "clang", "-O2"),
        _row("b" * 64, 20, 40, "gcc", "-O0"),
        _row("c" * 64, 8, 8, "rustc", "s"),
        _row("nope", 1, 1, "gcc", "-O3", status="fail", reason="unsupported_insn"),
    ]
    variants = unique_variants(rows, tasks)
    assert len(variants) == 3
    fastest = min(variants, key=lambda v: v["insn_count"])
    assert fastest["sha256"] == "c" * 64
    assert fastest["rank_speed"] == 1
    smallest = min(variants, key=lambda v: v["size"])
    assert smallest["sha256"] == "a" * 64
    shared = next(v for v in variants if v["sha256"] == "a" * 64)
    assert set(shared["compilers"]) == {"gcc", "clang"}
    sft = sft_records(variants, rows)
    goals = {row["goal"]: row["sha256"] for row in sft}
    assert goals["size"] == "a" * 64
    assert goals["speed"] == "c" * 64
    pairs = pair_records(variants, sft)
    assert pairs
    assert all(p["chosen"]["sha256"] != p["rejected"]["sha256"] for p in pairs)


def _row(sha, size, insn, compiler, opt, status="pass", reason="pass"):
    return {
        "config_id": f"{compiler}-{opt}-{sha[:6]}",
        "task_id": "add_i64",
        "compiler": compiler,
        "flags": [opt, "-march=x86-64"] if compiler != "rustc" else ["-C", f"opt-level={opt}"],
        "seed": None,
        "passes": None,
        "knobs": {
            "opt": opt if compiler != "rustc" else opt,
            "cpu": "x86-64",
            "feat": "",
            "toggle": "",
            "overflow": "off",
            "seed": None,
        },
        "status": status,
        "reason": reason,
        "detail": "",
        "sha256": sha if status == "pass" else "",
        "size": size,
        "insn_count": insn,
        "insn_per_test": [insn],
        "asm": f"{compiler} {opt}\n",
        "bytes_hex": "90" * size,
    }


def test_aggregate_files(tmp_path: Path):
    tasks = {"add_i64": {"id": "add_i64", "spec": "add", "signature": "int64_t add_i64(int64_t a, int64_t b)"}}
    rows = [_row("d" * 64, 5, 9, "gcc", "-O2"), _row("e" * 64, 9, 12, "gcc", "-Os")]
    # Make the -O2 row the canonical default.
    rows[0]["knobs"]["opt"] = "-O2"
    stats = write_outputs(
        out_dir=tmp_path,
        rows=rows,
        tasks_by_id=tasks,
        tools={"gcc": "gcc test"},
        compilers=["gcc"],
        grid_note="test grid\n",
    )
    assert stats["variants"] == 2
    assert (tmp_path / "SWEEP_REPORT.md").is_file()
    sft = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert {row["goal"] for row in sft} == {"size", "speed", "default"}


@pytest.mark.skipif(find_tool("gcc") is None, reason="gcc missing")
def test_mini_sweep_resumes(tmp_path: Path):
    tasks = Path(__file__).resolve().parents[1] / "tasks" / "starter.jsonl"
    if not tasks.is_file():
        pytest.skip("starter jsonl not written yet")
    out = tmp_path / "data"
    # Two gcc configs of the first task only: build via a tiny custom run.
    from e2m.data.grid import _base_job

    task_dicts = []
    # Use evaluate path through run_sweep's own loader by limiting compilers
    # and replacing the grid: call run_sweep with limit after monkeypatch.
    jobs = [
        _base_job(
            "add_i64",
            "gcc",
            ["-O0", "-march=x86-64"],
            {"opt": "-O0", "cpu": "x86-64", "feat": "", "toggle": "", "seed": None},
        ),
        _base_job(
            "add_i64",
            "gcc",
            ["-O2", "-march=x86-64"],
            {"opt": "-O2", "cpu": "x86-64", "feat": "", "toggle": "", "seed": None},
        ),
    ]
    from e2m.data import sweep as sweep_mod

    original = sweep_mod.build_jobs
    sweep_mod.build_jobs = lambda task_ids, compilers, llvm_seeds=0, **_kw: [
        job for job in jobs if job["task_id"] in task_ids and job["compiler"] in compilers
    ]
    try:
        stats = run_sweep(
            tasks_path=tasks,
            out_dir=out,
            jobs=2,
            compilers=["gcc"],
            llvm_seeds=0,
            resume=True,
        )
        assert stats["passed"] >= 2 or stats["variants"] >= 1
        cache_lines = (out / "sweep-cache.jsonl").read_text().strip().splitlines()
        assert len(cache_lines) == 2
        again = run_sweep(
            tasks_path=tasks,
            out_dir=out,
            jobs=2,
            compilers=["gcc"],
            llvm_seeds=0,
            resume=True,
        )
        assert again["pending"] == 0
        cache_lines_2 = (out / "sweep-cache.jsonl").read_text().strip().splitlines()
        assert len(cache_lines_2) == 2
        variants = [json.loads(line) for line in (out / "variants.jsonl").read_text().splitlines()]
        assert variants
        assert all(v["task_id"] == "add_i64" for v in variants)
    finally:
        sweep_mod.build_jobs = original


def test_bind_used_by_eval_helpers():
    # Import side: evaluate_job requires _init. Smoke the symbol so refactors notice.
    assert callable(_init)
