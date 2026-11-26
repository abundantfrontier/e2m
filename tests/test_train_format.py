"""Prompt split and scoring helpers. No model and no GPU."""

from e2m.train.format import (
    categorize_run,
    extract_asm,
    extract_asm_from_c_then_asm,
    extract_c_from_c_then_asm,
    format_c_then_asm_response,
    pass_at_k,
    val_task_ids,
)


def test_val_split_is_stable_and_about_three_percent():
    ids = [f"task_{i}" for i in range(3000)]
    first = val_task_ids(ids, frac=0.03, seed=0)
    second = val_task_ids(ids, frac=0.03, seed=0)
    assert first == second
    assert len(first) == 90
    assert len(set(first)) == 90


def test_pass_at_k_matches_the_unbiased_estimator():
    assert pass_at_k(5, 0, 5) == 0.0
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(5, 5, 1) == 1.0
    # 1 correct out of 5, pass@1 = 1/5.
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9


def test_extract_asm_prefers_a_fence_and_the_text_directive():
    raw = "Here is the function:\n```asm\n.text\n.globl foo\nfoo:\n  ret\n```\n"
    assert extract_asm(raw).startswith(".text")
    assert "foo:" in extract_asm(raw)


def test_c_then_asm_extracts_both_halves():
    blob = format_c_then_asm_response(
        "int64_t foo(int64_t x) { return x; }",
        ".text\n.globl foo\nfoo:\n\tmovq %rdi, %rax\n\tret",
    )
    assert extract_c_from_c_then_asm(blob).startswith("int64_t foo")
    assert extract_asm_from_c_then_asm(blob).startswith(".text")
    fenced = "```c\nint foo() { return 1; }\n```\n```asm\n.text\nret\n```"
    assert "int foo" in extract_c_from_c_then_asm(fenced)
    assert ".text" in extract_asm_from_c_then_asm(fenced)


def test_failure_categories():
    assert categorize_run("as: boom", "ok", False) == "doesnt_assemble"
    assert categorize_run(None, "timeout", False) == "timeout"
    assert categorize_run(None, "insn_limit", False) == "timeout"
    assert categorize_run(None, "abi", False) == "abi"
    assert categorize_run(None, "ok", False) == "wrong_answer"
    assert categorize_run(None, "fault", False) == "crash"
    assert categorize_run(None, "ok", True) == "pass"
