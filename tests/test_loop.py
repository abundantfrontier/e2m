"""Repair loop, visible/hidden split, leakage, mix. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

from e2m.harness.starter import STARTER_TASKS
from e2m.harness.tasks import Task, TestCase as HarnessCase, load_jsonl
from e2m.loop.feedback import (
    append_repair,
    feedback_leaks_hidden,
    format_feedback,
    visible_failure,
)
from e2m.loop.leakage import HeldoutIndex, collision, cosine, jaccard
from e2m.loop.mix import mix_sft
from e2m.loop.repair import pass_at_budget, repair_task, score_text
from e2m.loop.rewrite import parse_rewrite
from e2m.loop.split import SPLIT_RULE, n_visible, split_counts, task_with_tests, visible_tests
from e2m.loop.summarize import summarize_loop

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "eval" / "heldout.jsonl"
HELDOUT_SPLIT = ROOT / "eval" / "heldout_visible.json"

ADD_OK = """.text
.globl add_i64
add_i64:
	movq %rdi, %rax
	addq %rsi, %rax
	ret
"""
ADD_ZERO = """.text
.globl add_i64
add_i64:
	xorq %rax, %rax
	ret
"""
ADD_C = "int64_t add_i64(int64_t a, int64_t b) { return a + b; }\n"


def _add() -> Task:
    return next(t for t in STARTER_TASKS if t.id == "add_i64")


def test_split_counts_match_the_frozen_rule():
    assert split_counts(1) == (1, 0)
    assert split_counts(2) == (1, 1)
    assert split_counts(5) == (3, 2)
    assert split_counts(6) == (3, 3)
    assert n_visible(0) == 0


def test_heldout_visible_file_matches_the_rule():
    tasks = load_jsonl(HELDOUT)
    payload = json.loads(HELDOUT_SPLIT.read_text(encoding="utf-8"))
    assert payload["rule"] == SPLIT_RULE
    assert payload["n_tasks"] == len(tasks)
    for task in tasks:
        rec = payload["tasks"][task.id]
        vis, hid = split_counts(len(task.tests))
        assert rec["n"] == len(task.tests)
        assert rec["n_visible"] == vis
        assert rec["n_hidden"] == hid
        assert rec["n_visible"] + rec["n_hidden"] == rec["n"]
        assert rec["n_visible"] >= 1


def test_feedback_omits_hidden_expected_values():
    hidden_ret = 987654321
    task = Task(
        id="toy",
        spec="add two integers",
        signature="int64_t add_i64(int64_t a, int64_t b)",
        reference_c=ADD_C,
        tests=[
            HarnessCase(args=[1, 2], expected_return=3, id="toy/0"),
            HarnessCase(args=[10, 20], expected_return=hidden_ret, id="toy/1"),
        ],
    )
    vis = visible_tests(task)
    assert len(vis) == 1
    run = score_text(task, ADD_ZERO, mode="asm")
    # Full run fails the hidden expected; feedback must still only see visible tests.
    vis_run = score_text(task_with_tests(task, vis), ADD_ZERO, mode="asm")
    info = visible_failure(vis_run, vis)
    text = format_feedback(info)
    assert str(hidden_ret) not in text
    assert "3" in text or "expected" in text
    assert not feedback_leaks_hidden(text, [task.tests[1]])
    # If someone passes the full run, visible_failure still skips the hidden case.
    leaked = format_feedback(visible_failure(run, vis))
    assert str(hidden_ret) not in leaked


def test_repair_loop_stops_at_first_full_pass_and_counts_calls():
    task = _add()
    texts = ["not assembly at all", ADD_ZERO, ADD_OK]

    def generate(messages):
        idx = sum(1 for m in messages if m["role"] == "assistant")
        # initial: 0 assistant turns -> texts[0]
        return texts[min(idx, len(texts) - 1)]

    result = repair_task(task, generate, k=8, rounds=3, mode="asm")
    assert result.calls == 3
    assert result.passed_full
    assert result.passed_visible
    assert result.first_full_call == 3
    assert result.attempts[0].category_visible == "doesnt_assemble"
    assert result.attempts[1].ok_visible is False
    assert result.attempts[2].ok_full is True
    assert result.best_pass is not None
    assert result.best_pass["size"] > 0


def test_repair_loop_does_not_feedback_hidden_failures():
    hidden_ret = 42424242
    task = Task(
        id="add_i64",
        spec="Return a+b.",
        signature="int64_t add_i64(int64_t a, int64_t b)",
        reference_c=ADD_C,
        tests=[
            HarnessCase(args=[1, 2], expected_return=3, id="v"),
            HarnessCase(args=[4, 5], expected_return=hidden_ret, id="h"),
        ],
    )
    seen: list[str] = []

    def generate(messages):
        blob = "\n".join(m["content"] for m in messages)
        seen.append(blob)
        if "failed" in blob.lower() or "error kind" in blob.lower():
            return ADD_OK
        return ADD_OK  # passes visible (1+2=3), fails hidden (9 != 42424242)

    result = repair_task(task, generate, k=2, rounds=3, mode="asm")
    assert result.passed_visible
    assert not result.passed_full
    assert result.calls == 2  # new sample, no hidden feedback repairs
    for blob in seen:
        assert str(hidden_ret) not in blob
        assert not feedback_leaks_hidden(blob, [task.tests[1]])


def test_batch_initial_counts_each_sequence():
    task = _add()
    texts = [ADD_ZERO, ADD_OK]

    def generate(messages, n=1):
        del messages
        return list(texts[:n])

    result = repair_task(task, generate, k=2, rounds=0, mode="asm", batch_initial=True)
    assert result.calls == 2
    assert result.passed_full
    assert result.first_full_call == 2


def test_c_route_compiles_and_passes():
    task = _add()
    result = repair_task(task, lambda _m: ADD_C, k=1, rounds=0, mode="c")
    assert result.passed_full
    assert result.first_full_call == 1


def test_pass_at_budget_and_summarize(tmp_path: Path):
    task = _add()

    def generate(_m):
        return ADD_OK

    a = repair_task(task, generate, k=1, rounds=0)
    b = repair_task(task, lambda _m: "nope", k=2, rounds=1)
    assert pass_at_budget([a, b], 1) == 0.5
    assert pass_at_budget([a, b], 4) == 0.5
    path = tmp_path / "loop.jsonl"
    rows = [
        {
            "model": "lora1",
            "split": "heldout",
            "task_id": "a",
            "passed_full": True,
            "passed_visible": True,
            "passed_hidden": True,
            "first_full_call": 1,
            "first_visible_call": 1,
            "first_hidden_call": 1,
            "calls": 1,
            "attempts": [{"category_full": "pass"}],
            "best_pass": {"size": 10, "insn": 20},
            "gcc": {"ok": True, "size": 10, "insn": 10},
        },
        {
            "model": "lora1",
            "split": "heldout",
            "task_id": "b",
            "passed_full": False,
            "passed_visible": True,
            "passed_hidden": False,
            "first_full_call": None,
            "first_visible_call": 4,
            "first_hidden_call": None,
            "calls": 8,
            "attempts": [{"category_full": "wrong_answer"}],
            "best_pass": None,
            "gcc": {"ok": True, "size": 10, "insn": 10},
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = summarize_loop(path, model="lora1", split="heldout")
    assert summary["n"] == 2
    assert summary["pass_full"] == 0.5
    assert summary["pass_visible"] == 1.0
    assert summary["budget_full"]["1"] == 0.5
    assert summary["budget_full"]["4"] == 0.5
    assert summary["budget_visible"]["4"] == 1.0
    assert summary["size_ratio_vs_gcc_o2"] == 1.0
    assert summary["insn_ratio_vs_gcc_o2"] == 2.0


def test_leakage_catches_name_and_ngram_overlap():
    index = HeldoutIndex(
        [
            {
                "id": "he13_gcd",
                "signature": "int64_t he_gcd(int64_t a, int64_t b)",
                "spec": "Return the greatest common divisor of a and b using Euclidean algorithm.",
            }
        ]
    )
    assert collision(task_id="x", signature="int64_t he_gcd(int64_t a, int64_t b)", text="sum", index=index)
    assert collision(
        task_id="other",
        signature="int64_t foo(int64_t a, int64_t b)",
        text="Return the greatest common divisor of a and b using Euclidean algorithm.",
        index=index,
    )
    assert (
        collision(
            task_id="other",
            signature="int64_t adc_voltage(int x)",
            text="Scale an ADC reading by a constant after subtracting a zero offset.",
            index=index,
        )
        is None
    )
    assert jaccard({"ab", "bc"}, {"bc", "cd"}) == 1 / 3
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_parse_rewrite_builds_prompt():
    parsed = parse_rewrite(
        {
            "docstring": "Return the sum of a and b.",
            "examples": [{"call": "add(1, 2)", "result": "3"}],
            "prompt": "",
        },
        spec="add them",
        signature="int64_t add(int64_t a, int64_t b)",
    )
    assert "sum of a and b" in parsed["spec"]
    assert "add(1, 2) -> 3" in parsed["spec"]
    assert parsed["spec_original"] == "add them"


def test_mix_sft_drops_val_and_heldout(tmp_path: Path):
    gcc = tmp_path / "sft.jsonl"
    gcc.write_text(
        json.dumps({"task_id": "keep", "goal": "default", "spec": "k", "signature": "int keep(void)", "asm": "ret"})
        + "\n"
        + json.dumps({"task_id": "val_task", "goal": "default", "spec": "v", "signature": "int v(void)", "asm": "ret"})
        + "\n"
        + json.dumps({"task_id": "held", "goal": "default", "spec": "h", "signature": "int h(void)", "asm": "ret"})
        + "\n"
        + json.dumps({"task_id": "keep", "goal": "size", "spec": "k", "signature": "int keep(void)", "asm": "ret"})
        + "\n",
        encoding="utf-8",
    )
    rewritten = tmp_path / "rw.jsonl"
    rewritten.write_text(
        json.dumps({"task_id": "keep", "spec": "rewritten keep", "signature": "int keep(void)"})
        + "\n"
        + json.dumps({"task_id": "held", "spec": "should drop", "signature": "int h(void)"})
        + "\n",
        encoding="utf-8",
    )
    reject = tmp_path / "rj.jsonl"
    reject.write_text(
        json.dumps(
            {
                "task_id": "keep",
                "spec": "k",
                "signature": "int keep(void)",
                "asm": "nop\nret",
                "source": "reject_loop",
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "val_task",
                "spec": "v",
                "signature": "int v(void)",
                "asm": "ret",
                "source": "reject_loop",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "mix.jsonl"
    stats = mix_sft(
        gcc_sft=gcc,
        rewritten=rewritten,
        reject=reject,
        out_path=out,
        val_ids={"val_task"},
        heldout_ids={"held"},
    )
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {r["task_id"] for r in rows}
    assert ids == {"keep"}
    assert stats["n_gcc_default"] == 1
    assert stats["n_rewritten_gcc"] == 1
    assert stats["n_reject"] == 1
    assert any(r.get("source") == "rewritten_gcc" for r in rows)
    assert any(r.get("source") == "reject_loop" for r in rows)


def test_append_repair_keeps_history():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = append_repair(messages, ADD_ZERO, {"kind": "wrong_answer", "detail": "return 0 != 3", "test": {"args": [1, 2], "expected_return": 3}})
    assert out[-2]["role"] == "assistant"
    assert out[-1]["role"] == "user"
    assert "wrong_answer" in out[-1]["content"]
    assert "1, 2" in out[-1]["content"] or "[1, 2]" in out[-1]["content"]
