"""Compiler baseline: every starter task must pass via gcc -O0/-O2/-Os through the harness."""

from __future__ import annotations

import pytest

from e2m.harness.starter import STARTER_TASKS, write_starter_jsonl
from e2m.harness.tasks import load_starter, run_gcc_baseline, starter_jsonl_path

OPTS = ("-O0", "-O2", "-Os")


@pytest.fixture(scope="session", autouse=True)
def _write_jsonl():
    write_starter_jsonl(starter_jsonl_path())


def test_starter_count():
    tasks = load_starter()
    assert len(tasks) >= 20
    assert {t.id for t in tasks} == {t.id for t in STARTER_TASKS}


@pytest.mark.parametrize("task", STARTER_TASKS, ids=lambda t: t.id)
@pytest.mark.parametrize("opt", OPTS)
def test_gcc_baseline(task, opt):
    run = run_gcc_baseline(task, opt=opt)
    assert run.assemble_error is None, run.assemble_error
    assert run.passed, f"{task.id} {opt} failed: {run.fail_reason}\n" + "\n".join(
        f"  {c.case_id}: {c.reason} status={c.emu.status} ret={c.emu.return_value!r}"
        for c in run.cases
        if not c.passed
    )
    assert run.code_size > 0
    assert run.insn_count > 0
