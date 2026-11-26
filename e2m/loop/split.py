"""Deterministic visible / hidden test split.

Visible tests may be shown to the model as repair feedback. Hidden tests are
only used for scoring. The rule is frozen: the first ``ceil(n / 2)`` tests
(list order) are visible, the rest are hidden. A task with one test has no
hidden split; hidden pass then equals full pass.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from e2m.harness.tasks import Task, TestCase, load_jsonl

SPLIT_RULE = "first ceil(n/2) tests visible, remainder hidden; n==1 => all visible"


def n_visible(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return (n + 1) // 2


def split_counts(n: int) -> tuple[int, int]:
    vis = n_visible(n)
    return vis, n - vis


def split_tests(tests: list[TestCase]) -> tuple[list[TestCase], list[TestCase]]:
    vis, _hid = split_counts(len(tests))
    return tests[:vis], tests[vis:]


def visible_tests(task: Task) -> list[TestCase]:
    vis, _ = split_tests(task.tests)
    return vis


def hidden_tests(task: Task) -> list[TestCase]:
    _, hid = split_tests(task.tests)
    return hid


def task_with_tests(task: Task, tests: list[TestCase]) -> Task:
    return replace(task, tests=list(tests))


def split_record(task: Task) -> dict[str, Any]:
    vis, hid = split_counts(len(task.tests))
    return {
        "n": len(task.tests),
        "n_visible": vis,
        "n_hidden": hid,
        "visible_ids": [c.id for c in task.tests[:vis]],
        "hidden_ids": [c.id for c in task.tests[vis:]],
    }


def heldout_split_payload(tasks: list[Task]) -> dict[str, Any]:
    return {
        "rule": SPLIT_RULE,
        "n_tasks": len(tasks),
        "tasks": {t.id: split_record(t) for t in tasks},
    }


def write_heldout_split(tasks_path: Path, out_path: Path) -> dict[str, Any]:
    tasks = load_jsonl(tasks_path)
    payload = heldout_split_payload(tasks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
