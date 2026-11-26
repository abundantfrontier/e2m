"""Every starter task's Rust reference must pass the harness."""

from __future__ import annotations

import pytest

from e2m.data.compile import find_tool
from e2m.data.sweep import _init, evaluate_job
from e2m.harness.rust_refs import RUST_REFS
from e2m.harness.starter import STARTER_TASKS

pytestmark = pytest.mark.skipif(find_tool("rustc") is None, reason="rustc not installed")

_CANON = {
    "compiler": "rustc",
    "flags": [
        "-C",
        "opt-level=2",
        "-C",
        "target-cpu=x86-64",
        "-C",
        "overflow-checks=off",
    ],
    "seed": None,
    "passes": None,
    "knobs": {
        "opt": "2",
        "cpu": "x86-64",
        "feat": "",
        "overflow": "off",
        "toggle": "",
        "seed": None,
    },
}


@pytest.fixture(scope="module", autouse=True)
def _bind_rustc():
    tasks = []
    for task in STARTER_TASKS:
        tasks.append(
            {
                "id": task.id,
                "spec": task.spec,
                "signature": task.signature,
                "reference_c": task.reference_c,
                "reference_rust": task.reference_rust,
                "tests": [
                    {
                        "args": c.args,
                        "ret": c.expected_return,
                        "bufs": c.expected_buffers,
                        "ret_cmp": c.ret_cmp,
                        "id": c.id,
                    }
                    for c in task.tests
                ],
            }
        )
    _init(tasks, {"rustc": find_tool("rustc")})


def test_rust_refs_cover_starters():
    assert set(RUST_REFS) == {t.id for t in STARTER_TASKS}
    for task in STARTER_TASKS:
        assert "extern \"C\" fn" in task.reference_rust
        assert task.reference_rust.strip().startswith("#[no_mangle]")


@pytest.mark.parametrize("task", STARTER_TASKS, ids=lambda t: t.id)
def test_rust_reference_passes(task):
    job = dict(_CANON)
    job["task_id"] = task.id
    job["config_id"] = f"rust-canon-{task.id}"
    result = evaluate_job(job)
    assert result["status"] == "pass", result.get("reason") + ": " + result.get("detail", "")
    assert result["size"] > 0
    assert result["insn_count"] > 0
    assert "call" not in result["asm"].split("#", 1)[0]
