import json
from pathlib import Path

from e2m.harness.score import pass_at_k, score_tasks
from e2m.harness.starter import STARTER_TASKS
from e2m.score import main as score_main


def test_pass_at_k_known_values():
    # From the Codex paper: if n=c, pass@k is 1; if c=0, 0.
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 0, 10) == 0.0
    v = pass_at_k(100, 1, 1)
    assert abs(v - 0.01) < 1e-12
    # n-c < k → 1
    assert pass_at_k(5, 3, 3) == 1.0


def test_score_perfect_add(tmp_path: Path):
    add = next(t for t in STARTER_TASKS if t.id == "add_i64")
    cand = tmp_path / "c.jsonl"
    cand.write_text(
        json.dumps({"task_id": "add_i64", "code": "add_i64:\n  movq %rdi, %rax\n  addq %rsi, %rax\n  ret\n"})
        + "\n",
        encoding="utf-8",
    )
    from e2m.harness.score import load_candidates

    report = score_tasks([add], load_candidates(cand), ks=(1, 10), baseline_opt="-O2")
    assert report["pass_at_k"]["1"] == 1.0
    assert report["tasks"][0]["c"] == 1
    assert report["tasks"][0]["baseline"]["ok"] is True


def test_cli_writes_files(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    # one tiny task
    tasks.write_text(
        json.dumps(
            {
                "id": "add_i64",
                "spec": "add",
                "signature": "int64_t add_i64(int64_t a, int64_t b)",
                "reference_c": "int64_t add_i64(int64_t a, int64_t b) { return a + b; }\n",
                "tests": [{"args": [1, 2], "ret": 3}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cands = tmp_path / "cands.jsonl"
    cands.write_text(
        json.dumps({"task_id": "add_i64", "hex": "4889f84801f0c3", "syntax": "hex"}) + "\n",
        encoding="utf-8",
    )
    outj = tmp_path / "out.json"
    outm = tmp_path / "out.md"
    rc = score_main(
        ["--tasks", str(tasks), "--candidates", str(cands), "--out-json", str(outj), "--out-md", str(outm)]
    )
    assert rc == 0
    data = json.loads(outj.read_text())
    assert data["pass_at_k"]["1"] == 1.0
    assert "pass@1" in outm.read_text()
