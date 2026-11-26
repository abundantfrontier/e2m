"""DPO pair split and loss. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from e2m.loop.guard import DEADLINE, PT, past_deadline, release_gpu_lock, write_gpu_lock
from e2m.train.dpo import blocked_task_ids, dpo_loss, load_pair_rows, pair_goal, pair_key, split_pairs
from datetime import datetime


def _row(tid: str, chosen: str, rejected: str, n: int = 0) -> dict:
    return {
        "task_id": tid,
        "spec": "add",
        "signature": "int64_t add(int64_t a, int64_t b)",
        "goal": "size",
        "chosen": {"asm": chosen, "sha256": f"c{tid}{n}"},
        "rejected": {"asm": rejected, "sha256": f"r{tid}{n}"},
    }


def test_split_pairs_is_stable_and_holds_out_eval_n(tmp_path: Path):
    rows = [_row(f"t{i}", "ok", "bad", n=i) for i in range(20)]
    first_train, first_eval = split_pairs(rows, blocked_ids=set(), eval_n=4, seed=0)
    second_train, second_eval = split_pairs(rows, blocked_ids=set(), eval_n=4, seed=0)
    assert [pair_key(r) for r in first_eval] == [pair_key(r) for r in second_eval]
    assert len(first_eval) == 4
    assert len(first_train) == 16
    eval_keys = {pair_key(r) for r in first_eval}
    assert not eval_keys.intersection({pair_key(r) for r in first_train})


def test_split_pairs_drops_blocked_task_ids():
    rows = [_row("keep", "ok", "bad"), _row("drop", "ok", "bad")]
    train, eval_rows = split_pairs(rows, blocked_ids={"drop"}, eval_n=0, seed=0)
    assert [r["task_id"] for r in train] == ["keep"]
    assert eval_rows == []


def test_load_pair_rows_skips_identical_asm(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        _row("a", "good", "bad"),
        _row("b", "same", "same"),
        {"task_id": "c", "chosen": {"asm": ""}, "rejected": {"asm": "x"}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = load_pair_rows(path)
    assert [r["task_id"] for r in loaded] == ["a"]


def test_blocked_task_ids_reads_val_and_heldout(tmp_path: Path):
    val = tmp_path / "val.json"
    held = tmp_path / "held.jsonl"
    val.write_text(json.dumps({"task_ids": ["v1", "v2"]}), encoding="utf-8")
    held.write_text(json.dumps({"id": "h1"}) + "\n", encoding="utf-8")
    blocked = blocked_task_ids(val, held)
    assert blocked == {"v1", "v2", "h1"}


def test_dpo_loss_is_lower_when_policy_ranks_chosen_higher():
    chosen = torch.tensor([1.5, 2.0])
    rejected = torch.tensor([0.0, 0.2])
    ref_c = torch.tensor([0.1, 0.1])
    ref_r = torch.tensor([0.1, 0.1])
    good, good_m = dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1)
    bad, _ = dpo_loss(rejected, chosen, ref_c, ref_r, beta=0.1)
    assert good.item() < bad.item()
    assert good_m["accuracy"] == 1.0
    assert good_m["reward_margin"] > 0


def test_dpo_rpo_term_adds_chosen_nll():
    chosen = torch.tensor([-2.0, -2.0])
    rejected = torch.tensor([-2.5, -2.5])
    ref = torch.tensor([-2.0, -2.0])
    base, base_m = dpo_loss(chosen, rejected, ref, ref, beta=0.2, rpo_weight=0.0)
    rpo, rpo_m = dpo_loss(chosen, rejected, ref, ref, beta=0.2, rpo_weight=1.0)
    assert rpo.item() > base.item()
    assert abs(rpo_m["nll_chosen"] - 2.0) < 1e-5
    assert pair_goal({"goal": "speed"}) == "speed"


def test_deadline_is_november_18_2026():
    assert DEADLINE.year == 2026 and DEADLINE.month == 11 and DEADLINE.day == 18
    assert not past_deadline(datetime(2026, 11, 18, 17, 59, tzinfo=PT))
    assert past_deadline(datetime(2026, 11, 18, 18, 0, tzinfo=PT))


def test_gpu_lock_write_and_release_roundtrip(tmp_path: Path):
    lock = tmp_path / "gpu.lock"
    write_gpu_lock(1234, "round 2 pipeline", path=lock)
    text = lock.read_text(encoding="utf-8")
    assert text.split()[0] == "e2m"
    assert "1234" in text
    assert not release_gpu_lock(owner_pid=9999, path=lock)
    assert lock.is_file()
    assert release_gpu_lock(owner_pid=1234, path=lock)
    assert not lock.is_file()
