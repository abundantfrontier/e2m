"""Round-5b gate, budget, synth retry, and distill mix. No GPU and no ~/gpu.lock."""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime
from pathlib import Path

from e2m.loop.guard import LOCK_PATH, PT
from e2m.loop.round5_data import distill_pairs, write_mix
from e2m.loop.round5_sample import sample_plan, synth_fail_update
from e2m.loop.round5b import (
    EVAL_RATE_S,
    Pipeline,
    allocate,
    compute_allocation,
    configure_paths,
    eval_cap,
    gate_decision,
    gate_wait,
)

ROOT = Path(__file__).resolve().parents[1]
OPEN = datetime.fromisoformat("2026-11-25T18:45:00-08:00")
BEFORE = datetime(2026, 11, 25, 16, 30, tzinfo=PT)


def _decision(**kwargs):
    base = dict(
        now=BEFORE,
        hold_exists=False,
        lock_text=None,
        self_pid=4242,
        saw_live_non_e2m=False,
        pid_alive_fn=lambda _pid: False,
    )
    base.update(kwargs)
    return gate_decision(**base)


def test_courtesy_waits_when_the_lock_is_absent_before_1845():
    view = _decision()
    assert view.action == "wait"
    assert view.saw_live_non_e2m is False
    assert view.reason is not None
    assert "2026-11-25T18:45:00-08:00" in view.reason


def test_live_non_e2m_lock_is_observed_and_not_taken():
    view = _decision(
        lock_text="other-job 3568456 2026-11-25 16:58:33 PST Wren scoring",
        pid_alive_fn=lambda pid: pid == 3568456,
    )
    assert view.action == "wait"
    assert view.saw_live_non_e2m is True
    assert "3568456" in (view.reason or "")


def test_stale_lock_is_not_a_live_non_e2m_observation():
    view = _decision(
        lock_text="other-job 99999999 2026-11-25 16:58:33 PST Wren scoring",
        pid_alive_fn=lambda _pid: False,
    )
    assert view.action == "wait"
    assert view.saw_live_non_e2m is False
    assert "2026-11-25T18:45:00-08:00" in (view.reason or "")


def test_seen_non_e2m_lock_then_cleared_may_take_before_1845():
    view = _decision(saw_live_non_e2m=True, lock_text=None)
    assert view.action == "take"
    assert view.reason is None


def test_clock_opens_the_gate_when_the_lock_is_absent():
    view = _decision(now=OPEN, lock_text="")
    assert view.action == "take"


def test_hold_blocks_after_the_clock_opens():
    view = _decision(now=OPEN, hold_exists=True, saw_live_non_e2m=True)
    assert view.action == "wait"
    assert view.reason == "round5b.hold exists"


def test_our_own_lock_waits_out_the_courtesy_then_takes():
    text = "e2m 4242 2026-11-25T16:00:00-08:00 round 5b"
    waiting = _decision(lock_text=text, pid_alive_fn=lambda pid: pid == 4242)
    assert waiting.action == "wait"
    assert waiting.saw_live_non_e2m is False
    taking = _decision(
        lock_text=text,
        pid_alive_fn=lambda pid: pid == 4242,
        saw_live_non_e2m=True,
    )
    assert taking.action == "take"


def test_gate_wait_source_has_the_audit_literals():
    text = (ROOT / "e2m" / "loop" / "round5b.py").read_text(encoding="utf-8")
    start = text.index("def gate_wait(")
    end = text.index("\ndef ", start + 1)
    body = text[start:end]
    assert "round5b.hold" in body
    assert "round 5b" in body
    assert "2026-11-25T18:45:00-08:00" in body


def test_gate_wait_does_not_touch_the_real_lock_or_steal_a_live_one(tmp_path: Path):
    real_before = LOCK_PATH.read_text(encoding="utf-8") if LOCK_PATH.is_file() else None
    real_state = ROOT / "runs" / "round5b" / "gate_state.json"
    real_state_before = real_state.read_text(encoding="utf-8") if real_state.is_file() else None
    os.environ.pop("E2M_SMOKE", None)
    lock = tmp_path / "gpu.lock"
    hold = tmp_path / "round5b.hold"
    state = tmp_path / "gate_state.json"
    before = BEFORE

    def now() -> datetime:
        return before

    try:
        assert gate_wait(
            4242,
            sleep_s=0,
            now_fn=now,
            lock_path=lock,
            hold_path=hold,
            state_path=state,
            max_loops=1,
        ) is False
        assert not lock.exists()

        lock.write_text("other-job %s 2026-11-25 16:58:33 PST Wren scoring\n" % os.getpid(), encoding="utf-8")
        original = lock.read_text(encoding="utf-8")
        assert gate_wait(
            4242,
            sleep_s=0,
            now_fn=now,
            lock_path=lock,
            hold_path=hold,
            state_path=state,
            max_loops=1,
        ) is False
        assert lock.read_text(encoding="utf-8") == original
        saved = json.loads(state.read_text(encoding="utf-8"))
        assert saved["saw_live_non_e2m"] is True
        assert str(saved["observed_lock"]).startswith("other-job ")

        lock.unlink()
        assert gate_wait(
            4242,
            sleep_s=0,
            now_fn=now,
            lock_path=lock,
            hold_path=hold,
            state_path=state,
            max_loops=2,
        ) is True
        taken = lock.read_text(encoding="utf-8")
        assert taken.startswith("e2m 4242 ")
        assert "round 5b" in taken

        hold.write_text("stop\n", encoding="utf-8")
        lock.unlink()
        assert gate_wait(
            4242,
            sleep_s=0,
            now_fn=lambda: OPEN,
            lock_path=lock,
            hold_path=hold,
            state_path=state,
            max_loops=1,
        ) is False
        assert not lock.exists()
    finally:
        real_after = LOCK_PATH.read_text(encoding="utf-8") if LOCK_PATH.is_file() else None
        assert real_after == real_before
        real_state_after = real_state.read_text(encoding="utf-8") if real_state.is_file() else None
        assert real_state_after == real_state_before


def test_courtesy_done_without_an_observed_lock_does_not_open_early(tmp_path: Path):
    os.environ.pop("E2M_SMOKE", None)
    state = tmp_path / "gate_state.json"
    state.write_text(
        json.dumps({"saw_live_non_e2m": True, "courtesy_done": True}) + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "gpu.lock"
    opened = gate_wait(
        4242,
        sleep_s=0,
        now_fn=lambda: BEFORE,
        lock_path=lock,
        hold_path=tmp_path / "round5b.hold",
        state_path=state,
        max_loops=1,
    )
    assert opened is False
    assert not lock.exists()


def test_one_ollama_500_retries_and_does_not_count_as_a_harness_reject():
    action, harness, ollama = synth_fail_update(
        RuntimeError("ollama: HTTP Error 500: Internal Server Error"),
        harness_streak=4,
        ollama_streak=0,
        ollama_giveup=5,
    )
    assert action == "retry"
    assert harness == 4
    assert ollama == 1


def test_harness_rejects_reset_the_ollama_streak():
    action, harness, ollama = synth_fail_update(
        RuntimeError("TypeError: int() argument must be a string"),
        harness_streak=3,
        ollama_streak=4,
        ollama_giveup=5,
    )
    assert (action, harness, ollama) == ("reject", 4, 0)
    action, harness, ollama = synth_fail_update(
        RuntimeError("ollama: HTTP Error 500: Internal Server Error"),
        harness_streak=harness,
        ollama_streak=ollama,
        ollama_giveup=5,
    )
    assert action == "retry"
    assert ollama == 1


def test_five_consecutive_ollama_errors_abort():
    harness = 2
    ollama = 0
    action = ""
    for _ in range(4):
        action, harness, ollama = synth_fail_update(
            RuntimeError("ollama: HTTP Error 500: Internal Server Error"),
            harness_streak=harness,
            ollama_streak=ollama,
            ollama_giveup=5,
        )
        assert action == "retry"
        assert harness == 2
    action, harness, ollama = synth_fail_update(
        RuntimeError("ollama: HTTP Error 500: Internal Server Error"),
        harness_streak=harness,
        ollama_streak=ollama,
        ollama_giveup=5,
    )
    assert action == "abort"
    assert harness == 2
    assert ollama == 5


def test_sft_cap_is_a_quarter_and_distill_stays_the_majority(tmp_path: Path):
    rows = []
    for index in range(3000):
        rows.append(
            {
                "task_id": f"d{index}",
                "spec": "add",
                "signature": "int64_t f(int64_t a)",
                "goal": "default",
                "asm": "ret\n",
                "origin": "train",
            }
        )
    for index in range(5000):
        rows.append(
            {
                "task_id": f"s{index % 80}",
                "spec": "old",
                "signature": "int64_t g(int64_t a)",
                "goal": "default",
                "asm": "ret\n",
                "origin": "sft_lora2",
                "sha256": str(index),
            }
        )
    stats = write_mix(tmp_path / "sft.jsonl", kept_rows=rows, sft_fraction=0.25)
    assert stats["by_origin"]["train"] == 3000
    assert stats["by_origin"]["sft_lora2"] == 1000
    assert stats["proportions"]["sft_lora2"] == 0.25
    assert stats["distill_majority"] is True

    small = []
    for index in range(5):
        small.append(
            {
                "task_id": f"d{index}",
                "spec": "add",
                "signature": "int64_t f(int64_t a)",
                "goal": "default",
                "asm": "ret\n",
                "origin": "mbpp",
            }
        )
    for index in range(40):
        small.append(
            {
                "task_id": f"s{index}",
                "spec": "old",
                "signature": "int64_t g(int64_t a)",
                "goal": "default",
                "asm": "ret\n",
                "origin": "sft_lora2",
            }
        )
    floored = write_mix(tmp_path / "small.jsonl", kept_rows=small, sft_fraction=0.25)
    assert floored["by_origin"]["sft_lora2"] == 1
    assert floored["proportions"]["sft_lora2"] < 0.25
    assert floored["distill_majority"] is True
    assert inspect.signature(distill_pairs).parameters["max_distinct"].default == 2


def test_sample_plan_tops_up_round5_rows_and_keeps_explicit_zero():
    tasks = [{"id": "mbpp1", "spec": "s", "signature": "int64_t mbpp1(int64_t a)"}]
    pending = sample_plan(
        tasks,
        {"mbpp1": {"done": True, "c_texts": ["int64_t mbpp1(int64_t a) { return a; }\n"]}},
        ensure_k=8,
        prior_k=4,
        default_k=4,
    )
    assert pending[0]["_have_k"] == 4
    assert pending[0]["_n_new"] == 4
    assert pending[0]["_greedy_done"] is True
    assert len(pending[0]["_c_texts"]) == 1

    failed = sample_plan(
        [{"id": "synth_1"}],
        {"synth_1": {"done": True, "k": 0, "greedy_done": False, "c_texts": []}},
        ensure_k=8,
        prior_k=4,
        default_k=4,
    )
    assert failed[0]["_have_k"] == 0
    assert failed[0]["_n_new"] == 8
    assert failed[0]["_greedy_done"] is False

    fresh = sample_plan([{"id": "new"}], {}, ensure_k=8, prior_k=4, default_k=4)
    assert fresh[0]["_have_k"] == 0
    assert fresh[0]["_n_new"] == 8
    assert fresh[0]["_greedy_done"] is False

    done = sample_plan(
        [{"id": "t"}],
        {"t": {"k": 8, "greedy_done": True, "c_texts": ["x"]}},
        ensure_k=8,
        prior_k=4,
        default_k=4,
    )
    assert done == []


def test_allocation_at_8h_with_52_heldout_done_is_not_the_0_7h_cap():
    plan = compute_allocation(
        remain_h=8.0,
        counts={
            "a-heldout": 52,
            "a-val": 0,
            "b-heldout": 0,
            "b-val": 0,
            "a-loop": 0,
            "b-loop": 0,
            "a-loop-asm": 0,
            "b-loop-val": 0,
            "a-loop-val": 0,
        },
        hours={},
        settled=set(),
        train_done=False,
        sample_topped=False,
        synth_done=False,
    )
    alloc = plan["allocation"]
    assert abs(alloc["a-eval-heldout"] - 0.5455) < 0.02
    assert abs(alloc["a-eval-val"] - 1.5959) < 0.02
    assert abs(alloc["b-train"] - 2.0) < 0.02
    assert abs(alloc["b-eval-heldout"] - 1.3831) < 0.02
    assert abs(alloc["b-synth"] - 0.35) < 0.02
    assert abs(alloc["b-sample"] - 1.50) < 0.02
    assert 0.4 < alloc["a-loop-c-then-asm"] < 1.0
    assert alloc["b-eval-val"] == 0.0
    assert alloc["a-eval-heldout"] != 0.7
    assert plan["rates"]["not_the_round5_eval_cap_h"] == 0.7
    assert abs(eval_cap(26) - (26 * EVAL_RATE_S / 3600.0 * 1.55)) < 1e-9
    direct = allocate(8.0, plan["wants"])
    assert direct["a-eval-heldout"] == alloc["a-eval-heldout"]


def test_resume_keeps_finished_stages_when_a_later_one_is_recorded(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("E2M_SMOKE", raising=False)
    import e2m.loop.round5b as mod

    old = (mod.PIPE_DIR, mod.HOURS_PATH, mod.STAGE_PATH, mod.STATE_PATH, mod.DONE_PATH)
    try:
        configure_paths(smoke=True, out=tmp_path)
        (tmp_path / "state.json").write_text(
            json.dumps(
                {
                    "stages": {
                        "a-eval-heldout": {"status": "ok", "code": 0, "reason": ""},
                        "a-eval-val": {"status": "ok", "code": 0, "reason": ""},
                    }
                }
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(smoke=True, resume=True, dry_run=False, only=None, env={})
        pipe.record("a-eval-heldout", "skipped", reason="already recorded ok")
        pipe.record("b-synth", "ok", code=0, reason="already at cap")
        saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert saved["stages"]["a-eval-heldout"]["status"] == "ok"
        assert saved["stages"]["a-eval-val"]["status"] == "ok"
        assert saved["stages"]["b-synth"]["status"] == "ok"
        assert "a-eval-heldout" in pipe.frozen_ok
    finally:
        mod.PIPE_DIR, mod.HOURS_PATH, mod.STAGE_PATH, mod.STATE_PATH, mod.DONE_PATH = old
