"""Resumable GPU pipeline: rewrite -> lora1 loops -> reject -> lora2 train/eval.

Each stage is a subprocess so VRAM is released between Ollama and training.
Skip a stage when its exit file is 0 and the expected artifacts exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from e2m.loop.guard import PT, check_gpu_lock, install_run_files, past_deadline

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL_15 = ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_LORA1 = ROOT / "runs" / "lora1" / "adapter"
ADAPTER_LORA2 = ROOT / "runs" / "lora2" / "adapter"
HOURS_PATH = ROOT / "runs" / "loop-pipeline" / "gpu_hours.jsonl"
STAGE_PATH = ROOT / "runs" / "loop-pipeline" / "stage.txt"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip().split()[0])
    except ValueError:
        return None


def _exit_ok(rundir: Path) -> bool:
    p = rundir / "exit"
    if not p.is_file():
        return False
    return p.read_text(encoding="utf-8").strip() == "0"


def _n_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _log_hours(name: str, seconds: float, ok: bool) -> None:
    HOURS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HOURS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "seconds": round(seconds, 2),
                    "ok": ok,
                    "ts": datetime.now(tz=PT).isoformat(),
                }
            )
            + "\n"
        )


def gpu_hours_used() -> float:
    if not HOURS_PATH.is_file():
        return 0.0
    total = 0.0
    with HOURS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                total += float(json.loads(line).get("seconds") or 0)
    return total / 3600.0


def remaining_ok(*, need_h: float = 0.25) -> bool:
    if past_deadline():
        print("deadline reached", flush=True)
        return False
    used = gpu_hours_used()
    if used >= 40.0:
        print(f"gpu-hour budget used: {used:.2f}", flush=True)
        return False
    if used + need_h > 40.5:
        print(f"gpu-hour budget would exceed 40h (used {used:.2f}, need {need_h})", flush=True)
        return False
    check_gpu_lock()
    return True


def run_cmd(name: str, argv: list[str], *, rundir: Path, env: dict | None = None) -> int:
    rundir.mkdir(parents=True, exist_ok=True)
    log = rundir / "log"
    STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_PATH.write_text(name + "\n", encoding="utf-8")
    print(f"stage start {name}: {' '.join(argv)}", flush=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# start {name} {datetime.now(tz=PT).isoformat()}\n")
        fh.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        (rundir / "wrapper.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        code = proc.wait()
    elapsed = time.time() - t0
    if not (rundir / "exit").is_file():
        (rundir / "exit").write_text(str(code) + "\n", encoding="utf-8")
    _log_hours(name, elapsed, code == 0)
    print(f"stage end {name} code={code} elapsed_s={elapsed:.1f}", flush=True)
    return code


def wait_if_running(rundir: Path) -> None:
    for pid_name in ("pid", "wrapper.pid"):
        pid = _read_pid(rundir / pid_name)
        if pid and _alive(pid):
            print(f"waiting for existing pid {pid} in {rundir}", flush=True)
            while _alive(pid):
                time.sleep(15)


def main(argv: list[str] | None = None) -> int:
    del argv
    pipe_dir = ROOT / "runs" / "loop-pipeline"
    state = install_run_files(pipe_dir)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    def py_mod(mod: str, *args: str) -> list[str]:
        return [PY, "-m", mod, *args]

    # --- Stage 2: rewrite specs (Ollama). Do this before any torch job. ---
    rewrite_dir = ROOT / "runs" / "rewrite"
    leakage = ROOT / "data" / "scaled" / "rewrite-leakage.json"
    rewritten = ROOT / "data" / "scaled" / "rewritten.jsonl"
    wait_if_running(rewrite_dir)
    if _exit_ok(rewrite_dir) and rewritten.is_file() and _n_jsonl(rewritten) >= 2000:
        print(f"skip rewrite (kept {_n_jsonl(rewritten)})", flush=True)
    else:
        if not remaining_ok(need_h=0.5):
            state["code"] = 3
            return 3
        code = run_cmd(
            "rewrite",
            py_mod("e2m.loop", "rewrite"),
            rundir=rewrite_dir,
            env=env,
        )
        if code != 0:
            state["code"] = code
            return code

    # --- Stage 1: lora1 repair loops ---
    def loop_stage(name: str, extra: list[str], min_n: int, need_h: float) -> int:
        rundir = ROOT / "runs" / name
        wait_if_running(rundir)
        summary = rundir / "summary.json"
        if _exit_ok(rundir) and summary.is_file():
            n = json.loads(summary.read_text(encoding="utf-8")).get("n") or 0
            if n >= min_n:
                print(f"skip {name} (n={n})", flush=True)
                return 0
        if not remaining_ok(need_h=need_h):
            return 3
        return run_cmd(name, py_mod("e2m.loop", "run", "--out", str(rundir), *extra), rundir=rundir, env=env)

    for name, extra, min_n, need_h in [
        (
            "loop-lora1-heldout",
            [
                "--name",
                "lora1",
                "--adapter",
                str(ADAPTER_LORA1),
                "--split",
                "heldout",
                "--k",
                "8",
                "--rounds",
                "3",
            ],
            78,
            2.0,
        ),
        (
            "loop-lora1-val",
            [
                "--name",
                "lora1",
                "--adapter",
                str(ADAPTER_LORA1),
                "--split",
                "val",
                "--k",
                "8",
                "--rounds",
                "3",
            ],
            90,
            1.5,
        ),
        (
            "loop-base-c-heldout",
            [
                "--name",
                "base-c",
                "--mode",
                "c",
                "--split",
                "heldout",
                "--k",
                "4",
                "--rounds",
                "2",
            ],
            78,
            1.0,
        ),
    ]:
        code = loop_stage(name, extra, min_n, need_h)
        if code not in {0, 3}:
            state["code"] = code
            return code
        if code == 3:
            state["code"] = 3
            return 3

    # --- Stage 3: rejection sampling on train tasks ---
    reject_dir = ROOT / "runs" / "loop-reject-train"
    wait_if_running(reject_dir)
    reject_jsonl = ROOT / "data" / "scaled" / "reject.jsonl"
    if _exit_ok(reject_dir) and (reject_dir / "summary.json").is_file():
        n = json.loads((reject_dir / "summary.json").read_text(encoding="utf-8")).get("n") or 0
        print(f"skip reject-train (n={n})", flush=True)
    else:
        if not remaining_ok(need_h=8.0):
            state["code"] = 3
            return 3
        code = run_cmd(
            "loop-reject-train",
            py_mod(
                "e2m.loop",
                "run",
                "--name",
                "lora1",
                "--adapter",
                str(ADAPTER_LORA1),
                "--split",
                "train",
                "--k",
                "4",
                "--rounds",
                "2",
                "--batch-initial",
                "--max-new",
                "768",
                "--max-hours",
                "12",
                "--out",
                str(reject_dir),
                "--spec-kind",
                "original",
            ),
            rundir=reject_dir,
            env=env,
        )
        if code != 0:
            state["code"] = code
            return code

    # Optional: rewritten-spec loop on train if time remains (k=2 R=1).
    rw_loop = ROOT / "runs" / "loop-reject-rewritten"
    if rewritten.is_file() and remaining_ok(need_h=6.0) and not (
        _exit_ok(rw_loop) and (rw_loop / "summary.json").is_file()
    ):
        wait_if_running(rw_loop)
        code = run_cmd(
            "loop-reject-rewritten",
            py_mod(
                "e2m.loop",
                "run",
                "--name",
                "lora1",
                "--adapter",
                str(ADAPTER_LORA1),
                "--split",
                "train",
                "--k",
                "2",
                "--rounds",
                "1",
                "--batch-initial",
                "--max-new",
                "768",
                "--max-hours",
                "6",
                "--out",
                str(rw_loop),
                "--specs",
                str(rewritten),
                "--spec-kind",
                "rewritten",
            ),
            rundir=rw_loop,
            env=env,
        )
        if code not in {0, 3}:
            print(f"rewritten reject failed ({code}); continuing with original-spec passes", flush=True)
    else:
        print("skip rewritten reject (budget or already done)", flush=True)

    collect_dir = ROOT / "runs" / "mix-sft"
    collect_dir.mkdir(parents=True, exist_ok=True)
    if reject_dir.joinpath("loop.jsonl").is_file():
        subprocess.run(
            py_mod("e2m.loop", "collect", "--loop", str(reject_dir / "loop.jsonl"), "--out", str(reject_jsonl), "--spec-kind", "original"),
            cwd=str(ROOT),
            check=False,
        )
    if rw_loop.joinpath("loop.jsonl").is_file():
        subprocess.run(
            py_mod(
                "e2m.loop",
                "collect",
                "--loop",
                str(rw_loop / "loop.jsonl"),
                "--out",
                str(reject_jsonl),
                "--spec-kind",
                "rewritten",
            ),
            cwd=str(ROOT),
            check=False,
        )
    mix_out = ROOT / "data" / "scaled" / "sft_lora2.jsonl"
    mix = subprocess.run(py_mod("e2m.loop", "mix", "--out", str(mix_out)), cwd=str(ROOT), check=False, capture_output=True, text=True)
    (collect_dir / "mix.json").write_text(mix.stdout or mix.stderr or "", encoding="utf-8")
    print(mix.stdout, flush=True)

    # --- Train lora2 ---
    train_dir = ROOT / "runs" / "lora2"
    wait_if_running(train_dir)
    if (train_dir / "summary.json").is_file() and (train_dir / "adapter" / "adapter_config.json").is_file():
        print("skip lora2 train", flush=True)
    else:
        if not remaining_ok(need_h=0.75):
            state["code"] = 3
            return 3
        if not mix_out.is_file() or _n_jsonl(mix_out) < 1000:
            print("mix too small, refusing to train", flush=True)
            state["code"] = 4
            return 4
        code = run_cmd(
            "lora2-train",
            py_mod(
                "e2m.train.sft_lora",
                "--model",
                str(MODEL_15),
                "--sft",
                str(mix_out),
                "--out",
                str(train_dir),
                "--val-split",
                str(ROOT / "eval" / "val_split.json"),
                "--epochs",
                "3",
                "--lr",
                "2e-4",
                "--rank",
                "32",
                "--alpha",
                "64",
                "--dropout",
                "0.05",
                "--batch-size",
                "2",
                "--grad-accum",
                "8",
            ),
            rundir=train_dir,
            env=env,
        )
        if code != 0:
            state["code"] = code
            return code

    # --- Eval lora2 greedy/pass@5 ---
    eval_path = train_dir / "eval.jsonl"
    def eval_rows(split: str) -> int:
        n = 0
        if not eval_path.is_file():
            return 0
        with eval_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model") == "lora2" and row.get("mode") == "asm" and row.get("split") == split:
                    n += 1
        return n

    for split, min_n in (("val", 90), ("heldout", 78)):
        if eval_rows(split) >= min_n:
            print(f"skip lora2 eval {split}", flush=True)
            continue
        if not remaining_ok(need_h=0.75):
            state["code"] = 3
            return 3
        tasks = ROOT / ("eval/heldout.jsonl" if split == "heldout" else "tasks/scaled.jsonl")
        code = run_cmd(
            f"lora2-eval-{split}",
            py_mod(
                "e2m.train.eval_generate",
                "--model",
                str(MODEL_15),
                "--adapter",
                str(ADAPTER_LORA2),
                "--name",
                "lora2",
                "--mode",
                "asm",
                "--split",
                split,
                "--tasks",
                str(tasks),
                "--out",
                str(eval_path),
            ),
            rundir=ROOT / "runs" / f"lora2-eval-{split}",
            env=env,
        )
        if code != 0:
            state["code"] = code
            return code

    # --- lora2 repair loops ---
    for name, extra, min_n, need_h in [
        (
            "loop-lora2-heldout",
            ["--name", "lora2", "--adapter", str(ADAPTER_LORA2), "--split", "heldout", "--k", "8", "--rounds", "3"],
            78,
            2.0,
        ),
        (
            "loop-lora2-val",
            ["--name", "lora2", "--adapter", str(ADAPTER_LORA2), "--split", "val", "--k", "8", "--rounds", "3"],
            90,
            1.5,
        ),
    ]:
        code = loop_stage(name, extra, min_n, need_h)
        if code == 3:
            state["code"] = 3
            return 3
        if code != 0:
            state["code"] = code
            return code

    print("pipeline complete", flush=True)
    print("gpu_hours", round(gpu_hours_used(), 3), flush=True)
    state["code"] = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
