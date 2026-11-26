"""Resumable GPU runner for the repair loop.

Writes pid / exit / log-friendly stdout. Skips task ids already in the JSONL.
Checks ~/gpu.lock each task and stops if another project holds the GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from e2m.harness.tasks import Task, load_jsonl
from e2m.loop.guard import check_gpu_lock, hold_requested, install_run_files, past_deadline
from e2m.loop.repair import repair_task
from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import _done_ids, _gcc_reference, _generate, _load_model

ROOT = Path(__file__).resolve().parents[2]


def load_spec_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dropped"):
                continue
            tid = str(row.get("task_id") or row.get("id") or "")
            spec = str(row.get("spec") or "")
            if tid and spec:
                out[tid] = spec
    return out


def select_tasks(
    *,
    split: str,
    tasks_path: Path,
    val_path: Path,
    limit: int,
) -> list[Task]:
    tasks = load_jsonl(tasks_path)
    if split == "val":
        wanted = set(json.loads(val_path.read_text(encoding="utf-8"))["task_ids"] if val_path.is_file() else [])
        tasks = [t for t in tasks if t.id in wanted]
    elif split == "train":
        blocked = set()
        if val_path.is_file():
            blocked.update(json.loads(val_path.read_text(encoding="utf-8"))["task_ids"])
        tasks = [t for t in tasks if t.id not in blocked]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def run_loop(
    *,
    model_path: Path,
    adapter: Path | None,
    model_name: str,
    mode: str,
    split: str,
    tasks_path: Path,
    val_path: Path,
    out_dir: Path,
    k: int,
    rounds: int,
    temperature: float,
    max_new: int,
    limit: int,
    spec_map_path: Path | None,
    spec_kind: str,
    load_4bit: bool,
    timeout_s: float,
    batch_initial: bool = False,
    max_hours: float = 0.0,
) -> dict[str, Any]:
    check_gpu_lock()
    if hold_requested():
        print("held", flush=True)
        return {"stopped": "held", "mode": mode, "split": split, "model": model_name}
    if past_deadline():
        print("GPU deadline 2026-11-18 18:00 PT reached", flush=True)
        raise SystemExit(3)
    state = install_run_files(out_dir)
    out_path = out_dir / "loop.jsonl"
    progress_path = out_dir / "progress.txt"
    tasks = select_tasks(split=split, tasks_path=tasks_path, val_path=val_path, limit=limit)
    spec_map = load_spec_map(spec_map_path)
    if spec_map_path is not None:
        tasks = [t for t in tasks if t.id in spec_map]
    done = _done_ids(out_path, model_name, mode, split)
    pending = [t for t in tasks if t.id not in done]
    print(
        f"loop {model_name} {mode} {split}: {len(tasks)} tasks, {len(done)} cached, {len(pending)} pending "
        f"k={k} R={rounds} T={temperature}",
        flush=True,
    )
    model, tokenizer = _load_model(model_path, adapter, load_4bit=load_4bit)

    def generate(messages: list[dict[str, str]], n: int = 1) -> list[str]:
        return _generate(
            model,
            tokenizer,
            messages,
            n=n,
            temperature=temperature,
            max_new=max_new,
        )

    started = time.time()
    n_pass = 0
    stopped: str | None = None
    with out_path.open("a", encoding="utf-8") as fh:
        for i, task in enumerate(pending, 1):
            if past_deadline():
                print("deadline reached, stopping", flush=True)
                stopped = "deadline"
                break
            if max_hours > 0 and (time.time() - started) / 3600.0 >= max_hours:
                print(f"max-hours {max_hours} reached, stopping", flush=True)
                stopped = "max-hours"
                break
            if hold_requested():
                print("held", flush=True)
                stopped = "held"
                break
            try:
                lock = check_gpu_lock()
            except SystemExit as exc:
                print(f"stopping: {exc}", flush=True)
                stopped = "gpu-lock"
                state["code"] = 2
                break
            spec = spec_map.get(task.id)
            try:
                loop = repair_task(
                    task,
                    generate,
                    k=k,
                    rounds=rounds,
                    mode=mode,
                    spec=spec,
                    timeout_s=timeout_s,
                    batch_initial=batch_initial,
                )
                gcc = _gcc_reference(task)
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"[:400]
                print(f"  task {task.id} failed: {detail}", flush=True)
                row = {
                    "model": model_name,
                    "mode": mode,
                    "split": split,
                    "task_id": task.id,
                    "source": task.source,
                    "spec_kind": spec_kind,
                    "spec": spec if spec is not None else task.spec,
                    "signature": task.signature,
                    "calls": 0,
                    "passed_visible": False,
                    "passed_hidden": False,
                    "passed_full": False,
                    "first_visible_call": None,
                    "first_hidden_call": None,
                    "first_full_call": None,
                    "n_visible": 0,
                    "n_hidden": 0,
                    "n_tests": len(task.tests),
                    "attempts": [],
                    "best_pass": None,
                    "gcc": {"ok": False, "size": 0, "insn": 0, "detail": detail},
                    "error": detail,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                continue
            row = {
                "model": model_name,
                "mode": mode,
                "split": split,
                "task_id": task.id,
                "source": task.source,
                "spec_kind": spec_kind,
                "spec": spec if spec is not None else task.spec,
                "signature": task.signature,
                **loop.as_dict(),
                "gcc": gcc,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if loop.passed_full:
                n_pass += 1
            msg = (
                f"  {i}/{len(pending)} {task.id} calls={loop.calls} "
                f"full={int(loop.passed_full)} vis={int(loop.passed_visible)} "
                f"hid={int(loop.passed_hidden)} last={(loop.attempts[-1].category_full if loop.attempts else 'none')}"
            )
            print(msg, flush=True)
            progress_path.write_text(
                f"{i}/{len(pending)} cached={len(done)} pass_this_run={n_pass} "
                f"elapsed_s={time.time() - started:.0f} lock={lock.split()[0]}\n",
                encoding="utf-8",
            )
    summary = summarize_loop(out_path, model=model_name, split=split)
    summary["elapsed_s"] = round(time.time() - started, 2)
    summary["k"] = k
    summary["rounds"] = rounds
    summary["temperature"] = temperature
    summary["mode"] = mode
    summary["stopped"] = stopped
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("loop summary", json.dumps(summary), flush=True)
    if stopped == "gpu-lock":
        state["code"] = 2
    elif stopped in {"deadline", "max-hours"}:
        state["code"] = 3
    elif stopped == "held":
        state["code"] = 4
    else:
        state["code"] = 0
    return summary


def collect_reject(loop_path: Path, out_path: Path, *, spec_kind: str = "original") -> dict[str, Any]:
    from e2m.train.format import extract_asm

    n_in = 0
    n_keep = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    existing = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                prev = json.loads(line)
                existing.add((str(prev.get("task_id")), str(prev.get("spec_kind") or "original")))
    with loop_path.open(encoding="utf-8") as fh, out_path.open("a", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_in += 1
            if not row.get("passed_full"):
                continue
            best = row.get("best_pass") or {}
            raw = str(best.get("text") or "")
            asm = extract_asm(raw) if raw else ""
            if not asm:
                continue
            kind = str(row.get("spec_kind") or spec_kind)
            key = (str(row["task_id"]), kind)
            if key in seen or key in existing:
                continue
            seen.add(key)
            spec = str(row.get("spec") or "")
            signature = str(row.get("signature") or "")
            if not spec or not signature:
                continue
            n_keep += 1
            out.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "spec": spec,
                        "signature": signature,
                        "asm": asm,
                        "size": best.get("size"),
                        "insn": best.get("insn"),
                        "source": "reject_loop",
                        "spec_kind": kind,
                        "calls": row.get("calls"),
                        "split": row.get("split"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"n_in": n_in, "n_keep": n_keep, "out": str(out_path)}
