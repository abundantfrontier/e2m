"""GPU workers for round 5: synthetic tasks, teacher C samples, embedding screen.

The orchestrator holds ~/gpu.lock before starting ``sample`` or a GPU ``embed``.
``synth`` also needs the lock because Ollama qwen3:14b loads on the GPU.
CPU embeddings pass ``num_gpu=0`` and do not take the lock.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from e2m.loop.guard import check_gpu_lock, hold_requested
from e2m.loop.rewrite import unload_model
from e2m.loop.round5_data import (
    EMBED_THRESHOLD,
    leakage_reason,
    load_protected_tasks,
    screen_embed,
    score_c_source,
    verify_reference,
    _indexes,
    _Ref,
)

ROOT = Path(__file__).resolve().parents[2]
HOST = "http://127.0.0.1:11434"
TEACHER = ROOT / "runs" / "c-only-7b" / "adapter"
MODEL_7B = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"

_THEMES = (
    ("int", "int64_t {name}(int64_t a, int64_t b)", "two integers a and b"),
    ("int", "int64_t {name}(int64_t n)", "one integer n"),
    ("array", "int64_t {name}(const int64_t *a, int64_t a_n)", "an array of integers a and its length a_n"),
    (
        "array",
        "int64_t {name}(const int64_t *a, int64_t a_n, int64_t k)",
        "an array a, its length a_n, and an integer k",
    ),
    ("string", "int64_t {name}(const char *s)", "a NUL-terminated C string s"),
)


def _stop(max_hours: float, started: float, *, gpu: bool) -> str | None:
    if hold_requested():
        print("held", flush=True)
        return "held"
    if max_hours > 0 and (time.time() - started) / 3600.0 >= max_hours:
        print(f"max-hours {max_hours} reached", flush=True)
        return "max-hours"
    if gpu:
        try:
            check_gpu_lock()
        except SystemExit as exc:
            print(f"gpu lock lost: {exc}", flush=True)
            return "gpu-lock"
    return None


def _exit_for(reason: str | None) -> int:
    if reason == "held":
        return 4
    if reason == "max-hours":
        return 3
    if reason == "gpu-lock":
        return 2
    return 0


def _chat(prompt: str, *, temperature: float, num_predict: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": "qwen3:14b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write one leaf C function and tests. Reply with one JSON object. "
                    "No markdown and no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ollama: {exc}") from exc
    content = ((body.get("message") or {}).get("content") or "").strip()
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise RuntimeError("json not object")
    return obj


def _done_ids(path: Path) -> set[str]:
    found = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("done"):
                found.add(str(row.get("task_id") or row.get("id") or ""))
    found.discard("")
    return found


def _latest_done_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Last done row per task id. Append-only sample files keep earlier k here."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("done"):
                continue
            tid = str(row.get("task_id") or row.get("id") or "")
            if tid:
                latest[tid] = row
    return latest


def synth_fail_update(
    exc: BaseException,
    *,
    harness_streak: int,
    ollama_streak: int,
    ollama_giveup: int,
) -> tuple[str, int, int]:
    """Classify a synth failure.

    Harness rejects (bad JSON, gcc, leakage) do not count as ollama failures.
    Only consecutive ``ollama:`` errors move ``ollama_streak``. One HTTP 500
    is a retry. ``ollama_giveup`` consecutive ollama errors abort the stage.
    """
    if "ollama:" in str(exc):
        ollama_streak += 1
        if ollama_streak >= ollama_giveup:
            return "abort", harness_streak, ollama_streak
        return "retry", harness_streak, ollama_streak
    return "reject", harness_streak + 1, 0


def sample_plan(
    tasks: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
    *,
    ensure_k: int,
    prior_k: int,
    default_k: int,
) -> list[dict[str, Any]]:
    """Tasks that still need teacher samples.

    ``ensure_k <= 0`` is the round-5 rule: any done row is finished.
    ``ensure_k > 0`` tops a row up to that many temperature samples. A done
    row with no ``k`` field is treated as ``prior_k`` (round 5 used 4).
    """
    pending: list[dict[str, Any]] = []
    for task in tasks:
        tid = str(task.get("id") or "")
        row = latest.get(tid)
        item = dict(task)
        if ensure_k <= 0:
            if row is not None:
                continue
            item["_have_k"] = 0
            item["_c_texts"] = []
            item["_greedy_done"] = False
            item["_n_new"] = default_k
            pending.append(item)
            continue
        if row is None:
            have = 0
            greedy_done = False
            old = []
        else:
            # A missing k is the round-5 file (prior_k). An explicit 0 is a
            # failed attempt that still needs a full top-up, not prior_k.
            have = int(row["k"]) if "k" in row and row.get("k") is not None else prior_k
            greedy_done = bool(row["greedy_done"]) if "greedy_done" in row else True
            old = list(row.get("c_texts") or [])
        if have >= ensure_k:
            continue
        item["_have_k"] = have
        item["_c_texts"] = old
        item["_greedy_done"] = greedy_done
        item["_n_new"] = ensure_k - have
        pending.append(item)
    return pending


def _protected_index() -> tuple[list[dict[str, Any]], list[_Ref], dict[str, Any]]:
    protected = load_protected_tasks()
    refs = [_Ref(row) for row in protected]
    indexes = _indexes(refs)
    for row in protected:
        source = str(row.get("source") or "")
        if "/" in source:
            indexes["sources"].add(source)
    return protected, refs, indexes


def cmd_synth(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        _protected, refs, indexes = _protected_index()
    except Exception as exc:  # noqa: BLE001
        print(f"synth setup failed: {exc}", flush=True)
        return 1
    kept_ids = _done_ids(out)
    kept = len(kept_ids)
    streak = 0
    ollama_streak = 0
    ollama_giveup = max(1, int(getattr(args, "ollama_giveup", 5) or 5))
    attempt = 0
    for tid in kept_ids:
        parts = tid.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            attempt = max(attempt, int(parts[1]))
    print(f"synth start kept={kept} cap={args.cap}", flush=True)
    try:
        while kept < args.cap and streak < args.fail_streak:
            reason = _stop(args.max_hours, started, gpu=True)
            if reason:
                return _exit_for(reason)
            attempt += 1
            kind, sig_t, blurb = _THEMES[(attempt - 1) % len(_THEMES)]
            name = f"synth_{attempt}_{kind}"
            signature = sig_t.format(name=name)
            prompt = (
                f"Write a short HumanEval-style leaf function named {name}.\n"
                f"Signature, use it exactly: {signature}\n"
                f"The function takes {blurb}. Return int64_t. No other functions, "
                "no library calls, no floats, no malloc, no recursion.\n"
                "JSON keys:\n"
                "spec: 2 to 4 English sentences. No code.\n"
                "reference_c: the function only, matching the signature.\n"
                "tests: 4 to 8 objects {args, ret}. "
                "Integer args are JSON numbers. Array args are "
                '{"id":"a","ptr":"i64","data":[ints]} and the next arg is the length. '
                'String args are {"id":"s","ptr":"cstr","data":"text"}. '
                "ret is an integer. Include a zero or empty edge case.\n"
            )
            try:
                obj = _chat(prompt, temperature=0.7, num_predict=900, timeout=120)
                spec = str(obj.get("spec") or "").strip()
                code = str(obj.get("reference_c") or "").strip()
                tests = obj.get("tests")
                if not spec or "{" in spec or name not in code or not isinstance(tests, list) or len(tests) < 2:
                    raise RuntimeError("malformed task")
                task = {
                    "id": name,
                    "spec": spec,
                    "signature": signature,
                    "reference_c": code if code.endswith("\n") else code + "\n",
                    "tests": tests[:8],
                    "source": f"synth/{attempt}",
                    "origin": "synth",
                }
                leak = leakage_reason(task, refs, indexes=indexes)
                if leak:
                    raise RuntimeError(f"leak {leak}")
                err = verify_reference(task)
                if err:
                    raise RuntimeError(err[:200])
            except Exception as exc:  # noqa: BLE001
                action, streak, ollama_streak = synth_fail_update(
                    exc,
                    harness_streak=streak,
                    ollama_streak=ollama_streak,
                    ollama_giveup=ollama_giveup,
                )
                print(
                    f"synth {action} {attempt} harness_streak={streak} "
                    f"ollama_streak={ollama_streak}: {exc}",
                    flush=True,
                )
                if action == "abort":
                    print("synth ollama unreachable", flush=True)
                    return 1
                if action == "retry":
                    reason = _stop(args.max_hours, started, gpu=True)
                    if reason:
                        return _exit_for(reason)
                    attempt -= 1
                    time.sleep(min(20.0, 2.0 * ollama_streak))
                continue
            streak = 0
            ollama_streak = 0
            kept += 1
            with out.open("a", encoding="utf-8") as fh:
                row = dict(task)
                row["done"] = True
                row["task_id"] = name
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"synth kept {kept}/{args.cap} {name}", flush=True)
    finally:
        unload_model(HOST, "qwen3:14b")
    print(f"synth end kept={kept} streak={streak}", flush=True)
    if kept == 0:
        return 1
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from e2m.train.eval_generate import _generate, _load_model
    from e2m.train.format import c_messages, extract_c

    tasks = []
    with Path(args.tasks).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                tasks.append(json.loads(line))
    if args.limit > 0:
        tasks = tasks[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_k = int(getattr(args, "ensure_k", 0) or 0)
    prior_k = int(getattr(args, "prior_k", 4) or 4)
    latest = _latest_done_rows(out) if ensure_k > 0 else {}
    if ensure_k > 0:
        pending = sample_plan(
            tasks,
            latest,
            ensure_k=ensure_k,
            prior_k=prior_k,
            default_k=args.k,
        )
    else:
        done = _done_ids(out)
        pending = sample_plan(
            tasks,
            {tid: {"done": True, "task_id": tid, "c_texts": []} for tid in done},
            ensure_k=0,
            prior_k=prior_k,
            default_k=args.k,
        )
    print(
        f"sample tasks={len(tasks)} pending={len(pending)} ensure_k={ensure_k} k={args.k}",
        flush=True,
    )
    started = time.time()
    reason = _stop(args.max_hours, started, gpu=True)
    if reason:
        return _exit_for(reason)
    if not pending:
        return 0
    model, tokenizer = _load_model(Path(args.model), Path(args.adapter), load_4bit=args.load_4bit)
    for i, task in enumerate(pending, 1):
        reason = _stop(args.max_hours, started, gpu=True)
        if reason:
            return _exit_for(reason)
        tid = str(task["id"])
        messages = c_messages(str(task.get("spec") or ""), str(task.get("signature") or ""))
        texts: list[str] = []
        n_new = int(task.get("_n_new") or args.k)
        do_greedy = not task.get("_greedy_done")
        try:
            greedy = (
                _generate(model, tokenizer, messages, n=1, temperature=0.0, max_new=args.max_new)
                if do_greedy
                else []
            )
            sampled = (
                _generate(
                    model,
                    tokenizer,
                    messages,
                    n=n_new,
                    temperature=args.temperature,
                    max_new=args.max_new,
                )
                if n_new > 0
                else []
            )
        except Exception as exc:  # noqa: BLE001
            print(f"sample {tid} generate failed: {exc}", flush=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "task_id": tid,
                            "origin": task.get("origin"),
                            "done": True,
                            "k": int(task.get("_have_k") or 0),
                            "greedy_done": bool(task.get("_greedy_done")),
                            "c_texts": list(task.get("_c_texts") or []),
                            "error": str(exc)[:200],
                        }
                    )
                    + "\n"
                )
            continue
        seen = set()
        passed: list[str] = []
        for old in task.get("_c_texts") or []:
            code = str(old).strip()
            key = " ".join(code.split())
            if not code or key in seen:
                continue
            seen.add(key)
            passed.append(code if code.endswith("\n") else code + "\n")
        for raw in [*greedy, *sampled]:
            code = extract_c(raw).strip()
            key = " ".join(code.split())
            if not code or key in seen:
                continue
            seen.add(key)
            texts.append(code)
            try:
                ok = score_c_source(task, code)
            except Exception as exc:  # noqa: BLE001
                print(f"  score {tid} {exc}", flush=True)
                ok = False
            if ok:
                passed.append(code if code.endswith("\n") else code + "\n")
        with out.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "task_id": tid,
                        "origin": task.get("origin") or "",
                        "done": True,
                        "k": int(ensure_k or args.k),
                        "greedy_done": True,
                        "n_pass": len(passed),
                        "c_texts": passed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"sample {i}/{len(pending)} {tid} pass={len(passed)} tried={len(texts)}", flush=True)
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    rows = []
    with Path(args.rows).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    protected = load_protected_tasks()
    started = time.time()

    def stop_check() -> str | None:
        return _stop(args.max_hours, started, gpu=not args.cpu)

    try:
        kept, dropped, summary = screen_embed(
            rows,
            protected,
            cache_path=Path(args.cache),
            cpu=args.cpu,
            threshold=EMBED_THRESHOLD,
            stop_check=stop_check,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"embed failed: {exc}", flush=True)
        return 1
    finally:
        unload_model(HOST, "bge-m3:latest")
    report = {
        **summary,
        "kept": len(kept),
        "dropped": len(dropped),
        "rows": len(rows),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dropped_path = Path(args.dropped)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    with dropped_path.open("w", encoding="utf-8") as fh:
        for row in dropped:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if summary.get("embedded"):
        with Path(args.kept).open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"embed kept {len(kept)} dropped {len(dropped)}", flush=True)
        return 0
    print(f"embed stopped {summary.get('stopped')} pending={summary.get('pending')}", flush=True)
    return _exit_for(str(summary.get("stopped") or ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-5 sample, synth, and embed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=1.0)
    p.add_argument("--cap", type=int, default=80)
    p.add_argument("--fail-streak", type=int, default=40)
    p.add_argument("--ollama-giveup", type=int, default=5)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("sample")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_7B)
    p.add_argument("--adapter", type=Path, default=TEACHER)
    p.add_argument("--4bit", action="store_true", dest="load_4bit")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--ensure-k", type=int, default=0, help="Top up existing rows to this many temperature samples")
    p.add_argument("--prior-k", type=int, default=4, help="k assumed for a done row that has no k field")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("embed")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--kept", type=Path, required=True)
    p.add_argument("--dropped", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_embed)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
