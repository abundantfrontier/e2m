"""Round 7 GPU stages (untrained Qwen2.5-Coder-7B-Instruct, 4-bit, no adapter).

  python -m e2m.loop.round7_gpu edits   --max-hours H   # stage 2: smaller/faster asm
  python -m e2m.loop.round7_gpu fasterc --max-hours H   # stage 4: faster C
Refuses to start unless ~/gpu.lock is an e2m lock; re-checks the lock and the
hold file between tasks. Resume-safe (skips rows already in the output).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from e2m.harness.assemble import AssembleError
from e2m.loop.guard import check_gpu_lock, hold_requested
from e2m.loop.round7_common import (OUT, ROOT, N_ACCEPT, accept, append_jsonl, compile_c, interleaved_tasks,
                                    measure, oracle_image, read_jsonl, strip_asm, SPEED_COMPILERS)
from e2m.train.format import extract_asm, extract_c

MODEL = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"
SYSTEM = ("You are an expert x86-64 assembly programmer and superoptimizer. You write GNU assembler "
          "AT&T syntax for Linux x86-64 (System V ABI), freestanding: no calls, no libc, no PIC/GOT.")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


def goal_text(goal: str, bar: dict[str, Any]) -> str:
    if goal == "size":
        return (f"GOAL: make the machine code strictly SMALLER than {bar['size']} bytes of .text "
                f"(the best of gcc -O2/-O3/-Os and clang -O2/-Oz; this listing is {bar['compiler']} {bar['opt']}). "
                "Use short encodings: 32-bit registers where the upper half is not needed, xor for zero, "
                "push/pop instead of movabs, lea for arithmetic, loop restructuring, removing alignment padding.")
    return (f"GOAL: make it FASTER: fewer executed instructions than {bar['insn']} over the test inputs "
            f"(the best of gcc -O2/-O3 and clang -O2; this listing is {bar['compiler']} {bar['opt']}). "
            "Aim for at most 90% of that. Remove redundant work from loops, use better algorithms, "
            "strength-reduce, keep values in registers. Code size does not matter for this goal.")


def edit_messages(task, c_src: str, bar: dict[str, Any], goal: str) -> list[dict[str, str]]:
    user = (f"Signature: {task.signature}\n\nSpecification:\n{task.spec.strip()}\n\n"
            f"A correct C implementation:\n```c\n{c_src.strip()}\n```\n\n"
            f"Best compiler output:\n```asm\n{strip_asm(bar['asm']).strip()}\n```\n\n{goal_text(goal, bar)}\n\n"
            f"The function must stay named `{task.sig().name}`, behave identically on ALL inputs, and follow the ABI "
            "(preserve rbx, rbp, r12-r15; return in rax/eax or xmm0). Reply with only the complete assembly "
            "in one ```asm block.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def feedback_text(best_fail: dict[str, Any] | None, best_ok: dict[str, Any] | None, goal: str, bar: dict[str, Any]) -> str:
    key = "size" if goal == "size" else "insn"
    unit = "bytes" if goal == "size" else "executed instructions"
    parts = []
    if best_ok:
        parts.append(f"Your best correct version has {best_ok[key]} {unit}; the bar is {bar[key]} "
                     f"({'beaten' if best_ok[key] < bar[key] else 'not beaten yet'}). Here it is:\n```asm\n{best_ok['asm'].strip()}\n```")
    if best_fail:
        why = best_fail.get("assemble_error") or best_fail.get("fail") or best_fail.get("diff_detail") or best_fail.get("stage")
        ce = best_fail.get("counterexample")
        parts.append(f"This attempt was rejected ({best_fail.get('stage')}): {str(why)[:400]}"
                     + (f"\nCounterexample input: {json.dumps(ce['args'])[:400]}; expected return {ce['ref_return']!r}, got {ce['cand_return']!r}" if ce else "")
                     + f"\n```asm\n{best_fail['asm'].strip()[:3000]}\n```")
    parts.append(f"Try again: {goal_text(goal, bar)} Reply with only the complete assembly in one ```asm block.")
    return "\n\n".join(parts)


def score_batch(task, oracle, texts: list[str], extract, bar_val: int, key: str, compile_fn=None):
    """Return list of verdict dicts (with asm); accept() only for test-passing candidates that beat the best so far."""
    out = []
    for raw in texts:
        src = extract(raw)
        if compile_fn:
            try:
                asm = compile_fn(src)
            except AssembleError as exc:
                out.append({"asm": src, "stage": "compile", "accepted": False, "assemble_error": str(exc)[:300], "passed": False})
                continue
        else:
            asm = src
        m = measure(task, asm)
        m["asm"] = asm if not compile_fn else asm
        m["src"] = src
        if not m["passed"]:
            m.update(stage="tests", accepted=False)
            out.append(m)
            continue
        a = accept(task, asm, oracle, n=200)
        a["asm"], a["src"] = asm, src
        out.append(a)
    return out


def edits(args) -> int:
    check_gpu_lock()
    from e2m.train.eval_generate import _generate, _load_model

    out = OUT / "edits.jsonl"
    done = {(r["task_id"], r["goal"]) for r in read_jsonl(out)}
    port = {r["task_id"]: r for r in read_jsonl(OUT / "portfolio.jsonl")}
    tasks = [(s, t) for s, t in interleaved_tasks() if t.id in port]
    jobs = [(s, t, g) for s, t in tasks for g in ("size", "speed")
            if (t.id, g) not in done and port[t.id].get("best_size" if g == "size" else "best_speed")]
    log(f"edits: {len(jobs)} jobs pending ({len(done)} done), k={args.k}, rounds={args.rounds}, cap {args.max_hours} h")
    t0 = time.time()
    model, tok = _load_model(MODEL, None, load_4bit=True)
    model.eval()
    log(f"model loaded in {time.time()-t0:.0f}s")
    wins = 0
    for split, task, goal in jobs:
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            log("edits: GPU-hour cap reached; stopping"); break
        if hold_requested():
            log("edits: hold requested; stopping at task boundary"); return 3
        check_gpu_lock()
        p = port[task.id]
        bar = p["best_size"] if goal == "size" else p["best_speed"]
        key = "size" if goal == "size" else "insn"
        oracle = oracle_image(task)
        c_src = p["pass_set_c"]
        msgs = edit_messages(task, c_src, bar, goal)
        best_ok = None; best_fail = None; calls = 0; tallies: dict[str, int] = {}
        ts = time.time()
        for rnd in range(args.rounds + 1):
            texts = _generate(model, tok, msgs, n=args.k, temperature=args.temperature, max_new=args.max_new)
            calls += 1
            for v in score_batch(task, oracle, texts, extract_asm, bar[key], key):
                tallies[v.get("stage", "?")] = tallies.get(v.get("stage", "?"), 0) + 1
                if v.get("accepted"):
                    if best_ok is None or v[key] < best_ok[key] or (v[key] == best_ok[key] and v["size"] < best_ok["size"]):
                        best_ok = {"asm": v["asm"], "size": v["size"], "insn": v["insn"], "round": rnd}
                else:
                    # prefer the most informative failure: diff > tests > assemble
                    rank = {"diff": 3, "tests": 2, "compile": 1}.get(v.get("stage"), 0)
                    if best_fail is None or rank > best_fail["_rank"]:
                        best_fail = {**{k: v.get(k) for k in ("asm", "stage", "assemble_error", "fail", "diff_detail", "counterexample")}, "_rank": rank}
            if best_ok and best_ok[key] < (bar[key] if goal == "size" else 0.9 * bar[key]):
                break
            if rnd < args.rounds:
                msgs = msgs + [{"role": "assistant", "content": f"```asm\n{(best_ok or best_fail or {'asm': ''})['asm'].strip()[:3000]}\n```"},
                               {"role": "user", "content": feedback_text(best_fail, best_ok, goal, bar)}]
        final = None
        if best_ok:
            a = accept(task, best_ok["asm"], oracle, n=N_ACCEPT, seed=7)
            if a["accepted"]:
                final = best_ok
        win = bool(final) and ((goal == "size" and final["size"] < bar["size"]) or (goal == "speed" and final["insn"] <= 0.9 * bar["insn"]))
        wins += win
        row = {"split": split, "task_id": task.id, "goal": goal, "bar": bar[key], "bar_from": f"{bar['compiler']} {bar['opt']}",
               "calls": calls, "tallies": tallies, "best": final, "best_val": final[key] if final else None, "win": win,
               "seconds": round(time.time() - ts, 1)}
        append_jsonl(out, row)
        log(f"  {split} {task.id} {goal}: bar={bar[key]} best={row['best_val']} win={win} calls={calls} {tallies} (wins {wins})")
    return 0


def fasterc(args) -> int:
    check_gpu_lock()
    from e2m.train.eval_generate import _generate, _load_model

    out = OUT / "fasterc.jsonl"
    done = {r["task_id"] for r in read_jsonl(out)}
    port = {r["task_id"]: r for r in read_jsonl(OUT / "portfolio.jsonl")}
    jobs = []
    for s, t in interleaved_tasks():
        p = port.get(t.id)
        if not p or t.id in done or not p.get("best_speed"):
            continue
        # pass-set C's own best speed compile vs the bar
        own = [v for v in p["variants"] if v["c_idx"] == 1 and v.get("passed") and (v["compiler"], v["opt"]) in SPEED_COMPILERS]
        if not own:
            continue
        own_insn = min(v["insn"] for v in own)
        ratio = own_insn / max(1, p["gcc_o2"]["insn"] or 1)
        if ratio > args.ratio:
            jobs.append((ratio, s, t, own_insn))
    jobs.sort(key=lambda j: -j[0])
    log(f"fasterc: {len(jobs)} tasks with pass-set C over {args.ratio}x gcc -O2 insn: {[(j[2].id, round(j[0], 2)) for j in jobs[:12]]}")
    if not jobs:
        return 0
    t0 = time.time()
    model, tok = _load_model(MODEL, None, load_4bit=True)
    model.eval()
    for ratio, split, task, own_insn in jobs:
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            log("fasterc: cap reached"); break
        if hold_requested():
            log("fasterc: hold requested"); return 3
        check_gpu_lock()
        p = port[task.id]
        oracle = oracle_image(task)
        user = (f"Signature: {task.signature}\n\nSpecification:\n{task.spec.strip()}\n\nThis C is correct but slow "
                f"({own_insn} executed instructions on the tests vs {p['gcc_o2']['insn']} for the reference):\n```c\n{p['pass_set_c'].strip()}\n```\n\n"
                "Rewrite it to be as fast as possible (better algorithm, fewer passes, early exits) with identical behaviour on all inputs. "
                "Freestanding C: no libc calls, only <stdint.h>/<stddef.h>/<stdbool.h> types. Reply with only the function in one ```c block.")
        msgs = [{"role": "system", "content": "You are an expert C performance engineer."}, {"role": "user", "content": user}]
        texts = _generate(model, tok, msgs, n=args.k, temperature=0.8, max_new=1024)
        best = None
        for raw in texts:
            c = extract_c(raw)
            for comp, opt in (("gcc", "-O2"), ("gcc", "-O3"), ("clang", "-O2")):
                try:
                    asm = compile_c(c, comp, opt)
                except AssembleError:
                    continue
                m = measure(task, asm)
                if m["passed"] and (best is None or m["insn"] < best["insn"]):
                    a = accept(task, asm, oracle, n=N_ACCEPT)
                    if a["accepted"]:
                        best = {"c": c, "asm": asm, "insn": m["insn"], "size": m["size"], "compiler": f"{comp} {opt}"}
        row = {"split": split, "task_id": task.id, "ratio_before": round(ratio, 3), "own_insn": own_insn,
               "gcc_o2_insn": p["gcc_o2"]["insn"], "bar_speed": p["best_speed"]["insn"], "best": best,
               "ratio_after": round(best["insn"] / max(1, p["gcc_o2"]["insn"]), 3) if best else None,
               "win": bool(best) and best["insn"] <= 0.9 * p["best_speed"]["insn"]}
        append_jsonl(out, row)
        log(f"  {task.id}: ratio {ratio:.2f} -> {row['ratio_after']} win={row['win']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("edits")
    a.add_argument("--k", type=int, default=8); a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--temperature", type=float, default=0.8); a.add_argument("--max-new", type=int, default=1024)
    a.add_argument("--max-hours", type=float, default=2.5)
    b = sub.add_parser("fasterc")
    b.add_argument("--k", type=int, default=8); b.add_argument("--ratio", type=float, default=1.5)
    b.add_argument("--max-hours", type=float, default=0.5)
    args = p.parse_args(argv)
    return {"edits": edits, "fasterc": fasterc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
