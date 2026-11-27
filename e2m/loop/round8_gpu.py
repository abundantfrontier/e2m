"""Round 8 stage D: untrained Qwen2.5-Coder-7B-Instruct (4-bit, no adapter) edits on search near-misses.

  python -m e2m.loop.round8_gpu nearmiss            # prints the held-out near-miss count (CPU, no lock)
  python -m e2m.loop.round8_gpu edits --max-hours H # needs ~/gpu.lock to be an e2m lock
A near-miss is a task-goal the search did not win where it found a tests-passing candidate past the win
threshold that v3 rejected, or (speed) an accepted candidate at <= 0.99x the bar. Prompts carry the bar,
the search's best listing, the rejected candidate with its counterexample, and a list of tricks. k=8, one
refinement round, v3 acceptance (2,000 inputs; 5,000 on the final). Stops after 0.5 h with no unique win
(a win the search did not have). E2M_R8_FAKE_GEN=1 replaces the model with echoes for CPU dry runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from e2m.loop.guard import check_gpu_lock, hold_requested
from e2m.loop.round7_common import ROOT, append_jsonl, read_jsonl, strip_asm
from e2m.loop.round8_common import N_FINAL, N_V3, OUT, accept3, tasks_by_id
from e2m.train.format import extract_asm

MODEL = ROOT / "runs" / "models" / "Qwen2.5-Coder-7B-Instruct"
FAKE = os.environ.get("E2M_R8_FAKE_GEN") == "1"
SYSTEM = ("You are an expert x86-64 assembly programmer and superoptimizer. You write GNU assembler "
          "AT&T syntax for Linux x86-64 (System V ABI), freestanding: no calls, no libc, no PIC/GOT.")
TRICKS = ("Tricks that often help: 32-bit registers when the upper half is not needed (writes zero-extend); xor reg,reg "
          "for zero; push imm8 / pop reg for small constants (3 bytes); lea for add/shift/copy without touching flags; "
          "one-operand mul when rdx is free; reuse a register that already holds a constant; drop test/cmp whose flags "
          "are already set by the previous arithmetic; setcc/cmov instead of branches; rotate loops so the test is at the "
          "bottom; dec+jnz counters; remove alignment padding (.p2align) for size.")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


def near_misses(split: str = "heldout") -> list[dict[str, Any]]:
    return [r for r in read_jsonl(OUT / "search.jsonl") if r.get("isa") == "base" and r["split"] == split and r.get("near_miss")]


def messages(task, goal: str, bar: dict[str, Any], srow: dict[str, Any]) -> list[dict[str, str]]:
    key = "size" if goal == "size" else "insn"
    target = (f"strictly SMALLER than {bar['size']} bytes of .text" if goal == "size"
              else f"at most {int(0.9 * bar['insn'])} executed instructions over the test inputs (the bar is {bar['insn']})")
    parts = [f"Signature: {task.signature}\n\nSpecification:\n{task.spec.strip()}\n\nReference C:\n```c\n{task.reference_c.strip()}\n```",
             f"Best compiler output ({bar['compiler']} {bar['opt']}, {bar[key]} {'bytes' if goal == 'size' else 'instructions'}):\n```asm\n{strip_asm(bar['asm']).strip()}\n```"]
    if srow.get("best"):
        parts.append(f"A verified-correct search result ({srow['best'][key]}):\n```asm\n{srow['best']['asm'].strip()}\n```")
    rb = srow.get("rejected_below_bar")
    if rb:
        parts.append(f"This shorter attempt ({rb['val']}) is WRONG on some inputs ({rb['detail'][:160]}; counterexample {rb['counterexample'][:300]}):\n"
                     f"```asm\n{rb['asm'].strip()[:2500]}\n```\nFix it while keeping it short.")
    parts.append(f"GOAL: {target}. {TRICKS}\nThe function must stay named `{task.sig().name}`, behave identically to the C on ALL "
                 "inputs, and follow the ABI (preserve rbx, rbp, r12-r15). Reply with only the complete assembly in one ```asm block.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def edits(args) -> int:
    if not FAKE:
        check_gpu_lock()
        from e2m.train.eval_generate import _generate, _load_model
    out = OUT / "edits.jsonl"
    done = {(r["task_id"], r["goal"]) for r in read_jsonl(out)}
    bars = {r["task_id"]: r for r in read_jsonl(OUT / "bars.jsonl")}
    search_win = {(r["task_id"], r["goal"]) for r in read_jsonl(OUT / "search.jsonl") if r.get("win")}
    jobs = [r for r in near_misses("heldout") + near_misses("val") if (r["task_id"], r["goal"]) not in done]
    log(f"edits: {len(jobs)} near-miss jobs pending ({len(done)} done), k={args.k}, cap {args.max_hours} h, fake={FAKE}")
    t0 = time.time()
    if FAKE:
        model = tok = None
        def gen(msgs, n):  # echo the bar / search listing
            import re
            blocks = re.findall(r"```asm\n(.*?)```", msgs[1]["content"], re.S)
            return [f"```asm\n{blocks[min(i, len(blocks) - 1)]}```" for i in range(n)]
    else:
        model, tok = _load_model(MODEL, None, load_4bit=True)
        model.eval()
        log(f"model loaded in {time.time()-t0:.0f}s")
        def gen(msgs, n):
            return _generate(model, tok, msgs, n=n, temperature=args.temperature, max_new=args.max_new)
    unique = 0
    for srow in jobs:
        el = time.time() - t0
        if args.max_hours and el > args.max_hours * 3600:
            log("edits: GPU-hour cap reached; stopping"); break
        if el > args.no_unique_stop_h * 3600 and unique == 0:
            log(f"edits: no unique win after {args.no_unique_stop_h} h; stopping"); break
        if hold_requested():
            log("edits: hold requested; stopping at task boundary"); return 3
        if not FAKE:
            check_gpu_lock()
        task = tasks_by_id()[srow["task_id"]]
        goal = srow["goal"]
        key = "size" if goal == "size" else "insn"
        bar = bars[task.id][goal]
        thr = bar["size"] if goal == "size" else 0.9 * bar["insn"]
        msgs = messages(task, goal, bar, srow)
        best_ok, best_fail, tallies, ts = None, None, {}, time.time()
        for rnd in range(args.rounds + 1):
            for raw in gen(msgs, args.k):
                asm = extract_asm(raw)
                a = accept3(task, asm, n=N_V3, seed=rnd)
                tallies[a["stage"]] = tallies.get(a["stage"], 0) + 1
                if a["accepted"]:
                    if best_ok is None or a[key] < best_ok[key]:
                        best_ok = {"asm": asm, "size": a["size"], "insn": a["insn"], "round": rnd}
                elif a["stage"] == "v3" or best_fail is None:
                    best_fail = {"asm": asm, "why": a.get("v3_detail") or a.get("fail") or a.get("assemble_error"),
                                 "ce": json.dumps(a.get("counterexample"), default=str)[:300]}
            if best_ok and (best_ok[key] < thr if goal == "size" else best_ok[key] <= thr):
                break
            if rnd < args.rounds:
                fb = (f"Your best correct version has {best_ok[key]} (target {thr}).\n" if best_ok else "") + \
                     (f"An attempt was rejected: {str(best_fail['why'])[:300]}; counterexample {best_fail['ce']}\n" if best_fail else "") + \
                     "Try again. Reply with only the complete assembly in one ```asm block."
                msgs = msgs + [{"role": "assistant", "content": f"```asm\n{(best_ok or best_fail or {'asm': ''})['asm'].strip()[:3000]}\n```"},
                               {"role": "user", "content": fb}]
        final = None
        if best_ok:
            a = accept3(task, best_ok["asm"], n=N_FINAL, seed=7)
            if a["accepted"]:
                final = best_ok
        win = bool(final) and (final[key] < bar["size"] if goal == "size" else final[key] <= 0.9 * bar["insn"])
        uniq = win and (task.id, goal) not in search_win
        unique += uniq
        row = {"split": srow["split"], "task_id": task.id, "goal": goal, "bar": bar[key], "tallies": tallies, "best": final,
               "best_val": final[key] if final else None, "win": win, "unique": uniq, "seconds": round(time.time() - ts, 1)}
        append_jsonl(out, row)
        log(f"  {srow['split']} {task.id} {goal}: bar={bar[key]} best={row['best_val']} win={win} unique={uniq} {tallies}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nearmiss")
    e = sub.add_parser("edits")
    e.add_argument("--max-hours", type=float, default=1.0); e.add_argument("--k", type=int, default=8)
    e.add_argument("--rounds", type=int, default=1); e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--max-new", type=int, default=1536); e.add_argument("--no-unique-stop-h", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.cmd == "nearmiss":
        print(len(near_misses("heldout")))
        return 0
    return edits(args)


if __name__ == "__main__":
    sys.exit(main())
