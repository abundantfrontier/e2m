"""Round 8 CPU stages (no torch; CUDA is never touched).

  python -m e2m.loop.round8 gate      # A: checker v3 gate (19 known-wrong round-7 wins, 10 equivalent pairs)
  python -m e2m.loop.round8 revalidate  # B: every round-7 win and bar through v3; erratum
  python -m e2m.loop.round8 search    # C: search with new moves, restarts, annealing
  python -m e2m.loop.round8 summary   # E: final 5,000-input v3 recheck, held-back audit, llvm-mca, summary
Outputs go to runs/round8 (E2M_R8_OUT overrides, for dry runs). Resume-safe; exit 3 = paused by the hold file.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from e2m.harness.assemble import AssembleError
from e2m.loop.round7_common import (COMPILERS, ROOT, SPEED_COMPILERS, append_jsonl, compile_c, measure,
                                    pass_set_candidates, read_jsonl, strip_asm, CLANG)
from e2m.loop.round7_cpu import RULES, mutate as mutate_r7, _is_insn
from e2m.loop.round8_common import (HOLD, N_FINAL, N_V3, OUT, R7, accept3, checker, consensus_images, round7_wins,
                                    split_of, tasks_by_id)

PT = ZoneInfo("America/Los_Angeles")
MCA = CLANG.parent / "llvm-mca"
ISA_FLAGS = ["-mpopcnt", "-mlzcnt", "-mbmi"]   # opt-in extension set (popcnt is x86-64-v2; lzcnt/tzcnt are ABM/BMI1)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


def held() -> bool:
    return HOLD.is_file()


def _pool(workers):
    return ProcessPoolExecutor(workers)


# ================================================================ A: gate
def _gate_job(job):
    from e2m.loop import round8_common as _rc
    _rc._CHECKERS.clear()
    kind, tid, asm, meta = job
    t = tasks_by_id()[tid]
    t0 = time.time()
    a = accept3(t, asm, n=N_V3, seed=1)
    return {"kind": kind, "task_id": tid, **meta, "accepted": a["accepted"], "stage": a["stage"],
            "detail": (a.get("v3_detail") or a.get("fail") or "")[:200], "v3_cases": a.get("v3_cases"),
            "edge_coverage": a.get("edge_coverage"), "edges": a.get("edges"), "pool": a.get("pool"),
            "ub_dropped": a.get("ub_dropped"), "notes": a.get("notes"), "seconds": round(time.time() - t0, 1)}


def gate(args) -> int:
    tasks = tasks_by_id()
    jobs = []
    for w in round7_wins():
        kind = "known_wrong" if w["known_wrong"] else "audited_win"
        jobs.append((kind, w["task_id"], w["asm"], {"src": w["src"], "goal": w["goal"], "why": w["why"]}))
    eq_ids = list(tasks)[:10]
    for i, tid in enumerate(eq_ids):
        comp, opt = (("gcc", "-Os"), ("clang", "-Oz"), ("gcc", "-O3"), ("clang", "-O2"))[i % 4]
        jobs.append(("equivalent", tid, compile_c(tasks[tid].reference_c, comp, opt), {"cand": f"{comp} {opt} of reference C"}))
    log(f"gate: {len(jobs)} checks ({sum(j[0]=='known_wrong' for j in jobs)} known-wrong, "
        f"{sum(j[0]=='equivalent' for j in jobs)} equivalent, {sum(j[0]=='audited_win' for j in jobs)} audited wins), workers={args.workers}")
    rows = []
    with _pool(args.workers) as ex:
        for r in ex.map(_gate_job, jobs):
            rows.append(r)
            log(f"  {r['kind']} {r['task_id']} {r.get('goal','')} accepted={r['accepted']} {r['detail'][:90]}")
    kw = [r for r in rows if r["kind"] == "known_wrong"]
    eq = [r for r in rows if r["kind"] == "equivalent"]
    au = [r for r in rows if r["kind"] == "audited_win"]
    res = {"checker": "v3", "n": N_V3, "known_wrong_rejected": sum(not r["accepted"] for r in kw), "n_known_wrong": len(kw),
           "equivalent_agreed": sum(r["accepted"] for r in eq), "n_equivalent": len(eq),
           "audited_wins_accepted": sum(r["accepted"] for r in au), "n_audited_wins": len(au),
           "audited_wins_rejected": [{k: r.get(k) for k in ("src", "task_id", "goal", "detail", "notes")} for r in au if not r["accepted"]],
           "median_edges": st.median([r["edges"] or 0 for r in rows]) if rows else None,
           "rows": rows}
    res["pass"] = res["known_wrong_rejected"] == len(kw) == 19 and res["equivalent_agreed"] == len(eq) == 10
    (OUT / "diff_selftest.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"gate: known-wrong rejected {res['known_wrong_rejected']}/{len(kw)}, equivalent agreed {res['equivalent_agreed']}/{len(eq)}, "
        f"audited wins still accepted {res['audited_wins_accepted']}/{len(au)} -> pass={res['pass']}")
    return 0


# ================================================================ B: re-validate round 7
def _reval_task(job):
    from e2m.loop import round8_common as _rc
    _rc._CHECKERS.clear()
    split, tid, port, wins = job
    t = tasks_by_id()[tid]
    cands = pass_set_candidates().get(tid, [])
    sources = [{"src": "reference", "c": t.reference_c}] + cands
    out = {"split": split, "task_id": tid}
    for goal, bkey, key in (("size", "best_size", "size"), ("speed", "best_speed", "insn")):
        old = port.get(bkey)
        if not old:
            out[goal] = None
            continue
        a = accept3(t, old["asm"], n=N_V3, seed=2)
        if a["accepted"]:
            out[goal] = {**old, "replaced": False, "v3_cases": a.get("v3_cases"), "edge_coverage": a.get("edge_coverage")}
            continue
        allowed = (lambda v: True) if goal == "size" else (lambda v: (v["compiler"], v["opt"]) in SPEED_COMPILERS)
        vs = [v for v in port["variants"] if v.get("passed") and allowed(v)]
        vs.sort(key=lambda v: (v[key], v["size" if key == "insn" else "insn"]))
        new = None
        for v in vs:
            if v["c_idx"] == old["c_idx"] and v["compiler"] == old["compiler"] and v["opt"] == old["opt"]:
                continue
            try:
                asm = compile_c(sources[v["c_idx"]]["c"], v["compiler"], v["opt"])
            except (AssembleError, IndexError):
                continue
            b = accept3(t, asm, n=N_V3, seed=2)
            if b["accepted"]:
                new = {"size": b["size"], "insn": b["insn"], "compiler": v["compiler"], "opt": v["opt"],
                       "c_src": v["c_src"], "c_idx": v["c_idx"], "asm": asm}
                break
        out[goal] = ({**new, "replaced": True, "old": {k: old[k] for k in ("size", "insn", "compiler", "opt", "c_src", "c_idx")},
                      "old_v3_detail": a.get("v3_detail") or a.get("fail")} if new else
                     {"replaced": True, "none_accepted": True, "old": {k: old[k] for k in ("size", "insn", "compiler", "opt", "c_src", "c_idx")},
                      "old_v3_detail": a.get("v3_detail") or a.get("fail")})
    wr = []
    for w in wins:
        a = accept3(t, w["asm"], n=N_V3, seed=3)
        wr.append({"src": w["src"], "goal": w["goal"], "known_wrong": w["known_wrong"], "why": w["why"], "v3_ok": a["accepted"],
                   "detail": (a.get("v3_detail") or a.get("fail") or "")[:200], "size": a.get("size"), "insn": a.get("insn"),
                   "notes": a.get("notes"), "asm": w["asm"]})
    out["r7_wins"] = wr
    return out


def revalidate(args) -> int:
    out = OUT / "bars.jsonl"
    done = {r["task_id"] for r in read_jsonl(out)}
    port = read_jsonl(R7 / "portfolio.jsonl")
    wins: dict[str, list] = {}
    for w in round7_wins():
        wins.setdefault(w["task_id"], []).append(w)
    jobs = [(r["split"], r["task_id"], r, wins.get(r["task_id"], [])) for r in port if r["task_id"] not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    log(f"revalidate: {len(jobs)} tasks pending ({len(done)} done), workers={args.workers}")
    t0 = time.time()
    with _pool(args.workers) as ex:
        futs = {ex.submit(_reval_task, j): j[1] for j in jobs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  {futs[f]}: error {exc!r}"); continue
            append_jsonl(out, row)
            flag = [g for g in ("size", "speed") if (row.get(g) or {}).get("replaced")]
            log(f"  {row['split']} {row['task_id']}: bars replaced={flag or '-'} r7 wins v3_ok="
                f"{[(w['src'], w['goal'], w['v3_ok']) for w in row['r7_wins']]}")
            if held():
                log("revalidate: hold requested; stopping"); [g.cancel() for g in futs]; return 3
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                log("revalidate: time cap reached; stopping"); [g.cancel() for g in futs]; break
    write_erratum()
    return 0


def write_erratum() -> dict[str, Any]:
    rows = read_jsonl(OUT / "bars.jsonl")
    rep = {"tasks": len(rows), "bars_replaced": [], "bars_without_accepted_compile": [], "r7_wins": {}}
    for r in rows:
        for g in ("size", "speed"):
            b = r.get(g) or {}
            if b.get("replaced"):
                (rep["bars_without_accepted_compile"] if b.get("none_accepted") else rep["bars_replaced"]).append(
                    {"split": r["split"], "task_id": r["task_id"], "goal": g, "old": b.get("old"),
                     "new": {k: b.get(k) for k in ("size", "insn", "compiler", "opt", "c_src")} if not b.get("none_accepted") else None,
                     "why": (b.get("old_v3_detail") or "")[:160]})
    for split in ("heldout", "val"):
        for goal in ("size", "speed"):
            ws = [(r, w) for r in rows if r["split"] == split for w in r["r7_wins"] if w["goal"] == goal and w["src"] in ("model", "search")]
            # a round-7 win stands if v3 accepts it and it still beats the (possibly corrected) bar
            def stands(r, w):
                b = r.get(goal) or {}
                if not w["v3_ok"] or b.get("none_accepted"):
                    return False
                bar = b.get("size" if goal == "size" else "insn")
                v = w["size" if goal == "size" else "insn"]
                return bar is not None and v is not None and (v < bar if goal == "size" else v <= 0.9 * bar)
            ok = {(r["task_id"]) for r, w in ws if stands(r, w)}
            rep["r7_wins"][f"{split}/{goal}"] = {"claimed_task_goals": len({r["task_id"] for r, w in ws}),
                                                 "stand_under_v3": len(ok), "tasks": sorted(ok),
                                                 "v3_rejects": sorted({f"{w['src']}:{r['task_id']}" for r, w in ws if not w["v3_ok"]})}
    (OUT / "erratum_round7.json").write_text(json.dumps(rep, indent=1))
    lines = ["# Round 7 erratum (round 8 stage B, checker v3)", "",
             f"Every round-7 bar (best of five compiler settings) and every claimed win was re-checked with checker v3 "
             f"(tests + {N_V3} coverage-guided, UB-filtered inputs). Tasks checked: {rep['tasks']}.", ""]
    lines += ["## Bars replaced", ""]
    if rep["bars_replaced"]:
        lines += ["| Set | Task | Goal | Old bar | New bar | Why the old one failed |", "|---|---|---|---|---|---|"]
        for b in rep["bars_replaced"]:
            o, n = b["old"], b["new"]
            k = "size" if b["goal"] == "size" else "insn"
            lines.append(f"| {b['split']} | {b['task_id']} | {b['goal']} | {o[k]} ({o['compiler']} {o['opt']}, {o['c_src']}) | "
                         f"{n[k]} ({n['compiler']} {n['opt']}, {n['c_src']}) | {b['why'][:80]} |")
    else:
        lines.append("None: every round-7 bar passes v3.")
    if rep["bars_without_accepted_compile"]:
        lines += ["", "Bars with no accepted compile (task dropped from round 8): " +
                  ", ".join(f"{b['task_id']} {b['goal']}" for b in rep["bars_without_accepted_compile"])]
    lines += ["", "## Round-7 wins under v3", "", "| Set/goal | Claimed task-goals | Stand under v3 | Rejected by v3 |", "|---|---:|---:|---|"]
    for k, v in rep["r7_wins"].items():
        lines.append(f"| {k} | {v['claimed_task_goals']} | {v['stand_under_v3']} | {', '.join(v['v3_rejects']) or '—'} |")
    (OUT / "erratum_round7.md").write_text("\n".join(lines) + "\n")
    return rep


# ================================================================ C: search
REG32 = {"rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx", "rsi": "esi", "rdi": "edi", "rbp": "ebp",
         "r8": "r8d", "r9": "r9d", "r10": "r10d", "r11": "r11d", "r12": "r12d", "r13": "r13d", "r14": "r14d", "r15": "r15d"}
REG64 = {v: k for k, v in REG32.items()}
FLAG_SET = re.compile(r"^\s*(add|sub|and|or|xor|cmp|test|inc|dec|neg|shl|shr|sar|sal|imul)[lqbw]?\s")
FLAG_USE = re.compile(r"^\s*(j(?!mp)\w+|set\w+|cmov\w+|adc\w*|sbb\w*)\s")
NEW_RULES = [
    # one-operand mul (clobbers rdx; the checker decides whether that matters)
    (re.compile(r"\bimulq\s+%(r\w+),\s*%rax\s*$"), lambda m: f"mulq\t%{m.group(1)}"),
    (re.compile(r"\bimull\s+%(e\w+|r\d+d),\s*%eax\s*$"), lambda m: f"mull\t%{m.group(1)}"),
    (re.compile(r"\bimulq\s+%rax,\s*%(r\w+)\s*$"), lambda m: f"mulq\t%{m.group(1)}"),
]
ISA_RULES = [
    (re.compile(r"\bbsf([lq])\b"), lambda m: f"tzcnt{m.group(1)}"),
    (re.compile(r"\brep\s+bsf([lq])\b"), lambda m: f"tzcnt{m.group(1)}"),
]


def _ins(lines, i):
    return lines[i].split("#", 1)[0].strip()


def mutate8(lines: list[str], rng: random.Random, isa: bool) -> list[str] | None:
    idx = [i for i, l in enumerate(lines) if _is_insn(l)]
    if not idx:
        return None
    r = rng.random()
    if r < 0.5:
        return mutate_r7(lines, rng)
    new = list(lines)
    r = rng.random()
    if r < 0.18:  # push/pop a small constant (3 bytes) instead of mov $imm (5-7 bytes)
        hits = [i for i in idx if re.match(r"^\s*mov[lq]\s+\$(-?\d+),\s*%(\w+)\s*$", new[i])]
        if not hits:
            return None
        i = rng.choice(hits)
        m = re.match(r"^\s*mov([lq])\s+\$(-?\d+),\s*%(\w+)\s*$", new[i])
        imm, reg = int(m.group(2)), m.group(3)
        reg64 = REG64.get(reg, reg)
        if not (-128 <= imm <= 127) or reg64 not in REG32 or (m.group(1) == "l" and imm < 0):
            return None
        new[i:i + 1] = [f"\tpushq\t${imm}", f"\tpopq\t%{reg64}"]
    elif r < 0.36:  # lea folding: mov a,b ; add c|$k,b  ->  lea
        hits = [i for i in idx if re.match(r"^\s*mov([lq])\s+%(\w+),\s*%(\w+)\s*$", new[i]) and i + 1 < len(new)]
        rng.shuffle(hits)
        for i in hits:
            m = re.match(r"^\s*mov([lq])\s+%(\w+),\s*%(\w+)\s*$", new[i])
            w, a, b = m.groups()
            n = re.match(rf"^\s*add{w}\s+(%\w+|\$-?\d+),\s*%{b}\s*$", new[i + 1])
            if not n:
                continue
            c = n.group(1)
            if w == "l":
                a64, c64 = REG64.get(a, a), (REG64.get(c[1:], c[1:]) if c.startswith("%") else None)
                mem = f"{c[1:]}(%{a64})" if c.startswith("$") else f"(%{a64},%{c64})"
                new[i:i + 2] = [f"\tleal\t{mem}, %{b}"]
            else:
                mem = f"{c[1:]}(%{a})" if c.startswith("$") else f"(%{a},{c})"
                new[i:i + 2] = [f"\tleaq\t{mem}, %{b}"]
            break
        else:
            return None
    elif r < 0.5:  # register / constant reuse
        consts = {}
        hits = []
        for i in idx:
            m = re.match(r"^\s*mov([lq])\s+\$(-?\d+),\s*%(\w+)\s*$", new[i])
            if m:
                k = (m.group(1), m.group(2))
                if k in consts and consts[k] != m.group(3):
                    hits.append((i, consts[k], m.group(1), m.group(3)))
                consts[k] = m.group(3)
        if not hits:
            return None
        i, src, w, dst = rng.choice(hits)
        new[i] = f"\tmov{w}\t%{src}, %{dst}"
    elif r < 0.68:  # dead store: a write to %R immediately overwritten by a mov that does not read %R
        hits = []
        for a, b in zip(idx, idx[1:]):
            m = re.match(r"^\s*(mov\w*|lea\w*|xor[lq])\s+(.+),\s*%(\w+)\s*$", new[a])
            n = re.match(r"^\s*(mov\w*|lea\w*)\s+(.+),\s*%(\w+)\s*$", new[b])
            if m and n and m.group(3) == n.group(3) and m.group(3) not in n.group(2) and \
                    REG64.get(m.group(3), m.group(3)) not in n.group(2):
                hits.append(a)
        if not hits:
            return None
        del new[rng.choice(hits)]
    elif r < 0.84:  # dead flags: test/cmp whose flags are overwritten before any use
        hits = []
        for k, a in enumerate(idx):
            if not re.match(r"^\s*(test|cmp)[lqbw]?\s", new[a]):
                continue
            for b in idx[k + 1:]:
                s = new[b]
                if FLAG_USE.match(s) or s.strip().startswith(("ret", "jmp", "call")):
                    break
                if FLAG_SET.match(s):
                    hits.append(a); break
        if not hits:
            return None
        del new[rng.choice(hits)]
    else:
        rules = NEW_RULES + (ISA_RULES if isa else []) + RULES
        hits = [(i, rule) for i in idx for rule in rules if rule[0].search(new[i])]
        if not hits:
            return None
        i, (pat, fn) = rng.choice(hits)
        rep = pat.sub(fn, new[i], count=1)
        if rep == new[i]:
            return None
        new[i] = rep
    return new


def _starts(t, bar: dict[str, Any], isa: bool) -> list[tuple[str, str]]:
    cands = pass_set_candidates().get(t.id, [])
    sources = [{"src": "reference", "c": t.reference_c}] + cands
    c = sources[bar["c_idx"]]["c"] if bar.get("c_idx") is not None and bar["c_idx"] < len(sources) else t.reference_c
    out = []
    for comp, opt in COMPILERS:
        try:
            asm = compile_c(c, comp, opt)
            if isa:
                asm2 = _compile_isa(c, comp, opt)
                if asm2 and asm2 != asm:
                    out.append((f"{comp} {opt} {' '.join(ISA_FLAGS)}", asm2))
                continue
            out.append((f"{comp} {opt}", asm))
        except AssembleError:
            pass
    if not isa and bar.get("asm"):
        out.append(("bar", bar["asm"]))
    return out


def _compile_isa(c: str, comp: str, opt: str) -> str | None:
    from e2m.loop import round7_common as rc
    from e2m.harness.tasks import C_PREAMBLE, GCC_COMMON
    body = c if "#include" in c else C_PREAMBLE + c
    with tempfile.TemporaryDirectory(prefix="e2m-r8-") as td:
        src, asm = Path(td) / "f.c", Path(td) / "f.s"
        src.write_text(body)
        exe = "gcc" if comp == "gcc" else str(CLANG)
        cmd = [exe, opt, "-S", *GCC_COMMON, *ISA_FLAGS, "-o", str(asm), str(src)] + (["-w"] if comp == "clang" else [])
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if p.returncode != 0 or not asm.is_file():
            return None
        text = asm.read_text()
    return rc._clean_clang(text) if comp == "clang" else text


def _search_job(job: dict[str, Any]) -> dict[str, Any]:
    t = tasks_by_id()[job["task_id"]]
    # memory: keep only this task's v3 pool in the worker (the first run grew to ~28G with 12 workers caching every task)
    from e2m.loop import round8_common as _rc
    import gc
    for k in [k for k in _rc._CHECKERS if not k.startswith(t.id + ":")]:
        del _rc._CHECKERS[k]
    gc.collect()
    goal, isa, budget_s = job["goal"], job["isa"], job["budget_s"]
    key = "size" if goal == "size" else "insn"
    bar = job["bar"]
    barv = bar[key]
    rng = random.Random(f"r8:{t.id}:{goal}:{isa}:{job['seed']}")
    t0 = time.time()
    starts = [(n, strip_asm(a)) for n, a in _starts(t, bar, isa)] + [(f"r7-win-{i}", strip_asm(a)) for i, a in enumerate(job.get("extra", []))]
    measured = []
    for name, a in starts:
        m = measure(t, a)
        if m["passed"]:
            measured.append((name, a.splitlines(), m[key]))
    if not measured:
        return {**_rowhead(job), "error": "no start passes tests", "win": False}
    checker(t)  # build the v3 pool once (counts in this job's time)
    measured.sort(key=lambda x: x[2])
    best_name, best_lines, best_val = measured[0]
    cur_lines, cur = best_lines, best_val
    accepted_best: dict[str, Any] | None = None
    tried = passed = proposals = v3_rejects = 0
    rejected_below_bar: dict[str, Any] | None = None
    seen: set[int] = set()
    n_restart = 0
    restart_every = max(30.0, budget_s / 8)
    last_restart = time.time()
    win_thr = barv if goal == "size" else 0.9 * barv
    while time.time() - t0 < budget_s:
        if held():
            return {**_rowhead(job), "interrupted": True}
        frac = (time.time() - t0) / budget_s
        T = max(1e-6, 0.03 * max(1, cur) * (1 - frac))
        cand = mutate8(cur_lines, rng, isa)
        if cand is not None and rng.random() < 0.3:
            cand = mutate8(cand, rng, isa) or cand
        if cand is None:
            continue
        h = hash("\n".join(cand))
        if h in seen:
            continue
        seen.add(h)
        tried += 1
        text = "\n".join(cand) + "\n"
        m = measure(t, text, timeout_s=0.2)
        if not m["passed"]:
            continue
        passed += 1
        v = m[key]
        if v < best_val:
            proposals += 1
            q = accept3(t, text, n=N_V3, seed=proposals, quick=300)
            if q["accepted"]:
                q = accept3(t, text, n=N_V3, seed=proposals)
            if not q["accepted"]:
                v3_rejects += 1
                if (v < barv if goal == "size" else v <= win_thr) and (rejected_below_bar is None or v < rejected_below_bar["val"]):
                    rejected_below_bar = {"val": v, "asm": text, "detail": (q.get("v3_detail") or "")[:200],
                                          "counterexample": json.dumps(q.get("counterexample"), default=str)[:400]}
                continue
            best_lines, best_val = cand, v
            cur_lines, cur = cand, v
            accepted_best = {"asm": text, "size": m["size"], "insn": m["insn"]}
        elif v <= cur or rng.random() < math.exp(-(v - cur) / T):
            cur_lines, cur = cand, v  # annealing: worse moves accepted with falling temperature
        if time.time() - last_restart > restart_every:
            n_restart += 1
            last_restart = time.time()
            _, cur_lines, cur = rng.choice(measured) if rng.random() < 0.5 else (None, best_lines, best_val)
    final = None
    if accepted_best:
        a = accept3(t, accepted_best["asm"], n=N_FINAL, seed=99)
        if a["accepted"]:
            final = {**accepted_best, "edge_coverage": a.get("edge_coverage"), "v3_cases": a.get("v3_cases")}
    win = bool(final) and (final[key] < barv if goal == "size" else final[key] <= 0.9 * barv)
    near = (not win) and (rejected_below_bar is not None or (goal == "speed" and final is not None and final[key] <= 0.99 * barv))
    return {**_rowhead(job), "bar": barv, "bar_from": f"{bar.get('compiler')} {bar.get('opt')}", "start": measured[0][2],
            "start_from": measured[0][0], "n_starts": len(measured), "tried": tried, "passed_tests": passed,
            "proposals": proposals, "v3_rejects": v3_rejects, "restarts": n_restart, "best": final,
            "best_val": final[key] if final else None, "win": win, "near_miss": near, "rejected_below_bar": rejected_below_bar,
            "improved_over_start": bool(final) and final[key] < measured[0][2], "seconds": round(time.time() - t0, 1)}


def _rowhead(job):
    return {"split": job["split"], "task_id": job["task_id"], "goal": job["goal"], "isa": "ext" if job["isa"] else "base"}


def search_jobs() -> list[dict[str, Any]]:
    bars = read_jsonl(OUT / "bars.jsonl")
    order = {t: i for i, t in enumerate(tasks_by_id())}
    jobs = []
    for r in bars:
        survivors = {}
        for w in r["r7_wins"]:
            if w["v3_ok"]:
                survivors.setdefault(w["goal"], []).append(w["asm"])
        for goal in ("size", "speed"):
            b = r.get(goal)
            if not b or b.get("none_accepted"):
                continue
            jobs.append({"split": r["split"], "task_id": r["task_id"], "goal": goal, "isa": False, "bar": b,
                         "extra": survivors.get(goal, [])})
    jobs.sort(key=lambda j: (order.get(j["task_id"], 1e9), j["goal"]))
    return jobs


def search(args) -> int:
    out = OUT / "search.jsonl"
    done = {(r["task_id"], r["goal"], r["isa"]) for r in read_jsonl(out)}
    used_f = OUT / "stages" / "search_used_s.txt"
    used = float(used_f.read_text()) if used_f.is_file() else 0.0
    base = search_jobs()
    isa_jobs = []
    if args.isa:
        for j in base:
            t = tasks_by_id()[j["task_id"]]
            if any(n != "bar" for n, _ in _starts(t, j["bar"], True)):
                isa_jobs.append({**j, "isa": True, "extra": []})
    jobs = [j for j in base + isa_jobs if (j["task_id"], j["goal"], "ext" if j["isa"] else "base") not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    total_s = max(0.0, args.max_hours * 3600 - used)
    per = max(20.0, min(args.per_task_s, total_s * args.workers / max(1, len(jobs)) * 0.92))
    log(f"search: {len(jobs)} jobs pending ({len(done)} done; {len(isa_jobs)} ISA-extension jobs), workers={args.workers}, "
        f"per-job {per:.0f}s, wall left {total_s/3600:.2f} h")
    t0 = time.time()
    wins = 0
    rc = 0
    with _pool(args.workers) as ex:
        futs = {ex.submit(_search_job, {**j, "budget_s": per, "seed": args.seed}): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                row = f.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  {j['task_id']} {j['goal']}: error {exc!r}"); continue
            if row.get("interrupted"):
                rc = 3
                continue
            append_jsonl(out, row)
            wins += bool(row.get("win"))
            log(f"  {row['split']} {row['task_id']} {row['goal']} [{row['isa']}]: bar={row.get('bar')} start={row.get('start')} "
                f"best={row.get('best_val')} win={row.get('win')} near={row.get('near_miss')} tried={row.get('tried')} (wins {wins})")
            if held() and rc == 0:
                log("search: hold requested; cancelling queued jobs, running ones stop early")
                for g in futs:
                    g.cancel()
                rc = 3
            if time.time() - t0 > total_s + per + 120:
                log("search: wall cap reached; cancelling the rest")
                for g in futs:
                    g.cancel()
                break
    used_f.parent.mkdir(parents=True, exist_ok=True)
    used_f.write_text(str(used + time.time() - t0))
    return rc


# ================================================================ E: audit + summary
def mca_cycles(asm: str, cpu: str) -> int | None:
    body = "\n".join(l for l in strip_asm(asm).splitlines() if not l.strip().startswith((".globl", ".p2align", ".align", ".text", ".section")))
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write(body + "\n")
        p = f.name
    try:
        r = subprocess.run([str(MCA), f"-mcpu={cpu}", "-iterations=100", p], capture_output=True, text=True, timeout=60)
        m = re.search(r"Total Cycles:\s+(\d+)", r.stdout)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        Path(p).unlink(missing_ok=True)


def _audit_job(job):
    from e2m.loop import round8_common as _rc
    _rc._CHECKERS.clear()
    from e2m.harness.diff_holdout import holdout_check
    from e2m.loop.round7_common import oracle_image
    src, split, tid, goal, asm, bar = job
    t = tasks_by_id()[tid]
    a = accept3(t, asm, n=N_FINAL, seed=424242)
    h = holdout_check(t, asm, oracle_image(t), n=N_FINAL, seed=777, consensus=consensus_images(t))
    mca = {}
    for cpu in ("znver3", "skylake"):
        mca[cpu] = {"cand": mca_cycles(asm, cpu), "bar": mca_cycles(bar["asm"], cpu)}
    mca_ok = all(v["cand"] is not None and v["bar"] is not None and v["cand"] < v["bar"] for v in mca.values())
    return {"src": src, "split": split, "task_id": tid, "goal": goal, "size": a.get("size"), "insn": a.get("insn"),
            "bar": bar["size" if goal == "size" else "insn"], "recheck_ok": a["accepted"],
            "recheck_detail": (a.get("v3_detail") or a.get("fail") or "")[:200], "edge_coverage": a.get("edge_coverage"),
            "holdout_ok": h.ok, "holdout_cases": h.n_cases, "holdout_detail": h.detail[:200],
            "holdout_counterexample": json.dumps(h.counterexample, default=str)[:300] if h.counterexample else None,
            "mca": mca, "mca_improved": mca_ok}


def summary(args) -> int:
    sel = json.loads((OUT / "diff_selftest.json").read_text()) if (OUT / "diff_selftest.json").is_file() else {}
    err = write_erratum() if (OUT / "bars.jsonl").is_file() else {}
    bars = {r["task_id"]: r for r in read_jsonl(OUT / "bars.jsonl")}
    search_rows = [r for r in read_jsonl(OUT / "search.jsonl")]
    edit_rows = read_jsonl(OUT / "edits.jsonl")
    gpu_h = sum(r["seconds"] for r in read_jsonl(OUT / "gpu_hours.jsonl") if r.get("gpu")) / 3600
    claimed = []
    for src, rows in (("search", [r for r in search_rows if r["isa"] == "base"]), ("search-isa", [r for r in search_rows if r["isa"] == "ext"]),
                      ("model", edit_rows)):
        for r in rows:
            if r.get("win") and r.get("best"):
                claimed.append((src, r["split"], r["task_id"], r["goal"], r["best"]["asm"], bars[r["task_id"]][r["goal"]]))
    log(f"summary: {len(claimed)} claimed wins -> 5,000-input v3 recheck + held-back audit + llvm-mca, workers={args.workers}")
    with _pool(args.workers) as ex:
        audit = list(ex.map(_audit_job, claimed))
    (OUT / "audit.json").write_text(json.dumps({"n": N_FINAL, "generator": "diff_holdout.py (held back; first used here)",
                                                "checked": len(audit), "recheck_failed": sum(not a["recheck_ok"] for a in audit),
                                                "holdout_failed": sum(a["recheck_ok"] and not a["holdout_ok"] for a in audit),
                                                "rows": audit}, indent=1))
    ok = lambda a: a["recheck_ok"] and a["holdout_ok"]  # noqa: E731
    res: dict[str, Any] = {"round": 8, "generated": datetime.now(tz=PT).isoformat(), "checker": "v3", "recheck_n": N_FINAL,
                           "selftest": {k: sel.get(k) for k in ("known_wrong_rejected", "n_known_wrong", "equivalent_agreed",
                                                                "n_equivalent", "audited_wins_accepted", "n_audited_wins", "pass")},
                           "erratum_round7": {k: err.get(k) for k in ("tasks", "r7_wins")} | {"bars_replaced": len(err.get("bars_replaced", []))} if err else {},
                           "compiler_beat": {}, "isa_extension": {}, "gpu_hours": round(gpu_h, 3)}
    cba: dict[str, Any] = {}
    for split in ("heldout", "val"):
        tids = [t for t, r in bars.items() if r["split"] == split]
        if not tids:
            continue
        blk = {"pass_set_tasks": len(tids)}
        cba[split] = {}
        for goal in ("size", "speed"):
            key = "size" if goal == "size" else "insn"
            g_claim = {src: {a["task_id"] for a in audit if a["split"] == split and a["goal"] == goal and a["src"] == src} for src in ("model", "search")}
            g_ok = {src: {a["task_id"] for a in audit if a["split"] == split and a["goal"] == goal and a["src"] == src and ok(a)} for src in ("model", "search")}
            either = g_ok["model"] | g_ok["search"]
            ratios = []
            for tid in either:
                vals = [a[key] for a in audit if a["task_id"] == tid and a["goal"] == goal and a["src"] in ("model", "search") and ok(a)]
                ratios.append(min(vals) / max(1, a_bar(bars, tid, goal)))
            mca_ok = {a["task_id"] for a in audit if a["split"] == split and a["goal"] == goal and ok(a) and a["mca_improved"] and a["src"] in ("model", "search")}
            blk[goal] = {"wins": {"model": len(g_ok["model"]), "search": len(g_ok["search"]), "both": len(either)},
                         "claimed": {"model": len(g_claim["model"]), "search": len(g_claim["search"])},
                         "win_tasks": {"model": sorted(g_ok["model"]), "search": sorted(g_ok["search"]), "both": sorted(either)},
                         "median_ratio": {"both": round(st.median(ratios), 3) if ratios else None},
                         "model_unique": sorted(g_ok["model"] - g_ok["search"]), "mca_improved_both_cpus": sorted(mca_ok)}
            cba[split][goal] = {"model": len(g_ok["model"]), "search": len(g_ok["search"]), "either": len(either),
                                "model_only": sorted(g_ok["model"] - g_ok["search"]), "both": sorted(g_ok["model"] & g_ok["search"]),
                                "median_ratio_of_wins": round(st.median(ratios), 3) if ratios else None,
                                "mca_improved_both_cpus": len(mca_ok)}
        res["compiler_beat"][split] = blk
    isa = [a for a in audit if a["src"] == "search-isa"]
    res["isa_extension"] = {"flags": ISA_FLAGS, "claimed": len(isa), "verified": sum(ok(a) for a in isa),
                            "tasks": sorted(f"{a['task_id']}:{a['goal']}" for a in isa if ok(a))}
    failed_claims = [a for a in audit if not ok(a)]
    res["audit"] = {"what": "held-back audit (round 8 stage E): every claimed win re-checked on 5,000 v3 inputs (fresh seed) "
                            "and 5,000 inputs from the held-back generator e2m/harness/diff_holdout.py, written before any "
                            "round-8 candidate and used only here; UB inputs (compilers of the reference C disagree) skipped.",
                    "checked": len(audit), "failed": len(failed_claims),
                    "failures": [{k: a[k] for k in ("src", "split", "task_id", "goal", "recheck_detail", "holdout_detail")} for a in failed_claims],
                    "compiler_beat_audited": cba}
    ho = cba.get("heldout", {})
    W = sum((ho.get(g) or {}).get("either", 0) for g in ("size", "speed"))
    mca5 = (ho.get("speed") or {}).get("mca_improved_both_cpus", 0)
    fails_ho = [a for a in failed_claims if a["split"] == "heldout" and a["src"] in ("model", "search")]
    gate_ok = bool(sel.get("pass"))
    model_unique = sum(len((res["compiler_beat"].get("heldout", {}).get(g) or {}).get("model_unique", [])) for g in ("size", "speed"))
    if not gate_ok:
        outcome = "Checker gate failed (stopped after stage B)"
    elif W >= 30 and not fails_ho and mca5 >= 5:
        outcome = "Win"
    elif W < 10:
        outcome = "No win"
    else:
        outcome = "Partial"
    res["decision"] = {"outcome": outcome, "heldout_verified_wins": W, "heldout_claims_failed": len(fails_ho),
                       "heldout_speed_wins_mca_both": mca5, "model_unique_wins": model_unique,
                       "model_unique_per_gpu_h": round(model_unique / gpu_h, 2) if gpu_h > 0.01 else None,
                       "model_worth_keeping": bool(gpu_h > 0.01 and model_unique / gpu_h >= 3),
                       "checker_gate": gate_ok}
    sr = [r for r in search_rows if r["isa"] == "base"]
    res["search_stats"] = {"jobs": len(sr), "tried": sum(r.get("tried", 0) for r in sr), "near_miss_heldout":
                           sum(1 for r in sr if r.get("near_miss") and r["split"] == "heldout"),
                           "v3_rejects": sum(r.get("v3_rejects", 0) for r in sr)}
    (OUT / "RESULTS.json").write_text(json.dumps(res, indent=1))
    _write_md(res, audit, sel, err)
    log(f"summary: outcome={outcome} heldout verified wins={W} claims failed={len(fails_ho)} mca5={mca5} gpu_h={gpu_h:.3f}")
    if not args.no_done:
        (ROOT / "logs" / "round8.done").write_text(datetime.now(tz=PT).isoformat() + "\n")
    return 0


def a_bar(bars, tid, goal):
    b = bars[tid][goal]
    return b["size" if goal == "size" else "insn"]


def _write_md(res, audit, sel, err) -> None:
    d = res["decision"]
    L = [f"# Round 8 summary", "", f"Generated {res['generated']} (PT). Checker v3; GPU hours {res['gpu_hours']}.", "",
         f"**Decision: {d['outcome']}.** Held-out verified wins (task-goals, either source): {d['heldout_verified_wins']}; "
         f"held-out claims that failed the recheck or held-back audit: {d['heldout_claims_failed']}; held-out speed wins that "
         f"also improve llvm-mca on znver3 and skylake: {d['heldout_speed_wins_mca_both']}; model-unique wins: {d['model_unique_wins']}.", "",
         "## Checker gate (stage A)", "",
         f"Known-wrong round-7 wins rejected: {sel.get('known_wrong_rejected')}/{sel.get('n_known_wrong')}; equivalent pairs agreed: "
         f"{sel.get('equivalent_agreed')}/{sel.get('n_equivalent')}; audited round-7 wins still accepted: "
         f"{sel.get('audited_wins_accepted')}/{sel.get('n_audited_wins')}. Gate pass: {sel.get('pass')}.", ""]
    if sel.get("audited_wins_rejected"):
        L.append("Audited round-7 wins v3 now rejects: " + "; ".join(f"{r['src']} {r['task_id']} {r['goal']} ({(r['detail'] or '')[:70]})" for r in sel["audited_wins_rejected"]))
        L.append("")
    L += ["## Round 7 erratum (stage B)", "", "See erratum_round7.md.", ""]
    for split, blk in res["compiler_beat"].items():
        L += [f"## {split}: beating the best compiler bar", "", "| Goal | Tasks | Model wins | Search wins | Either | Median ratio | llvm-mca better (both CPUs) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for g in ("size", "speed"):
            x = blk.get(g)
            if x:
                L.append(f"| {g} | {blk['pass_set_tasks']} | {x['wins']['model']} | {x['wins']['search']} | {x['wins']['both']} | "
                         f"{x['median_ratio']['both']} | {len(x['mca_improved_both_cpus'])} |")
        L.append("")
    L += ["## ISA extension (reported separately)", "", f"Flags {' '.join(res['isa_extension']['flags'])}: claimed {res['isa_extension']['claimed']}, "
          f"verified {res['isa_extension']['verified']}: {', '.join(res['isa_extension']['tasks']) or '—'}.", ""]
    L += ["## Held-back audit", "", f"{res['audit']['checked']} claimed wins checked; {res['audit']['failed']} failed.", ""]
    for f in res["audit"]["failures"]:
        L.append(f"- {f['src']} {f['split']} {f['task_id']} {f['goal']}: {f['recheck_detail'] or f['holdout_detail']}")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    fa = ["# Round 8 failure analysis", "", f"Search: {res['search_stats']}.", "",
          "Claims removed by the final checks:", ""]
    for a in audit:
        if not (a["recheck_ok"] and a["holdout_ok"]):
            fa.append(f"- {a['src']} {a['task_id']} {a['goal']}: recheck_ok={a['recheck_ok']} holdout_ok={a['holdout_ok']} "
                      f"{a['recheck_detail'] or a['holdout_detail']} {a.get('holdout_counterexample') or ''}")
    fa += ["", "Speed wins whose llvm-mca estimate is not better on both CPUs:", ""]
    for a in audit:
        if a["goal"] == "speed" and a["recheck_ok"] and a["holdout_ok"] and not a["mca_improved"]:
            fa.append(f"- {a['src']} {a['task_id']}: {json.dumps(a['mca'])}")
    (OUT / "failure_analysis.md").write_text("\n".join(fa) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "revalidate", "search", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=14)
        s.add_argument("--max-hours", type=float, default=1.0)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--per-task-s", type=float, default=3600.0)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--isa", action="store_true")
        s.add_argument("--no-done", action="store_true")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stages").mkdir(exist_ok=True)
    return {"gate": gate, "revalidate": revalidate, "search": search, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
