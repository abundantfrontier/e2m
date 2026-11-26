"""Round 7 CPU stages.

  python -m e2m.loop.round7_cpu selftest   # stage 0: diff checker unit test
  python -m e2m.loop.round7_cpu portfolio  # stage 1: compiler portfolio bars
  python -m e2m.loop.round7_cpu search     # stage 3: STOKE-lite search
No torch; CUDA is never touched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from e2m.harness.assemble import AssembleError, assemble
from e2m.harness.diff import diff_check
from e2m.harness.tasks import load_jsonl
from e2m.loop.round7_common import (COMPILERS, OUT, ROOT, SPEED_COMPILERS, N_ACCEPT, accept, append_jsonl,
                                    compile_c, interleaved_tasks, measure, oracle_image, pass_set_candidates,
                                    read_jsonl, strip_asm)
from e2m.loop.split import visible_tests

HOLD = Path(os.environ.get("E2M_HOLD_FILE", str(ROOT / "round7.hold")))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


# ---------------------------------------------------------------- stage 0
def selftest(args) -> int:
    tasks = {t.id: t for _, t in interleaved_tasks()}
    rows = []
    # 10 equivalent pairs: gcc -Os / clang -Oz of the reference C vs the oracle
    eq_ids = [tid for tid in list(tasks)[:40]][:10]
    for i, tid in enumerate(eq_ids):
        t = tasks[tid]
        comp, opt = (("gcc", "-Os"), ("clang", "-Oz"), ("gcc", "-O3"), ("clang", "-O2"))[i % 4]
        cand = compile_c(t.reference_c, comp, opt)
        d = diff_check(t, cand, oracle_image(t), n=args.n, seed=1, seed_tests=visible_tests(t), include_originals=False)
        rows.append({"kind": "equivalent", "task_id": tid, "cand": f"{comp} {opt} of reference C", "ok": d.ok,
                     "n_cases": d.n_cases, "skipped": d.n_skipped, "detail": d.detail, "expect": True})
    # 10 different pairs: round-6 loop programs that passed visible tests but failed hidden ones
    diffs = []
    for p in ("runs/loop-c-only-7b-heldout/loop.jsonl", "runs/loop-c-only-7b-val/loop.jsonl"):
        for r in read_jsonl(ROOT / p):
            for a in r["attempts"]:
                if a.get("ok_visible") and not a.get("ok_full") and r["task_id"] in tasks:
                    if all(x[0] != r["task_id"] for x in diffs):
                        diffs.append((r["task_id"], a))
    for tid, a in diffs[:args.n_diff]:
        t = tasks[tid]
        from e2m.train.format import extract_c
        try:
            cand = compile_c(extract_c(a["text"]), "gcc", "-O2")
        except AssembleError as exc:
            rows.append({"kind": "different", "task_id": tid, "ok": None, "detail": f"compile: {exc}", "expect": False})
            continue
        # visible tests only as seeds, originals excluded: the checker never sees the hidden tests
        d = diff_check(t, cand, oracle_image(t), n=args.n, seed=1, seed_tests=visible_tests(t), include_originals=False)
        rows.append({"kind": "different", "task_id": tid, "cand": f"r6 loop call {a.get('call')} (visible-only pass)",
                     "ok": d.ok, "n_cases": d.n_cases, "skipped": d.n_skipped, "detail": d.detail, "expect": False,
                     "counterexample": d.counterexample})
    # hand-kept adversarial cases (from early stage-3 dry runs)
    extra = json.loads((Path(__file__).with_name("round7_selftest_cases.json")).read_text())
    for c in extra:
        t = tasks[c["task_id"]]
        d = diff_check(t, c["asm"], oracle_image(t), n=args.n, seed=1, seed_tests=visible_tests(t), include_originals=False)
        rows.append({"kind": "adversarial" if not c["expect"] else "control", "task_id": t.id, "cand": c["why"], "ok": d.ok,
                     "n_cases": d.n_cases, "skipped": d.n_skipped, "detail": d.detail, "expect": c["expect"]})
    out = OUT / "diff_selftest.json"
    eq_ok = sum(1 for r in rows if r["kind"] == "equivalent" and r["ok"] is True)
    df_ok = sum(1 for r in rows if r["kind"] == "different" and r["ok"] is False)
    n_eq = sum(1 for r in rows if r["kind"] == "equivalent")
    n_df = sum(1 for r in rows if r["kind"] == "different")
    res = {"n": args.n, "equivalent_agreed": eq_ok, "n_equivalent": n_eq, "different_caught": df_ok,
           "n_different": n_df, "extra_as_expected": sum(1 for r in rows if r["kind"] in ("adversarial", "control") and r["ok"] is r["expect"]),
           "n_extra": len(extra), "pass": all(r["ok"] is r["expect"] for r in rows), "rows": rows}
    out.write_text(json.dumps(res, indent=1, default=str))
    log(f"selftest: equivalent agreed {eq_ok}/{n_eq}, different caught {df_ok}/{n_df}, "
        f"extra as expected {res['extra_as_expected']}/{len(extra)} -> {out}")
    for r in rows:
        if r["ok"] is not r["expect"]:
            log(f"  unexpected: {r['kind']} {r['task_id']} ok={r['ok']} {r.get('detail','')}")
    return 0


# ---------------------------------------------------------------- stage 1
def _portfolio_task(split: str, tid: str, cands: list[dict[str, str]]) -> dict[str, Any]:
    tasks = {t.id: t for s, t in interleaved_tasks()}
    t = tasks[tid]
    oracle = oracle_image(t)
    variants = []
    sources = [{"src": "reference", "c": t.reference_c}] + cands
    for si, c in enumerate(sources):
        for comp, opt in COMPILERS:
            try:
                asm = compile_c(c["c"], comp, opt)
            except AssembleError as exc:
                variants.append({"c_src": c["src"], "c_idx": si, "compiler": comp, "opt": opt, "passed": False,
                                 "fail": f"compile: {str(exc)[:100]}"})
                continue
            m = measure(t, asm)
            variants.append({"c_src": c["src"], "c_idx": si, "compiler": comp, "opt": opt, "asm": asm, **m})
    accepted_cache: dict[int, dict[str, Any]] = {}

    def acc(i: int) -> bool:
        if i not in accepted_cache:
            v = accepted_cache[i] = accept(t, variants[i]["asm"], oracle, n=N_ACCEPT)
            variants[i]["diff_ok"] = bool(v["accepted"])
            variants[i]["diff_detail"] = v.get("diff_detail", "")
        return bool(accepted_cache[i]["accepted"])

    def best(key: str, allowed) -> int | None:
        idx = [i for i, v in enumerate(variants) if v.get("passed") and allowed(v)]
        for i in sorted(idx, key=lambda i: (variants[i][key], variants[i]["size" if key == "insn" else "insn"])):
            if acc(i):
                return i
        return None

    bs = best("size", lambda v: True)
    bsp = best("insn", lambda v: (v["compiler"], v["opt"]) in SPEED_COMPILERS)
    bs_ref = best("size", lambda v: v["c_src"] == "reference")
    bsp_ref = best("insn", lambda v: v["c_src"] == "reference" and (v["compiler"], v["opt"]) in SPEED_COMPILERS)
    bs_ps = best("size", lambda v: v["c_idx"] == 1)
    bsp_ps = best("insn", lambda v: v["c_idx"] == 1 and (v["compiler"], v["opt"]) in SPEED_COMPILERS)
    o2 = next((v for v in variants if v["c_src"] == "reference" and v["compiler"] == "gcc" and v["opt"] == "-O2"), {})
    os_ = next((v for v in variants if v["c_src"] == "reference" and v["compiler"] == "gcc" and v["opt"] == "-Os"), {})

    def pick(i):
        if i is None:
            return None
        v = variants[i]
        return {"size": v["size"], "insn": v["insn"], "compiler": v["compiler"], "opt": v["opt"], "c_src": v["c_src"],
                "c_idx": v["c_idx"], "asm": v["asm"]}

    return {"split": split, "task_id": tid, "n_c": len(sources), "c_sources": [s["src"] for s in sources],
            "pass_set_c": cands[0]["c"], "pass_set_src": cands[0]["src"],
            "best_size": pick(bs), "best_speed": pick(bsp), "ref_best_size": pick(bs_ref), "ref_best_speed": pick(bsp_ref),
            "ps_best_size": pick(bs_ps), "ps_best_speed": pick(bsp_ps),
            "gcc_o2": {"size": o2.get("size"), "insn": o2.get("insn")}, "gcc_os": {"size": os_.get("size"), "insn": os_.get("insn")},
            "variants": [{k: v.get(k) for k in ("c_src", "c_idx", "compiler", "opt", "passed", "size", "insn", "fail", "diff_ok", "diff_detail")} for v in variants]}


def portfolio(args) -> int:
    out = OUT / "portfolio.jsonl"
    done = {r["task_id"] for r in read_jsonl(out)}
    cands = pass_set_candidates()
    order = [(s, t.id) for s, t in interleaved_tasks() if t.id in cands and t.id not in done]
    log(f"portfolio: {len(cands)} pass-set tasks, {len(done)} done, {len(order)} pending, workers={args.workers}")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_portfolio_task, s, tid, cands[tid]): tid for s, tid in order}
        for f in as_completed(futs):
            tid = futs[f]
            try:
                row = f.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  {tid}: error {exc!r}")
                continue
            append_jsonl(out, row)
            bs, bsp = row["best_size"], row["best_speed"]
            log(f"  {row['split']} {tid}: best_size={bs and (bs['size'], bs['compiler'], bs['opt'], bs['c_src'])} "
                f"best_speed={bsp and (bsp['insn'], bsp['compiler'], bsp['opt'], bsp['c_src'])} gcc-O2={row['gcc_o2']}")
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                log("portfolio: time cap reached; stopping")
                for g in futs:
                    g.cancel()
                break
    rows = read_jsonl(out)
    summarize_portfolio(rows)
    return 0


def summarize_portfolio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics as st
    res: dict[str, Any] = {}
    for split in ("heldout", "val"):
        rs = [r for r in rows if r["split"] == split and r["best_size"]]
        if not rs:
            continue
        win = {}
        for r in rs:
            k = f"{r['best_size']['compiler']} {r['best_size']['opt']}"
            win[k] = win.get(k, 0) + 1
        sz_o2 = [r["best_size"]["size"] / r["gcc_o2"]["size"] for r in rs if r["gcc_o2"]["size"]]
        sz_os = [r["best_size"]["size"] / r["gcc_os"]["size"] for r in rs if r["gcc_os"]["size"]]
        sp = [r["best_speed"]["insn"] / r["gcc_o2"]["insn"] for r in rs if r["best_speed"] and r["gcc_o2"]["insn"]]
        model_c_wins = sum(1 for r in rs if r["best_size"]["c_src"] != "reference")
        res[split] = {"tasks": len(rs), "best_size_winner": dict(sorted(win.items(), key=lambda x: -x[1])),
                      "median_best_size_over_gcc_O2": round(st.median(sz_o2), 3),
                      "median_best_size_over_gcc_Os": round(st.median(sz_os), 3),
                      "median_best_speed_over_gcc_O2": round(st.median(sp), 3) if sp else None,
                      "best_size_from_model_c": model_c_wins,
                      "best_speed_from_model_c": sum(1 for r in rs if r["best_speed"] and r["best_speed"]["c_src"] != "reference")}
    (OUT / "portfolio_summary.json").write_text(json.dumps(res, indent=1))
    log("portfolio summary: " + json.dumps(res))
    return res


# ---------------------------------------------------------------- stage 3
REG64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]
REG32 = {"rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx", "rsi": "esi", "rdi": "edi",
         "r8": "r8d", "r9": "r9d", "r10": "r10d", "r11": "r11d"}
RULES: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"\bmov([lq])\s+\$0,\s*%(\w+)"), lambda m: f"xorl\t%{REG32.get(m.group(2), m.group(2))}, %{REG32.get(m.group(2), m.group(2))}"),
    (re.compile(r"\bmovq\s+\$(\d+),\s*%(r\w+)"), lambda m: f"movl\t${m.group(1)}, %{REG32.get(m.group(2), m.group(2))}" if int(m.group(1)) < 2**31 else m.group(0)),
    (re.compile(r"\bmovq\s+%(r\w+),\s*%(r\w+)"), lambda m: f"movl\t%{REG32.get(m.group(1), m.group(1))}, %{REG32.get(m.group(2), m.group(2))}"),
    (re.compile(r"\bcmp([lq])\s+\$0,\s*%(\w+)"), lambda m: f"test{m.group(1)}\t%{m.group(2)}, %{m.group(2)}"),
    (re.compile(r"\badd([lq])\s+\$1,\s*(%\w+)"), lambda m: f"inc{m.group(1)}\t{m.group(2)}"),
    (re.compile(r"\bsub([lq])\s+\$1,\s*(%\w+)"), lambda m: f"dec{m.group(1)}\t{m.group(2)}"),
    (re.compile(r"\btestq\s+%(r\w+),\s*%(r\w+)"), lambda m: f"testl\t%{REG32.get(m.group(1), m.group(1))}, %{REG32.get(m.group(2), m.group(2))}"),
    (re.compile(r"\bmovslq\s+%(e\w+),\s*%(r\w+)"), lambda m: "cltq" if (m.group(1), m.group(2)) == ("eax", "rax") else m.group(0)),
    (re.compile(r"\bleaq\s+(-?\d+)\(%(r\w+)\),\s*%\2\b"), lambda m: f"addq\t${m.group(1)}, %{m.group(2)}"),
    (re.compile(r"\bimul([lq])\s+\$2,\s*(%\w+),\s*\2"), lambda m: f"add{m.group(1)}\t{m.group(2)}, {m.group(2)}"),
]


def _is_insn(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith(".") and not s.endswith(":") and not s.startswith("#")


def mutate(lines: list[str], rng: random.Random) -> list[str] | None:
    idx = [i for i, l in enumerate(lines) if _is_insn(l)]
    al = [i for i, l in enumerate(lines) if l.strip().startswith((".p2align", ".align", ".balign"))]
    if not idx:
        return None
    new = list(lines)
    r = rng.random()
    if r < 0.12 and al:
        del new[rng.choice(al)]
    elif r < 0.45:
        i = rng.choice(idx)
        if new[i].strip().startswith(("ret",)):
            return None
        del new[i]
    elif r < 0.65:
        i = rng.choice(idx)
        j = i + 1
        if j >= len(new) or not _is_insn(new[j]):
            return None
        new[i], new[j] = new[j], new[i]
    elif r < 0.9:
        hits = [(i, rule) for i in idx for rule in RULES if rule[0].search(new[i])]
        if not hits:
            return None
        i, (pat, fn) = rng.choice(hits)
        rep = pat.sub(fn, new[i], count=1)
        if rep == new[i]:
            return None
        new[i] = rep
    else:  # copy a random instruction to another spot (helps after deletions)
        i, j = rng.choice(idx), rng.choice(idx)
        new.insert(j, new[i])
    return new


def _search_task(split: str, tid: str, goal: str, start_asm: str, bar: int, budget_s: float, seed: int) -> dict[str, Any]:
    tasks = {t.id: t for s, t in interleaved_tasks()}
    t = tasks[tid]
    oracle = oracle_image(t)
    rng = random.Random(f"{tid}:{goal}:{seed}")
    key = "size" if goal == "size" else "insn"
    start = strip_asm(start_asm)
    m0 = measure(t, start)
    if not m0["passed"]:
        return {"split": split, "task_id": tid, "goal": goal, "error": "start fails tests", "bar": bar}
    cur_lines = start.splitlines()
    cur = m0[key]
    best_lines, best_val = cur_lines, cur
    accepted_best: dict[str, Any] | None = None
    tried = passed = proposals = 0
    seen: set[int] = set()
    t0 = time.time()
    while time.time() - t0 < budget_s:
        if HOLD.is_file():
            time.sleep(5)
            continue
        cand = mutate(cur_lines, rng)
        if rng.random() < 0.3 and cand is not None:
            cand = mutate(cand, rng) or cand
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
            a = accept(t, text, oracle, n=200, seed=proposals)
            if not a["accepted"]:
                continue
            best_lines, best_val = cand, v
            cur_lines, cur = cand, v
            accepted_best = {"asm": text, "size": m["size"], "insn": m["insn"]}
        elif v == cur or (v < cur * 1.05 and rng.random() < 0.05):
            cur_lines, cur = cand, v  # neutral / slightly worse moves keep the walk going
        if rng.random() < 0.01:
            cur_lines, cur = best_lines, best_val
    final = None
    if accepted_best:
        a = accept(t, accepted_best["asm"], oracle, n=N_ACCEPT, seed=99)
        if a["accepted"]:
            final = accepted_best
    win = bool(final) and ((goal == "size" and final["size"] < bar) or (goal == "speed" and final["insn"] <= 0.9 * bar))
    return {"split": split, "task_id": tid, "goal": goal, "bar": bar, "start": m0[key], "tried": tried, "passed_tests": passed,
            "best": final, "best_val": (final[key] if final else None), "win": win,
            "improved_over_start": bool(final) and final[key] < m0[key], "seconds": round(time.time() - t0, 1)}


def search(args) -> int:
    out = OUT / "search.jsonl"
    done = {(r["task_id"], r["goal"]) for r in read_jsonl(out)}
    port = read_jsonl(OUT / "portfolio.jsonl")
    jobs = []
    for r in port:
        for goal, k in (("size", "best_size"), ("speed", "best_speed")):
            if r.get(k) and (r["task_id"], goal) not in done:
                jobs.append((r["split"], r["task_id"], goal, r[k]["asm"], r[k]["size" if goal == "size" else "insn"]))
    order = {tid: i for i, (s, t) in enumerate(interleaved_tasks()) for tid in [t.id]}
    jobs.sort(key=lambda j: (order.get(j[1], 1e9), j[2]))
    total_s = args.max_hours * 3600
    per = max(20.0, min(args.per_task_s, total_s * args.workers / max(1, len(jobs))))
    log(f"search: {len(jobs)} jobs, workers={args.workers}, per-job budget {per:.0f}s, wall cap {args.max_hours} h")
    t0 = time.time()
    wins = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_search_task, *j, per, args.seed): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                row = f.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  {j[1]} {j[2]}: error {exc!r}")
                continue
            append_jsonl(out, row)
            wins += bool(row.get("win"))
            log(f"  {row['split']} {row['task_id']} {row['goal']}: bar={row['bar']} start={row.get('start')} "
                f"best={row.get('best_val')} win={row.get('win')} tried={row.get('tried')} (wins so far {wins})")
            if time.time() - t0 > total_s + per + 60:
                log("search: wall cap reached; cancelling the rest")
                for g in futs:
                    g.cancel()
                break
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("selftest"); a.add_argument("--n", type=int, default=500); a.add_argument("--n-diff", type=int, default=10)
    b = sub.add_parser("portfolio"); b.add_argument("--workers", type=int, default=8); b.add_argument("--max-hours", type=float, default=0.5)
    c = sub.add_parser("search"); c.add_argument("--workers", type=int, default=8); c.add_argument("--max-hours", type=float, default=2.0)
    c.add_argument("--per-task-s", type=float, default=240.0); c.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"selftest": selftest, "portfolio": portfolio, "search": search}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
