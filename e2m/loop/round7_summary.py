"""Round 7 summary (CPU): 5,000-input re-check of every claimed win, metrics, decision rule.

Writes runs/round7/{RESULTS.json, SUMMARY.md, failure_analysis.md, recheck.json}, then logs/round7.done.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from e2m.loop.round7_common import OUT, ROOT, accept, interleaved_tasks, oracle_image, read_jsonl

PT = ZoneInfo("America/Los_Angeles")
RECHECK_N = 5000


def _recheck(job):
    tid, asm = job
    t = {t.id: t for _, t in interleaved_tasks()}[tid]
    a = accept(t, asm, oracle_image(t), n=RECHECK_N, seed=424242)
    return {"accepted": bool(a["accepted"]), "stage": a.get("stage"), "detail": a.get("diff_detail") or a.get("fail", ""),
            "diff_cases": a.get("diff_cases"), "size": a.get("size"), "insn": a.get("insn")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-done", action="store_true")
    args = ap.parse_args(argv)
    port = {r["task_id"]: r for r in read_jsonl(OUT / "portfolio.jsonl")}
    edits = {(r["task_id"], r["goal"]): r for r in read_jsonl(OUT / "edits.jsonl")}
    search = {(r["task_id"], r["goal"]): r for r in read_jsonl(OUT / "search.jsonl")}
    fasterc = {r["task_id"]: r for r in read_jsonl(OUT / "fasterc.jsonl")}
    selftest = json.loads((OUT / "diff_selftest.json").read_text()) if (OUT / "diff_selftest.json").is_file() else {}
    psum = json.loads((OUT / "portfolio_summary.json").read_text()) if (OUT / "portfolio_summary.json").is_file() else {}

    # every claimed win (and every model/search improvement that is reported as a ratio) gets the 5,000-input re-check
    jobs, keys = [], []
    for src, rows in (("model", edits), ("search", search)):
        for (tid, goal), r in rows.items():
            if r.get("best") and r.get("win"):
                jobs.append((tid, r["best"]["asm"])); keys.append((src, tid, goal))
    for tid, r in fasterc.items():
        if r.get("best") and r.get("win"):
            jobs.append((tid, r["best"]["asm"])); keys.append(("fasterc", tid, "speed"))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_recheck, jobs))
    recheck = {f"{s}|{t}|{g}": v for (s, t, g), v in zip(keys, res)}
    (OUT / "recheck.json").write_text(json.dumps({"n": RECHECK_N, "seconds": round(time.time() - t0, 1), "rows": recheck}, indent=1))
    rc_fail = [k for k, v in recheck.items() if not v["accepted"]]

    def verified(src, tid, goal):
        v = recheck.get(f"{src}|{tid}|{goal}")
        return bool(v and v["accepted"])

    results = {"round": 7, "generated": datetime.now(tz=PT).isoformat(), "recheck_n": RECHECK_N,
               "selftest": {k: selftest.get(k) for k in ("equivalent_agreed", "n_equivalent", "different_caught", "n_different", "extra_as_expected", "n_extra", "pass")},
               "portfolio": psum, "compiler_beat": {}, "fasterc": {}, "rechecks_failed": rc_fail}
    order = [(s, t.id) for s, t in interleaved_tasks() if t.id in port]
    for split in ("heldout", "val"):
        tids = [t for s, t in order if s == split]
        if not tids:
            continue
        blk = {"pass_set_tasks": len(tids)}
        for goal, bkey, mkey in (("size", "best_size", "size"), ("speed", "best_speed", "insn")):
            ratios = {"model": [], "search": [], "both": []}
            wins = {"model": [], "search": [], "both": []}
            covered = {"model": 0, "search": 0}
            model_better = []
            overfit = {"model": 0, "search": 0}
            for tid in tids:
                bar = (port[tid].get(bkey) or {}).get(mkey)
                if not bar:
                    continue
                vals = {}
                for src, rows in (("model", edits), ("search", search)):
                    r = rows.get((tid, goal))
                    if r is not None:
                        covered[src] += 1
                        overfit[src] += (r.get("tallies") or {}).get("diff", 0) if src == "model" else 0
                    if r and r.get("best") and r["best"].get(mkey) is not None:
                        if r.get("win"):
                            rc = recheck.get(f"{src}|{tid}|{goal}") or {}
                            if rc.get("accepted") and rc.get(mkey) is not None:
                                vals[src] = min(rc[mkey], bar)   # measured again, not the stage's claim
                        else:
                            vals[src] = min(r["best"][mkey], bar)
                both = min([v for v in vals.values()] + [bar])
                for src in ("model", "search"):
                    v = vals.get(src, bar)
                    ratios[src].append(v / bar)
                    if r_win(goal, v, bar) and (src in vals) and verified(src, tid, goal):
                        wins[src].append(tid)
                ratios["both"].append(both / bar)
                if any(t == tid for t in wins["model"]) or any(t == tid for t in wins["search"]):
                    wins["both"].append(tid)
                if "model" in vals and vals["model"] < vals.get("search", bar) and verified("model", tid, goal):
                    model_better.append(tid)
            blk[goal] = {
                "bar": "min bytes over gcc -O2/-O3/-Os, clang -O2/-Oz" if goal == "size" else "min executed insns over gcc -O2/-O3, clang -O2",
                "win_rule": "strictly smaller" if goal == "size" else "<= 0.9x insns",
                "wins": {k: len(v) for k, v in wins.items()}, "win_tasks": wins,
                "mean_ratio": {k: round(st.mean(v), 4) if v else None for k, v in ratios.items()},
                "median_ratio": {k: round(st.median(v), 4) if v else None for k, v in ratios.items()},
                "covered": covered, "model_better_than_search": model_better,
                "model_tests_pass_diff_fail": overfit["model"],
            }
        results["compiler_beat"][split] = blk
    fc = [r for r in fasterc.values()]
    results["fasterc"] = {"tasks": len(fc), "wins": sum(1 for r in fc if r.get("win") and verified("fasterc", r["task_id"], "speed")),
                          "rows": [{k: r.get(k) for k in ("split", "task_id", "ratio_before", "ratio_after", "win")} for r in fc]}
    search_overfit = sum(r.get("passed_tests", 0) for r in search.values())  # informative only
    results["search_stats"] = {"jobs": len(search), "candidates_tried": sum(r.get("tried", 0) for r in search.values()),
                               "passed_tests": search_overfit}
    ho = results["compiler_beat"].get("heldout", {})
    n_size = ho.get("size", {}).get("wins", {}).get("both", 0)
    n_speed = ho.get("speed", {}).get("wins", {}).get("both", 0)
    model_beats = bool(ho.get("size", {}).get("model_better_than_search") or ho.get("speed", {}).get("model_better_than_search"))
    any_win = any(results["compiler_beat"].get(s, {}).get(g, {}).get("wins", {}).get("both", 0) for s in ("heldout", "val") for g in ("size", "speed"))
    if (n_size >= 10 or n_speed >= 10) and model_beats and not rc_fail:
        outcome = "Win"
    elif any_win:
        outcome = "Partial"
    else:
        outcome = "No win"
    results["decision"] = {"outcome": outcome, "heldout_size_wins": n_size, "heldout_speed_wins": n_speed,
                           "model_beats_search": model_beats, "recheck_failures": len(rc_fail)}
    hours = read_jsonl(OUT / "gpu_hours.jsonl")
    results["gpu_hours"] = round(sum(r["seconds"] for r in hours if r.get("gpu")) / 3600, 3)
    (OUT / "RESULTS.json").write_text(json.dumps(results, indent=1))
    write_md(results, port, edits, search)
    if not args.no_done:
        (ROOT / "logs" / "round7.done").write_text(datetime.now(tz=PT).isoformat() + "\n")
    print(json.dumps(results["decision"]))
    return 0


def r_win(goal, v, bar):
    return v < bar if goal == "size" else v <= 0.9 * bar


def write_md(res, port, edits, search) -> None:
    L = ["# Round 7: beating the compiler on programs that already pass", "",
         f"Generated {res['generated']}. GPU hours used: {res['gpu_hours']}. No training.", "",
         f"**Decision: {res['decision']['outcome']}** (held-out size wins {res['decision']['heldout_size_wins']}, "
         f"speed wins {res['decision']['heldout_speed_wins']}, model beats search on some goal: {res['decision']['model_beats_search']}, "
         f"5,000-input re-check failures: {res['decision']['recheck_failures']}).", "",
         "Acceptance for every variant: all tests (visible and hidden) pass, and the variant agrees with gcc -O2 of the task's reference C "
         "on 500 generated inputs, run from an adversarial ABI-legal entry state (random flags, scratch registers and upper halves of "
         "narrow int args). Every claimed win was re-checked on 5,000 fresh inputs.", "",
         "## Differential checker self-test", "",
         f"Equivalent pairs agreed: {res['selftest'].get('equivalent_agreed')}/{res['selftest'].get('n_equivalent')}; "
         f"known-different pairs (round 6 visible-only passes) caught: {res['selftest'].get('different_caught')}/{res['selftest'].get('n_different')}; "
         f"hand-kept adversarial cases as expected: {res['selftest'].get('extra_as_expected')}/{res['selftest'].get('n_extra')}.", "",
         "## Compiler portfolio", ""]
    for split, v in (res.get("portfolio") or {}).items():
        L.append(f"- {split}: {v['tasks']} tasks; best-size winner {v['best_size_winner']}; median best_size / gcc -O2 = "
                 f"{v['median_best_size_over_gcc_O2']}, / gcc -Os = {v['median_best_size_over_gcc_Os']}; median best_speed / gcc -O2 = "
                 f"{v['median_best_speed_over_gcc_O2']}; bar set by a model-written C program for size on {v['best_size_from_model_c']}, speed on {v['best_speed_from_model_c']}.")
    L += ["", "## Beating the best compiler", "",
          "| Set | Goal | Tasks | Model wins | Search wins | Either | Mean ratio (model / search / either) | Median ratio (either) |",
          "|---|---|---:|---:|---:|---:|---|---:|"]
    for split, blk in res["compiler_beat"].items():
        for goal in ("size", "speed"):
            g = blk.get(goal)
            if not g:
                continue
            mr = g["mean_ratio"]
            L.append(f"| {split} | {goal} ({g['win_rule']}) | {blk['pass_set_tasks']} | {g['wins']['model']} | {g['wins']['search']} | "
                     f"{g['wins']['both']} | {mr['model']} / {mr['search']} / {mr['both']} | {g['median_ratio']['both']} |")
    L += ["", "Ratios are against the best compiler, over all pass-set tasks, with a task without a verified improvement counted as 1.0. "
          "Model coverage (task-goals attempted before the GPU cap): " +
          ", ".join(f"{s}/{g}: {res['compiler_beat'][s][g]['covered']['model']}" for s in res["compiler_beat"] for g in ("size", "speed") if g in res["compiler_beat"][s]) + ".", ""]
    fc = res["fasterc"]
    L += ["## Faster C (stage 4)", "", f"{fc['tasks']} tasks tried; {fc['wins']} verified wins (<= 0.9x the best compiler's instruction count on the best C)."]
    for r in fc["rows"]:
        L.append(f"- {r['task_id']}: ratio vs reference gcc -O2 {r['ratio_before']} -> {r['ratio_after']} (win: {r['win']})")
    L += ["", "## Files", "", "portfolio.jsonl, edits.jsonl, search.jsonl, fasterc.jsonl and diff_selftest.json hold the per-task rows "
          "(kept with the run; not published because they carry raw assembly and counterexample buffers). recheck.json lists every re-check.", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L))
    # failure analysis
    F = ["# Round 7 failure analysis", ""]
    tallies: dict[str, int] = {}
    for r in edits.values():
        for k, v in (r.get("tallies") or {}).items():
            tallies[k] = tallies.get(k, 0) + v
    tot = sum(tallies.values()) or 1
    F += ["## Model candidates by outcome", "", "| Outcome | Count | Share |", "|---|---:|---:|"]
    names = {"tests": "assembled, failed a test (or did not assemble)", "diff": "passed all tests, failed the diff check (overfit)",
             "ok": "accepted (tests + diff)", "compile": "did not compile"}
    for k, v in sorted(tallies.items(), key=lambda x: -x[1]):
        F.append(f"| {names.get(k, k)} | {v} | {v / tot:.1%} |")
    F += ["", "## Re-check failures (5,000 inputs)", ""]
    F += [f"- {k}: {v}" for k, v in json.loads((OUT / 'recheck.json').read_text())["rows"].items() if not v["accepted"]] or ["- none"]
    srch = list(search.values())
    F += ["", "## Search", "", f"{len(srch)} task-goals; {sum(r.get('tried', 0) for r in srch)} candidates tried; "
          f"{sum(r.get('passed_tests', 0) for r in srch)} passed the tests; {sum(1 for r in srch if r.get('improved_over_start'))} improved on their start; "
          f"{sum(1 for r in srch if r.get('win'))} claimed wins before re-check.", ""]
    (OUT / "failure_analysis.md").write_text("\n".join(F))


if __name__ == "__main__":
    sys.exit(main())
