"""Round 6 CPU summary: RESULTS.json, SUMMARY.md, failure_analysis.md, logs/round6.done.

CPU only. Reads the loop and eval outputs written by round6_run.sh, adds a
size-bucketed re-score of existing held-out/val results and gcc -Os on the
reference C. Never touches the GPU or ~/gpu.lock.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from e2m.loop.summarize import summarize_loop
from e2m.train.eval_generate import summarize as summarize_eval

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "runs" / "round6"
PT = ZoneInfo("America/Los_Angeles")
SMALL = 40  # bytes of gcc -O2 on the reference C
LOOPS = {
    "loop-c-only-7b-heldout": ("c-only-7b", "heldout"),
    "loop-c-only-7b-val": ("c-only-7b", "val"),
}
EVALS = [  # (path, model, mode, split, round label)
    ("runs/base7b-c/eval.jsonl", "base7b-c", "c", "heldout", "6"),
    ("runs/base7b-c/eval.jsonl", "base7b-c", "c", "val", "6"),
]
BUCKET_SOURCES = [  # existing per-task greedy results to re-score by size
    ("runs/c-only-7b/eval.jsonl", "c-only-7b", "c"),
    ("runs/asm-distill-7b/eval.jsonl", "asm-distill-7b", "asm"),
    ("runs/asm-distill-7b-v2/eval.jsonl", "asm-distill-7b-v2", "asm"),
    ("runs/c-then-asm/eval.jsonl", "c-then-asm", "c-then-asm"),
    ("runs/base7b-c/eval.jsonl", "base7b-c", "c"),
]


def _rows(path: Path):
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _src(task_id: str) -> str:
    return "MBPP" if task_id.startswith("mbpp") else ("HumanEval" if task_id.startswith("he") else "ExeBench")


def gpu_hours():
    rows = _rows(PIPE / "gpu_hours.jsonl")
    return rows, round(sum(r["seconds"] for r in rows if r.get("gpu")) / 3600.0, 4)


def stage_status():
    out = {}
    for d in sorted((PIPE / "stages").glob("*")):
        ex = (d / "exit").read_text().strip() if (d / "exit").is_file() else "missing"
        out[d.name] = {"exit": ex, "status": {"0": "ok", "3": "stopped (max-hours)", "4": "held", "skipped": "skipped (gpu-hour budget)"}.get(ex, f"code {ex}")}
    return out


def ref_size_map():
    """task_id -> gcc -O2 reference size, from any eval row that carries it."""
    m = {}
    for p, *_ in BUCKET_SOURCES:
        for r in _rows(ROOT / p):
            g = r.get("gcc") or {}
            if g.get("ok") and g.get("size"):
                m.setdefault(r["task_id"], int(g["size"]))
    return m


def buckets(sizes):
    out = {}
    for p, model, mode in BUCKET_SOURCES:
        by = {}
        for r in _rows(ROOT / p):
            if r.get("model") != model or r.get("mode") != mode:
                continue
            by[(r["split"], r["task_id"])] = bool((r.get("greedy") or {}).get("ok"))
        if not by:
            continue
        res = {}
        for split in ("heldout", "val"):
            c = Counter()
            for (s, t), ok in by.items():
                if s != split or t not in sizes:
                    continue
                b = "small" if sizes[t] <= SMALL else "large"
                c[b + "_n"] += 1
                c[b + "_pass"] += ok
            if c:
                res[split] = dict(c)
        out[f"{model}/{mode}"] = res
    return out


def os_reference():
    """gcc -Os vs -O2 on the reference C (CPU)."""
    try:
        from e2m.harness.tasks import load_jsonl, run_gcc_baseline
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}
    val_ids = set(json.loads((ROOT / "eval/val_split.json").read_text())["task_ids"])
    sets = {
        "heldout": load_jsonl(ROOT / "eval/heldout.jsonl"),
        "val": [t for t in load_jsonl(ROOT / "tasks/scaled.jsonl") if t.id in val_ids],
    }
    out = {}
    per_task = {}
    for split, tasks in sets.items():
        ratios, ok_n = [], 0
        for t in tasks:
            try:
                o2 = run_gcc_baseline(t, opt="-O2")
                os_ = run_gcc_baseline(t, opt="-Os")
            except Exception:  # noqa: BLE001
                continue
            if o2.passed and os_.passed and o2.code_size:
                ok_n += 1
                ratios.append(os_.code_size / o2.code_size)
                per_task[t.id] = {"o2": int(o2.code_size), "os": int(os_.code_size)}
        out[split] = {"n": ok_n, "mean_os_over_o2_size": (sum(ratios) / len(ratios)) if ratios else None}
    # hybrid -Os passes against the reference at -Os (apples to apples)
    hyb = [r for r in _rows(ROOT / "runs/round4/hybrid.jsonl") if r.get("mode") == "hybrid-Os"]
    for model in sorted({r["model"] for r in hyb}):
        for split in ("heldout", "val"):
            rs = [r for r in hyb if r["model"] == model and r["split"] == split and (r.get("greedy") or {}).get("ok") and r["task_id"] in per_task]
            if rs:
                v = [r["greedy"]["size"] / per_task[r["task_id"]]["os"] for r in rs if per_task[r["task_id"]]["os"]]
                out[f"hybrid-Os/{model}/{split}"] = {"n": len(v), "mean_size_vs_ref_Os": sum(v) / len(v) if v else None}
    (PIPE / "os_reference.json").write_text(json.dumps({"summary": out, "per_task": per_task}, indent=1))
    return out


def loop_breakdown(path: Path):
    rows = {r["task_id"]: r for r in _rows(path)}
    c = Counter()
    for t, r in rows.items():
        s = _src(t)
        c[s + "_n"] += 1
        c[s + "_pass"] += bool(r.get("passed_full"))
    return dict(c), rows


def main() -> int:
    hours_rows, used = gpu_hours()
    stages = stage_status()
    greedy, loops, notes = {}, {}, []
    # comparison rows (copied)
    for split in ("heldout", "val"):
        s = summarize_eval(ROOT / "runs/c-only-7b/eval.jsonl", "c-only-7b", "c", split)
        if s.get("n"):
            s["copied_from"] = "round 4"
            greedy[f"c-only-7b/c/{split}/round4"] = s
    for p, model, mode, split, _ in EVALS:
        s = summarize_eval(ROOT / p, model, mode, split)
        if s.get("n"):
            greedy[f"{model}/{mode}/{split}"] = s
    breakdown = {}
    loop_rows = {}
    for name, (model, split) in LOOPS.items():
        lp = ROOT / "runs" / name / "loop.jsonl"
        if not lp.is_file():
            continue
        s = summarize_loop(lp, model=model, split=split)
        expected = 78 if split == "heldout" else 90
        s["expected"] = expected
        s["partial"] = s.get("n", 0) < expected
        s["mode"] = "c"
        loops[name] = s
        breakdown[name], loop_rows[name] = loop_breakdown(lp)
    sizes = ref_size_map()
    bk = buckets(sizes)
    osref = os_reference()

    # failure analysis: c-only-7b C loop held-out vs its greedy C and the round 4 hybrid
    fa = ["# Round 6 failure analysis", ""]
    lh = loop_rows.get("loop-c-only-7b-heldout", {})
    greedy_c = {r["task_id"]: bool(r["greedy"]["ok"]) for r in _rows(ROOT / "runs/c-only-7b/eval.jsonl") if r.get("split") == "heldout" and r.get("mode") == "c"}
    hyb48 = {r["task_id"] for r in _rows(ROOT / "runs/round4/hybrid.jsonl") if r.get("split") == "heldout" and r.get("model") == "c-then-asm" and r.get("mode") == "hybrid-O2" and (r.get("greedy") or {}).get("ok")}
    if lh:
        rescued = sorted(t for t, r in lh.items() if r.get("passed_full") and not greedy_c.get(t))
        lost = sorted(t for t, r in lh.items() if not r.get("passed_full") and greedy_c.get(t))
        vis_only = sorted(t for t, r in lh.items() if r.get("passed_visible") and not r.get("passed_full"))
        never = sorted(t for t, r in lh.items() if not r.get("passed_visible"))
        cats = Counter()
        for t in never:
            att = lh[t].get("attempts") or []
            cats[(att[-1].get("category_full") or att[-1].get("category_visible") if att and isinstance(att[-1], dict) else None) or "unknown"] += 1
        fa += [f"C repair loop (c-only-7b, k=8, R=3) on {len(lh)} held-out tasks.", "",
               f"- Pass full tests: {sum(bool(r.get('passed_full')) for r in lh.values())}; greedy C alone passed {sum(greedy_c.get(t, False) for t in lh)} of these tasks.",
               f"- Rescued by the loop (greedy C failed, loop passed): {len(rescued)} {rescued}",
               f"- Greedy C passed but loop's final program failed: {len(lost)} {lost}",
               f"- Passed visible tests only (overfit to visible): {len(vis_only)} {vis_only}",
               f"- Never passed visible tests: {len(never)}; last-attempt categories {dict(cats)}",
               f"- Union with the round 4 hybrid's 48 passes: {len(hyb48 | {t for t, r in lh.items() if r.get('passed_full')})}/78",
               f"- By source: {breakdown.get('loop-c-only-7b-heldout')}", ""]
    else:
        fa.append("Held-out C loop produced no rows.")
    # untrained 7B (base7b-c) vs fine-tuned c-only-7b, greedy C, same eval protocol
    fa += ["## Untrained 7B vs fine-tuned c-only-7b (greedy C, gcc -O2)", ""]
    ft = {(r["split"], r["task_id"]): r for r in _rows(ROOT / "runs/c-only-7b/eval.jsonl") if r.get("mode") == "c" and r.get("model") == "c-only-7b"}
    bs = {(r["split"], r["task_id"]): r for r in _rows(ROOT / "runs/base7b-c/eval.jsonl") if r.get("mode") == "c" and r.get("model") == "base7b-c"}
    for split in ("heldout", "val"):
        ks = [k for k in bs if k[0] == split and k in ft]
        both = sum(ft[k]["greedy"]["ok"] and bs[k]["greedy"]["ok"] for k in ks)
        bo = sorted(k[1] for k in ks if bs[k]["greedy"]["ok"] and not ft[k]["greedy"]["ok"])
        fo = sorted(k[1] for k in ks if ft[k]["greedy"]["ok"] and not bs[k]["greedy"]["ok"])
        cats = Counter(ft[(split, t)]["greedy"]["category"] for t in bo)
        fa += [f"- {split}: both pass {both}; base only {len(bo)}; fine-tuned only {len(fo)}; neither {len(ks) - both - len(bo) - len(fo)}. Fine-tuned categories on base-only tasks: {dict(cats)}",
               f"  - base only: {bo}", f"  - fine-tuned only: {fo}"]
    fa.append("")
    (PIPE / "failure_analysis.md").write_text("\n".join(fa) + "\n")

    results = {
        "title": "E2M round 6",
        "goal": "Ceiling of the C path (model writes C, gcc -O2 lowers it) with a matched repair loop; untrained 7B in C mode; size buckets; gcc -Os reference.",
        "budget_h": 5.0,
        "gpu_hours_used": used,
        "greedy": greedy,
        "loops": loops,
        "loop_source_breakdown": breakdown,
        "size_buckets": {"threshold_bytes_gcc_O2": SMALL, "by_model": bk},
        "os_reference": osref,
        "decision_rule": "C loop held-out >= 55/78: next round superoptimizes the hybrid's gcc output. < 50/78: next round improves the C side.",
        "stages": stages,
        "heldout_order": "runs/round6/heldout_order.json (MBPP and HumanEval interleaved)",
    }
    (PIPE / "RESULTS.json").write_text(json.dumps(results, indent=1, default=str))

    def frac(p, n):
        return "—" if p is None or not n else f"{round(p * n)}/{n}"
    L = ["# Round 6: the C path's ceiling", "",
         f"GPU hours: {used} / 5.0.", "",
         "## C repair loops (c-only-7b, mode c, k=8, R=3, visible-test feedback; gcc -O2)", ""]
    for name, s in loops.items():
        L.append(f"- {name}: pass {s.get('passed_full')}/{s.get('n')} full, {s.get('passed_visible')}/{s.get('n')} visible, mean calls {s.get('mean_calls', 0):.1f}; partial={s['partial']} (of {s['expected']}). Budget curve {s.get('budget_full')}. By source {breakdown.get(name)}.")
    L += ["", "## Greedy C (pass@1 / pass@5)", ""]
    for k, s in greedy.items():
        L.append(f"- {k}: n={s['n']} pass@1 {frac(s.get('pass_at_1'), s['n'])} pass@5 {frac(s.get('pass_at_5'), s['n'])} categories {s.get('greedy_categories')}")
    L += ["", "Comparison (copied, not re-scored): round 4 hybrid gcc -O2 48/78 (c-then-asm C), c-only-7b C 46/78, 1.5B untrained C loop 44/78 (k=4, R=2), best direct asm 7/78.", "",
          f"## Size buckets (reference gcc -O2 <= {SMALL} bytes = small)", "", "```", json.dumps(bk, indent=1), "```", "",
          "## gcc -Os on the reference C", "", "```", json.dumps(osref, indent=1, default=str), "```", "",
          "## Stages", "", "| stage | status |", "|---|---|"]
    L += [f"| {k} | {v['status']} |" for k, v in stages.items()]
    L += ["", "Decision rule: " + results["decision_rule"], ""]
    (PIPE / "SUMMARY.md").write_text("\n".join(L))
    done = {"ts": datetime.now(tz=PT).isoformat(), "gpu_hours_used": used, "stages": {k: v["status"] for k, v in stages.items()}}
    (ROOT / "logs" / "round6.done").write_text(json.dumps(done, indent=2))
    print("round6 summary written; gpu_hours", used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
