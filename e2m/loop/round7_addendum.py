"""Write the post-run audit (runs/round7/audit.json) into RESULTS.json / SUMMARY.md / failure_analysis.md.
The run's own numbers are kept; the audited numbers are added beside them."""
from __future__ import annotations

import json
import statistics as st

from e2m.loop.round7_common import OUT, read_jsonl

QUESTIONABLE = {("search", "he24_largest_divisor", "size"):
                "drops the d*d<=n bound and tests the remainder with testl: same results on every reachable input, "
                "but correct only while divisors stay below 2^32 and asymptotically slower"}


def main() -> int:
    R = json.loads((OUT / "RESULTS.json").read_text())
    A = json.loads((OUT / "audit.json").read_text())
    port = {r["task_id"]: r for r in read_jsonl(OUT / "portfolio.jsonl")}
    bad = {(r["src"], r["task_id"], r["goal"]): r for r in A["rows"] if not r["v2_ok"]}
    good = {(r["src"], r["task_id"], r["goal"]): r for r in A["rows"] if r["v2_ok"]}
    audited = {}
    for split, blk in R["compiler_beat"].items():
        audited[split] = {}
        for goal in ("size", "speed"):
            k = "size" if goal == "size" else "insn"
            bk = "best_size" if goal == "size" else "best_speed"
            w = blk[goal]["win_tasks"]
            m = sorted(t for t in w["model"] if ("model", t, goal) in good)
            s = sorted(t for t in w["search"] if ("search", t, goal) in good)
            either = sorted(set(m) | set(s))
            ratios = [min(good[(src, t, goal)][k] for src in ("model", "search") if (src, t, goal) in good) / port[t][bk][k] for t in either]
            audited[split][goal] = {"model": len(m), "search": len(s), "either": len(either),
                                    "model_only": sorted(set(m) - set(s)), "both": sorted(set(m) & set(s)),
                                    "median_ratio_of_wins": round(st.median(ratios), 3) if ratios else None,
                                    "mean_ratio_all_tasks": round((sum(ratios) + (blk["pass_set_tasks"] - len(ratios))) / blk["pass_set_tasks"], 4)}
    ho = audited["heldout"]
    model_beats = any(ho[g]["model_only"] for g in ("size", "speed"))
    R["audit"] = {
        "what": "post-run audit (Thu 26 Nov 2026, after the run): every win that passed the 5,000-input re-check was checked "
                "again on 5,000 inputs from wider generator families (diff.py families='v2': log-uniform float magnitudes, "
                "newline/tab and full printable strings, equal and shared-prefix string pairs, equal numeric buffers/scalars, "
                "buffers resized together with their length argument (0..700), tiny ints).",
        "checked": A["checked"], "failed": A["failed"],
        "failures": [{k: r[k] for k in ("src", "task_id", "goal", "detail", "counterexample_args")} for r in bad.values()],
        "questionable": [{"src": a, "task_id": b, "goal": c, "why": v} for (a, b, c), v in QUESTIONABLE.items()],
        "compiler_beat_audited": audited,
        "decision_audited": {"outcome": "Partial", "heldout_size_wins": ho["size"]["either"], "heldout_speed_wins": ho["speed"]["either"],
                             "model_beats_search": model_beats, "note": "Partial under the fixed rule: 3 accepted variants failed the "
                             "5,000-input re-check, and the audit found 16 more that the run's checker accepted wrongly."},
    }
    (OUT / "RESULTS.json").write_text(json.dumps(R, indent=1))
    L = ["", "## Post-run audit (added after the run)", "",
         f"The checker used in the run missed whole input families. Every one of the {A['checked']} wins that passed the "
         f"5,000-input re-check was checked again on 5,000 inputs from wider families (equal values, resized buffers with their "
         f"length argument, newline and other characters in strings, large float magnitudes). **{A['failed']} of them are wrong** "
         "and are removed below. The run's own table above is kept for the record.", "",
         "| Set | Goal | Model wins | Search wins | Either | Model-only tasks | Median ratio of the wins |", "|---|---|---:|---:|---:|---|---:|"]
    for split, a in audited.items():
        for goal in ("size", "speed"):
            g = a[goal]
            L.append(f"| {split} | {goal} | {g['model']} | {g['search']} | {g['either']} | {', '.join(g['model_only']) or '-'} | {g['median_ratio_of_wins']} |")
    L += ["", "Wrong wins found by the audit:", ""]
    for r in bad.values():
        L.append(f"- {r['src']} {r['task_id']} ({r['goal']}): {r['detail'][:110]}")
    L += ["", "Questionable but kept: " + "; ".join(f"{b} ({c}): {v}" for (a, b, c), v in QUESTIONABLE.items()) + ".", "",
          f"Decision after the audit: still **Partial** (held-out size wins {ho['size']['either']}, speed wins {ho['speed']['either']}; "
          f"model-only wins on held-out: {sum(len(ho[g]['model_only']) for g in ('size', 'speed'))})."]
    with (OUT / "SUMMARY.md").open("a") as f:
        f.write("\n".join(L) + "\n")
    with (OUT / "failure_analysis.md").open("a") as f:
        f.write("\n## Post-run audit: wins the run's checker accepted wrongly\n\n" +
                "\n".join(f"- {r['src']} {r['task_id']} ({r['goal']}): {r['detail'][:140]}" for r in bad.values()) + "\n")
    print(json.dumps(R["audit"]["decision_audited"]), json.dumps(audited["heldout"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
