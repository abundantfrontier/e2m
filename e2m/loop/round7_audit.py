"""Post-run audit of round 7 wins (CPU): re-check every verified win with the v2 input families
(diff.py mutate_case_v2), 5,000 inputs, fresh seed. Writes runs/round7/audit.json."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

from e2m.harness.diff import diff_check
from e2m.loop.round7_common import OUT, interleaved_tasks, measure, oracle_image, read_jsonl


def _one(job):
    src, tid, goal, asm = job
    t = {t.id: t for _, t in interleaved_tasks()}[tid]
    m = measure(t, asm)
    d = diff_check(t, asm, oracle_image(t), n=5000, seed=8080, families="v2", time_budget_s=300)
    ce = d.counterexample
    return {"src": src, "task_id": tid, "goal": goal, "tests_pass": m["passed"], "size": m["size"], "insn": m["insn"],
            "v2_ok": d.ok, "v2_cases": d.n_cases, "v2_skipped": d.n_skipped, "detail": d.detail[:200],
            "counterexample_args": json.dumps(ce["args"])[:300] if ce else None,
            "ref_return": ce["ref_return"] if ce else None, "cand_return": ce["cand_return"] if ce else None}


def main() -> int:
    rc = json.loads((OUT / "recheck.json").read_text())["rows"]
    rows = {}
    for src, f in (("model", "edits.jsonl"), ("search", "search.jsonl"), ("fasterc", "fasterc.jsonl")):
        for r in read_jsonl(OUT / f):
            rows[(src, r["task_id"], r.get("goal", "speed"))] = r
    jobs = []
    for key, v in rc.items():
        src, tid, goal = key.split("|")
        if v["accepted"]:
            jobs.append((src, tid, goal, rows[(src, tid, goal)]["best"]["asm"]))
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_one, jobs))
    bad = [r for r in res if not r["v2_ok"]]
    (OUT / "audit.json").write_text(json.dumps({"n": 5000, "families": "v2", "checked": len(res), "failed": len(bad), "rows": res}, indent=1))
    print(f"audit: {len(res)} verified wins re-checked with v2 families; {len(bad)} fail")
    for r in bad:
        print(" ", r["src"], r["task_id"], r["goal"], r["detail"][:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
