"""Round 8 shared helpers (on top of round7_common): v3 acceptance, consensus images, known-wrong list."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from e2m.harness.assemble import Assembled, AssembleError, assemble
from e2m.harness.diff3 import Checker3
from e2m.loop.round7_common import (COMPILERS, ROOT, compile_c, interleaved_tasks, measure, oracle_image, read_jsonl)

OUT = Path(os.environ.get("E2M_R8_OUT", str(ROOT / "runs" / "round8")))
R7 = ROOT / "runs" / "round7"
HOLD = Path(os.environ.get("E2M_HOLD_FILE", str(ROOT / "round8.hold")))
N_V3 = 2000
N_FINAL = 5000


@lru_cache(maxsize=None)
def tasks_by_id():
    return {t.id: t for _, t in interleaved_tasks()}


def split_of() -> dict[str, str]:
    return {t.id: s for s, t in interleaved_tasks()}


def consensus_images(task) -> list[Assembled]:
    imgs = []
    for comp, opt in COMPILERS:
        if (comp, opt) == ("gcc", "-O2"):
            continue
        try:
            imgs.append(assemble(compile_c(task.reference_c, comp, opt), syntax="att"))
        except (AssembleError, Exception):  # noqa: BLE001
            pass
    return imgs


_CHECKERS: dict[str, Checker3] = {}


def checker(task, seed: int = 0, **kw) -> Checker3:
    """one Checker3 per task per process (the input pool is the expensive part)"""
    k = f"{task.id}:{seed}"
    if k not in _CHECKERS:
        _CHECKERS[k] = Checker3(task, oracle_image(task), seed=seed, consensus=consensus_images(task), **kw)
    return _CHECKERS[k]


def accept3(task, asm: str, *, n: int = N_V3, seed: int = 0, quick: int | None = None) -> dict[str, Any]:
    """all tests, then v3 (coverage-guided, UB-filtered, diversity guard) on n inputs"""
    m = measure(task, asm)
    if not m["passed"]:
        m.update(accepted=False, stage="tests")
        return m
    c = checker(task)
    r = c.check(asm, n=n, seed=seed, quick=quick)
    m.update(accepted=bool(r.ok), stage="ok" if r.ok else "v3", v3_cases=r.n_cases, v3_detail=r.detail[:300],
             edge_coverage=r.edge_coverage, edges=len(c.edges), pool=len(c.pool), ub_dropped=c.ub_dropped,
             notes=r.notes, counterexample=r.counterexample)
    return m


def round7_wins() -> list[dict[str, Any]]:
    """every round-7 claimed win with its asm and its known verdict (recheck / v2 audit)"""
    rows = {}
    for src, f in (("model", "edits.jsonl"), ("search", "search.jsonl"), ("fasterc", "fasterc.jsonl")):
        for r in read_jsonl(R7 / f):
            rows[(src, r["task_id"], r.get("goal", "speed"))] = r
    rc = json.loads((R7 / "recheck.json").read_text())["rows"]
    au = {(r["src"], r["task_id"], r["goal"]): r["v2_ok"] for r in json.loads((R7 / "audit.json").read_text())["rows"]}
    sp = split_of()
    out = []
    for k, v in rc.items():
        s, t, g = k.split("|")
        known_wrong = (not v["accepted"]) or (au.get((s, t, g)) is False)
        why = "recheck" if not v["accepted"] else ("v2 audit" if known_wrong else "")
        out.append({"src": s, "task_id": t, "goal": g, "split": sp.get(t), "asm": rows[(s, t, g)]["best"]["asm"],
                    "known_wrong": known_wrong, "why": why, "bar": rows[(s, t, g)]["bar"]})
    return out
