"""Differential checker: candidate code vs a reference image on generated inputs.

Inputs are type-preserving mutations of the task's own test cases, so buffer
lengths and the integers that describe them stay consistent:
- int scalars are re-drawn unless they equal a buffer length in the same case;
- input buffer elements are re-drawn in a range scaled to the originals;
- C strings keep their length and get random printable characters;
- output buffers keep their initial contents;
- floats are perturbed.
A generated case is used only when the reference runs cleanly on it (no crash,
no timeout, ABI ok). The candidate must match the reference's return value and
every output buffer on every such case.
"""
from __future__ import annotations

import copy
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any

from e2m.harness.assemble import Assembled, AssembleError, assemble
from e2m.harness.emu import emulate
from e2m.harness.tasks import Task

EDGE_INTS = [0, 1, -1, 2, -2, 3, 7, 8, 10, 15, 16, 31, 32, 63, 64, 100, 127, 255, 1000]
PRINTABLE = string.ascii_letters + string.digits + " .,!?-_aeiouAEIOU"


@dataclass
class DiffResult:
    ok: bool
    n_cases: int = 0
    n_skipped: int = 0
    detail: str = ""
    counterexample: dict[str, Any] | None = None
    insn_ref: int = 0
    insn_cand: int = 0
    notes: list[str] = field(default_factory=list)


def _buf_lengths(args: list[Any]) -> set[int]:
    out = set()
    for a in args:
        if isinstance(a, dict):
            d = a.get("data", a.get("value"))
            if isinstance(d, (list, str, bytes)):
                out.add(len(d))
    return out


def _rand_int(rng: random.Random, v: int) -> int:
    span = max(16, 2 * abs(int(v)))
    r = rng.random()
    if r < 0.04:
        return rng.choice([2**31 - 1, -2**31, 2**32 + 5, 2**40 + 3, -(2**40) - 7, 2**62, 123456789, -987654321])
    if r < 0.09:  # k*2^32 + small: low 32 bits look harmless, the full value does not
        v = (rng.randint(1, 3) << 32) + rng.choice([0, 1, 2, 3, 5, 7, 11, 13, 15, rng.randint(-3, 40)])
        return -v if rng.random() < 0.2 else v
    if r < 0.15:  # wide values: catch 32-bit truncation of 64-bit args
        v = rng.getrandbits(rng.randint(33, 62))
        return -v if rng.random() < 0.3 else v
    if r < 0.25:
        return rng.choice(EDGE_INTS) * (1 if rng.random() < 0.8 else -1)
    if r < 0.5:
        return int(v) + rng.randint(-3, 3)
    return rng.randint(-span, span)


def mutate_case(args: list[Any], rng: random.Random, unsigned: list[bool] | None = None) -> list[Any]:
    lens = _buf_lengths(args)
    chars = sorted({ch for a in args if isinstance(a, dict) and isinstance(a.get("data"), str) for ch in a["data"]})
    alphabet = "".join(chars) + "ab " if len(chars) < 3 else "".join(chars)
    out = []
    for ai, a in enumerate(args):
        if isinstance(a, bool):
            out.append(rng.random() < 0.5)
        elif isinstance(a, int):
            v = a if (a in lens and a >= 0) else _rand_int(rng, a)
            if unsigned and ai < len(unsigned) and unsigned[ai]:
                v = abs(v)
            out.append(v)
        elif isinstance(a, float):
            out.append(a * rng.uniform(-2, 2) + rng.uniform(-10, 10) if rng.random() < 0.7 else rng.choice([0.0, 1.0, -1.0, 0.5]))
        elif isinstance(a, dict):
            b = copy.deepcopy(a)
            data = b.get("data", b.get("value"))
            key = "data" if "data" in b else "value"
            if b.get("out") and key in b and not isinstance(data, str):
                # pure/initialized output buffers: keep initial contents
                if b.get("ptr") not in ("cstr", "str", "char") and not _is_input_output(b):
                    out.append(b)
                    continue
            if isinstance(data, str):
                n = len(data)
                if not b.get("out") and not any(isinstance(x, int) and not isinstance(x, bool) and x == n for x in args) and rng.random() < 0.6:
                    n = rng.choice([0, 1, 2, rng.randint(0, 2 * n + 3)])
                alpha = alphabet if rng.random() < 0.8 else PRINTABLE
                b[key] = "".join(rng.choice(alpha) for _ in range(n))
            elif isinstance(data, list) and data and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data):
                if all(isinstance(x, int) for x in data):
                    m = max(16, 2 * max(abs(x) for x in data))
                    wide = "64" in str(b.get("ptr", "")) and rng.random() < 0.3
                    b[key] = [(_rand_int(rng, x) if wide and rng.random() < 0.15 else
                               rng.randint(-m, m) if rng.random() < 0.85 else rng.choice(EDGE_INTS)) for x in data]
                    if str(b.get("ptr", "")).startswith("u"):
                        b[key] = [abs(x) for x in b[key]]
                else:
                    b[key] = [x * rng.uniform(-2, 2) + rng.uniform(-5, 5) for x in data]
            out.append(b)
        else:
            out.append(copy.deepcopy(a))
    return out


WIDE_CHARS = PRINTABLE + "\n\t\r0123456789;:/\\\"'()[]{}<>=+*&^%$#@~|`"


def mutate_case_v2(args: list[Any], rng: random.Random, unsigned: list[bool] | None = None) -> list[Any]:
    """Round-8 families on top of v1 (added after the round-7 audit found constant-output 'wins'):
    log-uniform float magnitudes, the full printable alphabet incl. newline/tab, equal or
    shared-prefix string pairs, length ints decoupled from their buffer, and tiny ints."""
    out = mutate_case(args, rng, unsigned)
    strs = [i for i, a in enumerate(out) if isinstance(a, dict) and isinstance(a.get("data"), str)]
    for i, a in enumerate(out):
        if isinstance(a, float) and not isinstance(a, bool) and rng.random() < 0.5:
            mag = 10 ** rng.uniform(-3, 9)
            out[i] = (-mag if rng.random() < 0.3 else mag) if rng.random() < 0.7 else args[i] * 10 ** rng.randint(1, 6)
        elif isinstance(a, bool):
            continue
        elif isinstance(a, int):
            r = rng.random()
            if r < 0.15:
                out[i] = rng.randint(0, 3)
            elif r < 0.35 and strs:
                n = len(out[rng.choice(strs)]["data"])
                out[i] = rng.randint(0, n + 3)
            if unsigned and i < len(unsigned) and unsigned[i]:
                out[i] = abs(out[i])
        elif isinstance(a, dict) and isinstance(a.get("data"), str) and rng.random() < 0.4:
            n = len(args[i]["data"]) if isinstance(args[i], dict) else len(a["data"])
            n = rng.choice([0, 1, n, rng.randint(0, 2 * n + 20)])
            a["data"] = "".join(rng.choice(WIDE_CHARS) for _ in range(n))
    # resize: grow or shrink every list buffer of one length together with the ints equal to it
    lists = [i for i, a in enumerate(out) if isinstance(a, dict) and isinstance(a.get("data"), list)]
    if lists and rng.random() < 0.3:
        L = len(out[rng.choice(lists)]["data"])
        newL = rng.choice([0, 1, 2, rng.randint(0, 40), rng.randint(250, 700)])
        coupled = [i for i, a in enumerate(out) if isinstance(a, int) and not isinstance(a, bool) and a == L]
        if coupled or L == 0:
            for i in lists:
                d = out[i]["data"]
                if len(d) != L:
                    continue
                if newL <= L:
                    out[i]["data"] = d[:newL]
                else:
                    fill = d if d else [0]
                    if out[i].get("out") and not _is_input_output(out[i]):
                        extra = [fill[0]] * (newL - L)
                    elif all(isinstance(x, int) for x in fill):
                        m = max(16, 2 * max(abs(x) for x in fill))
                        extra = [rng.randint(-m, m) for _ in range(newL - L)]
                        if str(out[i].get("ptr", "")).startswith("u"):
                            extra = [abs(x) for x in extra]
                    else:
                        extra = [rng.uniform(-100, 100) for _ in range(newL - L)]
                    out[i]["data"] = d + extra
            for i in coupled:
                out[i] = newL
    # equal values: copy one numeric buffer into another of the same length, or one scalar into another
    if len(lists) >= 2 and rng.random() < 0.25:
        i, j = rng.sample(lists, 2)
        if not out[j].get("out") or _is_input_output(out[j]):
            k = min(len(out[i]["data"]), len(out[j]["data"]))
            if rng.random() < 0.5:
                out[j]["data"] = list(out[i]["data"][:k]) + list(out[j]["data"][k:])
            else:  # equal on a random subset of positions
                out[j]["data"] = [out[i]["data"][q] if q < k and rng.random() < 0.5 else x for q, x in enumerate(out[j]["data"])]
    for kind in (int, float):
        sc = [i for i, a in enumerate(out) if isinstance(a, kind) and not isinstance(a, bool)]
        if len(sc) >= 2 and rng.random() < 0.2:
            i, j = rng.sample(sc, 2)
            out[j] = out[i]
    for i in lists:
        d = out[i]["data"]
        if d and all(isinstance(x, float) for x in d) and rng.random() < 0.2:
            v = d[rng.randrange(len(d))]
            out[i]["data"] = [v if rng.random() < 0.5 else x for x in d]
    if len(strs) >= 2 and rng.random() < 0.35:
        i, j = rng.sample(strs, 2)
        src = out[i]["data"]
        if rng.random() < 0.5:
            out[j]["data"] = src
        else:
            k = rng.randint(0, len(src))
            out[j]["data"] = src[:k] + "".join(rng.choice(WIDE_CHARS) for _ in range(rng.randint(0, 5)))
    return out


def _is_input_output(b: dict[str, Any]) -> bool:
    """An 'out' buffer whose initial data is not a constant fill is treated as in/out."""
    d = b.get("data")
    return isinstance(d, list) and len(set(map(str, d))) > 1


def _image(src: Assembled | str) -> Assembled:
    return src if isinstance(src, Assembled) else assemble(src, syntax="auto")


def _run(img: Assembled, task: Task, args: list[Any], insn_limit: int, timeout_s: float, noise: int | None = None):
    return emulate(img, signature=task.sig(), args=args, entry=task.sig().name,
                   insn_limit=insn_limit, timeout_s=timeout_s, check_abi=True, entry_noise=noise)


def _same(a, b) -> bool:
    if a.return_value != b.return_value:
        if isinstance(a.return_value, float) and isinstance(b.return_value, float):
            if not (a.return_value != a.return_value and b.return_value != b.return_value):  # both NaN ok
                return False
        else:
            return False
    return dict(a.buffers) == dict(b.buffers)


def diff_check(task: Task, candidate: Assembled | str, reference: Assembled | str, *, n: int = 500,
               seed: int = 0, insn_limit: int | None = None, timeout_s: float = 0.3,
               time_budget_s: float = 120.0,
               seed_tests: list | None = None, include_originals: bool = True, families: str = "v1") -> DiffResult:
    try:
        cand = _image(candidate)
    except (AssembleError, Exception) as exc:  # noqa: BLE001
        return DiffResult(False, detail=f"candidate does not assemble: {str(exc)[:200]}")
    try:
        ref = _image(reference)
    except (AssembleError, Exception) as exc:  # noqa: BLE001
        return DiffResult(False, detail=f"reference does not assemble: {str(exc)[:200]}")
    rng = random.Random(f"{task.id}:{seed}")
    seeds = [t.args for t in (seed_tests if seed_tests is not None else task.tests)] or [[]]
    res = DiffResult(True)
    # the original tests first (as plain inputs), then mutations
    cases = [copy.deepcopy(s) for s in seeds] if include_originals else []
    try:
        unsigned = [p.ctype.kind != "ptr" and p.ctype.kind != "float" and not p.ctype.signed for p in task.sig().params]
    except Exception:  # noqa: BLE001
        unsigned = None
    while len(cases) < n:
        mut = mutate_case_v2 if (families == "v2" and rng.random() < 0.6) else mutate_case
        cases.append(mut(rng.choice(seeds), rng, unsigned))
    if insn_limit is None:
        # generated inputs may make the reference loop for a long time (e.g. an unsigned count of 2^40);
        # bound it relative to the reference's own cost on the task's tests, and skip such inputs
        base = [_run(ref, task, copy.deepcopy(s_), 2_000_000, 1.0) for s_ in seeds]
        top = max([b.insn_count for b in base if b.ok] or [10_000])
        insn_limit = int(min(1_000_000, max(20_000, 10 * top)))
    t_start = time.monotonic()
    for args in cases[:n]:
        if time.monotonic() - t_start > time_budget_s:
            res.notes.append(f"time budget {time_budget_s:.0f}s reached after {res.n_cases + res.n_skipped} inputs")
            break
        if res.n_skipped >= 60 and res.n_skipped > 3 * res.n_cases:
            res.notes.append(f"reference rejects most generated inputs ({res.n_skipped} skipped)")
            break
        r = _run(ref, task, copy.deepcopy(args), insn_limit, timeout_s)
        if not r.ok or not r.abi_ok:
            res.n_skipped += 1
            continue
        # the candidate runs from an adversarial (ABI-legal) entry state: random flags,
        # scratch registers and upper halves of narrow int args
        c = _run(cand, task, copy.deepcopy(args), max(10_000, 4 * r.insn_count + 1000), timeout_s * 4,
                 noise=rng.getrandbits(32))
        res.n_cases += 1
        res.insn_ref += r.insn_count
        res.insn_cand += c.insn_count
        if not c.ok or not c.abi_ok or not _same(r, c):
            res.ok = False
            res.detail = (f"mismatch: ref ret={r.return_value!r} cand ret={c.return_value!r} "
                          f"cand status={c.status} {c.message[:120]}").strip()
            res.counterexample = {"args": args, "ref_return": r.return_value, "cand_return": c.return_value,
                                  "ref_buffers": {k: v.hex() for k, v in r.buffers.items()},
                                  "cand_buffers": {k: v.hex() for k, v in c.buffers.items()},
                                  "cand_status": c.status}
            return res
    if res.n_cases < max(10, n // 10):
        res.ok = False
        res.detail = f"too few valid generated cases ({res.n_cases} of {n})"
    elif res.notes and res.n_cases < n // 2:
        res.ok = False
        res.detail = f"diff check incomplete: {res.notes[-1]}; {res.n_cases} valid inputs"
    return res
