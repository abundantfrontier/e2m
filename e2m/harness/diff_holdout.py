"""Round-8 held-back input generator (written before any round-8 candidate existed; used only in stage E).

Independent of diff.py/diff3.py's families: structured inputs built from each test's shape.
- integer arrays: sorted, reverse-sorted, all-equal, alternating signs, element-type extremes
  (INT_MIN/INT_MAX of the element width) at random positions, powers of two +-1, n = 0 or 1 with a
  non-empty buffer;
- strings: one repeated char, palindromes, digits only, upper-case only, words and spaces, single char, empty;
- floats (scalars and arrays): +-0, +-inf, nan, subnormals, 1e308, integral floats, k/10;
- int scalars: type min/max, 0, +-1, 2^k +- 1, all scalars equal.
Length parameters stay within their array (diff3._length_params). Inputs the oracle cannot run, or on which
the other compilations of the reference C disagree with it (undefined behaviour), are skipped.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

from e2m.harness.assemble import Assembled, assemble
from e2m.harness.diff3 import Check3, _key, _length_params, agrees
from e2m.harness.emu import emulate
from e2m.harness.tasks import Task

FLOAT_SPECIALS = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 2.2250738585072014e-308, 1e308, -1e308,
                  1.0, -1.0, 2.0, 3.0, -3.0, 0.5, 0.1, 0.3, 100.0, 1e15, 4503599627370496.0]


def _elem_range(ptr: str) -> tuple[int, int] | None:
    p = str(ptr)
    for bits in (8, 16, 32, 64):
        if p in (f"i{bits}", f"int{bits}"):
            return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if p in (f"u{bits}", f"uint{bits}"):
            return 0, (1 << bits) - 1
    return None


def _scalar_range(task: Task, i: int) -> tuple[int, int]:
    try:
        c = task.sig().params[i].ctype
        bits = 8 * c.size
        return ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if c.signed else (0, (1 << bits) - 1))
    except Exception:  # noqa: BLE001
        return -(1 << 63), (1 << 63) - 1


def _int_array(rng: random.Random, d: list[int], ptr: str) -> list[int]:
    lo, hi = _elem_range(ptr) or (-(1 << 31), (1 << 31) - 1)
    n = len(d) if d and rng.random() < 0.6 else rng.choice([2, 3, 5, 8, 16, 33])
    m = max([16] + [abs(x) for x in d])
    base = [rng.randint(max(lo, -m), min(hi, m)) for _ in range(n)]
    fam = rng.randrange(7)
    if fam == 0:
        out = sorted(base)
    elif fam == 1:
        out = sorted(base, reverse=True)
    elif fam == 2:
        out = [base[0]] * n if n else []
    elif fam == 3:
        out = [abs(x) * (1 if k % 2 == 0 else -1) for k, x in enumerate(base)]
    elif fam == 4:
        out = list(base)
        for _ in range(rng.randint(1, max(1, n // 2))):
            if out:
                out[rng.randrange(n)] = rng.choice([lo, hi, lo + 1, hi - 1, 0])
    elif fam == 5:
        out = [rng.choice([1, -1]) * ((1 << rng.randint(0, 30)) + rng.choice([-1, 0, 1])) for _ in range(n)]
    else:
        out = [rng.choice([0, 1, -1]) for _ in range(n)]
    return [min(hi, max(lo, x)) for x in out]


def _string(rng: random.Random, s: str) -> str:
    n = rng.choice([0, 1, 2, len(s), len(s) + 5, rng.randint(3, 40)])
    fam = rng.randrange(7)
    if fam == 0:
        return rng.choice("aZ0 !x9") * n
    if fam == 1:
        half = "".join(rng.choice("abcxyz") for _ in range(n // 2))
        return half + (rng.choice("abc") if n % 2 else "") + half[::-1]
    if fam == 2:
        return "".join(rng.choice("0123456789") for _ in range(n))
    if fam == 3:
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
    if fam == 4:
        return " ".join("".join(rng.choice("abcde") for _ in range(rng.randint(1, 5))) for _ in range(max(1, n // 4)))
    if fam == 5:
        return rng.choice("aA1 ")
    return ""


def holdout_case(task: Task, seed_args: list[Any], rng: random.Random, lenp: dict[int, int]) -> list[Any]:
    out = copy.deepcopy(seed_args)
    sig = task.sig()
    ints = [i for i, a in enumerate(out) if isinstance(a, int) and not isinstance(a, bool) and i not in lenp]
    for i, a in enumerate(out):
        if isinstance(a, bool):
            continue
        if isinstance(a, float):
            out[i] = rng.choice(FLOAT_SPECIALS) if rng.random() < 0.6 else float(rng.randint(-20, 20)) / rng.choice([1, 10])
        elif isinstance(a, int) and i not in lenp:
            lo, hi = _scalar_range(task, i)
            r = rng.random()
            if r < 0.35:
                v = rng.choice([lo, hi, lo + 1, hi - 1, 0, 1, -1])
            elif r < 0.7:
                v = rng.choice([1, -1]) * ((1 << rng.randint(0, 62)) + rng.choice([-1, 0, 1]))
            else:
                v = rng.randint(-64, 64)
            out[i] = min(hi, max(lo, v))
        elif isinstance(a, dict):
            d = a.get("data", a.get("value"))
            key = "data" if "data" in a else "value"
            if a.get("out") and not (isinstance(d, list) and len(set(map(str, d))) > 1) and not isinstance(d, str):
                continue
            if isinstance(d, str):
                a[key] = _string(rng, d)
            elif isinstance(d, list) and d and all(isinstance(x, int) and not isinstance(x, bool) for x in d):
                a[key] = _int_array(rng, d, a.get("ptr", ""))
            elif isinstance(d, list) and d and all(isinstance(x, (int, float)) for x in d):
                n = len(d) if rng.random() < 0.6 else rng.choice([1, 2, 4, 9])
                a[key] = [rng.choice(FLOAT_SPECIALS) if rng.random() < 0.5 else float(rng.randint(-9, 9)) for _ in range(n)]
    if len(ints) >= 2 and rng.random() < 0.15:
        v = out[rng.choice(ints)]
        for i in ints:
            out[i] = v
    # out buffers keep their size relation; length params follow their array (or 0/1 on a non-empty array)
    for i, j in lenp.items():
        if j < len(out) and isinstance(out[j], dict):
            d = out[j].get("data", out[j].get("value"))
            L = len(d) if isinstance(d, list) else 0
            out[i] = rng.choice([0, 1]) if (L >= 1 and rng.random() < 0.15) else L
            out[i] = min(out[i], L)
    # output buffers sized from a length arg: keep them at least as long as any input array
    maxL = max([len(x["data"]) for x in out if isinstance(x, dict) and isinstance(x.get("data"), list) and not x.get("out")] or [0])
    for x in out:
        if isinstance(x, dict) and x.get("out") and isinstance(x.get("data"), list) and x["data"] and len(x["data"]) < maxL:
            x["data"] = x["data"] + [x["data"][0]] * (maxL - len(x["data"]))
    return out


def holdout_check(task: Task, candidate: Assembled | str, oracle: Assembled | str, *, n: int = 5000, seed: int = 0,
                  timeout_s: float = 0.3, max_tries: int | None = None,
                  consensus: list[Assembled] | None = None) -> Check3:
    cand = candidate if isinstance(candidate, Assembled) else assemble(candidate, syntax="auto")
    orc = oracle if isinstance(oracle, Assembled) else assemble(oracle, syntax="auto")
    sig = task.sig()
    rng = random.Random(f"holdout:{task.id}:{seed}")
    lenp = _length_params(task)
    seeds = [t.args for t in task.tests] or [[]]
    res = Check3(True)
    skipped = 0
    keys = set()
    tries = 0
    max_tries = max_tries or 3 * n
    while res.n_cases < n and tries < max_tries:
        tries += 1
        args = holdout_case(task, rng.choice(seeds), rng, lenp)
        r = emulate(orc, signature=sig, args=copy.deepcopy(args), entry=sig.name, insn_limit=1_000_000,
                    timeout_s=timeout_s, check_abi=True)
        if not r.ok or not r.abi_ok or not agrees(consensus or [], sig, args, r, 1_000_000, timeout_s):
            skipped += 1
            continue
        c = emulate(cand, signature=sig, args=copy.deepcopy(args), entry=sig.name,
                    insn_limit=min(5_000_000, max(20_000, 20 * r.insn_count + 1000)), timeout_s=timeout_s * 6,
                    check_abi=True, entry_noise=rng.getrandbits(32))
        res.n_cases += 1
        if not c.ok or not c.abi_ok or _key(c) != _key(r):
            res.ok = False
            res.detail = f"holdout mismatch: ref ret={r.return_value!r} cand ret={c.return_value!r} status={c.status}"[:240]
            res.counterexample = {"args": args, "ref_return": r.return_value, "cand_return": c.return_value, "cand_status": c.status}
            break
        keys.add(_key(c))
    res.cand_outputs_distinct = len(keys)
    res.notes.append(f"skipped {skipped} inputs the oracle could not run")
    if res.ok and res.n_cases < min(n, 200):
        res.ok = False
        res.detail = f"too few valid held-back inputs ({res.n_cases})"
    return res
