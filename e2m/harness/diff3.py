"""Round-8 equivalence checker (v3): coverage-guided differential testing.

For one task and oracle image, ``Checker3`` builds an input pool once:
- the task's tests;
- mutations from diff.py (v1 and v2 families), with a wide-int stress pass
  (about half of the scalar ints redrawn from 31..62 bits);
- coverage-guided growth: an input that reaches a new basic-block edge of the
  oracle joins the corpus and is mutated further (AFL-style), within a budget;
- argument-dependence pairs: for each argument, input pairs that differ only in
  that argument and change the oracle's output (so a candidate that ignores the
  argument is caught).
Inputs on which the oracle faults, times out or breaks the ABI are dropped, and so are inputs on which
the other compilations of the reference C (``consensus`` images: gcc -O3/-Os, clang -O2/-Oz) disagree
with the oracle: such inputs hit undefined behaviour (signed overflow, INT_MIN negation, ...), where no
single answer is right.

The acceptance set is a greedy edge cover (one input per discovered edge), the
dependence pairs, then random pool inputs up to n. ``check`` runs a candidate on
it from an adversarial entry state and compares return value and buffers with
the oracle. It also applies an output-diversity guard (oracle outputs vary,
candidate's do not) and reports edge coverage of the set (fraction of the edges
the pool discovered that the set reaches; by construction 1.0 when the set holds
the cover) and static branch-direction coverage of the oracle (capstone).
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any

from e2m.harness import diff as D
from e2m.harness.assemble import Assembled, assemble
from e2m.harness.emu import emulate
from e2m.harness.tasks import Task


@dataclass
class Check3:
    ok: bool
    n_cases: int = 0
    detail: str = ""
    counterexample: dict[str, Any] | None = None
    edge_coverage: float = 0.0
    branch_dir_coverage: float | None = None
    cand_outputs_distinct: int = 0
    oracle_outputs_distinct: int = 0
    notes: list[str] = field(default_factory=list)


def _key(r) -> tuple:
    rv = r.return_value
    if isinstance(rv, float) and rv != rv:
        rv = "nan"
    return (rv, tuple(sorted((k, bytes(v)) for k, v in r.buffers.items())))


def agrees(images, sig, args, ref, insn_limit: int, timeout_s: float) -> bool:
    """True when every consensus image computes the oracle's result on args (no UB disagreement)."""
    k = _key(ref)
    for img in images:
        c = emulate(img, signature=sig, args=copy.deepcopy(args), entry=sig.name,
                    insn_limit=min(5_000_000, max(insn_limit, 20 * ref.insn_count + 1000)), timeout_s=timeout_s * 6, check_abi=True)
        if not c.ok or _key(c) != k:
            return False
    return True


def _wide(rng: random.Random) -> int:
    v = rng.getrandbits(rng.randint(31, 62))
    if rng.random() < 0.2:
        v = (rng.randint(1, 3) << 32) + rng.randint(-3, 40)
    return -v if rng.random() < 0.3 else v


def _digit_rich(rng: random.Random) -> int:
    """decimal numbers built from a small digit subset (digit-product/sum tasks overflow 32 bits)"""
    digs = rng.sample("123456789" if rng.random() < 0.7 else "13579", rng.randint(1, 3))
    s = "".join(rng.choice(digs) for _ in range(rng.randint(6, 18)))
    return int(s)


def _length_params(task: Task) -> dict[int, int]:
    """int param index -> buffer arg index whose length it equals in every test"""
    tests = [t.args for t in task.tests]
    if not tests:
        return {}
    out: dict[int, int] = {}
    for i, a in enumerate(tests[0]):
        if not isinstance(a, int) or isinstance(a, bool):
            continue
        for j, b in enumerate(tests[0]):
            if not isinstance(b, dict):
                continue
            ok = True
            for args in tests:
                bj = args[j] if j < len(args) else None
                d = bj.get("data", bj.get("value")) if isinstance(bj, dict) else None
                if not isinstance(d, list) or args[i] != len(d):  # arrays only: n > strlen is valid for strings
                    ok = False
                    break
            if ok:
                out[i] = j
                break
    return out


def _int_params(task: Task) -> list[tuple[bool, bool]]:
    """per param: (is_int_scalar, unsigned)"""
    out = []
    for p in task.sig().params:
        c = p.ctype
        out.append((c.kind not in ("ptr", "float"), not getattr(c, "signed", True)))
    return out


def _clip_scalar(task: Task, i: int, v: int) -> int:
    try:
        c = task.sig().params[i].ctype
        bits = 8 * c.size
        if c.signed:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        else:
            lo, hi = 0, (1 << bits) - 1
        return max(lo, min(hi, v)) if bits < 64 else v
    except Exception:  # noqa: BLE001
        return v


class Checker3:
    def __init__(self, task: Task, oracle: Assembled | str, *, seed: int = 0, explore_n: int = 3000,
                 budget_s: float = 90.0, insn_limit: int | None = None, timeout_s: float = 0.3,
                 consensus: list[Assembled] | None = None):
        self.task = task
        self.oracle = oracle if isinstance(oracle, Assembled) else assemble(oracle, syntax="auto")
        self.rng = random.Random(f"v3:{task.id}:{seed}")
        self.timeout_s = timeout_s
        self.sig = task.sig()
        self.consensus = list(consensus or [])
        self.ub_dropped = 0
        self.ip = _int_params(task)
        self.unsigned = [u and s for s, u in self.ip]
        self.lenp = _length_params(task)
        base = [self._ref(t.args, 2_000_000, 1.0) for t in task.tests]
        top = max([b.insn_count for b in base if b.ok] or [10_000])
        self.insn_limit = insn_limit or int(min(1_000_000, max(20_000, 10 * top)))
        self.pool: list[dict[str, Any]] = []   # {"args", "key", "insn", "edges"}
        self.edges: set = set()
        self.cover: list[int] = []
        self.dep_pairs: list[int] = []
        self._explore(explore_n, budget_s)
        self.static_dirs = self._static_branch_dirs()

    # -------------------------------------------------------------- oracle runs
    def _ref(self, args, limit=None, timeout=None, cov=None):
        return emulate(self.oracle, signature=self.sig, args=copy.deepcopy(args), entry=self.sig.name,
                       insn_limit=limit or self.insn_limit, timeout_s=timeout or self.timeout_s, check_abi=True,
                       coverage=cov)

    def _add(self, args) -> bool:
        cov: set = set()
        r = self._ref(args, cov=cov)
        if not r.ok or not r.abi_ok:
            return False
        if not agrees(self.consensus, self.sig, args, r, self.insn_limit, self.timeout_s):
            self.ub_dropped += 1
            return False
        new = cov - self.edges
        self.pool.append({"args": args, "key": _key(r), "insn": r.insn_count, "edges": cov})
        if new:
            self.edges |= new
            self.cover.append(len(self.pool) - 1)
        return bool(new)

    def _mutate(self, seed_args) -> list[Any]:
        rng = self.rng
        r = rng.random()
        if r < 0.45:
            out = D.mutate_case_v2(seed_args, rng, self.unsigned)
        else:
            out = D.mutate_case(seed_args, rng, self.unsigned)
        if rng.random() < 0.1:  # late events: long arrays whose first 256+ elements are tame (byte-width index bugs)
            self._late_event(out, rng)
        if rng.random() < 0.3:  # wide-int stress: about half of the non-length ints get 31..62-bit values
            for i, a in enumerate(out):
                if i < len(self.ip) and self.ip[i][0] and isinstance(a, int) and not isinstance(a, bool) \
                        and i not in self.lenp and rng.random() < 0.5:
                    v = _wide(rng) if rng.random() < 0.7 else _digit_rich(rng)
                    if self.ip[i][1]:
                        v = abs(v)
                    out[i] = _clip_scalar(self.task, i, v)
        # length params stay within their buffer (a longer or negative n is outside the precondition)
        for i, j in self.lenp.items():
            if i < len(out) and j < len(out) and isinstance(out[j], dict):
                d = out[j].get("data", out[j].get("value"))
                L = len(d) if isinstance(d, (list, str)) else 0
                if not isinstance(out[i], int) or out[i] < 0 or out[i] > L:
                    out[i] = L if rng.random() < 0.7 else rng.randint(0, L)
        return out

    def _late_event(self, out: list[Any], rng: random.Random) -> None:
        lists = [i for i, a in enumerate(out) if isinstance(a, dict) and isinstance(a.get("data"), list) and a["data"]
                 and all(isinstance(x, int) and not isinstance(x, bool) for x in a["data"])]
        ins = [i for i in lists if not out[i].get("out") or D._is_input_output(out[i])]
        if not ins:
            return
        j = rng.choice(ins)
        L0 = len(out[j]["data"])
        L = rng.randint(257, 700)
        k = rng.randint(256, L - 1)
        d = out[j]["data"]
        m = max(1, min(20, max(abs(x) for x in d)))
        fam = rng.random()
        if fam < 0.4:
            new = [rng.randint(0, m) for _ in range(L)]
        elif fam < 0.7:
            v = rng.randint(0, m); new = [v] * L
        else:
            new = sorted(rng.randint(0, 4 * m) for _ in range(L))
        for q in range(k, min(L, k + rng.randint(1, 3))):
            new[q] = rng.choice([-(10 ** rng.randint(3, 9)), -1, 0, 10 ** rng.randint(3, 9), m + 1])
        if str(out[j].get("ptr", "")).startswith("u"):
            new = [abs(x) for x in new]
        out[j]["data"] = new
        for i in lists:  # buffers that had the same length follow (outputs keep their fill value)
            if i != j and len(out[i]["data"]) == L0:
                f = out[i]["data"]
                out[i]["data"] = (f + [f[0]] * (L - L0)) if out[i].get("out") and not D._is_input_output(out[i]) else \
                    [rng.randint(0, m) for _ in range(L)]
        for i, jj in self.lenp.items():
            if i < len(out) and jj < len(out) and isinstance(out[jj], dict) and isinstance(out[jj].get("data"), list):
                out[i] = len(out[jj]["data"])

    def _explore(self, n: int, budget_s: float) -> None:
        t0 = time.monotonic()
        for t in self.task.tests:
            self._add(copy.deepcopy(t.args))
        seeds = [t.args for t in self.task.tests] or [[]]
        tries = 0
        while tries < n and time.monotonic() - t0 < budget_s:
            tries += 1
            corpus = [self.pool[i]["args"] for i in self.cover]
            src = self.rng.choice(corpus) if corpus and self.rng.random() < 0.5 else self.rng.choice(seeds)
            self._add(self._mutate(src))
        # argument-dependence pairs
        nparams = len(self.sig.params)
        for ai in range(nparams):
            found = 0
            for _ in range(60):
                if found >= 4 or not self.pool or time.monotonic() - t0 > budget_s * 1.3:
                    break
                p = self.rng.choice(self.pool)
                alt = self._mutate(p["args"])
                cand = copy.deepcopy(p["args"])
                if ai < len(alt) and ai < len(cand):
                    cand[ai] = alt[ai]
                else:
                    continue
                r = self._ref(cand)
                if r.ok and r.abi_ok and _key(r) != p["key"] and \
                        agrees(self.consensus, self.sig, cand, r, self.insn_limit, self.timeout_s):
                    self.pool.append({"args": cand, "key": _key(r), "insn": r.insn_count, "edges": set()})
                    self.dep_pairs += [self.pool.index(p), len(self.pool) - 1]
                    found += 1
        self.explore_s = time.monotonic() - t0
        self.oracle_distinct = len({p["key"] for p in self.pool})

    def _static_branch_dirs(self) -> int | None:
        try:
            import capstone
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
            n = 0
            for ins in md.disasm(self.oracle.text, self.oracle.text_addr):
                if ins.mnemonic.startswith("j") and ins.mnemonic != "jmp":
                    n += 1
            return 2 * n
        except Exception:  # noqa: BLE001
            return None

    # -------------------------------------------------------------- selection
    def selection(self, n: int, seed: int = 0) -> list[int]:
        rng = random.Random(f"sel:{self.task.id}:{seed}:{n}")
        sel = list(dict.fromkeys(self.cover + self.dep_pairs + list(range(min(len(self.task.tests), len(self.pool))))))
        rest = [i for i in range(len(self.pool)) if i not in set(sel)]
        rng.shuffle(rest)
        sel += rest[: max(0, n - len(sel))]
        return sel

    def extend(self, n: int, budget_s: float = 120.0) -> None:
        """grow the pool with fresh mutations (for the 5,000-input final check)"""
        t0 = time.monotonic()
        seeds = [p["args"] for p in self.pool[:50]] or [t.args for t in self.task.tests]
        while len(self.pool) < n and time.monotonic() - t0 < budget_s:
            self._add(self._mutate(self.rng.choice(seeds)))

    # -------------------------------------------------------------- check
    def check(self, candidate: Assembled | str, n: int = 2000, seed: int = 0, quick: int | None = None,
              time_budget_s: float = 400.0) -> Check3:
        try:
            cand = candidate if isinstance(candidate, Assembled) else assemble(candidate, syntax="auto")
        except Exception as exc:  # noqa: BLE001
            return Check3(False, detail=f"candidate does not assemble: {str(exc)[:160]}")
        if n > len(self.pool):
            self.extend(n)
        sel = self.selection(n, seed)
        if quick:
            sel = sel[:quick]
        rng = random.Random(f"noise:{self.task.id}:{seed}")
        cov: set = set()
        keys = set()
        res = Check3(True, oracle_outputs_distinct=self.oracle_distinct)
        t_start = time.monotonic()
        for i in sel:
            if time.monotonic() - t_start > time_budget_s:
                res.ok = False
                res.detail = f"unverified: time budget {time_budget_s:.0f}s ran out after {res.n_cases} inputs"
                res.notes.append(res.detail)
                return res
            p = self.pool[i]
            c = emulate(cand, signature=self.sig, args=copy.deepcopy(p["args"]), entry=self.sig.name,
                        insn_limit=min(5_000_000, max(20_000, 20 * p["insn"] + 1000)), timeout_s=self.timeout_s * 6,
                        check_abi=True, entry_noise=rng.getrandbits(32))
            res.n_cases += 1
            cov |= p["edges"]
            if not c.ok or not c.abi_ok or _key(c) != p["key"]:
                ref = self._ref(p["args"])
                res.ok = False
                if c.status == "insn_limit":
                    res.notes.append("unverified: candidate exceeded 20x the oracle's instruction count")
                res.detail = (f"mismatch: ref ret={ref.return_value!r} cand ret={c.return_value!r} "
                              f"cand status={c.status}")[:240]
                res.counterexample = {"args": p["args"], "ref_return": ref.return_value, "cand_return": c.return_value,
                                      "cand_status": c.status}
                return res
            keys.add(_key(c))
        res.cand_outputs_distinct = len(keys)
        res.edge_coverage = round(len(cov) / max(1, len(self.edges)), 4)
        if self.static_dirs:
            # directions observed = distinct edges leaving a block that ends in a conditional jump; approximated
            res.branch_dir_coverage = round(min(1.0, len(self.edges) / max(1, self.static_dirs + 1)), 4)
        if self.oracle_distinct >= 2 and len(keys) <= 1 and not quick:
            res.ok = False
            res.detail = "output-diversity guard: oracle outputs vary, candidate's do not"
        if not quick and res.edge_coverage < 0.95:
            res.ok = False
            res.detail = f"acceptance set covers only {res.edge_coverage:.0%} of the oracle's discovered edges"
        if len(self.pool) < 20 and not quick:
            res.ok = False
            res.detail = f"too few valid inputs ({len(self.pool)})"
        return res
