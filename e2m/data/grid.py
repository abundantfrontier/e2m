"""Configurable compiler grids for the dataset sweep."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable

# Full factorial. Each point is one compile. YMM/FMA outputs are rejected
# later by the harness check; -mno-avx and 128-bit AVX stay executable.
C_OPTS = ("-O0", "-O1", "-O2", "-O3", "-Os", "-Oz")
C_CPUS = ("x86-64", "x86-64-v2", "x86-64-v3", "znver3")
C_FEATS: tuple[tuple[str, ...], ...] = (
    (),
    ("-mno-avx", "-mno-avx2"),
    ("-mavx2", "-mno-fma", "-mprefer-vector-width=128"),
    ("-mavx2",),
    ("-mno-avx", "-mno-avx2", "-mno-bmi", "-mno-bmi2", "-mno-sse4.2"),
)
C_TOGGLES: tuple[tuple[str, ...], ...] = (
    (),
    ("-funroll-loops",),
    ("-fno-unroll-loops",),
    ("-ftree-vectorize",),
    ("-fno-tree-vectorize", "-fno-tree-slp-vectorize"),
    ("--param=max-inline-insns-single=1",),
    ("--param=max-inline-insns-single=1000",),
    ("-fno-align-loops", "-fno-align-functions", "-fno-align-jumps", "-fno-align-labels"),
)

# Reduced grid: ~72 gcc/clang configs per task (fits 3k tasks in a few hours).
C_OPTS_REDUCED = ("-O0", "-O2", "-O3", "-Os")
C_CPUS_REDUCED = ("x86-64", "x86-64-v3")
C_FEATS_REDUCED: tuple[tuple[str, ...], ...] = (
    (),
    ("-mno-avx", "-mno-avx2"),
    ("-mavx2",),
)
C_TOGGLES_REDUCED: tuple[tuple[str, ...], ...] = (
    (),
    ("-funroll-loops",),
    ("-fno-tree-vectorize", "-fno-tree-slp-vectorize"),
)

RUST_OPTS = ("0", "1", "2", "3", "s", "z")
RUST_CPUS = C_CPUS
RUST_FEATS = (
    "",
    "-avx,-avx2",
    "+avx2,-fma",
    "+avx2",
    "-avx,-avx2,-bmi,-bmi2,-sse4.2",
)
RUST_OVERFLOW = ("off", "on")
RUST_LLVM = (
    "",
    "-unroll-threshold=0",
    "-unroll-threshold=300",
    "-inline-threshold=1",
    "-inline-threshold=1000",
    "-vectorize-loops=false",
)

# Function passes the new PM accepts in a flat pipeline, plus a few loop
# forms. Random order is allowed to fail; the harness drops bad code.
LLVM_SAFE_PASSES = (
    "mem2reg",
    "sroa",
    "instcombine",
    "reassociate",
    "simplifycfg",
    "gvn",
    "early-cse",
    "dce",
    "adce",
    "sccp",
    "jump-threading",
    "tailcallelim",
    "correlated-propagation",
    "bdce",
    "memcpyopt",
    "dse",
    "aggressive-instcombine",
)
LLVM_EXTRA_PASSES = (
    "loop(loop-rotate)",
    "loop-mssa(licm)",
    "loop(loop-unroll-full)",
    "loop(indvars)",
    "loop(loop-deletion)",
    "slp-vectorizer",
    "loop-vectorize",
)
# LLVM 23 dropped opt/llc -Os and -Oz ("use O2 with optsize/minsize").
# Each row is (label, opt -passes, llc -O, function attributes).
LLVM_DEFAULT_PIPELINES = (
    ("default<O0>", "default<O0>", "-O0", ""),
    ("default<O1>", "default<O1>", "-O1", ""),
    ("default<O2>", "default<O2>", "-O2", ""),
    ("default<O3>", "default<O3>", "-O3", ""),
    ("default<Os>", "default<O2>", "-O2", "optsize"),
    ("default<Oz>", "default<O2>", "-O2", "minsize optsize"),
)
LLVM_CPUS = C_CPUS


def config_id(job: dict[str, Any]) -> str:
    payload = {
        "task_id": job["task_id"],
        "compiler": job["compiler"],
        "flags": list(job["flags"]),
        "seed": job.get("seed"),
        "passes": job.get("passes"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _base_job(task_id: str, compiler: str, flags: list[str], knobs: dict[str, Any]) -> dict[str, Any]:
    job = {
        "task_id": task_id,
        "compiler": compiler,
        "flags": list(flags),
        "seed": knobs.get("seed"),
        "passes": knobs.get("passes"),
        "knobs": knobs,
    }
    job["config_id"] = config_id(job)
    return job


def iter_c_jobs(
    task_ids: Iterable[str],
    compiler: str,
    *,
    reduced: bool = False,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    opts = C_OPTS_REDUCED if reduced else C_OPTS
    cpus = C_CPUS_REDUCED if reduced else C_CPUS
    feats = C_FEATS_REDUCED if reduced else C_FEATS
    toggles = C_TOGGLES_REDUCED if reduced else C_TOGGLES
    for task_id in task_ids:
        for opt in opts:
            for cpu in cpus:
                for feat in feats:
                    for toggle in toggles:
                        flags = [opt, f"-march={cpu}", *feat, *toggle]
                        knobs = {
                            "opt": opt,
                            "cpu": cpu,
                            "feat": " ".join(feat),
                            "toggle": " ".join(toggle),
                            "seed": None,
                        }
                        jobs.append(_base_job(task_id, compiler, flags, knobs))
    return jobs


def iter_rust_jobs(task_ids: Iterable[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task_id in task_ids:
        for opt in RUST_OPTS:
            for cpu in RUST_CPUS:
                for feat in RUST_FEATS:
                    for overflow in RUST_OVERFLOW:
                        for llvm_args in RUST_LLVM:
                            flags = [
                                "-C",
                                f"opt-level={opt}",
                                "-C",
                                f"target-cpu={cpu}",
                                "-C",
                                f"overflow-checks={overflow}",
                            ]
                            if feat:
                                flags.extend(["-C", f"target-feature={feat}"])
                            if llvm_args:
                                flags.extend(["-C", f"llvm-args={llvm_args}"])
                            knobs = {
                                "opt": opt,
                                "cpu": cpu,
                                "feat": feat,
                                "overflow": overflow,
                                "toggle": llvm_args,
                                "seed": None,
                            }
                            jobs.append(_base_job(task_id, "rustc", flags, knobs))
    return jobs


def random_pass_pipeline(seed: int) -> str:
    rng = random.Random(seed)
    passes = list(LLVM_SAFE_PASSES)
    rng.shuffle(passes)
    # Keep a couple of cleanup passes at the end so random order still lowers.
    passes.extend(["instcombine", "simplifycfg"])
    extras = list(LLVM_EXTRA_PASSES)
    rng.shuffle(extras)
    for extra in extras[: rng.randint(0, 3)]:
        slot = rng.randint(0, len(passes))
        passes.insert(slot, extra)
    return ",".join(passes)


def iter_llvm_jobs(task_ids: Iterable[str], n_seeds: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if n_seeds < 0:
        n_seeds = 0
    for task_id in task_ids:
        for label, passes, llc_opt, attrs in LLVM_DEFAULT_PIPELINES:
            cpu = "x86-64"
            flags = [f"-passes={passes}", llc_opt, f"-mcpu={cpu}"]
            if attrs:
                flags.append(f"function-attrs={attrs}")
            knobs = {
                "opt": label,
                "cpu": cpu,
                "feat": "",
                "toggle": attrs,
                "seed": label,
                "passes": passes,
                "ir_attrs": attrs,
            }
            jobs.append(_base_job(task_id, "llvm-opt", flags, knobs))
        for seed in range(n_seeds):
            passes = random_pass_pipeline(seed)
            cpu = LLVM_CPUS[seed % len(LLVM_CPUS)]
            # llc on LLVM 23 accepts -O0..-O3 only. -Os/-Oz are invalid.
            opt_level = ("-O0", "-O1", "-O2", "-O3")[seed % 4]
            flags = [f"-passes={passes}", opt_level, f"-mcpu={cpu}"]
            knobs = {
                "opt": opt_level,
                "cpu": cpu,
                "feat": "",
                "toggle": "random-pipeline",
                "seed": str(seed),
                "passes": passes,
                "ir_attrs": "",
            }
            jobs.append(_base_job(task_id, "llvm-opt", flags, knobs))
    return jobs


def build_jobs(
    task_ids: list[str],
    compilers: list[str],
    *,
    llvm_seeds: int,
    grid: str = "full",
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    reduced = grid == "reduced"
    for compiler in compilers:
        if compiler in {"gcc", "clang"}:
            jobs.extend(iter_c_jobs(task_ids, compiler, reduced=reduced))
        elif compiler == "rustc":
            jobs.extend(iter_rust_jobs(task_ids))
        elif compiler == "llvm-opt":
            jobs.extend(iter_llvm_jobs(task_ids, 0 if reduced else llvm_seeds))
        else:
            raise ValueError(f"unknown compiler {compiler!r}")
    return jobs
