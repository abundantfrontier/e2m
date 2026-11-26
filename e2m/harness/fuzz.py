"""Robustness fuzz: random bytes and mutated real functions, never crash the harness."""

from __future__ import annotations

import os
import random
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

from e2m.harness.emu import emulate

_SEED_ASM = [
    b"\xc3",  # ret
    bytes.fromhex("4889f84801f0c3"),  # mov rax,rdi; add rax,rsi; ret
    bytes.fromhex("31c0c3"),  # xor eax,eax; ret
    bytes.fromhex("4889f8c3"),  # mov rax,rdi; ret
    bytes.fromhex("f20f58c1c3"),  # addsd xmm0,xmm1; ret
    bytes.fromhex("e800000000c3"),  # call +0; ret  (will recurse / fault)
    bytes.fromhex("0f05"),  # syscall
    bytes.fromhex("cd80"),  # int 0x80
    bytes.fromhex("0f34"),  # sysenter
    bytes.fromhex("ebfe"),  # jmp $
    bytes.fromhex("f4c3"),  # hlt; ret
    bytes.fromhex("0f0b"),  # ud2
    bytes.fromhex("cc"),  # int3
]


@dataclass
class FuzzReport:
    n: int
    harness_crashes: int = 0
    hangs: int = 0
    elapsed_s: float = 0.0
    by_status: dict[str, int] = field(default_factory=dict)
    crash_samples: list[str] = field(default_factory=list)
    rss_delta_kb: int = 0

    @property
    def runs_per_sec(self) -> float:
        return self.n / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def ok(self) -> bool:
        return self.harness_crashes == 0 and self.hangs == 0


def make_corpus(n: int, rng: random.Random) -> list[bytes]:
    blobs: list[bytes] = []
    # Always include the fixed seeds and a few empty / tiny cases.
    blobs.extend(_SEED_ASM)
    blobs.append(b"")
    blobs.append(b"\x00")
    blobs.append(b"\x90" * 64 + b"\xc3")
    while len(blobs) < n:
        kind = rng.randrange(5)
        if kind == 0:
            length = rng.choice([1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            blobs.append(bytes(rng.getrandbits(8) for _ in range(length)))
        elif kind == 1:
            base = rng.choice(_SEED_ASM)
            blobs.append(_mutate(base, rng))
        elif kind == 2:
            # Insert a privileged / stop opcode somewhere in a real function.
            base = bytearray(rng.choice(_SEED_ASM[1:5]))
            op = rng.choice([b"\x0f\x05", b"\xcd\x80", b"\x0f\x34", b"\xcc", b"\xf4", b"\xeb\xfe"])
            i = rng.randrange(len(base) + 1)
            blobs.append(bytes(base[:i] + op + base[i:]))
        elif kind == 3:
            blobs.append(os.urandom(rng.randrange(1, 64)))
        else:
            # Random length with a trailing ret, often still well-formed enough to run.
            length = rng.randrange(0, 48)
            blobs.append(bytes(rng.getrandbits(8) for _ in range(length)) + b"\xc3")
    return blobs[:n]


def _mutate(blob: bytes, rng: random.Random) -> bytes:
    data = bytearray(blob or b"\x90")
    for _ in range(rng.randrange(1, 6)):
        op = rng.randrange(4)
        if op == 0 and data:
            data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        elif op == 1:
            data.insert(rng.randrange(len(data) + 1), rng.getrandbits(8))
        elif op == 2 and data:
            del data[rng.randrange(len(data))]
        elif data:
            j = rng.randrange(len(data))
            data[j : j + 1] = bytes([rng.getrandbits(8), rng.getrandbits(8)])
    return bytes(data)


def _run_one(blob: bytes) -> dict[str, Any]:
    try:
        result = emulate(
            blob,
            signature="int64_t f(int64_t a, int64_t b)",
            args=[1, 2],
            insn_limit=4_000,
            timeout_s=0.05,
            check_abi=False,
            count_instructions=False,
        )
        if result.status == "harness_error":
            return {"harness_crash": True, "status": result.status, "message": result.message}
        return {"harness_crash": False, "status": result.status, "message": result.message}
    except Exception as exc:  # noqa: BLE001
        return {
            "harness_crash": True,
            "status": "exception",
            "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _run_batch(blobs: list[bytes]) -> list[dict[str, Any]]:
    return [_run_one(b) for b in blobs]


def _rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


def run_fuzz(
    n: int = 10_000,
    *,
    workers: int | None = None,
    seed: int = 1,
    batch_size: int = 32,
    batch_timeout_s: float = 30.0,
) -> FuzzReport:
    rng = random.Random(seed)
    blobs = make_corpus(n, rng)
    if workers is None:
        workers = min(os.cpu_count() or 2, 8)
    workers = max(1, workers)

    chunks = [blobs[i : i + batch_size] for i in range(0, len(blobs), batch_size)]
    report = FuzzReport(n=len(blobs))
    rss_before = _rss_kb()
    t0 = time.perf_counter()

    ctx = get_context("fork")
    hangs = 0
    crashes = 0
    by_status: dict[str, int] = {}
    samples: list[str] = []

    with ctx.Pool(processes=workers) as pool:
        asyncs = [pool.apply_async(_run_batch, (chunk,)) for chunk in chunks]
        for ar in asyncs:
            try:
                results = ar.get(timeout=batch_timeout_s)
            except Exception as exc:  # timeout or worker death
                hangs += 1
                samples.append(f"batch hang/fail: {exc}")
                continue
            for rec in results:
                st = rec.get("status") or "unknown"
                by_status[st] = by_status.get(st, 0) + 1
                if rec.get("harness_crash"):
                    crashes += 1
                    if len(samples) < 8:
                        samples.append(rec.get("message") or st)

    report.elapsed_s = time.perf_counter() - t0
    report.harness_crashes = crashes
    report.hangs = hangs
    report.by_status = dict(sorted(by_status.items(), key=lambda kv: -kv[1]))
    report.crash_samples = samples
    report.rss_delta_kb = _rss_kb() - rss_before
    return report
