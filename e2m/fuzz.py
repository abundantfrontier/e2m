"""CLI: python -m e2m.fuzz --n 10000"""

from __future__ import annotations

import argparse
import json
import sys

from e2m.harness.fuzz import run_fuzz


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fuzz the E2M Unicorn harness")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    report = run_fuzz(n=args.n, workers=args.workers, seed=args.seed)
    payload = {
        "n": report.n,
        "harness_crashes": report.harness_crashes,
        "hangs": report.hangs,
        "elapsed_s": report.elapsed_s,
        "runs_per_sec": report.runs_per_sec,
        "by_status": report.by_status,
        "rss_delta_kb": report.rss_delta_kb,
        "crash_samples": report.crash_samples,
        "ok": report.ok,
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"fuzz n={report.n} crashes={report.harness_crashes} hangs={report.hangs}")
        print(f"elapsed {report.elapsed_s:.2f}s  throughput {report.runs_per_sec:.1f} runs/s")
        print(f"status histogram: {report.by_status}")
        if report.crash_samples:
            print("samples:")
            for s in report.crash_samples:
                print(" ---")
                print(s[:1000])
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
