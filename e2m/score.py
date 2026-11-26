"""CLI: python -m e2m.score --tasks tasks.jsonl --candidates cands.jsonl"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from e2m.harness.score import load_candidates, render_markdown, score_tasks, write_report
from e2m.harness.starter import write_starter_jsonl
from e2m.harness.tasks import load_jsonl, load_starter, starter_jsonl_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score E2M assembly candidates against tasks")
    p.add_argument("--tasks", type=Path, default=None, help="tasks JSONL (default: starter set)")
    p.add_argument("--candidates", type=Path, required=True, help="candidates JSONL")
    p.add_argument("--k", default="1,10", help="comma-separated k values for pass@k")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument(
        "--baseline",
        default="O2",
        help="gcc opt level without a dash (O0, O2, Os) or 'none' to skip",
    )
    args = p.parse_args(argv)

    if args.tasks is None:
        path = starter_jsonl_path()
        if not path.is_file():
            write_starter_jsonl(path)
        tasks = load_starter()
    else:
        tasks = load_jsonl(args.tasks)
    cands = load_candidates(args.candidates)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    if args.baseline in {"none", "off", ""}:
        baseline = None
    else:
        baseline = args.baseline if str(args.baseline).startswith("-") else f"-{args.baseline}"
    report = score_tasks(tasks, cands, ks=ks, baseline_opt=baseline)
    md = render_markdown(report)
    sys.stdout.write(md)
    write_report(report, args.out_json, args.out_md)
    if args.out_json is None:
        # still emit JSON to stdout after the markdown if no file was given? keep md-only on stdout.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
