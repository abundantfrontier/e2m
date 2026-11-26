"""Verify held-out tasks at gcc -O0 and -O2 and write eval/heldout.jsonl.

The set is never a training source. A task is dropped when its function name
or normalized body is a near-duplicate of tasks/scaled.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2m.harness.tasks import assemble_reference, dump_jsonl, run_task, task_from_dict
from e2m.heldout.catalog import tasks as catalog_tasks
from e2m.heldout.near_dup import collision, load_scaled_index

ROOT = Path(__file__).resolve().parents[2]


def build(*, scaled_path: Path, out_path: Path) -> dict:
    rows = catalog_tasks()
    names, bodies = load_scaled_index(scaled_path)
    kept = []
    dropped = []
    for row in rows:
        why = collision(row, names, bodies)
        if why:
            dropped.append({"id": row["id"], "source": row["source"], "why": why})
            continue
        kept.append(row)
    if len(kept) < 60:
        raise SystemExit(f"only {len(kept)} held-out tasks survived near-dup filtering")
    sources = {r["source"].split("/")[0] for r in kept}
    if "HumanEval" not in sources or "MBPP" not in sources:
        raise SystemExit(f"held-out sources missing a benchmark: {sources}")

    verified = []
    failures = []
    for row in kept:
        task = task_from_dict(row)
        for opt in ("-O0", "-O2"):
            try:
                image = assemble_reference(task, opt=opt)
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": row["id"], "opt": opt, "reason": f"assemble: {exc}"[:300]})
                break
            run = run_task(task, image, timeout_s=2.0)
            if not run.passed:
                failures.append({"id": row["id"], "opt": opt, "reason": run.fail_reason[:300]})
                break
        else:
            verified.append(row)
    if failures:
        detail = "\n".join(f"  {f['id']} {f['opt']}: {f['reason']}" for f in failures[:20])
        raise SystemExit(f"{len(failures)} held-out tasks failed the harness:\n{detail}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl((task_from_dict(r) for r in verified), out_path)
    report = {
        "count": len(verified),
        "humaneval": sum(1 for r in verified if r["source"].startswith("HumanEval/")),
        "mbpp": sum(1 for r in verified if r["source"].startswith("MBPP/")),
        "dropped": dropped,
        "ids": [r["id"] for r in verified],
    }
    (out_path.parent / "heldout_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Held-out evaluation set",
        "",
        "This file is never a training source. Do not add these tasks to `data/scaled/sft.jsonl` or any other train split.",
        "",
        f"- Tasks: {report['count']} ({report['humaneval']} HumanEval, {report['mbpp']} MBPP).",
        "- Each task has an English spec, a SysV signature, a leaf C reference, and tests.",
        "- Every task passed the harness at `gcc -O0` and `gcc -O2`.",
        "- Near-duplicate check against `tasks/scaled.jsonl`: function name, task id, exact normalized body, and difflib ratio >= 0.92.",
        f"- Dropped by that check: {len(dropped)}.",
        "",
    ]
    if dropped:
        lines.append("Dropped:")
        lines.append("")
        for item in dropped:
            lines.append(f"- `{item['id']}` ({item['source']}): {item['why']}")
        lines.append("")
    (out_path.parent / "HELDOUT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"heldout: {report['count']} tasks "
        f"(HumanEval {report['humaneval']}, MBPP {report['mbpp']}), dropped {len(dropped)}",
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build eval/heldout.jsonl")
    parser.add_argument("--scaled", type=Path, default=ROOT / "tasks" / "scaled.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    args = parser.parse_args(argv)
    build(scaled_path=args.scaled, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
