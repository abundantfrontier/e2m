"""Re-verify scaled-sweep ``unsupported_insn`` rows on the QEMU backend.

Unicorn 2.1.4 drops YMM, FMA, and non-destructive VEX. Those rows are
compiled again and executed only under QEMU (``E2M_EMU_BACKEND=qemu``).
The correctness pass does not count instructions. A second pass counts
dynamic instructions once per ``(task_id, sha256)`` and copies that count
onto every recovered row. A missing count becomes ``INSN_COUNT_UNKNOWN``
so the variant is not ranked as the fastest.

The recover jsonl is append-only and resumable. Merge overlays those
rows onto the sweep cache by ``config_id`` and regenerates the scaled
datasets. It does not append a second copy of a config.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from e2m.data.aggregate import load_cache, write_outputs
from e2m.data.sweep import (
    GRID_NOTE_REDUCED,
    _init,
    _task_dicts,
    _worker,
    detect_compilers,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "scaled" / "sweep-cache.jsonl"
DEFAULT_TASKS = ROOT / "tasks" / "scaled.jsonl"
DEFAULT_OUT = ROOT / "data" / "scaled" / "qemu-recover.jsonl"
DEFAULT_INSN = ROOT / "data" / "scaled" / "qemu-insn.jsonl"
DEFAULT_TMP = ROOT / "runs" / "qemu-tmp"

# Larger than any real dynamic count we record, so unknown rows sort last
# on the speed ranking (fewer instructions is better).
INSN_COUNT_UNKNOWN = 10**12


def job_from_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": row["config_id"],
        "task_id": row["task_id"],
        "compiler": row["compiler"],
        "flags": list(row.get("flags") or []),
        "seed": row.get("seed"),
        "passes": row.get("passes"),
        "knobs": row.get("knobs") or {},
    }


def iter_unsupported_jobs(cache_path: Path) -> Iterable[dict[str, Any]]:
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            if "unsupported_insn" not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "fail" and row.get("reason") == "unsupported_insn":
                if row.get("config_id") and row.get("task_id") and row.get("compiler"):
                    yield job_from_cache_row(row)


def load_jsonl_by_id(path: Path, key: str = "config_id") -> dict[str, dict[str, Any]]:
    """Last complete record wins. A truncated final line is ignored."""
    found: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ident = row.get(key)
            if ident:
                found[str(ident)] = row
    return found


def overlay_rows(
    cache_rows: Iterable[dict[str, Any]],
    recovered_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace cache rows with a recovered result that shares ``config_id``.

    Configs that are not in the cache are not appended.
    """
    out: list[dict[str, Any]] = []
    for row in cache_rows:
        cid = str(row.get("config_id") or "")
        if cid and cid in recovered_by_id:
            out.append(recovered_by_id[cid])
        else:
            out.append(row)
    return out


def apply_insn_counts(
    rows: Iterable[dict[str, Any]],
    counts: dict[tuple[str, str], int],
) -> int:
    """Write dynamic counts onto passing rows. Returns how many were unknown."""
    unknown = 0
    for row in rows:
        if row.get("status") != "pass" or not row.get("sha256"):
            continue
        key = (str(row["task_id"]), str(row["sha256"]))
        if key in counts and int(counts[key]) > 0:
            row["insn_count"] = int(counts[key])
        elif int(row.get("insn_count") or 0) <= 0:
            row["insn_count"] = INSN_COUNT_UNKNOWN
            unknown += 1
    return unknown


def _recover_init(task_dicts: list[dict[str, Any]], tools: dict[str, str], timeout_s: str, tmp: str) -> None:
    os.environ["E2M_EMU_BACKEND"] = "qemu"
    os.environ["E2M_COUNT_INSNS"] = "0"
    os.environ["E2M_TEST_TIMEOUT"] = timeout_s
    os.environ["TMPDIR"] = tmp
    Path(tmp).mkdir(parents=True, exist_ok=True)
    _init(task_dicts, tools)


def _count_init(task_dicts: list[dict[str, Any]], timeout_s: str, tmp: str) -> None:
    os.environ["E2M_EMU_BACKEND"] = "qemu"
    os.environ["E2M_COUNT_INSNS"] = "1"
    os.environ["TMPDIR"] = tmp
    Path(tmp).mkdir(parents=True, exist_ok=True)
    _init(task_dicts, {})
    global _COUNT_TIMEOUT
    _COUNT_TIMEOUT = float(timeout_s)


_COUNT_TIMEOUT = 3.0


def _count_one(item: dict[str, Any]) -> dict[str, Any]:
    """Assemble a stored AT&T variant and count instructions under QEMU."""
    from e2m.harness.assemble import AssembleError, assemble
    from e2m.harness.tasks import run_task, task_from_dict
    from e2m.data.sweep import _TASKS

    task_id = item["task_id"]
    sha = item["sha256"]
    try:
        task = task_from_dict(_TASKS[task_id])
        image = assemble(item["asm"], syntax="att")
        run = run_task(
            task,
            image,
            syntax="att",
            timeout_s=_COUNT_TIMEOUT,
            backend="qemu",
            count_instructions=True,
            insn_limit=1_000_000,
        )
    except AssembleError as exc:
        return {"task_id": task_id, "sha256": sha, "insn_count": 0, "status": "assemble_error", "detail": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": task_id,
            "sha256": sha,
            "insn_count": 0,
            "status": "worker_error",
            "detail": f"{exc}\n{traceback.format_exc()[-300:]}",
        }
    if not run.passed or run.insn_count <= 0:
        return {
            "task_id": task_id,
            "sha256": sha,
            "insn_count": 0,
            "status": "fail",
            "detail": (run.fail_reason or "zero_count")[:300],
        }
    return {"task_id": task_id, "sha256": sha, "insn_count": int(run.insn_count), "status": "pass", "detail": ""}


def _append(path: Path, row: dict[str, Any], fh) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _default_sft_digest(sft_path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    if not sft_path.is_file():
        return ""
    with sft_path.open(encoding="utf-8") as fh:
        for line in fh:
            if '"goal"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("goal") != "default":
                continue
            h.update(str(row.get("task_id", "")).encode())
            h.update(b"\0")
            h.update(str(row.get("asm", "")).encode())
            h.update(b"\n")
    return h.hexdigest()


def run_correctness(
    *,
    cache_path: Path,
    tasks_path: Path,
    out_path: Path,
    jobs: int,
    timeout_s: float,
    limit: int,
    tmp: Path,
) -> dict[str, Any]:
    jobs_list = list(iter_unsupported_jobs(cache_path))
    done = load_jsonl_by_id(out_path)
    pending = [job for job in jobs_list if job["config_id"] not in done]
    if limit > 0:
        pending = pending[:limit]
    task_dicts = _task_dicts(tasks_path)
    compilers, _ = detect_compilers(["gcc", "clang"])
    from e2m.data.sweep import _active_tools

    tools = {name: path for name, path in _active_tools(compilers).items()}
    tmp.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"qemu recover: {len(jobs_list)} unsupported_insn, {len(done)} cached, "
        f"{len(pending)} to run, {jobs} workers, timeout {timeout_s}s",
        flush=True,
    )
    started = time.time()
    passed = failed = 0
    reasons: Counter[str] = Counter()
    # Recount already-cached rows so a resume reports the full total.
    for row in done.values():
        if row.get("status") == "pass":
            passed += 1
        else:
            failed += 1
            reasons[str(row.get("reason") or "fail")] += 1
    if pending:
        ctx = mp.get_context("spawn")
        progress = out_path.with_name("qemu-recover-progress.txt")
        with out_path.open("a", encoding="utf-8") as fh:
            with ctx.Pool(
                processes=max(1, jobs),
                initializer=_recover_init,
                initargs=(task_dicts, tools, str(timeout_s), str(tmp)),
                maxtasksperchild=24,
            ) as pool:
                for i, result in enumerate(pool.imap_unordered(_worker, pending, chunksize=1), 1):
                    _append(out_path, result, fh)
                    if result.get("status") == "pass":
                        passed += 1
                    else:
                        failed += 1
                        reasons[str(result.get("reason") or "fail")] += 1
                    if i % 25 == 0 or i == len(pending):
                        elapsed = time.time() - started
                        rate = i / elapsed if elapsed else 0
                        line = (
                            f"{i}/{len(pending)} pass={passed} fail={failed} "
                            f"{rate:.2f} cfg/s elapsed={elapsed:.0f}s"
                        )
                        print("  " + line, flush=True)
                        progress.write_text(line + "\n", encoding="utf-8")
    stats = {
        "unsupported_insn": len(jobs_list),
        "cached": len(done),
        "ran": len(pending),
        "pass": passed,
        "fail": failed,
        "reasons": dict(reasons),
        "elapsed_s": round(time.time() - started, 2),
    }
    (out_path.with_name("qemu-recover-stats.json")).write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print("qemu recover correctness", stats, flush=True)
    return stats


def run_counts(
    *,
    tasks_path: Path,
    recover_path: Path,
    insn_path: Path,
    jobs: int,
    timeout_s: float,
    tmp: Path,
) -> dict[str, Any]:
    recovered = load_jsonl_by_id(recover_path)
    counted_keys = _load_count_keys(insn_path)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recovered.values():
        if row.get("status") != "pass" or not row.get("sha256") or not row.get("asm"):
            continue
        key = (str(row["task_id"]), str(row["sha256"]))
        if key in counted_keys or key in unique:
            continue
        unique[key] = {"task_id": key[0], "sha256": key[1], "asm": row["asm"]}
    pending = list(unique.values())
    print(
        f"qemu insn count: {len(counted_keys)} cached, {len(pending)} unique variants, {jobs} workers",
        flush=True,
    )
    task_dicts = _task_dicts(tasks_path)
    tmp.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = fail = 0
    if pending:
        ctx = mp.get_context("spawn")
        with insn_path.open("a", encoding="utf-8") as fh:
            with ctx.Pool(
                processes=max(1, jobs),
                initializer=_count_init,
                initargs=(task_dicts, str(timeout_s), str(tmp)),
                maxtasksperchild=16,
            ) as pool:
                for i, result in enumerate(pool.imap_unordered(_count_one, pending, chunksize=1), 1):
                    _append(insn_path, result, fh)
                    if result.get("status") == "pass":
                        ok += 1
                    else:
                        fail += 1
                    if i % 20 == 0 or i == len(pending):
                        elapsed = time.time() - started
                        rate = i / elapsed if elapsed else 0
                        print(
                            f"  count {i}/{len(pending)} ok={ok} fail={fail} {rate:.2f}/s",
                            flush=True,
                        )
    stats = {"unique_new": len(pending), "ok": ok, "fail": fail, "elapsed_s": round(time.time() - started, 2)}
    print("qemu insn count", stats, flush=True)
    return stats


def _load_count_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id") and row.get("sha256"):
                keys.add((str(row["task_id"]), str(row["sha256"])))
    return keys


def load_count_map(path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "pass":
                continue
            key = (str(row.get("task_id")), str(row.get("sha256")))
            n = int(row.get("insn_count") or 0)
            if key[0] and key[1] and n > 0:
                counts[key] = n
    return counts


def merge_scaled(
    *,
    cache_path: Path,
    tasks_path: Path,
    recover_path: Path,
    insn_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    sft_path = out_dir / "sft.jsonl"
    before = _default_sft_digest(sft_path)
    recovered = load_jsonl_by_id(recover_path)
    counts = load_count_map(insn_path)
    unknown = apply_insn_counts(recovered.values(), counts)
    print(
        f"merge: overlay {len(recovered)} configs, insn counts for {len(counts)} variants, "
        f"{unknown} passing rows still unknown",
        flush=True,
    )
    # Stream the cache once. Passing rows carry asm, so this is the large step.
    rows = overlay_rows(load_cache(cache_path), recovered)
    task_dicts = _task_dicts(tasks_path)
    tools_path = out_dir / "sweep-tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8")) if tools_path.is_file() else {}
    compilers, _ = detect_compilers(["gcc", "clang"])
    stats = write_outputs(
        out_dir=out_dir,
        rows=rows,
        tasks_by_id={t["id"]: t for t in task_dicts},
        tools=tools,
        compilers=compilers,
        grid_note=GRID_NOTE_REDUCED,
    )
    after = _default_sft_digest(sft_path)
    recovered_pass = sum(1 for r in recovered.values() if r.get("status") == "pass")
    note = (
        "\n## QEMU re-check of unsupported_insn\n\n"
        f"- Re-ran {len(recovered)} Unicorn `unsupported_insn` configs under QEMU "
        f"(`qemu-x86_64 -cpu max`) after giving short arrays a read-only tail page.\n"
        f"- Recovered passing configs: {recovered_pass}. "
        f"Failing configs stay failed and are not added as variants.\n"
        f"- Dynamic instruction counts were measured once per recovered `(task, sha256)` "
        f"with `-one-insn-per-tb -d exec`. {unknown} passing rows had no count and use "
        f"insn_count={INSN_COUNT_UNKNOWN} so they do not rank as fastest.\n"
        f"- goal=default SFT digest before merge `{before}` and after `{after}` "
        f"({'unchanged' if before == after else 'CHANGED'}).\n"
        "- The recover jsonl is gitignored. This report and the jsonl datasets include the overlay.\n"
    )
    report = out_dir / "SWEEP_REPORT.md"
    report.write_text(report.read_text(encoding="utf-8") + note, encoding="utf-8")
    stats.update(
        {
            "recovered_configs": len(recovered),
            "recovered_pass": recovered_pass,
            "insn_counts": len(counts),
            "insn_unknown_rows": unknown,
            "default_sft_unchanged": before == after,
            "default_sft_digest": after,
        }
    )
    (out_dir / "qemu-recover-stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print("merge", stats, flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QEMU re-check of scaled unsupported_insn configs")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--insn-out", type=Path, default=DEFAULT_INSN)
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--count-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tmp", type=Path, default=DEFAULT_TMP)
    parser.add_argument("--count", action="store_true", help="only the instruction-count pass")
    parser.add_argument("--merge", action="store_true", help="overlay recover results and regenerate datasets")
    parser.add_argument("--all", action="store_true", help="correctness, then counts, then merge")
    args = parser.parse_args(argv)

    if args.merge and not args.all:
        merge_scaled(
            cache_path=args.cache,
            tasks_path=args.tasks,
            recover_path=args.out,
            insn_path=args.insn_out,
            out_dir=args.out.parent,
        )
        return 0
    if args.count and not args.all:
        run_counts(
            tasks_path=args.tasks,
            recover_path=args.out,
            insn_path=args.insn_out,
            jobs=args.jobs,
            timeout_s=args.count_timeout,
            tmp=args.tmp,
        )
        return 0
    run_correctness(
        cache_path=args.cache,
        tasks_path=args.tasks,
        out_path=args.out,
        jobs=args.jobs,
        timeout_s=args.timeout,
        limit=args.limit,
        tmp=args.tmp,
    )
    if args.all and args.limit == 0:
        run_counts(
            tasks_path=args.tasks,
            recover_path=args.out,
            insn_path=args.insn_out,
            jobs=args.jobs,
            timeout_s=args.count_timeout,
            tmp=args.tmp,
        )
        merge_scaled(
            cache_path=args.cache,
            tasks_path=args.tasks,
            recover_path=args.out,
            insn_path=args.insn_out,
            out_dir=args.out.parent,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
