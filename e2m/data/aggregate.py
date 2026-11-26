"""Dedupe, rank, and report a finished sweep cache."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GOALS = ("size", "speed", "default")
PAIR_CAP = 8


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _flag_set(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "compiler": row["compiler"],
        "flags": list(row.get("flags") or []),
        "seed": row.get("seed"),
        "passes": row.get("passes"),
        "knobs": row.get("knobs") or {},
    }


def unique_variants(
    rows: Iterable[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One record per (task, sha256), carrying every flag set that produced it."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "pass" or not row.get("sha256"):
            continue
        key = (row["task_id"], row["sha256"])
        slot = groups.get(key)
        fs = _flag_set(row)
        if slot is None:
            task = tasks_by_id.get(row["task_id"], {})
            groups[key] = {
                "task_id": row["task_id"],
                "spec": task.get("spec", ""),
                "signature": task.get("signature", ""),
                "sha256": row["sha256"],
                "size": int(row["size"]),
                "insn_count": int(row["insn_count"]),
                "insn_per_test": list(row.get("insn_per_test") or []),
                "asm": row.get("asm") or "",
                "bytes": row.get("bytes_hex") or "",
                "rodata": row.get("rodata_hex") or "",
                "compilers": [row["compiler"]],
                "flag_sets": [fs],
            }
        else:
            if row["compiler"] not in slot["compilers"]:
                slot["compilers"].append(row["compiler"])
            slot["flag_sets"].append(fs)
    variants = list(groups.values())
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        by_task[variant["task_id"]].append(variant)
    for task_id, group in by_task.items():
        by_size = sorted(group, key=lambda v: (v["size"], v["insn_count"], v["sha256"]))
        by_speed = sorted(group, key=lambda v: (v["insn_count"], v["size"], v["sha256"]))
        size_rank = {v["sha256"]: i + 1 for i, v in enumerate(by_size)}
        speed_rank = {v["sha256"]: i + 1 for i, v in enumerate(by_speed)}
        for variant in group:
            variant["rank_size"] = size_rank[variant["sha256"]]
            variant["rank_speed"] = speed_rank[variant["sha256"]]
        _ = task_id
    variants.sort(key=lambda v: (v["task_id"], v["rank_size"], v["sha256"]))
    return variants


def _canonical_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """gcc -O2 -march=x86-64 with no extra feature or toggle, if it passed."""
    for compiler in ("gcc", "clang"):
        for row in rows:
            if row.get("status") != "pass" or row.get("compiler") != compiler:
                continue
            knobs = row.get("knobs") or {}
            if (
                knobs.get("opt") == "-O2"
                and knobs.get("cpu") == "x86-64"
                and not knobs.get("feat")
                and not knobs.get("toggle")
            ):
                return row
    for row in rows:
        if row.get("status") != "pass" or row.get("compiler") != "rustc":
            continue
        knobs = row.get("knobs") or {}
        if (
            knobs.get("opt") == "2"
            and knobs.get("cpu") == "x86-64"
            and not knobs.get("feat")
            and knobs.get("overflow") == "off"
            and not knobs.get("toggle")
        ):
            return row
    return None


def _variant_for_row(variants: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    for variant in variants:
        if variant["task_id"] == row["task_id"] and variant["sha256"] == row["sha256"]:
            return variant
    return None


def _balanced_key(variant: dict[str, Any], min_size: int, min_insn: int) -> tuple:
    size_term = variant["size"] / max(min_size, 1)
    insn_term = variant["insn_count"] / max(min_insn, 1)
    return (size_term + insn_term, variant["size"], variant["insn_count"], variant["sha256"])


def _pick_sft(
    task_variants: list[dict[str, Any]],
    passing_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map goal -> chosen variant."""
    if not task_variants:
        return {}
    by_size = sorted(task_variants, key=lambda v: (v["size"], v["insn_count"], v["sha256"]))
    by_speed = sorted(task_variants, key=lambda v: (v["insn_count"], v["size"], v["sha256"]))
    min_size = by_size[0]["size"]
    min_insn = by_speed[0]["insn_count"]
    by_default = sorted(task_variants, key=lambda v: _balanced_key(v, min_size, min_insn))
    canon = _canonical_row(passing_rows)
    default_variant = by_default[0]
    if canon is not None:
        found = _variant_for_row(task_variants, canon)
        if found is not None:
            default_variant = found
    return {"size": by_size[0], "speed": by_speed[0], "default": default_variant}


def _represent_flags(variant: dict[str, Any]) -> dict[str, Any]:
    """Prefer a gcc flag set when several compilers emitted the same bytes."""
    order = {"gcc": 0, "clang": 1, "llvm-opt": 2, "rustc": 3}
    sets = sorted(
        variant["flag_sets"],
        key=lambda fs: (order.get(fs["compiler"], 9), len(fs["flags"]), json.dumps(fs["flags"])),
    )
    return sets[0]


def sft_records(
    variants: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        by_task[variant["task_id"]].append(variant)
    for row in rows:
        if row.get("status") == "pass":
            rows_by_task[row["task_id"]].append(row)
    out: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        chosen = _pick_sft(by_task[task_id], rows_by_task[task_id])
        for goal, variant in chosen.items():
            flags = _represent_flags(variant)
            out.append(
                {
                    "task_id": task_id,
                    "spec": variant["spec"],
                    "signature": variant["signature"],
                    "goal": goal,
                    "asm": variant["asm"],
                    "sha256": variant["sha256"],
                    "size": variant["size"],
                    "insn_count": variant["insn_count"],
                    "compiler": flags["compiler"],
                    "flags": flags["flags"],
                    "seed": flags.get("seed"),
                }
            )
    return out


def _rank_for_goal(
    task_variants: list[dict[str, Any]],
    goal: str,
    default_sha: str | None,
) -> list[dict[str, Any]]:
    if goal == "size":
        return sorted(task_variants, key=lambda v: (v["size"], v["insn_count"], v["sha256"]))
    if goal == "speed":
        return sorted(task_variants, key=lambda v: (v["insn_count"], v["size"], v["sha256"]))
    # default: distance from the chosen default variant
    origin = next((v for v in task_variants if v["sha256"] == default_sha), task_variants[0])
    base_s = max(origin["size"], 1)
    base_i = max(origin["insn_count"], 1)

    def dist(v: dict[str, Any]) -> tuple:
        score = abs(v["size"] - origin["size"]) / base_s + abs(v["insn_count"] - origin["insn_count"]) / base_i
        return (score, v["size"], v["insn_count"], v["sha256"])

    return sorted(task_variants, key=dist)


def _sample_worse(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worse = ranked[1:]
    if len(worse) <= PAIR_CAP:
        return worse
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    last = len(worse) - 1
    for i in range(PAIR_CAP):
        idx = round(i * last / (PAIR_CAP - 1))
        sha = worse[idx]["sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        picked.append(worse[idx])
    return picked


def _brief_variant(variant: dict[str, Any]) -> dict[str, Any]:
    flags = _represent_flags(variant)
    return {
        "asm": variant["asm"],
        "sha256": variant["sha256"],
        "size": variant["size"],
        "insn_count": variant["insn_count"],
        "compiler": flags["compiler"],
        "flags": flags["flags"],
        "seed": flags.get("seed"),
    }


def pair_records(variants: list[dict[str, Any]], sft: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        by_task[variant["task_id"]].append(variant)
    default_sha = {
        row["task_id"]: row["sha256"]
        for row in sft
        if row["goal"] == "default"
    }
    out: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        group = by_task[task_id]
        if len(group) < 2:
            continue
        spec = group[0]["spec"]
        signature = group[0]["signature"]
        for goal in GOALS:
            ranked = _rank_for_goal(group, goal, default_sha.get(task_id))
            best = ranked[0]
            chosen = _brief_variant(best)
            seen: set[tuple[str, str]] = set()
            for other in _sample_worse(ranked):
                key = (best["sha256"], other["sha256"])
                if key in seen or other["sha256"] == best["sha256"]:
                    continue
                # Skip a pair that is not strictly worse on the goal's metric.
                if goal == "size" and other["size"] < best["size"]:
                    continue
                if goal == "speed" and other["insn_count"] < best["insn_count"]:
                    continue
                if goal == "size" and other["size"] == best["size"] and other["insn_count"] == best["insn_count"]:
                    continue
                if goal == "speed" and other["insn_count"] == best["insn_count"] and other["size"] == best["size"]:
                    continue
                seen.add(key)
                rejected = _brief_variant(other)
                out.append(
                    {
                        "task_id": task_id,
                        "spec": spec,
                        "signature": signature,
                        "goal": goal,
                        "chosen": chosen,
                        "rejected": rejected,
                        "margin": {
                            "size": other["size"] - best["size"],
                            "insn_count": other["insn_count"] - best["insn_count"],
                        },
                    }
                )
    return out


def _knob_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("knobs") or {}).keys())
    names.discard("passes")
    names.discard("seed")
    return sorted(names)


def knob_diversity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How often varying one knob changes the machine code, holding the rest fixed."""
    passing = [r for r in rows if r.get("status") == "pass" and r.get("sha256")]
    knobs = [k for k in _knob_names(passing) if k in {"opt", "cpu", "feat", "toggle", "overflow"}]
    report: list[dict[str, Any]] = []
    for knob in knobs:
        groups: dict[tuple, set[str]] = defaultdict(set)
        for row in passing:
            kn = row.get("knobs") or {}
            if knob not in kn:
                continue
            frozen = tuple(
                sorted(
                    (key, "" if val is None else str(val))
                    for key, val in kn.items()
                    if key != knob
                    and key not in {"passes", "ir_attrs"}
                    # Default LLVM rows copy the pipeline label into seed, which
                    # would hide opt diversity. Random seeds stay in the key.
                    and not (key == "seed" and str(val) == str(kn.get("opt")))
                )
            )
            groups[(row["task_id"], row["compiler"], frozen)].add(row["sha256"])
        if not groups:
            continue
        counts = [len(shas) for shas in groups.values()]
        changed = sum(1 for n in counts if n > 1)
        report.append(
            {
                "knob": knob,
                "groups": len(counts),
                "groups_with_diversity": changed,
                "fraction": changed / len(counts),
                "mean_unique": statistics.fmean(counts),
                "max_unique": max(counts),
            }
        )
    report.sort(key=lambda r: (-r["fraction"], -r["mean_unique"], r["knob"]))
    return report


def summarize(rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_tc: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        key = (row.get("task_id", "?"), row.get("compiler", "?"))
        by_tc[key]["generated"] += 1
        if row.get("status") == "pass":
            by_tc[key]["pass"] += 1
        else:
            by_tc[key]["fail"] += 1
            by_tc[key]["fail:" + str(row.get("reason") or "unknown")] += 1
    uniq: Counter = Counter()
    for variant in variants:
        for compiler in variant["compilers"]:
            uniq[(variant["task_id"], compiler)] += 1
    reasons = Counter(str(r.get("reason") or "unknown") for r in rows if r.get("status") != "pass")
    return {"by_tc": by_tc, "uniq": uniq, "reasons": reasons}


def _spread(variants: list[dict[str, Any]], field: str) -> str:
    if not variants:
        return "-"
    vals = sorted(int(v[field]) for v in variants)
    if len(vals) == 1:
        return str(vals[0])
    mid = vals[len(vals) // 2]
    return f"{vals[0]} / {mid} / {vals[-1]}"


def render_report(
    *,
    rows: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    tools: dict[str, str],
    compilers: list[str],
    grid_note: str,
) -> str:
    summary = summarize(rows, variants)
    diversity = knob_diversity(rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        by_task[variant["task_id"]].append(variant)
    task_ids = sorted({r.get("task_id", "?") for r in rows} | set(by_task))
    lines: list[str] = []
    lines.append("# Compiler sweep report")
    lines.append("")
    lines.append("Every kept variant was assembled and executed under Unicorn by the E2M harness. No generated code ran on the host CPU.")
    lines.append("")
    lines.append("## Tools")
    lines.append("")
    if not tools:
        lines.append("No compiler versions were recorded.")
    for name, version in tools.items():
        lines.append(f"- `{name}`: {version}")
    lines.append("")
    lines.append("## Grid")
    lines.append("")
    lines.append(grid_note.rstrip())
    lines.append("")
    lines.append("Compilers in this run: " + ", ".join(f"`{c}`" for c in compilers) + ".")
    lines.append("")
    total = len(rows)
    passed = sum(1 for r in rows if r.get("status") == "pass")
    failed = total - passed
    lines.append("## Totals")
    lines.append("")
    lines.append("| | count |")
    lines.append("|---|---:|")
    lines.append(f"| configs generated | {total} |")
    lines.append(f"| passing | {passed} |")
    lines.append(f"| failing | {failed} |")
    lines.append(f"| unique variants (task, sha256) | {len(variants)} |")
    lines.append("")
    lines.append("## Per task and compiler")
    lines.append("")
    header = "| task | " + " | ".join(f"{c} gen/pass/uniq" for c in compilers) + " | size min/med/max | insn min/med/max |"
    sep = "|---|" + "|".join("---:" for _ in compilers) + "|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for task_id in task_ids:
        cells = []
        for compiler in compilers:
            key = (task_id, compiler)
            gen = summary["by_tc"][key]["generated"] if key in summary["by_tc"] else 0
            ok = summary["by_tc"][key]["pass"] if key in summary["by_tc"] else 0
            nu = summary["uniq"][key]
            cells.append(f"{gen}/{ok}/{nu}")
        lines.append(
            f"| {task_id} | "
            + " | ".join(cells)
            + f" | {_spread(by_task[task_id], 'size')} | {_spread(by_task[task_id], 'insn_count')} |"
        )
    lines.append("")
    lines.append("## Per compiler")
    lines.append("")
    lines.append("| compiler | generated | passing | failing | unique |")
    lines.append("|---|---:|---:|---:|---:|")
    for compiler in compilers:
        gen = ok = fail = uniq = 0
        for task_id in task_ids:
            key = (task_id, compiler)
            gen += summary["by_tc"][key]["generated"]
            ok += summary["by_tc"][key]["pass"]
            fail += summary["by_tc"][key]["fail"]
            uniq += summary["uniq"][key]
        lines.append(f"| {compiler} | {gen} | {ok} | {fail} | {uniq} |")
    lines.append("")
    lines.append("Size and instruction-count columns above are min / median / max over unique passing variants of that task. Instruction count is the sum of Unicorn retired instructions across the task's tests, the same proxy the harness scorer uses.")
    lines.append("")
    lines.append("## Which knobs moved the bytes")
    lines.append("")
    lines.append("For each knob, configs are grouped by task, compiler, and every other knob. `fraction` is how often a group that varies that knob produced more than one unique passing sha256. Groups where every config failed do not count.")
    lines.append("")
    if not diversity:
        lines.append("No passing configs, so knob diversity is empty.")
    else:
        lines.append("| knob | groups | groups with >1 sha | fraction | mean unique | max unique |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in diversity:
            lines.append(
                f"| {row['knob']} | {row['groups']} | {row['groups_with_diversity']} | "
                f"{row['fraction']:.3f} | {row['mean_unique']:.2f} | {row['max_unique']} |"
            )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    if not summary["reasons"]:
        lines.append("No failures.")
    else:
        lines.append("| reason | count |")
        lines.append("|---|---:|")
        for reason, count in summary["reasons"].most_common():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        lines.append("Reason keys:")
        lines.append("")
        lines.append("- `compile_error`: the compiler rejected the flags or crashed.")
        lines.append("- `external_call`: the function contains a `call`, so it is not a leaf.")
        lines.append("- `external_reloc`: a relocation targets a symbol outside the extracted function and its rodata.")
        lines.append("- `unsupported_insn`: YMM, FMA, `vpbroadcast*`, or a VEX instruction whose `vvvv` source is not the destination. Unicorn 2.1.4 cannot run those faithfully, so those variants are not kept.")
        lines.append("- `test_fail`: assembled and ran, but a harness test failed.")
        lines.append("- `extract` / `assemble_error` / `worker_error`: other drops.")
        # A few example details
        examples: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if row.get("status") == "pass":
                continue
            reason = str(row.get("reason") or "unknown")
            if reason in seen:
                continue
            seen.add(reason)
            detail = str(row.get("detail") or "").replace("\n", " ")[:180]
            examples.append(f"- `{reason}` on `{row.get('task_id')}` via `{row.get('compiler')}`: {detail}")
            if len(examples) >= 8:
                break
        if examples:
            lines.append("")
            lines.append("Examples:")
            lines.append("")
            lines.extend(examples)
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Dedup key is sha256 of the linked `.text` bytes of the extracted function.")
    lines.append("- `goal=size` prefers fewer bytes, then fewer instructions. `goal=speed` prefers fewer instructions, then fewer bytes. `goal=default` is the plain gcc `-O2 -march=x86-64` variant when it passed (else clang, else rustc `opt-level=2`), which is the usual imitation target.")
    lines.append("- Preference pairs sample the ranked list (best against a spread of worse unique variants), capped at 8 per task per goal.")
    lines.append("- Rust references use wrapping arithmetic and an inline `idiv` for remainder so the object does not call `core::panicking`. `-C overflow-checks=on` therefore often matches `off` on these tasks. Byte loops load through `read_volatile` so LLVM does not emit a `strlen`/`memcpy`/`memcmp` call.")
    lines.append("- LLVM 23 has no `opt`/`llc` `-Os` or `-Oz`, and `default<Os>` is rejected. Size rows run `default<O2>` with `optsize` or `minsize optsize` on the functions. IR is clang `-O0 -disable-O0-optnone` so later pipelines are not blocked by `optnone`.")
    lines.append("- `+avx2` and `-march=x86-64-v3`/`znver3` stay in the grid. Outputs that use YMM or FMA fail `unsupported_insn` and are dropped. `-mno-avx` and `-mavx2 -mno-fma -mprefer-vector-width=128` are the executable vector points.")
    lines.append("- Unicorn 2.1.4 ignores `VEX.vvvv` and computes `dest = dest OP rm`. A 128-bit VEX instruction is kept only when `vvvv` is unused or names the destination (the destructive form). Non-destructive AVX is `unsupported_insn`, not a passing variant. SSE without a VEX prefix is unaffected.")
    lines.append("- `sum_to` under gcc `-mavx2` sometimes emits `.set .LC2, .LC0+24` inside a 32-byte constant pool. The extractor does not follow `.set`, so those configs fail `external_reloc`. The pool is YMM-sized and would be dropped as `unsupported_insn` anyway.")
    lines.append("- A unique sha256 that two compilers both emit counts once in the total and once in each compiler's unique column.")
    lines.append("- llvm-opt random seeds 6–9 were first recorded with a different `llc` `-O` (the old 6-level cycle) and then again after the cycle was limited to `-O0..-O3`. Both pairings passed and stay in the cache, so llvm-opt's generated count is 80 above the current grid (440 recorded, 360 in the grid).")
    lines.append("- Resumable cache: `sweep-cache.jsonl` next to this report. A rerun skips `config_id`s already recorded. That cache is gitignored; the jsonl datasets and this report are not.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
    tools: dict[str, str],
    compilers: list[str],
    grid_note: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = unique_variants(rows, tasks_by_id)
    sft = sft_records(variants, rows)
    pairs = pair_records(variants, sft)
    report = render_report(
        rows=rows,
        variants=variants,
        tools=tools,
        compilers=compilers,
        grid_note=grid_note,
    )
    _dump_jsonl(out_dir / "variants.jsonl", variants)
    _dump_jsonl(out_dir / "sft.jsonl", sft)
    _dump_jsonl(out_dir / "pairs.jsonl", pairs)
    (out_dir / "SWEEP_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "variants": len(variants),
        "sft": len(sft),
        "pairs": len(pairs),
        "rows": len(rows),
        "passed": sum(1 for r in rows if r.get("status") == "pass"),
    }


def _dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
