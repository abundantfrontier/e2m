"""Resumable ExeBench/AnghaBench → verified tasks/scaled.jsonl pipeline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from e2m.data.compile import CompileError, compile_c_asm, find_tool
from e2m.data.extract import ExtractError, extract_function_asm
from e2m.harness.abi import parse_signature
from e2m.harness.assemble import AssembleError, assemble
from e2m.harness.emu import emulate
from e2m.harness.tasks import (
    TestCase,
    Task,
    run_task,
)
from e2m.scale.heldout import is_heldout_name
from e2m.scale.io_convert import exebench_inputs, heuristic_inputs
from e2m.scale.llm import LLMError, write_spec
from e2m.scale.parse_c import (
    body_hash,
    cheap_reject,
    clean_signature,
    line_count,
)

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SCALE_DIR = ROOT / "data" / "scale"
TASKS_OUT = ROOT / "tasks" / "scaled.jsonl"

_GCC: str | None = None


def _gcc() -> str:
    global _GCC
    if _GCC is None:
        path = find_tool("gcc")
        if not path:
            raise SystemExit("gcc not found")
        _GCC = path
    return _GCC


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id"):
                ids.add(str(obj["id"]))
    return ids


def _load_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id"):
                out[str(obj["id"])] = obj
    return out


def _iter_exebench(split_dir: Path) -> Iterable[dict[str, Any]]:
    import zstandard as zstd

    shards = sorted(split_dir.glob("*.jsonl.zst"))
    for shard in shards:
        dctx = zstd.ZstdDecompressor()
        with shard.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield row.get("text") or row


_ANGHA_FN = re.compile(
    r"(?:^|\n)((?:(?:static|inline|extern|__attribute__\s*\(\([^)]*\)\))\s+)*)"
    r"([\w\s\*]+?)\s+([A-Za-z_]\w*)\s*\(([^;{]*)\)\s*\{",
)


def _iter_angha(root: Path) -> Iterable[dict[str, Any]]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.c")):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matches = list(_ANGHA_FN.finditer(src))
        if not matches:
            continue
        m = matches[-1]
        body = src[m.start(1) if m.start(1) >= 0 else m.start() :].strip()
        fname = m.group(3)
        yield {
            "path": str(path.relative_to(root)),
            "fname": fname,
            "func_def": body,
            "real_io_pairs": [],
            "synth_io_pairs": [],
            "source": "angha",
        }


def _make_id(source: str, fname: str, func_def: str, path: str) -> str:
    h = hashlib.sha256(f"{path}\n{func_def}".encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_]", "_", fname)[:40] or "fn"
    return f"{source}_{safe}_{h}"


def _candidate_from_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    func_def = row.get("func_def") or ""
    fname = row.get("fname") or ""
    if not fname and func_def:
        m = re.search(r"([A-Za-z_]\w*)\s*\(", clean_signature(func_def))
        fname = m.group(1) if m else ""
    path = str(row.get("path") or "")
    reason = cheap_reject(func_def, fname)
    if reason:
        return {
            "id": _make_id(source, fname or "x", func_def, path),
            "status": "reject",
            "reason": reason,
            "source": source,
            "fname": fname,
        }
    if is_heldout_name(fname):
        return {
            "id": _make_id(source, fname, func_def, path),
            "status": "reject",
            "reason": "heldout",
            "source": source,
            "fname": fname,
        }
    try:
        sig = parse_signature(clean_signature(func_def))
    except Exception as exc:
        return {
            "id": _make_id(source, fname, func_def, path),
            "status": "reject",
            "reason": f"bad_signature:{exc}"[:80],
            "source": source,
            "fname": fname,
        }
    io_pairs = list(row.get("real_io_pairs") or []) + list(row.get("synth_io_pairs") or [])
    inputs = exebench_inputs(sig, io_pairs)
    if len(inputs) < 3:
        extra = heuristic_inputs(sig)
        seen = {repr(a) for a in inputs}
        for a in extra:
            if repr(a) not in seen:
                inputs.append(a)
                seen.add(repr(a))
            if len(inputs) >= 8:
                break
    if len(inputs) < 3:
        return {
            "id": _make_id(source, fname, func_def, path),
            "status": "reject",
            "reason": "too_few_inputs",
            "source": source,
            "fname": fname,
        }
    cid = _make_id(source, fname, func_def, path)
    return {
        "id": cid,
        "status": "keep",
        "reason": "keep",
        "source": source,
        "fname": fname,
        "path": path,
        "signature": sig.spelling,
        "reference_c": func_def if func_def.endswith("\n") else func_def + "\n",
        "inputs": inputs,
        "nlines": line_count(func_def),
        "body_hash": body_hash(func_def),
    }


def cheap_scan(
    *,
    reasons: Counter[str],
    seen_bodies: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    exe_dir = DATASETS / "exebench" / "train_real_simple_io"
    n = 0
    if exe_dir.is_dir():
        for row in _iter_exebench(exe_dir):
            n += 1
            cand = _candidate_from_row(row, "exebench")
            if cand is None:
                continue
            if cand["status"] != "keep":
                reasons[str(cand["reason"]).split(":")[0]] += 1
                continue
            bh = cand.get("body_hash") or ""
            if bh in seen_bodies:
                reasons["duplicate_body"] += 1
                continue
            seen_bodies.add(bh)
            kept.append(cand)
            if len(kept) >= limit:
                break
    angha_dir = DATASETS / "angha"
    if len(kept) < limit and angha_dir.is_dir():
        for row in _iter_angha(angha_dir):
            cand = _candidate_from_row(row, "angha")
            if cand is None:
                continue
            if cand["status"] != "keep":
                reasons[str(cand["reason"]).split(":")[0]] += 1
                continue
            bh = cand.get("body_hash") or ""
            if bh in seen_bodies:
                reasons["duplicate_body"] += 1
                continue
            seen_bodies.add(bh)
            kept.append(cand)
            if len(kept) >= limit:
                break
    print(f"cheap scan: {n} exebench rows, {len(kept)} kept after cheap filters", flush=True)
    return kept


def _compile_ok(cand: dict[str, Any]) -> dict[str, Any]:
    """Compile -O2, extract, assemble. Reject externals."""
    src = cand["reference_c"]
    name = cand["fname"]
    try:
        asm = compile_c_asm(src, _gcc(), ["-O2"])
    except CompileError as exc:
        return {**cand, "status": "reject", "reason": "compile_error", "detail": str(exc)[:200]}
    try:
        extracted = extract_function_asm(asm, name)
    except ExtractError as exc:
        msg = exc.message.lower()
        if "call" in msg:
            reason = "external_call"
        elif "reloc" in msg or "undefined" in msg:
            reason = "external_reloc"
        else:
            reason = "extract"
        return {**cand, "status": "reject", "reason": reason, "detail": exc.message[:200]}
    try:
        assemble(extracted, syntax="att")
    except AssembleError as exc:
        return {**cand, "status": "reject", "reason": "assemble_error", "detail": str(exc)[:200]}
    return {**cand, "status": "keep", "reason": "compile_ok"}


def _eval_case_capture(task: Task, args: list[Any], image) -> TestCase | None:
    emu = emulate(
        image,
        signature=task.sig(),
        args=args,
        entry=task.sig().name,
        insn_limit=200_000,
        timeout_s=0.15,
        check_abi=True,
    )
    if not emu.ok:
        return None
    bufs: dict[str, Any] = {}
    sig = task.sig()
    for ident, raw in emu.buffers.items():
        # Store as list of ints (bytes or i32 depending on pointer).
        elem = None
        for i, p in enumerate(sig.params):
            name = f"arg{i}"
            a = args[i] if i < len(args) else None
            if isinstance(a, dict):
                name = str(a.get("id", name))
            if name == ident and p.ctype.kind == "ptr":
                elem = p.ctype.pointee
                if p.ctype.cstr:
                    # keep original string if it was a cstr
                    if isinstance(a, dict) and isinstance(a.get("data"), str):
                        # compare via bytes; store original-length string without forcing
                        bufs[ident] = list(raw[:-1] if raw.endswith(b"\x00") else raw)
                    else:
                        bufs[ident] = list(raw)
                    elem = None
                break
        if ident in bufs:
            continue
        if elem is not None and elem.kind == "int" and elem.size == 4:
            n = len(raw) // 4
            bufs[ident] = [
                int.from_bytes(raw[i * 4 : (i + 1) * 4], "little", signed=elem.signed)
                for i in range(n)
            ]
        elif elem is not None and elem.kind == "int" and elem.size == 8:
            n = len(raw) // 8
            bufs[ident] = [
                int.from_bytes(raw[i * 8 : (i + 1) * 8], "little", signed=elem.signed)
                for i in range(n)
            ]
        elif elem is not None and elem.kind == "float":
            import struct as _st

            fmt = "<d" if elem.size == 8 else "<f"
            n = len(raw) // elem.size
            bufs[ident] = list(_st.unpack("<" + fmt[-1] * n, raw[: n * elem.size])) if n else []
        else:
            bufs[ident] = list(raw)
    return TestCase(args=args, expected_return=emu.return_value, expected_buffers=bufs)


def _stub_source(sig, kind: str) -> str:
    ret = sig.ret.spelling or "int"
    params = ", ".join(f"{p.ctype.spelling} {p.name}" for p in sig.params) or "void"
    if kind == "zero":
        if sig.ret.kind == "void":
            body = "return;"
        elif sig.ret.kind == "float":
            body = "return 0.0;"
        else:
            body = "return 0;"
    elif kind == "arg0" and sig.params and sig.ret.kind == sig.params[0].ctype.kind:
        body = f"return {sig.params[0].name};"
    else:
        body = "return;" if sig.ret.kind == "void" else "return 0;"
    return f"{ret} {sig.name}({params}) {{ {body} }}\n"


def _stub_passes(task: Task) -> bool:
    sig = task.sig()
    kinds = ["zero"]
    if sig.params and sig.ret.kind != "void":
        kinds.append("arg0")
    for kind in kinds:
        stub = Task(
            id=task.id + "_stub",
            spec=task.spec,
            signature=task.signature,
            reference_c=_stub_source(sig, kind),
            tests=task.tests,
        )
        try:
            from e2m.harness.tasks import assemble_reference

            image = assemble_reference(stub, opt="-O0")
        except AssembleError:
            continue
        run = run_task(stub, image, insn_limit=50_000, timeout_s=0.1, check_abi=False)
        if run.passed:
            return True
    return False


def _verify_one(cand: dict[str, Any]) -> dict[str, Any]:
    from e2m.harness.tasks import assemble_reference

    try:
        sig = parse_signature(cand["signature"])
    except Exception as exc:
        return {**cand, "status": "reject", "reason": f"bad_signature:{exc}"[:80]}
    tests: list[TestCase] = []
    task = Task(
        id=cand["id"],
        spec="",
        signature=cand["signature"],
        reference_c=cand["reference_c"],
        tests=[],
        source=cand.get("source", ""),
    )
    try:
        img0 = assemble_reference(task, opt="-O0")
    except AssembleError as exc:
        return {**cand, "status": "reject", "reason": "compile_error", "detail": str(exc)[:200]}
    for args in cand.get("inputs") or []:
        case = _eval_case_capture(task, args, img0)
        if case is None:
            continue
        tests.append(case)
        if len(tests) >= 8:
            break
    if len(tests) < 3:
        return {**cand, "status": "reject", "reason": "capture_fail", "n_ok": len(tests)}
    task.tests = tests
    run0 = run_task(task, img0, insn_limit=200_000, timeout_s=0.15)
    if not run0.passed:
        return {**cand, "status": "reject", "reason": "o0_fail", "detail": run0.fail_reason[:200]}
    try:
        img2 = assemble_reference(task, opt="-O2")
    except AssembleError as exc:
        return {**cand, "status": "reject", "reason": "o2_compile", "detail": str(exc)[:200]}
    run2 = run_task(task, img2, insn_limit=200_000, timeout_s=0.15)
    if not run2.passed:
        return {**cand, "status": "reject", "reason": "o2_mismatch", "detail": run2.fail_reason[:200]}
    if _stub_passes(task):
        return {**cand, "status": "reject", "reason": "stub_passes"}
    serialized = []
    for c in tests:
        d: dict[str, Any] = {"args": c.args}
        if c.expected_return is not None:
            d["ret"] = c.expected_return
        if c.expected_buffers:
            d["bufs"] = c.expected_buffers
        serialized.append(d)
    return {
        **cand,
        "status": "pass",
        "reason": "verified",
        "tests": serialized,
        "o0_size": run0.code_size,
        "o2_size": run2.code_size,
    }


def _mp_compile(cand: dict[str, Any]) -> dict[str, Any]:
    try:
        return _compile_ok(cand)
    except Exception as exc:  # noqa: BLE001
        return {**cand, "status": "reject", "reason": "worker_error", "detail": str(exc)[:200]}


def _mp_verify(cand: dict[str, Any]) -> dict[str, Any]:
    try:
        return _verify_one(cand)
    except Exception as exc:  # noqa: BLE001
        return {**cand, "status": "reject", "reason": "worker_error", "detail": str(exc)[:200]}


def write_download_log() -> None:
    path = SCALE_DIR / "DOWNLOAD_LOG.md"
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Dataset / tool downloads\n"]
    exe = DATASETS / "exebench"
    if exe.is_dir():
        tar = exe / "train_real_simple_io.tar.gz"
        split = exe / "train_real_simple_io"
        n = 0
        if split.is_dir():
            import zstandard as zstd

            for shard in split.glob("*.jsonl.zst"):
                dctx = zstd.ZstdDecompressor()
                with shard.open("rb") as fh, dctx.stream_reader(fh) as r:
                    n += sum(1 for _ in io.TextIOWrapper(r, encoding="utf-8"))
        lines.append(
            f"- ExeBench `train_real_simple_io`: tar {tar.stat().st_size if tar.is_file() else 0} bytes, "
            f"{n} rows, dir {split}\n"
        )
    ang = DATASETS / "angha"
    if ang.is_dir():
        n_c = sum(1 for _ in ang.rglob("*.c"))
        lines.append(f"- AnghaBench sample: {n_c} `.c` files under {ang} ({_du(ang)} bytes)\n")
    q = ROOT / "tools" / "qemu" / "qemu-x86_64"
    if q.is_file():
        lines.append(f"- qemu-x86_64 user-static 8.2.2: {q} ({q.stat().st_size} bytes)\n")
    path.write_text("".join(lines), encoding="utf-8")


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_pipeline(
    *,
    target: int = 3000,
    max_tasks: int = 5000,
    llm_hours: float = 6.0,
    jobs: int = 16,
    min_adequacy: int = 2,
) -> dict[str, Any]:
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    write_download_log()
    filter_path = SCALE_DIR / "filter-cache.jsonl"
    verify_path = SCALE_DIR / "verify-cache.jsonl"
    llm_path = SCALE_DIR / "llm-cache.jsonl"
    reject_path = SCALE_DIR / "reject-counts.json"
    TASKS_OUT.parent.mkdir(parents=True, exist_ok=True)

    reasons: Counter[str] = Counter()
    if reject_path.is_file():
        try:
            reasons.update(json.loads(reject_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    existing_final = _load_ids(TASKS_OUT)
    verify_map = _load_map(verify_path)
    llm_map = _load_map(llm_path)

    print(
        f"resume: {len(existing_final)} scaled tasks, {len(verify_map)} verified cache, {len(llm_map)} llm cache",
        flush=True,
    )

    # Cheap + compile filter. Scan more than target because later stages drop.
    scan_cap = max(max_tasks * 6, 25000)
    seen_bodies: set[str] = set()
    cheap = cheap_scan(reasons=reasons, seen_bodies=seen_bodies, limit=scan_cap)
    # Dedup already-compiled
    need_compile = [c for c in cheap if c["id"] not in verify_map and c["id"] not in existing_final]
    print(f"compile filter: {len(need_compile)} candidates, {jobs} workers", flush=True)
    compiled: list[dict[str, Any]] = []
    if need_compile:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, jobs), maxtasksperchild=32) as pool:
            for i, row in enumerate(pool.imap_unordered(_mp_compile, need_compile, chunksize=4), 1):
                if row.get("status") == "keep":
                    compiled.append(row)
                else:
                    reasons[str(row.get("reason") or "compile_error")] += 1
                    _append_jsonl(filter_path, {"id": row["id"], "status": "reject", "reason": row.get("reason")})
                if i % 200 == 0 or i == len(need_compile):
                    print(f"  compile {i}/{len(need_compile)} kept={len(compiled)}", flush=True)
    # Include previously verified as already done.
    print(f"compile kept {len(compiled)} new leaf functions", flush=True)

    need_verify = compiled[: max(0, max_tasks * 3)]
    verified_new = 0
    print(f"verify: {len(need_verify)} to run", flush=True)
    if need_verify:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, jobs), maxtasksperchild=16) as pool:
            for i, row in enumerate(pool.imap_unordered(_mp_verify, need_verify, chunksize=1), 1):
                _append_jsonl(verify_path, {k: row[k] for k in row if k != "inputs"})
                if row.get("status") == "pass":
                    verify_map[row["id"]] = row
                    verified_new += 1
                else:
                    reasons[str(row.get("reason") or "verify_fail")] += 1
                if i % 100 == 0 or i == len(need_verify):
                    print(
                        f"  verify {i}/{len(need_verify)} pass_new={verified_new} cache={len(verify_map)}",
                        flush=True,
                    )

    reject_path.write_text(json.dumps(reasons, indent=2) + "\n", encoding="utf-8")

    # LLM specs for verified rows not yet in scaled.jsonl
    pending = [
        verify_map[i]
        for i in verify_map
        if verify_map[i].get("status") == "pass" and i not in existing_final
    ]
    # Prefer exebench, then angha
    pending.sort(key=lambda r: (0 if r.get("source") == "exebench" else 1, r.get("id")))
    llm_budget = llm_hours * 3600.0
    llm_used = float(sum(v.get("seconds") or 0 for v in llm_map.values()))
    print(f"llm: {len(pending)} pending, {llm_used:.1f}s already used, budget {llm_budget:.0f}s", flush=True)

    written = 0
    llm_fail = 0
    t_llm = time.time()
    for row in pending:
        if len(existing_final) >= target:
            break
        if len(existing_final) >= max_tasks:
            break
        if llm_used >= llm_budget:
            print("llm time budget reached", flush=True)
            break
        spec = llm_map.get(row["id"])
        if spec is None:
            try:
                obj, elapsed = write_spec(row["signature"], row["reference_c"])
            except LLMError as exc:
                llm_fail += 1
                _append_jsonl(llm_path, {"id": row["id"], "status": "fail", "detail": str(exc)[:200]})
                continue
            llm_used += elapsed
            adequacy = int(obj.get("adequacy") or 0)
            if adequacy and adequacy < min_adequacy:
                reasons["low_adequacy"] += 1
                _append_jsonl(
                    llm_path,
                    {"id": row["id"], "status": "low_adequacy", "adequacy": adequacy, "seconds": elapsed},
                )
                continue
            spec = {
                "id": row["id"],
                "status": "ok",
                "spec": obj["spec"],
                "user_prompt": obj["user_prompt"],
                "adequacy": adequacy,
                "seconds": elapsed,
            }
            _append_jsonl(llm_path, spec)
            llm_map[row["id"]] = spec
        if spec.get("status") != "ok":
            continue
        task_obj = {
            "id": row["id"],
            "spec": spec["spec"],
            "user_prompt": spec.get("user_prompt") or "",
            "signature": row["signature"],
            "reference_c": row["reference_c"],
            "reference_rust": "",
            "source": row.get("source") or "exebench",
            "tests": row["tests"],
        }
        _append_jsonl(TASKS_OUT, task_obj)
        existing_final.add(row["id"])
        written += 1
        if written % 50 == 0:
            print(
                f"  wrote {written} specs, total {len(existing_final)}, llm {llm_used:.0f}s, fail {llm_fail}",
                flush=True,
            )

    reject_path.write_text(json.dumps(reasons, indent=2) + "\n", encoding="utf-8")
    stats = {
        "scaled_tasks": len(existing_final),
        "written_this_run": written,
        "llm_seconds": round(llm_used, 1),
        "llm_fail": llm_fail,
        "verify_cache": len(verify_map),
        "reasons": dict(reasons),
        "llm_wall_s": round(time.time() - t_llm, 1),
    }
    (SCALE_DIR / "pipeline-stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("pipeline stats:", json.dumps({k: stats[k] for k in stats if k != "reasons"}), flush=True)
    print("top reject reasons:", reasons.most_common(12), flush=True)
    return stats


_AVX_TASKS: dict[str, Task] = {}


def _avx_init(task_dicts: list[dict[str, Any]]) -> None:
    from e2m.harness.tasks import task_from_dict

    global _AVX_TASKS
    _AVX_TASKS = {d["id"]: task_from_dict(d) for d in task_dicts}


def _avx_one(r: dict[str, Any]) -> dict[str, Any]:
    task = _AVX_TASKS.get(r["task_id"])
    if task is None:
        return {**r, "qemu_status": "missing_task"}
    from e2m.data.extract import extract_function_asm
    from e2m.harness.assemble import assemble

    compiler = str(r.get("compiler") or "")
    tool = find_tool("gcc") if compiler == "gcc" else find_tool(compiler)
    if not tool:
        return {**r, "qemu_status": "no_compiler"}
    try:
        asm = compile_c_asm(task.reference_c, tool, list(r.get("flags") or []))
        extracted = extract_function_asm(asm, task.sig().name)
        image = assemble(extracted, syntax="att")
    except Exception as exc:  # noqa: BLE001
        return {**r, "qemu_status": "rebuild_fail", "detail": str(exc)[:200]}
    run = run_task(task, image, backend="qemu", timeout_s=0.4)
    return {
        "task_id": r["task_id"],
        "compiler": r.get("compiler"),
        "flags": r.get("flags"),
        "qemu_status": "pass" if run.passed else "fail",
        "detail": run.fail_reason[:200] if not run.passed else "",
    }


def recover_avx(*, limit: int = 0, jobs: int = 8) -> dict[str, Any]:
    """Re-verify starter-sweep unsupported_insn rows under QEMU."""
    cache = ROOT / "data" / "sweep-cache.jsonl"
    if not cache.is_file():
        return {"error": "no starter sweep cache"}
    from e2m.harness.tasks import load_starter

    tasks = {t.id: t for t in load_starter()}
    rows = []
    with cache.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                obj.get("reason") == "unsupported_insn"
                and obj.get("status") == "fail"
                and obj.get("compiler") in {"gcc", "clang"}
            ):
                rows.append(obj)
    # unique by task+flags
    seen = set()
    uniq = []
    for r in rows:
        key = (r.get("task_id"), tuple(r.get("flags") or []))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    if limit:
        uniq = uniq[:limit]
    print(f"avx recover: {len(rows)} unsupported rows, {len(uniq)} unique configs", flush=True)

    passed = failed = skip = 0
    out_path = SCALE_DIR / "avx-recover.jsonl"
    if out_path.exists():
        out_path.unlink()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max(1, min(jobs, 8)),
        initializer=_avx_init,
        initargs=([{"id": t.id, "spec": t.spec, "signature": t.signature, "reference_c": t.reference_c, "tests": [
            {"args": c.args, "ret": c.expected_return, "bufs": c.expected_buffers, "ret_cmp": c.ret_cmp}
            for c in t.tests
        ]} for t in tasks.values()],),
        maxtasksperchild=8,
    ) as pool:
        for i, row in enumerate(pool.imap_unordered(_avx_one, uniq, chunksize=1), 1):
            _append_jsonl(out_path, row)
            st = row.get("qemu_status")
            if st == "pass":
                passed += 1
            elif st in {"skip_non_c", "missing_task", "no_compiler"}:
                skip += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(uniq):
                print(f"  avx {i}/{len(uniq)} pass={passed} fail={failed} skip={skip}", flush=True)
    stats = {
        "unsupported_rows": len(rows),
        "unique_configs": len(uniq),
        "qemu_pass": passed,
        "qemu_fail": failed,
        "skip": skip,
    }
    (SCALE_DIR / "avx-recover-stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("avx recover", stats, flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scale E2M tasks from ExeBench/AnghaBench")
    p.add_argument("--target", type=int, default=3000)
    p.add_argument("--max-tasks", type=int, default=5000)
    p.add_argument("--llm-hours", type=float, default=6.0)
    p.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--min-adequacy", type=int, default=2)
    p.add_argument("--avx-recover", action="store_true")
    p.add_argument("--avx-limit", type=int, default=0)
    args = p.parse_args(argv)
    if args.avx_recover:
        recover_avx(limit=args.avx_limit, jobs=max(1, args.jobs // 2))
        return 0
    run_pipeline(
        target=args.target,
        max_tasks=args.max_tasks,
        llm_hours=args.llm_hours,
        jobs=args.jobs,
        min_adequacy=args.min_adequacy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
