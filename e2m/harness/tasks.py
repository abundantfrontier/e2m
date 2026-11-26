"""Task format, JSONL loader, gcc baseline, and per-case evaluation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from e2m.harness.abi import CType, Signature, pack_data, parse_signature
from e2m.harness.assemble import AssembleError, Assembled, assemble
from e2m.harness.emu import EmuResult, emulate

GCC_COMMON = [
    "-fno-asynchronous-unwind-tables",
    "-fno-unwind-tables",
    "-fcf-protection=none",
    "-fno-ident",
    "-fno-stack-protector",
    "-fno-pic",
    "-fno-plt",
    "-ffreestanding",
    "-m64",
]

C_PREAMBLE = "#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n\n"


@dataclass
class TestCase:
    args: list[Any]
    expected_return: Any = None
    expected_buffers: dict[str, Any] = field(default_factory=dict)
    ret_cmp: str = "eq"  # "eq" | "sign"
    id: str | None = None


@dataclass
class Task:
    id: str
    spec: str
    signature: str
    reference_c: str
    tests: list[TestCase]
    reference_rust: str = ""
    user_prompt: str = ""
    source: str = ""
    parsed: Signature | None = None

    def sig(self) -> Signature:
        if self.parsed is None:
            self.parsed = parse_signature(self.signature)
        return self.parsed


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    emu: EmuResult
    reason: str = ""


@dataclass
class TaskRun:
    task_id: str
    passed: bool
    cases: list[CaseResult]
    code_size: int = 0
    insn_count: int = 0  # sum over cases that ran
    assemble_error: str | None = None
    source_kind: str = ""

    @property
    def fail_reason(self) -> str:
        if self.assemble_error:
            return f"assemble: {self.assemble_error}"
        for c in self.cases:
            if not c.passed:
                return c.reason or c.emu.status
        return ""


def load_jsonl(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            tasks.append(task_from_dict(obj))
    return tasks


def task_from_dict(obj: dict[str, Any]) -> Task:
    tests: list[TestCase] = []
    for i, t in enumerate(obj.get("tests", [])):
        tests.append(
            TestCase(
                args=list(t.get("args", [])),
                expected_return=t.get("ret", t.get("expected_return")),
                expected_buffers=dict(t.get("bufs", t.get("expected_buffers", {}))),
                ret_cmp=t.get("ret_cmp", t.get("compare", "eq")),
                id=t.get("id", f"{obj.get('id', 'task')}/{i}"),
            )
        )
    return Task(
        id=str(obj["id"]),
        spec=str(obj.get("spec", "")),
        signature=str(obj["signature"]),
        reference_c=str(obj.get("reference_c", obj.get("c", ""))),
        tests=tests,
        reference_rust=str(obj.get("reference_rust", "")),
        user_prompt=str(obj.get("user_prompt", "")),
        source=str(obj.get("source", "")),
    )


def dump_jsonl(tasks: Iterable[Task], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            obj = {
                "id": task.id,
                "spec": task.spec,
                "signature": task.signature,
                "reference_c": task.reference_c,
                "reference_rust": task.reference_rust,
                "tests": [
                    _case_to_json(c)
                    for c in task.tests
                ],
            }
            if task.user_prompt:
                obj["user_prompt"] = task.user_prompt
            if task.source:
                obj["source"] = task.source
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _case_to_json(c: TestCase) -> dict[str, Any]:
    d: dict[str, Any] = {"args": c.args}
    if c.expected_return is not None:
        d["ret"] = c.expected_return
    if c.expected_buffers:
        d["bufs"] = c.expected_buffers
    if c.ret_cmp != "eq":
        d["ret_cmp"] = c.ret_cmp
    return d


def compile_reference_asm(c_source: str, opt: str = "-O2") -> str:
    """Compile a C function to GNU AT&T assembly with gcc -S."""
    if not opt.startswith("-"):
        opt = "-" + opt
    src_body = c_source if "#include" in c_source else C_PREAMBLE + c_source
    with tempfile.TemporaryDirectory(prefix="e2m-gcc-") as td:
        src = Path(td) / "f.c"
        asm = Path(td) / "f.s"
        src.write_text(src_body, encoding="utf-8")
        cmd = ["gcc", opt, "-S", *GCC_COMMON, "-o", str(asm), str(src)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
        if proc.returncode != 0 or not asm.is_file():
            raise AssembleError(
                f"gcc {opt} -S failed",
                details=(proc.stderr or proc.stdout or "").strip(),
            )
        return asm.read_text(encoding="utf-8", errors="replace")


def assemble_reference(task: Task, opt: str = "-O2") -> Assembled:
    asm = compile_reference_asm(task.reference_c, opt=opt)
    return assemble(asm, syntax="att")


def run_task(
    task: Task,
    source: Assembled | str | bytes,
    *,
    syntax: str = "auto",
    insn_limit: int = 1_000_000,
    timeout_s: float = 0.2,
    check_abi: bool = True,
    backend: str | None = None,
    count_instructions: bool = True,
) -> TaskRun:
    """Assemble a candidate and run every test case."""
    if isinstance(source, Assembled):
        image = source
    else:
        try:
            image = assemble(source, syntax=syntax)  # type: ignore[arg-type]
        except AssembleError as exc:
            return TaskRun(
                task_id=task.id,
                passed=False,
                cases=[],
                assemble_error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return TaskRun(
                task_id=task.id,
                passed=False,
                cases=[],
                assemble_error=f"harness: {exc}",
            )

    cases: list[CaseResult] = []
    insn_sum = 0
    all_ok = True
    for i, test in enumerate(task.tests):
        emu = emulate(
            image,
            signature=task.sig(),
            args=test.args,
            entry=task.sig().name,
            insn_limit=insn_limit,
            timeout_s=timeout_s,
            check_abi=check_abi,
            backend=backend,
            count_instructions=count_instructions,
        )
        passed, reason = _eval_case(task, test, emu)
        cases.append(CaseResult(case_id=test.id or str(i), passed=passed, emu=emu, reason=reason))
        insn_sum += emu.insn_count
        if not passed:
            all_ok = False
    return TaskRun(
        task_id=task.id,
        passed=all_ok and bool(cases),
        cases=cases,
        code_size=image.code_size,
        insn_count=insn_sum,
        source_kind=image.source_kind,
    )


def run_gcc_baseline(task: Task, opt: str = "-O2", *, backend: str | None = None) -> TaskRun:
    try:
        image = assemble_reference(task, opt=opt)
    except AssembleError as exc:
        return TaskRun(task_id=task.id, passed=False, cases=[], assemble_error=str(exc))
    return run_task(task, image, backend=backend)


def _eval_case(task: Task, test: TestCase, emu: EmuResult) -> tuple[bool, str]:
    if not emu.ok:
        return False, f"{emu.status}: {emu.message}".strip(": ")
    sig = task.sig()
    if not _return_matches(sig.ret, emu.return_value, test.expected_return, test.ret_cmp):
        return False, f"return {emu.return_value!r} != expected {test.expected_return!r}"
    for ident, expected in test.expected_buffers.items():
        if ident not in emu.buffers:
            return False, f"missing output buffer {ident!r}"
        got = emu.buffers[ident]
        exp_bytes = _expected_buffer_bytes(task, test, ident, expected)
        if got != exp_bytes:
            return False, f"buffer {ident!r} mismatch: got {got.hex()} expected {exp_bytes.hex()}"
    return True, "ok"


def _return_matches(ret: CType, got: Any, expected: Any, mode: str) -> bool:
    if ret.kind == "void":
        return True
    if expected is None:
        return True
    if ret.kind == "float":
        if got is None:
            return False
        return float(got) == float(expected)
    if got is None:
        return False
    got_i = int(got)
    exp_i = int(expected)
    if mode == "sign":
        def sgn(x: int) -> int:
            return (x > 0) - (x < 0)

        return sgn(got_i) == sgn(exp_i)
    return got_i == exp_i


def _expected_buffer_bytes(task: Task, test: TestCase, ident: str, expected: Any) -> bytes:
    if isinstance(expected, str):
        return expected.encode("utf-8") + (b"" if expected.endswith("\x00") else b"\x00")
    if isinstance(expected, (bytes, bytearray)):
        return bytes(expected)
    # Match the original arg's element type when possible.
    elem: CType | None = None
    for i, param in enumerate(task.sig().params):
        raw = test.args[i] if i < len(test.args) else None
        name = f"arg{i}"
        if isinstance(raw, dict):
            name = str(raw.get("id", raw.get("name", name)))
        if name == ident and param.ctype.kind == "ptr" and param.ctype.pointee is not None:
            elem = param.ctype.pointee
            if param.ctype.cstr:
                elem = CType("int", 1, True, cstr=True, spelling="char")
            break
    if elem is None:
        if isinstance(expected, list) and expected and isinstance(expected[0], float):
            elem = CType("float", 8, True, spelling="double")
        else:
            elem = CType("int", 4, True, spelling="int32_t")
    return pack_data(elem, expected)


def starter_jsonl_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tasks" / "starter.jsonl"


def load_starter() -> list[Task]:
    path = starter_jsonl_path()
    if path.is_file():
        return load_jsonl(path)
    from e2m.harness.starter import STARTER_TASKS

    return list(STARTER_TASKS)
