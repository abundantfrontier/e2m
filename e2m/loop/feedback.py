"""Repair-prompt construction. Hidden tests never appear in the prompt."""

from __future__ import annotations

import json
from typing import Any

from e2m.harness.tasks import Task, TaskRun, TestCase
from e2m.train.format import SYSTEM_ASM, SYSTEM_C, SYSTEM_C_THEN_ASM, user_asm, user_c, user_c_then_asm

SYSTEM_ASM_REPAIR = (
    SYSTEM_ASM
    + " When a previous attempt failed, output a complete corrected function, not a diff."
)
SYSTEM_C_REPAIR = (
    SYSTEM_C
    + " When a previous attempt failed, output a complete corrected function, not a diff."
)
SYSTEM_C_THEN_ASM_REPAIR = (
    SYSTEM_C_THEN_ASM
    + " When a previous attempt failed, output a complete corrected C function and assembly, not a diff."
)

_MAX_DETAIL = 800
_MAX_VALUE = 400


def _clip(text: str, n: int = _MAX_DETAIL) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _jsonish(value: Any) -> str:
    try:
        dumped = json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        dumped = repr(value)
    return _clip(dumped, _MAX_VALUE)


def format_args(args: list[Any]) -> str:
    return _jsonish(args)


def format_case(case: TestCase) -> str:
    parts = [f"args={format_args(case.args)}"]
    if case.expected_return is not None:
        parts.append(f"expected_return={_jsonish(case.expected_return)}")
    if case.expected_buffers:
        parts.append(f"expected_buffers={_jsonish(case.expected_buffers)}")
    return ", ".join(parts)


def visible_failure(run: TaskRun, visible: list[TestCase]) -> dict[str, Any]:
    """Structured failure drawn only from `run` (already restricted to visible tests)."""
    if run.assemble_error:
        return {
            "kind": "doesnt_assemble",
            "detail": _clip(str(run.assemble_error)),
            "test": None,
        }
    vis_ids = {c.id for c in visible}
    for i, case in enumerate(run.cases):
        if case.passed:
            continue
        if case.case_id not in vis_ids and visible:
            # Caller passed a full run by mistake; skip non-visible cases.
            continue
        test = visible[i] if i < len(visible) else None
        status = case.emu.status if case.emu is not None else "crash"
        got = None
        if case.emu is not None:
            got = case.emu.return_value
        kind = "wrong_answer"
        if status in {"timeout", "insn_limit"}:
            kind = "timeout"
        elif status == "abi":
            kind = "abi"
        elif status != "ok":
            kind = "crash"
        info: dict[str, Any] = {
            "kind": kind,
            "detail": _clip(case.reason or (case.emu.message if case.emu else status) or status),
            "status": status,
            "got": got,
            "test": None,
        }
        if test is not None:
            info["test"] = {
                "id": test.id,
                "args": test.args,
                "expected_return": test.expected_return,
                "expected_buffers": test.expected_buffers,
            }
        return info
    return {"kind": "crash", "detail": _clip(run.fail_reason or "failed"), "test": None}


def format_feedback(info: dict[str, Any]) -> str:
    kind = info.get("kind") or "crash"
    lines = [
        "The previous function failed the visible tests.",
        f"Error kind: {kind}",
    ]
    if info.get("detail"):
        lines.append("Details:")
        lines.append(_clip(str(info["detail"])))
    test = info.get("test")
    if test:
        lines.append("First failing visible test:")
        lines.append(f"  args: {_jsonish(test.get('args'))}")
        if test.get("expected_return") is not None:
            lines.append(f"  expected return: {_jsonish(test.get('expected_return'))}")
        if test.get("expected_buffers"):
            lines.append(f"  expected buffers: {_jsonish(test.get('expected_buffers'))}")
        if info.get("got") is not None and kind == "wrong_answer":
            lines.append(f"  got return: {_jsonish(info.get('got'))}")
    lines.append("Write a complete corrected function. Do not explain.")
    return "\n".join(lines) + "\n"


def initial_messages(spec: str, signature: str, *, mode: str = "asm") -> list[dict[str, str]]:
    if mode == "c":
        return [
            {"role": "system", "content": SYSTEM_C_REPAIR},
            {"role": "user", "content": user_c(spec, signature)},
        ]
    if mode == "c-then-asm":
        return [
            {"role": "system", "content": SYSTEM_C_THEN_ASM_REPAIR},
            {"role": "user", "content": user_c_then_asm(spec, signature, "default")},
        ]
    return [
        {"role": "system", "content": SYSTEM_ASM_REPAIR},
        {"role": "user", "content": user_asm(spec, signature, "default")},
    ]


def append_repair(
    messages: list[dict[str, str]],
    previous: str,
    info: dict[str, Any],
) -> list[dict[str, str]]:
    out = list(messages)
    out.append({"role": "assistant", "content": previous.strip() + "\n"})
    out.append({"role": "user", "content": format_feedback(info)})
    return out


def feedback_leaks_hidden(text: str, hidden: list[TestCase]) -> bool:
    """True when `text` contains a hidden test's distinctive expected value."""
    blob = text.lower()
    for case in hidden:
        for value in _distinctive_values(case):
            if value and value in blob:
                return True
    return False


def _distinctive_values(case: TestCase) -> list[str]:
    values: list[str] = []
    for raw in (case.expected_return, *list(case.expected_buffers.values())):
        if raw is None:
            continue
        if isinstance(raw, (int, float)) and raw not in {0, 1, -1, 2}:
            values.append(str(raw))
        elif isinstance(raw, str) and len(raw) >= 6:
            values.append(raw.lower())
        elif isinstance(raw, list) and len(raw) >= 3:
            dumped = json.dumps(raw, default=str)
            if len(dumped) >= 8:
                values.append(dumped.lower())
    return values
