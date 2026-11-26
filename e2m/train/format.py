"""Prompt, split, and scoring helpers shared by training and evaluation.

The supervised target is AT&T syntax for GNU as: the extracted function,
including ``.text`` / ``.globl`` and any ``.rodata`` the extractor kept.
Evaluation assembles model text with ``syntax="auto"`` (AT&T, then Intel).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

SYSTEM_ASM = (
    "You translate an English specification into x86-64 assembly for one SysV AMD64 function. "
    "Reply with AT&T syntax for GNU as only. Include the .text directive, .globl, the function "
    "label, and any .rodata the function uses. Do not write a main function. Do not explain."
)

SYSTEM_C = (
    "You write exactly one C function with the given signature. "
    "The function must be a leaf: no library calls and no other functions. "
    "Reply with only the function."
)

C_THEN_ASM_C_MARK = "===C==="
C_THEN_ASM_ASM_MARK = "===ASM==="

SYSTEM_C_THEN_ASM = (
    "You translate an English specification into one leaf C function and then into "
    "x86-64 assembly for that same SysV AMD64 function. Reply with exactly two parts, "
    f"in this order, using these delimiters.\n{C_THEN_ASM_C_MARK}\n"
    "the C function with the given signature (no main, no library calls)\n"
    f"{C_THEN_ASM_ASM_MARK}\n"
    "AT&T syntax for GNU as: .text, .globl, the function label, and any .rodata. "
    "Do not explain."
)

_FENCE_RE = re.compile(r"```(?:asm|s|att|gas|c|cpp)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_C_MARK_RE = re.compile(r"^=== *C *===", re.MULTILINE | re.IGNORECASE)
_ASM_MARK_RE = re.compile(r"^=== *ASM *===", re.MULTILINE | re.IGNORECASE)


def user_asm(spec: str, signature: str, goal: str = "default") -> str:
    return f"Goal: {goal}\nSignature: {signature}\nSpecification:\n{spec.strip()}\n"


def user_c(spec: str, signature: str) -> str:
    return f"Signature: {signature}\nSpecification:\n{spec.strip()}\n"


def asm_messages(spec: str, signature: str, goal: str = "default", asm: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_ASM},
        {"role": "user", "content": user_asm(spec, signature, goal)},
    ]
    if asm is not None:
        messages.append({"role": "assistant", "content": asm.strip() + "\n"})
    return messages


def c_messages(spec: str, signature: str, code: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_C},
        {"role": "user", "content": user_c(spec, signature)},
    ]
    if code is not None:
        messages.append({"role": "assistant", "content": code.strip() + "\n"})
    return messages


def user_c_then_asm(spec: str, signature: str, goal: str = "default") -> str:
    return f"Goal: {goal}\nSignature: {signature}\nSpecification:\n{spec.strip()}\n"


def format_c_then_asm_response(c_src: str, asm: str) -> str:
    return f"{C_THEN_ASM_C_MARK}\n{c_src.strip()}\n{C_THEN_ASM_ASM_MARK}\n{asm.strip()}\n"


def c_then_asm_messages(
    spec: str,
    signature: str,
    goal: str = "default",
    response: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_C_THEN_ASM},
        {"role": "user", "content": user_c_then_asm(spec, signature, goal)},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response.strip() + "\n"})
    return messages


def split_c_then_asm(text: str) -> tuple[str, str]:
    """Return (c_source, asm_text) from a delimited C-then-assembly reply."""
    body = (text or "").strip()
    c_match = _C_MARK_RE.search(body)
    a_match = _ASM_MARK_RE.search(body)
    if c_match and a_match and a_match.start() > c_match.start():
        c_part = body[c_match.end() : a_match.start()]
        a_part = body[a_match.end() :]
        return extract_c(c_part), extract_asm(a_part)
    fences = list(_FENCE_RE.finditer(body))
    if len(fences) >= 2:
        return extract_c(fences[0].group(1)), extract_asm(fences[-1].group(1))
    if fences:
        inner = fences[0].group(1)
        lang = body[fences[0].start() : fences[0].start() + 12].lower()
        if "c" in lang and "asm" not in lang:
            return extract_c(inner), extract_asm(body[fences[0].end() :])
        return extract_c(body), extract_asm(inner)
    return extract_c(body), extract_asm(body)


def extract_asm_from_c_then_asm(text: str) -> str:
    _c, asm = split_c_then_asm(text)
    return asm


def extract_c_from_c_then_asm(text: str) -> str:
    c_src, _asm = split_c_then_asm(text)
    return c_src


def val_task_ids(task_ids: list[str], frac: float = 0.03, seed: int = 0) -> list[str]:
    """Stable 3% holdout. The same id always lands on the same side of the split."""
    unique = sorted(set(task_ids))
    ranked = sorted(unique, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    n = max(1, round(len(ranked) * frac))
    return ranked[:n]


def extract_fenced(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_asm(text: str) -> str:
    body = extract_fenced(text)
    indexes = [body.find(marker) for marker in (".text", ".globl", ".global") if body.find(marker) >= 0]
    if indexes:
        return body[min(indexes):].strip()
    return body.strip()


def extract_c(text: str) -> str:
    return extract_fenced(text).strip()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k. n is the sample count and c is how many were correct."""
    if n < k or c < 0:
        return float("nan")
    if c == n or n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def categorize_run(assemble_error: str | None, status: str, passed: bool) -> str:
    if assemble_error:
        return "doesnt_assemble"
    if passed:
        return "pass"
    if status in {"timeout", "insn_limit"}:
        return "timeout"
    if status == "abi":
        return "abi"
    if status == "ok":
        return "wrong_answer"
    return "crash"


def score_task_run(run: Any) -> dict[str, Any]:
    """Summarize one harness TaskRun. `run` is duck-typed so tests can pass a stub."""
    if getattr(run, "assemble_error", None):
        return {
            "ok": False,
            "category": "doesnt_assemble",
            "size": 0,
            "insn": 0,
            "detail": str(run.assemble_error)[:400],
        }
    status = "ok"
    detail = ""
    if not run.passed:
        for case in run.cases:
            if not case.passed:
                status = case.emu.status
                detail = (case.reason or case.emu.message or status)[:400]
                break
        if not run.cases:
            status = "crash"
            detail = "no cases"
    return {
        "ok": bool(run.passed),
        "category": categorize_run(None, status, bool(run.passed)),
        "size": int(run.code_size or 0),
        "insn": int(run.insn_count or 0),
        "detail": detail,
    }
