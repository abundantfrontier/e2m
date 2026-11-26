"""Cheap C filters: signature cleanup, size, triviality, forbidden constructs."""

from __future__ import annotations

import hashlib
import re

from e2m.harness.abi import Signature, parse_signature

_ATTR = re.compile(r"__attribute__\s*\(\(.*?\)\)", re.S)
_STORAGE = re.compile(
    r"\b(static|inline|extern|__inline__|__inline|__restrict|restrict|volatile|_Noreturn)\b"
)
_ARRAY_PARAM = re.compile(r"([A-Za-z_]\w*(?:\s+\w+)*)\s+(\*?\s*[A-Za-z_]\w*)\s*\[(?:\d+)?\]")
_LINE_DIR = re.compile(r"^\s*#\s*(line|\d+)\b", re.M)
_STRUCT = re.compile(r"\b(struct|union|enum)\s")
_VARARGS = re.compile(r"\.\.\.")
_FORBIDDEN_CALL = re.compile(
    r"\b(printf|sprintf|snprintf|fprintf|scanf|fopen|fread|fwrite|malloc|calloc|"
    r"realloc|free|exit|abort|syscall|system|execve|fork|mmap|dlopen|longjmp|"
    r"setjmp|va_start|va_arg)\s*\("
)

GENERIC_NAMES = {
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "abs",
    "sum",
    "len",
    "cmp",
    "compare",
    "copy",
    "move",
    "swap",
    "find",
    "sort",
    "count",
    "init",
    "free",
    "open",
    "close",
    "read",
    "write",
    "main",
    "test",
    "foo",
    "bar",
    "helper",
    "dummy",
    "func",
}


def extract_head(func_def: str) -> str:
    text = func_def.strip()
    if "{" not in text:
        return text
    return text.split("{", 1)[0].strip()


def extract_body(func_def: str) -> str:
    if "{" not in func_def:
        return ""
    return func_def.split("{", 1)[1]


def clean_signature(func_def: str) -> str:
    head = extract_head(func_def)
    head = _LINE_DIR.sub("", head)
    for _ in range(6):
        nxt = _ATTR.sub("", head)
        if nxt == head:
            break
        head = nxt
    head = _STORAGE.sub(" ", head)
    head = re.sub(r"\bunsigned\s+int\b", "unsigned", head)
    head = re.sub(r"\bsigned\s+int\b", "int", head)
    head = re.sub(r"\blong\s+int\b", "long", head)
    head = re.sub(r"\bshort\s+int\b", "short", head)
    head = _ARRAY_PARAM.sub(r"\1 *\2", head)
    head = " ".join(head.split())
    return head


def line_count(func_def: str) -> int:
    return sum(1 for ln in func_def.splitlines() if ln.strip())


def normalize_body(func_def: str) -> str:
    body = extract_body(func_def)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"^\s*#.*", "", body, flags=re.M)
    compact = re.sub(r"\s+", "", body)
    return compact.rstrip("}")


def body_hash(func_def: str) -> str:
    return hashlib.sha256(normalize_body(func_def).encode()).hexdigest()[:16]


def is_source_trivial(func_def: str, sig: Signature | None = None) -> bool:
    compact = normalize_body(func_def)
    if compact in {"", "}", "return;", "return0;", "return1;", "return-1;", "returnNULL;"}:
        return True
    if sig and sig.params:
        name = sig.params[0].name
        if compact in {f"return{name};", f"return({name});"}:
            return True
    return False


def cheap_reject(func_def: str, fname: str) -> str | None:
    """Return a rejection reason, or None if the function survives cheap checks."""
    if not fname or fname == "main":
        return "main_or_empty_name"
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fname):
        return "bad_name"
    nlines = line_count(func_def)
    if nlines > 300:
        return "too_long"
    if nlines < 3 and is_source_trivial(func_def):
        return "trivial_source"
    if _VARARGS.search(extract_head(func_def)):
        return "varargs"
    if _STRUCT.search(func_def):
        return "struct_or_union"
    if _FORBIDDEN_CALL.search(func_def):
        return "forbidden_call"
    if re.search(r"\bgoto\b", func_def):
        return "goto"
    try:
        sig = parse_signature(clean_signature(func_def))
    except Exception as exc:
        return f"bad_signature:{exc}"[:80]
    if is_source_trivial(func_def, sig):
        return "trivial_source"
    n_int = sum(1 for p in sig.params if p.ctype.is_integer)
    n_fp = sum(1 for p in sig.params if p.ctype.is_float)
    if n_int > 6:
        return "too_many_int_args"
    if n_fp > 8:
        return "too_many_fp_args"
    if any(p.ctype.kind == "ptr" and p.ctype.pointee and p.ctype.pointee.kind == "ptr" for p in sig.params):
        return "nested_pointer"
    if sig.ret.kind == "ptr":
        return "pointer_return"
    return None
