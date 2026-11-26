"""SysV AMD64 ABI helpers: C signature parsing and argument classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CType:
    kind: str  # "int" | "float" | "ptr" | "void"
    size: int
    signed: bool = True
    pointee: "CType | None" = None
    cstr: bool = False  # pointer to NUL-terminated bytes
    spelling: str = ""

    @property
    def is_integer(self) -> bool:
        return self.kind in ("int", "ptr")

    @property
    def is_float(self) -> bool:
        return self.kind == "float"


_INT_MAP: dict[str, tuple[int, bool]] = {
    "int8_t": (1, True),
    "uint8_t": (1, False),
    "int16_t": (2, True),
    "uint16_t": (2, False),
    "int32_t": (4, True),
    "uint32_t": (4, False),
    "int64_t": (8, True),
    "uint64_t": (8, False),
    "size_t": (8, False),
    "ptrdiff_t": (8, True),
    "intptr_t": (8, True),
    "uintptr_t": (8, False),
    "char": (1, True),
    "signed char": (1, True),
    "unsigned char": (1, False),
    "short": (2, True),
    "signed short": (2, True),
    "unsigned short": (2, False),
    "int": (4, True),
    "signed int": (4, True),
    "unsigned int": (4, False),
    "unsigned": (4, False),
    "long": (8, True),
    "signed long": (8, True),
    "unsigned long": (8, False),
    "long long": (8, True),
    "signed long long": (8, True),
    "unsigned long long": (8, False),
    "bool": (1, False),
    "_bool": (1, False),
}

_FLOAT_MAP = {
    "float": 4,
    "double": 8,
    "f32": 4,
    "f64": 8,
}


@dataclass(frozen=True)
class Param:
    name: str
    ctype: CType


@dataclass(frozen=True)
class Signature:
    name: str
    ret: CType
    params: tuple[Param, ...]
    spelling: str = ""


_SIG_RE = re.compile(
    r"^\s*(?P<ret>.+?)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*)\)\s*;?\s*$",
    re.DOTALL,
)


def parse_signature(spelling: str) -> Signature:
    """Parse a C function signature used by a task.

    Understands stdint names, pointers, ``const``, and ``void``.
    """
    text = " ".join(spelling.strip().split())
    m = _SIG_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse signature: {spelling!r}")
    ret = parse_type(m.group("ret"))
    name = m.group("name")
    args_raw = m.group("args").strip()
    params: list[Param] = []
    if args_raw and args_raw != "void":
        parts = _split_args(args_raw)
        for i, part in enumerate(parts):
            pname, ptype = _split_decl(part)
            if not pname:
                pname = f"arg{i}"
            params.append(Param(name=pname, ctype=ptype))
    return Signature(name=name, ret=ret, params=tuple(params), spelling=text)


def parse_type(spelling: str) -> CType:
    raw = spelling.strip()
    s = raw
    s = s.replace("*", " * ")
    tokens = [t for t in s.split() if t]
    is_const = False
    filtered: list[str] = []
    stars = 0
    for t in tokens:
        if t == "const":
            is_const = True
            continue
        if t == "*":
            stars += 1
            continue
        if t in ("restrict", "volatile"):
            continue
        filtered.append(t)
    if not filtered:
        raise ValueError(f"empty type: {spelling!r}")
    if filtered[-1] in {"struct", "union", "enum"}:
        raise ValueError(f"unsupported type: {spelling!r}")
    # last token might be a dummy name if someone passed a full decl; ignore words
    # that look like identifiers after a known type? we only get types here.
    type_tokens = filtered
    joined = " ".join(type_tokens)
    if joined == "void" and stars == 0:
        return CType(kind="void", size=0, signed=False, spelling=raw)

    cstr = False
    base: CType
    if joined in _FLOAT_MAP:
        base = CType(kind="float", size=_FLOAT_MAP[joined], signed=True, spelling=raw)
    elif joined in _INT_MAP:
        size, signed = _INT_MAP[joined]
        base = CType(kind="int", size=size, signed=signed, spelling=raw)
    elif joined in ("signed",):
        base = CType(kind="int", size=4, signed=True, spelling=raw)
    else:
        # "unsigned long" already in map; try dropping a trailing ident (param name)
        if len(type_tokens) >= 2:
            maybe = " ".join(type_tokens[:-1])
            if maybe in _INT_MAP or maybe in _FLOAT_MAP or maybe == "void":
                return parse_type(maybe + " * " * stars)
        raise ValueError(f"unsupported type: {spelling!r}")

    if stars == 0:
        _ = is_const
        return base

    # pointer. One level is enough for v0 tasks; nested pointers stay as ptr-to-ptr.
    inner = base
    for i in range(stars):
        is_last = i == stars - 1
        # Only char* / signed char* are C strings. uint8_t* is a byte buffer.
        make_cstr = is_last and joined in ("char", "signed char")
        inner = CType(
            kind="ptr",
            size=8,
            signed=False,
            pointee=inner,
            cstr=make_cstr,
            spelling=raw,
        )
    if inner.cstr:
        # char * treated as C string when packing a Python str.
        pass
    return inner


def _split_args(args: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in args:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _split_decl(decl: str) -> tuple[str, CType]:
    s = decl.strip()
    s = s.replace("*", " * ")
    tokens = [t for t in s.split() if t and t not in ("const", "restrict", "volatile")]
    if not tokens:
        raise ValueError(f"empty parameter: {decl!r}")
    # If last token is an identifier and not a type keyword, it is the name.
    name = ""
    type_str = decl
    last = tokens[-1].strip("*")
    known = set(_INT_MAP) | set(_FLOAT_MAP) | {"void", "signed", "unsigned", "long", "short"}
    if last and last not in known and re.match(r"^[A-Za-z_]\w*$", last):
        name = last
        type_str = decl[: decl.rfind(last)].strip()
    return name, parse_type(type_str)


def pack_int(value: int, size: int, signed: bool) -> bytes:
    bits = size * 8
    if signed:
        min_v = -(1 << (bits - 1))
        max_v = (1 << (bits - 1)) - 1
        if value < min_v or value > max_v:
            # Allow wrapping for harness robustness; mask into range.
            value = (value + (1 << bits)) % (1 << bits)
            if value >= 1 << (bits - 1):
                value -= 1 << bits
        return int(value).to_bytes(size, "little", signed=True)
    value = int(value) & ((1 << bits) - 1)
    return value.to_bytes(size, "little", signed=False)


def unpack_int(raw: int, size: int, signed: bool) -> int:
    bits = size * 8
    raw &= (1 << bits) - 1
    if signed and raw >= 1 << (bits - 1):
        return raw - (1 << bits)
    return raw


def pack_data(elem: CType, data: Any) -> bytes:
    """Serialize a JSON value into a buffer of ``elem`` (or bytes for cstr)."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if elem.cstr or (elem.kind == "int" and elem.size == 1 and isinstance(data, str)):
        if isinstance(data, str):
            return data.encode("utf-8") + b"\x00"
        if isinstance(data, list):
            return bytes(int(x) & 0xFF for x in data) + (b"\x00" if not data or data[-1] != 0 else b"")
    if isinstance(data, str):
        return data.encode("utf-8") + b"\x00"
    if elem.kind == "float":
        import struct

        fmt = "<f" if elem.size == 4 else "<d"
        if isinstance(data, list):
            return b"".join(struct.pack(fmt, float(x)) for x in data)
        return struct.pack(fmt, float(data))
    if isinstance(data, list):
        return b"".join(pack_int(int(x), elem.size, elem.signed) for x in data)
    return pack_int(int(data), elem.size, elem.signed)


def unpack_data(elem: CType, buf: bytes) -> list[Any] | str:
    if elem.cstr or (elem.kind == "int" and elem.size == 1 and buf.endswith(b"\x00") and b"\x00" in buf[:-1] or False):
        # Prefer list of bytes unless the original was a string; caller decides.
        pass
    if elem.kind == "float":
        import struct

        fmt = "<f" if elem.size == 4 else "<d"
        n = len(buf) // elem.size
        return list(struct.unpack("<" + fmt[-1] * n, buf[: n * elem.size])) if n else []
    if elem.size == 1 and elem.kind == "int":
        return list(buf)
    out: list[int] = []
    for i in range(0, len(buf) // elem.size * elem.size, elem.size):
        chunk = buf[i : i + elem.size]
        val = int.from_bytes(chunk, "little", signed=elem.signed)
        out.append(val)
    return out
