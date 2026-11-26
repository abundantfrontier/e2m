"""Pull one function out of compiler assembly and reject non-leaf code.

The kept assembly is something GNU ``as`` can assemble on its own: the target
function, plus any ``.rodata`` blobs it references. External calls, GOT/PLT,
and relocations to symbols we do not have are rejected here. A later capstone
pass rejects instructions Unicorn 2.1.4 cannot execute faithfully (256-bit YMM,
FMA, and VEX forms whose ``vvvv`` source is not the destination).
"""

from __future__ import annotations

import re

from capstone import CS_ARCH_X86, CS_MODE_64, Cs

_LABEL_DEF = re.compile(r"^([.A-Za-z_$][\w.$]*):\s*$")


def _nocomment(line: str) -> str:
    """Strip a trailing ``#`` comment. Compiler asm uses ``#`` that way."""
    return line.split("#", 1)[0].strip()
_SIZE_LINE = re.compile(r"^\s*\.size\s+([.A-Za-z_$][\w.$]*)\b")
_DOT_SYM = re.compile(r"(\.[A-Za-z_][\w.]*)")
_CALL = re.compile(r"^(callq?|syscall|sysenter|int|int3)\b", re.IGNORECASE)
_BAD_RELOC = re.compile(r"@((GOT|PLT|TLSCALL|TPOFF|GOTPCREL|GOTPC)[A-Z0-9]*)", re.IGNORECASE)

_HEADER_STARTS = (
    ".globl",
    ".global",
    ".weak",
    ".type",
    ".prefalign",
    ".p2align",
    ".align",
    ".balign",
    ".section",
    ".text",
    ".set",
)

_DROP_PREFIXES = (
    ".cfi_",
    ".loc",
    ".file",
    ".ident",
    ".addrsig",
    ".cv_",
    ".prefalign",
    ".Lfunc_begin",
)

# Mnemonics Unicorn 2.1.4's TCG rejects even when the vector length is 128.
_UNSUPPORTED_MNEMONIC_PREFIXES = (
    "vfmadd",
    "vfmsub",
    "vfnmadd",
    "vfnmsub",
    "vfmaddsub",
    "vfmsubadd",
)
# Present in the encoding but UC_ERR_INSN_INVALID on this Unicorn build.
_UNIMPLEMENTED_MNEMONIC_PREFIXES = ("vpbroadcast",)


class ExtractError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def extract_function_asm(asm: str, name: str) -> str:
    """Return assembleable AT&T text for ``name`` only.

    Raises :class:`ExtractError` when the function is missing, calls out, or
    references a symbol that is not part of the extracted unit.
    """
    lines = asm.splitlines()
    start = _find_label(lines, name)
    if start is None:
        raise ExtractError(f"function {name!r} not found in assembly")
    begin = _header_begin(lines, start, name)
    end = _function_end(lines, start, name)
    body = _clean_lines(lines[begin : end + 1])
    body = _drop_leading_align(body, name)
    text = "\n".join(body).rstrip() + "\n"
    text = _attach_rodata(lines, text)
    _reject_control(text, name)
    if not any(_nocomment(ln) == f"{name}:" for ln in text.splitlines()):
        raise ExtractError(f"extracted assembly lost label {name}")
    return text


def _find_label(lines: list[str], name: str) -> int | None:
    target = f"{name}:"
    for i, line in enumerate(lines):
        if _nocomment(line) == target:
            return i
    return None


def _header_begin(lines: list[str], start: int, name: str) -> int:
    begin = start
    while begin > 0:
        prev = _nocomment(lines[begin - 1])
        if prev == "" or prev.startswith("//"):
            begin -= 1
            continue
        if prev.startswith(".cfi") or prev.startswith(".loc") or prev.startswith(".file"):
            break
        if _is_header(prev, name):
            begin -= 1
            continue
        break
    # Do not swallow a previous function's label.
    for i in range(begin, start):
        m = _LABEL_DEF.match(_nocomment(lines[i]))
        if m and m.group(1) != name and not m.group(1).startswith("."):
            begin = i + 1
    return begin


def _is_header(stripped: str, name: str) -> bool:
    if stripped.startswith(_HEADER_STARTS):
        if stripped.startswith((".globl", ".global", ".weak", ".type", ".set")):
            return name in stripped
        return True
    return False


def _function_end(lines: list[str], start: int, name: str) -> int:
    for j in range(start + 1, len(lines)):
        m = _SIZE_LINE.match(_nocomment(lines[j]))
        if m and m.group(1) == name:
            return j
        label = _LABEL_DEF.match(_nocomment(lines[j]))
        if label and not label.group(1).startswith(".") and label.group(1) != name:
            return j - 1
    return len(lines) - 1


def _drop_leading_align(lines: list[str], name: str) -> list[str]:
    """Drop alignment directives that sit before the entry label.

    They emit NOP padding in front of the symbol, so the function would not
    start at the first byte of ``.text``. Alignment inside the body stays.
    """
    label_at = None
    for i, line in enumerate(lines):
        if _nocomment(line) == f"{name}:":
            label_at = i
            break
    if label_at is None:
        return lines
    kept: list[str] = []
    for i, line in enumerate(lines):
        stripped = _nocomment(line)
        if i < label_at and stripped.startswith((".p2align", ".align", ".balign", ".prefalign")):
            continue
        kept.append(line)
    return kept


def _clean_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if out and out[-1] != "":
                out.append("")
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if stripped.startswith(_DROP_PREFIXES):
            continue
        if stripped.startswith(".section") and ("note" in stripped or "debug" in stripped or "eh_frame" in stripped):
            continue
        out.append(line.rstrip())
    while out and out[-1] == "":
        out.pop()
    return out


def _attach_rodata(all_lines: list[str], text: str) -> str:
    """Include local data labels referenced by instructions but defined outside."""
    defined = _defined_labels(text)
    needed = _insn_dot_labels(text) - defined
    if not needed:
        return text
    blobs: list[str] = []
    # Iterate a few times so a blob can reference another blob.
    for _ in range(4):
        if not needed:
            break
        progress = False
        for label in sorted(needed):
            blob = _data_blob(all_lines, label)
            if blob is None:
                continue
            blobs.append(blob)
            defined |= _defined_labels(blob)
            progress = True
        if not progress:
            break
        needed = set()
        for blob in blobs:
            needed |= _insn_dot_labels(blob)
            for line in blob.splitlines():
                stripped = line.strip()
                if not stripped.startswith("."):
                    continue
                head, _, rest = stripped.partition(" ")
                if head in {".quad", ".long", ".byte", ".short", ".word", ".4byte", ".8byte", ".value"}:
                    needed |= set(_DOT_SYM.findall(rest))
        needed -= defined
        needed -= _directive_names(needed)
    if needed:
        # Labels defined in the function body are fine (.L1 in `.quad .L1`).
        still = sorted(lab for lab in needed if lab not in defined)
        if still:
            raise ExtractError(
                "unresolved local symbol(s): " + ", ".join(still[:8])
            )
    if not blobs:
        return text
    return text.rstrip() + "\n\n" + "\n".join(blobs).rstrip() + "\n"


def _directive_names(labels: set[str]) -> set[str]:
    # Section names and alignment directives sometimes look like dot-labels.
    drop = set()
    for lab in labels:
        base = lab.split(".")[1] if lab.startswith(".") else lab
        if base in {
            "text",
            "rodata",
            "data",
            "bss",
            "section",
            "globl",
            "global",
            "type",
            "size",
            "p2align",
            "align",
            "byte",
            "long",
            "quad",
            "zero",
            "space",
            "string",
            "ascii",
            "asciz",
            "file",
            "ident",
        }:
            drop.add(lab)
    return drop


def _defined_labels(text: str) -> set[str]:
    found = set()
    for line in text.splitlines():
        m = _LABEL_DEF.match(_nocomment(line))
        if m:
            found.add(m.group(1))
    return found


def _insn_dot_labels(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        s = _nocomment(line)
        if not s or s.startswith(".") or s.endswith(":"):
            continue
        found.update(_DOT_SYM.findall(s))
    return found


def _data_blob(lines: list[str], label: str) -> str | None:
    idx = _find_label(lines, label)
    if idx is None:
        return None
    # Only pull data, never another function.
    if not _is_data_label(lines, idx):
        return None
    begin = idx
    while begin > 0:
        prev = _nocomment(lines[begin - 1])
        if prev == "":
            begin -= 1
            continue
        if prev.startswith((
            ".section",
            ".p2align",
            ".align",
            ".balign",
            ".rodata",
            ".text",
        )) or prev.startswith(".type") or prev.startswith(".globl") or prev.startswith(".local"):
            begin -= 1
            continue
        break
    end = idx
    for j in range(idx + 1, len(lines)):
        s = _nocomment(lines[j])
        if not s:
            end = j
            continue
        if s.endswith(":") and _LABEL_DEF.match(s):
            break
        if s.startswith(".section") or s.startswith(".text") or s.startswith(".globl") or s.startswith(".type"):
            break
        if s.startswith(".") or s.endswith(":"):
            end = j
            continue
        # An instruction means this was code.
        break
    blob_lines = _clean_lines(lines[begin : end + 1])
    if not blob_lines:
        return None
    return "\n".join(blob_lines).rstrip() + "\n"


def _is_data_label(lines: list[str], idx: int) -> bool:
    """True when ``lines[idx]`` is a constant-pool label, not a code block."""
    section = ".text"
    for line in lines[:idx]:
        stripped = _nocomment(line)
        if stripped.startswith(".section") or stripped in {".text", ".data", ".rodata"} or stripped.startswith(".rodata"):
            section = stripped
    if "rodata" in section or ".data" in section or section.startswith(".data"):
        return True
    for line in lines[idx + 1 : idx + 8]:
        stripped = _nocomment(line)
        if not stripped or stripped.endswith(":"):
            continue
        if stripped.startswith((".section", ".text", ".globl", ".global", ".type", ".size")):
            return False
        # A directive (.quad, .byte, .long, ...) is data. An instruction is code.
        return stripped.startswith(".")
    return False


def _reject_control(text: str, name: str) -> None:
    defined = _defined_labels(text)
    defined.add(name)
    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw or raw.startswith(".") or raw.endswith(":"):
            continue
        if _BAD_RELOC.search(raw):
            raise ExtractError(f"forbidden relocation in: {raw}")
        if _CALL.match(raw):
            raise ExtractError(f"external call in: {raw}")
        # Tail call: a direct jump to a symbol that is not a local label.
        m = re.match(r"(jmpq?|jmp)\s+\*?\s*([A-Za-z_][\w.]*)\b", raw)
        if m:
            target = m.group(2)
            if target not in defined and not target.startswith("."):
                raise ExtractError(f"jump to external symbol {target}")
        for sym in _DOT_SYM.findall(raw):
            if sym not in defined:
                raise ExtractError(f"reference to undefined label {sym}")


_MD = Cs(CS_ARCH_X86, CS_MODE_64)


def _vex_vvvv_mismatch(raw: bytes) -> bool:
    """True when this VEX/EVEX insn is not faithful on Unicorn 2.1.4.

    Unicorn computes ``dest = dest OP rm`` and ignores ``VEX.vvvv``. That
    matches the ISA only when ``vvvv`` is unused (15) or names the same
    register as ``ModRM.reg`` (the destructive form compilers also emit).
    EVEX (AVX-512) is rejected outright.
    """
    if not raw:
        return False
    if raw[0] == 0x62:
        return True
    if raw[0] == 0xC5:
        if len(raw) < 4:
            return False
        prefix = raw[1]
        modrm = raw[3]
        r_ext = 0 if prefix & 0x80 else 8
        stored = (prefix >> 3) & 0xF
    elif raw[0] == 0xC4:
        if len(raw) < 5:
            return False
        prefix = raw[1]
        prefix2 = raw[2]
        modrm = raw[4]
        r_ext = 0 if prefix & 0x80 else 8
        stored = (prefix2 >> 3) & 0xF
    else:
        return False
    # The nibble is stored in 1's complement. 1111b means "no vvvv operand".
    if stored == 0xF:
        return False
    vvvv = (~stored) & 0xF
    dest = ((modrm >> 3) & 7) | r_ext
    return vvvv != dest


def unsupported_instruction(blob: bytes, addr: int = 0) -> str | None:
    """Return a reason if ``blob`` uses an instruction Unicorn 2.1.4 cannot run.

    Rejected: 256-bit YMM (VEX.L), FMA, EVEX, and VEX instructions whose
    ``VEX.vvvv`` source is not the destination. SSE and destructive 128-bit
    VEX (``vpaddq xmm1, xmm1, xmm0``) are fine.
    """
    if not blob:
        return "empty code"
    for insn in _MD.disasm(blob, addr):
        mnem = insn.mnemonic.lower()
        ops = insn.op_str.lower()
        if "ymm" in ops or "zmm" in ops or mnem in {"vzeroall"}:
            return f"ymm/zmm not emulated: {insn.mnemonic} {insn.op_str}".strip()
        if mnem.startswith(_UNSUPPORTED_MNEMONIC_PREFIXES):
            return f"fma not emulated: {insn.mnemonic} {insn.op_str}".strip()
        if mnem.startswith(_UNIMPLEMENTED_MNEMONIC_PREFIXES):
            return f"unimplemented in unicorn: {insn.mnemonic} {insn.op_str}".strip()
        start = insn.address - addr
        raw = bytes(getattr(insn, "bytes", b"") or blob[start : start + insn.size])
        # BMI2 (andn, mulx, shlx, …) is also VEX-encoded, and Unicorn gets
        # those GPR forms right. The vvvv bug shows up on the AVX mnemonics.
        if mnem.startswith("v") and raw and raw[0] in (0xC4, 0xC5, 0x62) and _vex_vvvv_mismatch(raw):
            return f"unicorn vex.vvvv: {insn.mnemonic} {insn.op_str}".strip()
    return None
