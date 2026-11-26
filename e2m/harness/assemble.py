"""Assemble model output to x86-64 bytes with GNU ``as`` / ``ld``.

Accepted inputs:
  * GNU AT&T assembly text
  * Intel-syntax assembly text (auto-detected, or ``syntax="intel"``)
  * raw machine-code bytes, or a hex dump of those bytes
  * an already-assembled ELF relocatable (``\\x7fELF``)

Anything that needs a relocation to an undefined symbol, or that allocates
a section other than ``.text`` / ``.rodata``, is rejected with a clear error.
Internal relocations (local labels, string literals in ``.rodata``) are
resolved by ``ld`` at a fixed load address.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

CODE_BASE = 0x400000
PAGE = 0x1000

Syntax = Literal["auto", "att", "intel", "hex"]

# Allocated sections we will load. Empty .data/.bss (gcc always emits them)
# are ignored; non-empty ones are rejected.
_ALLOWED_ALLOC_NONEMPTY = (".text", ".rodata")
_IGNORED_SECTIONS = {
    "",
    ".symtab",
    ".strtab",
    ".shstrtab",
    ".note.GNU-stack",
    ".comment",
    ".eh_frame",
    ".eh_frame_hdr",
    ".rela.text",
    ".rela.rodata",
    ".rela.eh_frame",
}

# Reloc types ld can apply for PIC-free function bodies.
# See elf.h R_X86_64_*.
_ALLOWED_RELOC = {
    0: "R_X86_64_NONE",
    1: "R_X86_64_64",
    2: "R_X86_64_PC32",
    4: "R_X86_64_PLT32",  # treated as PC32 for a locally defined target
    10: "R_X86_64_32",
    11: "R_X86_64_32S",
    24: "R_X86_64_PC64",
}

_FORBIDDEN_RELOC = {
    3: "R_X86_64_GOT32",
    6: "R_X86_64_GLOB_DAT",
    7: "R_X86_64_JUMP_SLOT",
    9: "R_X86_64_GOTPCREL",
    19: "R_X86_64_TLSGD",
    20: "R_X86_64_TLSLD",
    22: "R_X86_64_GOTTPOFF",
    23: "R_X86_64_TPOFF32",
    41: "R_X86_64_GOTPCRELX",
    42: "R_X86_64_REX_GOTPCRELX",
}

_LINKER_SCRIPT = f"""\
OUTPUT_FORMAT("elf64-x86-64")
OUTPUT_ARCH(i386:x86-64)
SECTIONS {{
  . = {CODE_BASE:#x};
  .text : {{ *(.text) *(.text.*) }}
  . = ALIGN({PAGE:#x});
  .rodata : {{ *(.rodata) *(.rodata.*) }}
  /DISCARD/ : {{
    *(.eh_frame*) *(.comment) *(.note*) *(.got*) *(.plt*)
    *(.data) *(.data.*) *(.bss) *(.bss.*) *(COMMON)
    *(.gcc_except_table*) *(.debug*)
  }}
}}
"""

_MNEMONIC_RE = re.compile(
    r"\b(mov|movq|movl|movb|movzbl|movslq|lea|leaq|add|sub|xor|and|or|cmp|"
    r"test|push|pop|call|jmp|je|jne|jg|jl|jge|jle|ja|jb|ret|nop|int|syscall|"
    r"pxor|movsd|mulsd|addsd|cvtsi2sd)\b",
    re.IGNORECASE,
)
_ATT_RE = re.compile(r"%[er]?([abcd]x|[sd]i|[sb]p|ip)|%[xyz]mm\d+|\$[-0-9]|-?\d+\(%")
_INTEL_RE = re.compile(
    r"\b(ptr|qword|dword|word|byte)\b|\[(r[a-z0-9]+|[er]?[abcd]x)",
    re.IGNORECASE,
)
_INTEL_REG_OP = re.compile(
    r"\b(mov|add|sub|xor|and|or|lea|cmp|test|push|pop)\s+"
    r"([er]?([abcd]x|[sd]i|[sb]p)|r\d+[d]?|[xyz]mm\d+)\s*,",
    re.IGNORECASE,
)


class AssembleError(Exception):
    """Raised when input cannot be turned into a loadable .text image."""

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\n{self.details}"
        return self.message


@dataclass
class Assembled:
    """Loadable image produced by :func:`assemble`."""

    text: bytes
    text_addr: int
    rodata: bytes = b""
    rodata_addr: int | None = None
    symbols: dict[str, int] = field(default_factory=dict)
    source_kind: str = "att"
    warnings: list[str] = field(default_factory=list)

    @property
    def code_size(self) -> int:
        return len(self.text)

    def entry(self, name: str | None = None) -> int:
        if name and name in self.symbols:
            return self.symbols[name]
        if name:
            alt = name.lstrip("_")
            if alt in self.symbols:
                return self.symbols[alt]
            underscored = "_" + name
            if underscored in self.symbols:
                return self.symbols[underscored]
        if name is None:
            for key, addr in self.symbols.items():
                if not key.startswith(".") and addr >= self.text_addr:
                    return addr
        return self.text_addr


def assemble(
    source: str | bytes,
    *,
    syntax: Syntax = "auto",
    as_bin: str = "as",
    ld_bin: str = "ld",
) -> Assembled:
    """Turn assembly text, hex, or bytes into a linked x86-64 image."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        blob = bytes(source)
        if blob.startswith(b"\x7fELF"):
            with tempfile.TemporaryDirectory(prefix="e2m-as-") as td:
                path = Path(td) / "in.o"
                path.write_bytes(blob)
                return _object_to_image(path, ld_bin=ld_bin, source_kind="elf")
        return _raw_bytes(blob)

    text = source.strip()
    if not text:
        raise AssembleError("empty assembly input")

    kind = syntax if syntax != "auto" else detect_syntax(text)
    if kind == "hex":
        return _raw_bytes(_parse_hex(text), source_kind="hex")

    with tempfile.TemporaryDirectory(prefix="e2m-as-") as td:
        td_path = Path(td)
        src_path = td_path / "in.s"
        obj_path = td_path / "in.o"
        body = text if text.endswith("\n") else text + "\n"
        src_path.write_text(body, encoding="utf-8")
        as_err = _run_as(src_path, obj_path, kind, as_bin)
        if as_err is not None:
            if syntax == "auto" and kind in ("att", "intel"):
                other = "intel" if kind == "att" else "att"
                as_err2 = _run_as(src_path, obj_path, other, as_bin)
                if as_err2 is None:
                    kind = other
                else:
                    raise AssembleError(
                        "assembler rejected the input as both AT&T and Intel syntax",
                        details=f"AT&T:\n{as_err}\n\nIntel:\n{as_err2}",
                    )
            else:
                raise AssembleError(f"GNU as failed ({kind} syntax)", details=as_err)
        return _object_to_image(obj_path, ld_bin=ld_bin, source_kind=kind)


def detect_syntax(text: str) -> str:
    """Guess ``att``, ``intel``, or ``hex`` from assembly text."""
    stripped = text.strip()
    if ".intel_syntax" in stripped:
        # Directives will switch the assembler; feed it as AT&T-mode input.
        return "att"
    if _looks_like_hex(stripped) and not _MNEMONIC_RE.search(stripped):
        return "hex"
    if "%" in stripped or re.search(r"\$\-?\d", stripped):
        return "att"
    intel = len(_INTEL_RE.findall(stripped)) + len(_INTEL_REG_OP.findall(stripped))
    if intel:
        return "intel"
    att = len(_ATT_RE.findall(stripped))
    if att:
        return "att"
    return "att"


def _looks_like_hex(text: str) -> bool:
    compact = re.sub(r"(0x|\\x|[\s,;:_h])", "", text, flags=re.IGNORECASE)
    return bool(compact) and len(compact) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", compact) is not None


def _parse_hex(text: str) -> bytes:
    s = text.strip()
    s = s.replace("\\x", " ")
    s = re.sub(r"0x", " ", s, flags=re.I)
    s = re.sub(r"[,;:_h]", " ", s, flags=re.I)
    s = re.sub(r"\s+", "", s)
    if not s:
        raise AssembleError("empty hex input")
    if len(s) % 2 != 0:
        raise AssembleError(f"odd-length hex string ({len(s)} nybbles)")
    try:
        return bytes.fromhex(s)
    except ValueError as exc:
        raise AssembleError(f"invalid hex: {exc}") from exc


def _raw_bytes(blob: bytes, source_kind: str = "bytes") -> Assembled:
    if not blob:
        raise AssembleError("empty machine-code input")
    if len(blob) > 1 << 20:
        raise AssembleError(f"machine-code input too large ({len(blob)} bytes; max 1 MiB)")
    return Assembled(
        text=blob,
        text_addr=CODE_BASE,
        symbols={"_start": CODE_BASE},
        source_kind=source_kind,
    )


def _run_as(src: Path, obj: Path, kind: str, as_bin: str) -> str | None:
    cmd = [as_bin, "--64", "-o", str(obj), str(src)]
    if kind == "intel":
        cmd[1:1] = ["-msyntax=intel", "-mnaked-reg"]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise AssembleError(f"GNU assembler not found: {as_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AssembleError("GNU as timed out") from exc
    if proc.returncode != 0 or not obj.is_file():
        return (proc.stderr or proc.stdout or f"as exit {proc.returncode}").strip()
    return None


def _object_to_image(obj_path: Path, *, ld_bin: str, source_kind: str) -> Assembled:
    warnings: list[str] = []
    _inspect_object(obj_path, warnings)
    with tempfile.TemporaryDirectory(prefix="e2m-ld-") as td:
        td_path = Path(td)
        script = td_path / "link.ld"
        out = td_path / "out.elf"
        script.write_text(_LINKER_SCRIPT)
        entry = _first_global_code_symbol(obj_path)
        cmd = [
            ld_bin,
            "-nostdlib",
            "-static",
            "-n",
            "-T",
            str(script),
            "-o",
            str(out),
            str(obj_path),
        ]
        if entry:
            cmd.extend(["-e", entry])
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise AssembleError(f"GNU ld not found: {ld_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AssembleError("GNU ld timed out") from exc
        if proc.returncode != 0 or not out.is_file():
            err = (proc.stderr or proc.stdout or f"ld exit {proc.returncode}").strip()
            if "undefined reference" in err or "undefined symbol" in err:
                raise AssembleError(
                    "relocation to an external symbol (ld could not resolve)",
                    details=err,
                )
            raise AssembleError("linker failed", details=err)
        if proc.stderr.strip():
            warnings.append(proc.stderr.strip())
        return _read_linked_elf(out, source_kind=source_kind, warnings=warnings)


def _inspect_object(path: Path, warnings: list[str]) -> None:
    with path.open("rb") as fh:
        elf = ELFFile(fh)
        if elf.get_machine_arch() not in ("x64", "x86-64", "AMD64"):
            raise AssembleError(f"unsupported architecture: {elf.get_machine_arch()} (need x86-64)")

        undefined: list[str] = []
        for sym in _iter_symbols(elf):
            name = sym.name
            shndx = sym["st_shndx"]
            bind = sym["st_info"]["bind"]
            if not name:
                continue
            if name in {"_GLOBAL_OFFSET_TABLE_", "_GLOBAL_OFFSET_TABLE"}:
                raise AssembleError(
                    "object references _GLOBAL_OFFSET_TABLE_ (PIC/GOT is not allowed)",
                )
            if shndx in ("SHN_UNDEF", 0) and bind in ("STB_GLOBAL", "STB_WEAK", "STB_GNU_UNIQUE"):
                undefined.append(name)

        if undefined:
            names = ", ".join(sorted(set(undefined)))
            raise AssembleError(
                "relocations to external symbols are not allowed",
                details=f"undefined: {names}",
            )

        for sect in elf.iter_sections():
            name = sect.name or ""
            flags = sect["sh_flags"]
            alloc = bool(flags & 0x2)  # SHF_ALLOC
            size = sect["sh_size"]
            if name.startswith(".rela") or name.startswith(".rel"):
                continue
            if name in _IGNORED_SECTIONS:
                continue
            if not alloc:
                continue
            allowed = any(
                name == allow or name.startswith(allow + ".") for allow in _ALLOWED_ALLOC_NONEMPTY
            )
            if allowed:
                continue
            if name in (".data", ".bss", ".data.rel.ro") and size == 0:
                continue
            if size == 0:
                warnings.append(f"ignoring empty allocated section {name!r}")
                continue
            raise AssembleError(
                "allocated section other than .text/.rodata is not allowed",
                details=f"section {name!r} size {size} flags {flags:#x}",
            )

        for sect in elf.iter_sections():
            if not isinstance(sect, RelocationSection):
                continue
            target = elf.get_section(sect["sh_info"])
            target_name = target.name if target is not None else ""
            if not _is_text_or_rodata(target_name):
                # CFI / debug relocs: ignore.
                continue
            for rel in sect.iter_relocations():
                rtype = rel["r_info_type"]
                if rtype in _FORBIDDEN_RELOC:
                    raise AssembleError(
                        f"forbidden relocation {_FORBIDDEN_RELOC[rtype]} "
                        f"in {target_name}+{rel['r_offset']:#x}",
                    )
                if rtype not in _ALLOWED_RELOC:
                    raise AssembleError(
                        f"unsupported relocation type {rtype} "
                        f"in {target_name}+{rel['r_offset']:#x}",
                    )
                sym = _reloc_symbol(elf, sect, rel)
                if sym is None:
                    continue
                shndx = sym["st_shndx"]
                if shndx in ("SHN_UNDEF", 0) and sym.name:
                    raise AssembleError(
                        "relocation to an external symbol is not allowed",
                        details=f"{sym.name} from {target_name}+{rel['r_offset']:#x}",
                    )
                if shndx in ("SHN_COMMON", "SHN_ABS"):
                    if shndx == "SHN_COMMON":
                        raise AssembleError(
                            f"relocation to COMMON symbol {sym.name!r} is not allowed",
                        )
                    continue
                if isinstance(shndx, int):
                    try:
                        dest = elf.get_section(shndx)
                    except Exception:
                        dest = None
                    dest_name = dest.name if dest is not None else ""
                    if dest_name and not _is_text_or_rodata(dest_name):
                        raise AssembleError(
                            "relocation to a section other than .text/.rodata is not allowed",
                            details=(
                                f"{sym.name or '<section>'} in {dest_name!r} "
                                f"from {target_name}+{rel['r_offset']:#x}"
                            ),
                        )


def _is_text_or_rodata(name: str) -> bool:
    return bool(name) and (
        name == ".text"
        or name == ".rodata"
        or name.startswith(".text.")
        or name.startswith(".rodata.")
    )


def _iter_symbols(elf: ELFFile):
    from elftools.elf.sections import SymbolTableSection

    for sect in elf.iter_sections():
        if isinstance(sect, SymbolTableSection):
            yield from sect.iter_symbols()


def _reloc_symbol(elf: ELFFile, rel_section: RelocationSection, rel):
    from elftools.elf.sections import SymbolTableSection

    link = rel_section["sh_link"]
    try:
        symtab = elf.get_section(link)
    except Exception:
        return None
    if not isinstance(symtab, SymbolTableSection):
        return None
    idx = rel["r_info_sym"]
    try:
        return symtab.get_symbol(idx)
    except Exception:
        return None


def _first_global_code_symbol(path: Path) -> str | None:
    with path.open("rb") as fh:
        elf = ELFFile(fh)
        for sym in _iter_symbols(elf):
            if not sym.name or sym.name.startswith("."):
                continue
            if sym["st_info"]["type"] not in ("STT_FUNC", "STT_NOTYPE"):
                continue
            if sym["st_info"]["bind"] not in ("STB_GLOBAL", "STB_WEAK", "STB_LOCAL"):
                continue
            shndx = sym["st_shndx"]
            if shndx in ("SHN_UNDEF", 0):
                continue
            if isinstance(shndx, int):
                try:
                    sect = elf.get_section(shndx)
                except Exception:
                    continue
                if sect is not None and _is_text_or_rodata(sect.name):
                    if sym["st_info"]["bind"] == "STB_GLOBAL":
                        return sym.name
        # second pass: any defined non-empty name
        for sym in _iter_symbols(elf):
            if sym.name and sym["st_shndx"] not in ("SHN_UNDEF", 0, "SHN_ABS"):
                return sym.name
    return None


def _read_linked_elf(path: Path, *, source_kind: str, warnings: list[str]) -> Assembled:
    with path.open("rb") as fh:
        elf = ELFFile(fh)
        text_sect = None
        rodata_sect = None
        for sect in elf.iter_sections():
            if sect.name == ".text" or sect.name.startswith(".text."):
                if text_sect is None:
                    text_sect = sect
            elif sect.name == ".rodata" or sect.name.startswith(".rodata."):
                if rodata_sect is None:
                    rodata_sect = sect
        if text_sect is None or text_sect["sh_size"] == 0:
            raise AssembleError("linked ELF has no .text")
        text = bytes(text_sect.data())
        text_addr = int(text_sect["sh_addr"])
        rodata = bytes(rodata_sect.data()) if rodata_sect is not None else b""
        rodata_addr = int(rodata_sect["sh_addr"]) if rodata_sect is not None and rodata else None
        symbols: dict[str, int] = {}
        for sym in _iter_symbols(elf):
            name = sym.name
            if not name or name.startswith("."):
                continue
            if sym["st_shndx"] in ("SHN_UNDEF", 0):
                continue
            symbols[name] = int(sym["st_value"])
        if not symbols:
            symbols["_start"] = text_addr
        return Assembled(
            text=text,
            text_addr=text_addr,
            rodata=rodata,
            rodata_addr=rodata_addr,
            symbols=symbols,
            source_kind=source_kind,
            warnings=warnings,
        )
