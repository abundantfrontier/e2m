"""QEMU linux-user backend for SysV x86-64 functions (AVX2/FMA capable).

Unicorn 2.1.4 cannot retire YMM, FMA, or non-destructive VEX. QEMU 8.2 TCG
can. This module builds a tiny ET_EXEC ELF around the assembled function,
runs it under ``qemu-x86_64`` (never on the host CPU), and parses a result
blob the trampoline writes with ``write(1, ...)`` then ``exit``.

Guest syscalls inside the *function* are rejected by a Capstone pre-scan.
The trampoline is the only trusted code allowed to syscall (write + exit).
The QEMU process is rlimited and wall-clock killed. Default backend stays
Unicorn; this is opt-in via ``backend="qemu"`` or ``E2M_EMU_BACKEND=qemu``.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import struct
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from capstone import CS_ARCH_X86, CS_MODE_64, Cs

from e2m.harness.abi import CType, Signature, parse_signature, unpack_int
from e2m.harness.assemble import CODE_BASE, PAGE, Assembled, assemble
from e2m.harness.emu import (
    BUFFER_BASE,
    CANARY,
    CODE_LIMIT,
    STACK_BASE,
    STACK_SIZE,
    STACK_TOP,
    TAIL_READ_PAD,
    CrashReport,
    EmuResult,
    _coerce_ptr_arg,
)

# Trampoline and result dump live away from Unicorn's code/stack/buffer map.
TRAMP_BASE = 0x00200000
RESULT_BASE = 0x30000000
RESULT_HDR = 128  # magic + rax + xmm0 + 7 callee-saved/rsp

PF_X = 1
PF_W = 2
PF_R = 4
PT_LOAD = 1
PT_GNU_STACK = 0x6474E551
PF_RX = PF_R | PF_X
PF_RW = PF_R | PF_W

_CS = Cs(CS_ARCH_X86, CS_MODE_64)
_CS.detail = False

# ``Trace 0: 0xHOST [cs_base/GUEST_PC/flags/cflags]``. Guest PC is field 2.
_EXEC_PC = re.compile(rb"Trace\s+\d+:\s+0x[0-9a-fA-F]+\s+\[[0-9a-fA-F]+/([0-9a-fA-F]+)/")

_FORBIDDEN_MNEMS = {
    "syscall",
    "sysenter",
    "sysexit",
    "sysret",
    "int",
    "int1",
    "int3",
    "into",
    "in",
    "out",
    "insb",
    "insw",
    "insd",
    "outsb",
    "outsw",
    "outsd",
}

MAGIC = b"E2MRES01"


def find_qemu() -> str | None:
    env = os.environ.get("E2M_QEMU")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    here = Path(__file__).resolve().parents[2] / "tools" / "qemu" / "qemu-x86_64"
    for cand in (here, Path.home() / "e2m" / "tools" / "qemu" / "qemu-x86_64"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which("qemu-x86_64") or shutil.which("qemu-x86_64-static")


def emulate_qemu(
    source: Assembled | str | bytes,
    *,
    signature: str | Signature = "void f(void)",
    args: list[Any] | None = None,
    entry: str | int | None = None,
    insn_limit: int = 1_000_000,
    timeout_s: float = 0.1,
    check_abi: bool = True,
    syntax: str = "auto",
    count_instructions: bool = True,
) -> EmuResult:
    """Same contract as ``e2m.harness.emu.emulate``. Always returns."""
    _ = insn_limit
    try:
        return _emulate_qemu_inner(
            source,
            signature=signature,
            args=args or [],
            entry=entry,
            timeout_s=timeout_s,
            check_abi=check_abi,
            syntax=syntax,
            count_instructions=count_instructions,
        )
    except Exception as exc:  # noqa: BLE001
        return EmuResult(
            status="harness_error",
            ok=False,
            message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _emulate_qemu_inner(
    source: Assembled | str | bytes,
    *,
    signature: str | Signature,
    args: list[Any],
    entry: str | int | None,
    timeout_s: float,
    check_abi: bool,
    syntax: str,
    count_instructions: bool,
) -> EmuResult:
    qemu = find_qemu()
    if not qemu:
        return EmuResult(status="harness_error", ok=False, message="qemu-x86_64 not found")

    if isinstance(source, Assembled):
        image = source
    else:
        try:
            image = assemble(source, syntax=syntax)  # type: ignore[arg-type]
        except Exception as exc:
            from e2m.harness.assemble import AssembleError

            status = "assemble_error" if isinstance(exc, AssembleError) else "harness_error"
            return EmuResult(status=status, ok=False, message=str(exc))

    if isinstance(signature, str):
        try:
            sig = parse_signature(signature)
        except Exception as exc:
            return EmuResult(status="harness_error", ok=False, message=f"bad signature: {exc}")
    else:
        sig = signature

    if not image.text:
        return EmuResult(status="assemble_error", ok=False, message="empty .text")
    if len(image.text) > CODE_LIMIT:
        return EmuResult(status="assemble_error", ok=False, message=".text exceeds 1 MiB")

    try:
        if isinstance(entry, int):
            start = entry
        elif isinstance(entry, str):
            start = image.entry(entry)
        else:
            start = image.entry(sig.name if sig.name in image.symbols else None)
    except Exception as exc:
        return EmuResult(status="assemble_error", ok=False, message=str(exc))

    forbidden = _scan_syscalls(image.text)
    if forbidden:
        return EmuResult(
            status=forbidden,
            ok=False,
            message=f"guest {forbidden} (pre-scan)",
            crash=CrashReport(
                fault_type=forbidden,
                fault_address=start,
                rip=start,
                disasm_window=forbidden,
                registers={},
            ),
        )

    buffers = _layout_buffers(sig, args)
    int_args, xmm_args = _classify_args(sig, args, buffers)

    tramp = _build_trampoline(
        func_addr=start,
        int_args=int_args,
        xmm_args=xmm_args,
        buffers=buffers,
        check_abi=check_abi,
    )

    result_span = RESULT_HDR + sum(len(b.data) for b in buffers)
    result_span = max(result_span, RESULT_HDR)
    result_data = bytearray(result_span)
    result_data[:8] = MAGIC

    text_addr = image.text_addr or CODE_BASE
    loads = [
        _Load(TRAMP_BASE, PF_RX, tramp),
        _Load(text_addr, PF_RX, image.text),
        _Load(STACK_BASE, PF_RW, b"", memsz=STACK_SIZE),
        _Load(RESULT_BASE, PF_RW, bytes(result_data), memsz=_align_up(result_span)),
    ]
    if image.rodata and image.rodata_addr is not None:
        loads.append(_Load(image.rodata_addr, PF_R, image.rodata))
    for buf in buffers:
        span = _align_up(max(len(buf.data), 1))
        page = bytearray(span)
        if buf.data:
            page[span - len(buf.data) :] = buf.data
        loads.append(_Load(buf.page, PF_RW, bytes(page), memsz=span))
        # Read-only tail (>= TAIL_READ_PAD bytes). Stores into it fault.
        pad = _align_up(TAIL_READ_PAD)
        loads.append(_Load(buf.page + span, PF_R, b"\x00" * pad, memsz=pad))

    elf = _build_elf(TRAMP_BASE, loads)
    wall = max(0.05, float(timeout_s))
    raw, status, message, insn_count = _run_qemu(
        qemu,
        elf,
        wall,
        text_range=(text_addr, text_addr + len(image.text)) if count_instructions else None,
    )
    if status != "ok":
        return EmuResult(
            status=status,
            ok=False,
            message=message,
            crash=CrashReport(
                fault_type=status,
                fault_address=None,
                rip=0,
                disasm_window=message[:400],
                registers={},
            ),
        )

    parsed = _parse_result(raw, buffers)
    if parsed is None:
        return EmuResult(
            status="fault",
            ok=False,
            message="qemu result blob missing or truncated",
            crash=CrashReport(
                fault_type="fault",
                fault_address=None,
                rip=0,
                disasm_window=raw[:64].hex() if raw else "",
                registers={},
            ),
        )

    rax, xmm0, regs, out_bufs = parsed
    abi_errors: list[str] = []
    if check_abi:
        for name, want in CANARY.items():
            got = regs.get(name, 0)
            if got != want:
                abi_errors.append(f"callee-saved {name} clobbered: got {got:#x} want {want:#x}")
        want_rsp = (STACK_TOP - 8) + 8
        if regs.get("rsp", 0) != want_rsp:
            abi_errors.append(
                f"rsp not restored: got {regs.get('rsp', 0):#x} want {want_rsp:#x}"
            )

    ret_val = _decode_return(sig.ret, rax, xmm0)
    ok = not abi_errors
    return EmuResult(
        status="abi" if abi_errors else "ok",
        ok=ok,
        return_value=ret_val,
        buffers=out_bufs,
        insn_count=insn_count,
        abi_ok=not abi_errors,
        abi_errors=abi_errors,
        message="; ".join(abi_errors),
    )


def _scan_syscalls(blob: bytes) -> str | None:
    if not blob:
        return None
    try:
        for insn in _CS.disasm(blob, 0):
            mnem = insn.mnemonic.lower()
            if mnem in _FORBIDDEN_MNEMS:
                if mnem == "int":
                    return "int"
                if mnem in {"in", "out", "insb", "insw", "insd", "outsb", "outsw", "outsd"}:
                    return "io"
                return mnem
    except Exception:
        if b"\x0f\x05" in blob:
            return "syscall"
    return None


class _Load:
    def __init__(self, vaddr: int, flags: int, data: bytes, memsz: int | None = None):
        self.vaddr = vaddr
        self.flags = flags
        self.data = data
        # QEMU rejects memsz > filesz on a non-writable PT_LOAD ("non-writable bss").
        if memsz is None:
            memsz = len(data) if not (flags & PF_W) else _align_up(max(len(data), 1))
        self.memsz = memsz


class _Buf:
    def __init__(self, ident: str, page: int, addr: int, data: bytes):
        self.ident = ident
        self.page = page
        self.addr = addr
        self.data = data


def _align_up(n: int, page: int = PAGE) -> int:
    return (n + page - 1) & ~(page - 1)


def _layout_buffers(sig: Signature, args: list[Any]) -> list[_Buf]:
    """Match Unicorn: right-aligned data, read-only tail page, unmapped guard."""
    placed: list[_Buf] = []
    cursor = BUFFER_BASE
    n = max(len(sig.params), len(args))
    for i in range(n):
        ctype = sig.params[i].ctype if i < len(sig.params) else CType("int", 8, True)
        raw = args[i] if i < len(args) else 0
        if ctype.kind != "ptr":
            continue
        ident, data, _writable = _coerce_ptr_arg(i, ctype, raw)
        need = max(len(data), 1)
        span = _align_up(need)
        offset = (span - len(data)) if data else 0
        addr = cursor + offset
        placed.append(_Buf(ident=ident, page=cursor, addr=addr, data=data))
        cursor += span + _align_up(TAIL_READ_PAD) + PAGE
    return placed


def _classify_args(
    sig: Signature, args: list[Any], buffers: list[_Buf]
) -> tuple[list[int], list[tuple[int, int]]]:
    int_args: list[int] = []
    xmm_args: list[tuple[int, int]] = []  # (bitpattern as u64, size)
    buf_i = 0
    for i, param in enumerate(sig.params):
        ctype = param.ctype
        raw = args[i] if i < len(args) else 0
        if ctype.kind == "float":
            if len(xmm_args) >= 8:
                raise ValueError("too many floating-point arguments (max 8 in registers)")
            val = float(raw["value"] if isinstance(raw, dict) and "value" in raw else raw)
            if ctype.size == 4:
                bits = struct.unpack("<I", struct.pack("<f", val))[0]
            else:
                bits = struct.unpack("<Q", struct.pack("<d", val))[0]
            xmm_args.append((bits, ctype.size))
            continue
        if len(int_args) >= 6:
            raise ValueError("too many integer arguments (max 6 in registers)")
        if ctype.kind == "ptr":
            value = buffers[buf_i].addr
            buf_i += 1
        else:
            value = int(raw["value"] if isinstance(raw, dict) and "value" in raw else raw)
            value = unpack_int(value, ctype.size, ctype.signed)
            if value < 0:
                value = (1 << 64) + value
            elif not ctype.signed:
                value &= (1 << (ctype.size * 8)) - 1
        int_args.append(value & ((1 << 64) - 1))
    return int_args, xmm_args


# x86-64 r64 ids for REX.B encoding of B8+rd.
_REG = {
    "rax": 0,
    "rcx": 1,
    "rdx": 2,
    "rbx": 3,
    "rsp": 4,
    "rbp": 5,
    "rsi": 6,
    "rdi": 7,
    "r8": 8,
    "r9": 9,
    "r10": 10,
    "r11": 11,
    "r12": 12,
    "r13": 13,
    "r14": 14,
    "r15": 15,
}
_INT_REGS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")


def _movabs(reg: str, imm: int) -> bytes:
    rid = _REG[reg]
    rex = 0x49 if rid >= 8 else 0x48
    return bytes([rex, 0xB8 + (rid & 7)]) + (imm & ((1 << 64) - 1)).to_bytes(8, "little")


def _build_trampoline(
    *,
    func_addr: int,
    int_args: list[int],
    xmm_args: list[tuple[int, int]],
    buffers: list[_Buf],
    check_abi: bool,
) -> bytes:
    """Machine code: set ABI state, jmp to the function, dump results, exit."""
    prefix = bytearray()
    prefix += _movabs("rsp", STACK_TOP - 8)
    after_slot = len(prefix)
    prefix += _movabs("rax", 0)  # placeholder AFTER
    prefix += b"\x48\x89\x04\x24"  # movq %rax, (%rsp)
    for reg, val in zip(_INT_REGS, int_args):
        prefix += _movabs(reg, val)
    for i, (bits, size) in enumerate(xmm_args):
        # movq imm, %rax; movq %rax, %xmmN
        prefix += _movabs("rax", bits)
        # movq xmmN, rax: 66 48 0F 6E /r with ModRM.reg = N, r/m = rax.
        prefix += bytes([0x66, 0x48, 0x0F, 0x6E, 0xC0 + (i << 3)])
        if size == 4:
            # zero the high half already from the 64-bit move of a 32-bit pattern
            pass
    if check_abi:
        for name, val in CANARY.items():
            prefix += _movabs(name, val)
    prefix += _movabs("rax", func_addr)
    prefix += b"\xff\xe0"  # jmp *%rax
    after = TRAMP_BASE + len(prefix)
    prefix[after_slot + 2 : after_slot + 10] = after.to_bytes(8, "little")

    body = bytearray(prefix)
    body += _movabs("rdi", RESULT_BASE)
    body += b"\x48\x89\x47\x08"  # movq %rax, 8(%rdi)
    body += b"\xf2\x0f\x11\x47\x10"  # movsd %xmm0, 16(%rdi)
    # callee-saved at 32,40,48,56,64,72 and rsp at 80
    body += b"\x48\x89\x5f\x20"  # rbx 32
    body += b"\x48\x89\x6f\x28"  # rbp 40
    body += b"\x4c\x89\x67\x30"  # r12 48
    body += b"\x4c\x89\x6f\x38"  # r13 56
    body += b"\x4c\x89\x77\x40"  # r14 64
    body += b"\x4c\x89\x7f\x48"  # r15 72
    body += b"\x48\x89\x67\x50"  # rsp 80
    dest = RESULT_HDR
    for buf in buffers:
        if not buf.data:
            continue
        body += b"\xfc"  # cld
        body += _movabs("rsi", buf.addr)
        body += _movabs("rdi", RESULT_BASE + dest)
        body += _movabs("rcx", len(buf.data))
        body += b"\xf3\xa4"  # rep movsb
        dest += len(buf.data)
    body += _movabs("rax", 1)  # write
    body += _movabs("rdi", 1)
    body += _movabs("rsi", RESULT_BASE)
    body += _movabs("rdx", dest if dest > RESULT_HDR else RESULT_HDR)
    body += b"\x0f\x05"
    body += _movabs("rax", 60)  # exit
    body += b"\x48\x31\xff"  # xor %rdi, %rdi
    body += b"\x0f\x05"
    return bytes(body)


def _u16(n: int) -> bytes:
    return n.to_bytes(2, "little")


def _u32(n: int) -> bytes:
    return n.to_bytes(4, "little")


def _u64(n: int) -> bytes:
    return n.to_bytes(8, "little")


def _build_elf(entry: int, loads: list[_Load]) -> bytes:
    """Minimal ELF64 ET_EXEC with PT_LOAD segments at absolute vaddrs."""
    nload = len(loads)
    nph = nload + 1  # + PT_GNU_STACK
    ehdr_sz = 64
    ph_sz = 56
    phoff = ehdr_sz
    # Place file payloads after the headers, page-aligned.
    cursor = _align_up(ehdr_sz + ph_sz * nph)
    placements: list[tuple[_Load, int]] = []  # load, file offset
    for ld in loads:
        filesz = len(ld.data)
        if filesz == 0:
            placements.append((ld, 0))
            continue
        # p_offset ≡ p_vaddr (mod PAGE)
        want = cursor + ((ld.vaddr - cursor) & (PAGE - 1))
        if want < cursor:
            want += PAGE
        placements.append((ld, want))
        cursor = want + filesz
    file_end = _align_up(cursor)

    phdrs = bytearray()
    for ld, off in placements:
        filesz = len(ld.data)
        phdrs += _u32(PT_LOAD)
        phdrs += _u32(ld.flags)
        phdrs += _u64(off)
        phdrs += _u64(ld.vaddr)
        phdrs += _u64(ld.vaddr)
        phdrs += _u64(filesz)
        phdrs += _u64(max(ld.memsz, filesz))
        phdrs += _u64(PAGE)
    phdrs += _u32(PT_GNU_STACK)
    phdrs += _u32(PF_RW)
    phdrs += _u64(0)
    phdrs += _u64(0)
    phdrs += _u64(0)
    phdrs += _u64(0)
    phdrs += _u64(0)
    phdrs += _u64(0x10)

    ehdr = bytearray(ehdr_sz)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # 64-bit
    ehdr[5] = 1  # LE
    ehdr[6] = 1  # version
    ehdr[16:18] = _u16(2)  # ET_EXEC
    ehdr[18:20] = _u16(62)  # EM_X86_64
    ehdr[20:24] = _u32(1)
    ehdr[24:32] = _u64(entry)
    ehdr[32:40] = _u64(phoff)
    ehdr[40:48] = _u64(0)
    ehdr[48:52] = _u32(0)
    ehdr[52:54] = _u16(ehdr_sz)
    ehdr[54:56] = _u16(ph_sz)
    ehdr[56:58] = _u16(nph)
    ehdr[58:60] = _u16(0)
    ehdr[60:62] = _u16(0)
    ehdr[62:64] = _u16(0)

    blob = bytearray(file_end)
    blob[:ehdr_sz] = ehdr
    blob[phoff : phoff + len(phdrs)] = phdrs
    for ld, off in placements:
        if ld.data:
            blob[off : off + len(ld.data)] = ld.data
    return bytes(blob)


def _preexec() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        # QEMU TCG spawns helper threads; do not cap NPROC here.
        gb = 2 << 30
        resource.setrlimit(resource.RLIMIT_AS, (gb, gb))
    except (ValueError, resource.error, OSError):
        pass


def _count_exec_log(blob: bytes, lo: int, hi: int) -> int:
    """Count guest instructions in ``[lo, hi)`` from a ``-one-insn-per-tb -d exec`` log."""
    n = 0
    for match in _EXEC_PC.finditer(blob):
        pc = int(match.group(1), 16)
        if lo <= pc < hi:
            n += 1
    return n


def _run_qemu(
    qemu: str,
    elf: bytes,
    timeout_s: float,
    text_range: tuple[int, int] | None = None,
) -> tuple[bytes, str, str, int]:
    with tempfile.TemporaryDirectory(prefix="e2m-qemu-") as td:
        path = Path(td) / "t.elf"
        path.write_bytes(elf)
        os.chmod(path, 0o755)
        cmd = [qemu, "-cpu", "max"]
        log_path: Path | None = None
        if text_range is not None:
            # One guest instruction per TB so each ``Trace`` line is one retire.
            log_path = Path(td) / "exec.log"
            cmd += ["-one-insn-per-tb", "-d", "exec", "-D", str(log_path)]
        cmd.append(str(path))
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=timeout_s + 0.4,
                preexec_fn=_preexec,
            )
        except subprocess.TimeoutExpired:
            return b"", "timeout", "qemu wall timeout", 0
        except OSError as exc:
            return b"", "harness_error", f"qemu exec: {exc}", 0
        insn_count = 0
        if log_path is not None and log_path.is_file() and text_range is not None:
            try:
                insn_count = _count_exec_log(log_path.read_bytes(), text_range[0], text_range[1])
            except OSError:
                insn_count = 0
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace")[:300]
            return proc.stdout or b"", "fault", f"qemu exit {proc.returncode}: {err}", insn_count
        return proc.stdout or b"", "ok", "", insn_count


def _parse_result(
    raw: bytes, buffers: list[_Buf]
) -> tuple[int, bytes, dict[str, int], dict[str, bytes]] | None:
    need = RESULT_HDR + sum(len(b.data) for b in buffers)
    if len(raw) < 8 or raw[:8] != MAGIC:
        return None
    if len(raw) < RESULT_HDR:
        return None
    rax = int.from_bytes(raw[8:16], "little")
    xmm0 = raw[16:32]
    names = ("rbx", "rbp", "r12", "r13", "r14", "r15", "rsp")
    regs = {}
    off = 32
    for name in names:
        regs[name] = int.from_bytes(raw[off : off + 8], "little")
        off += 8
    out: dict[str, bytes] = {}
    cursor = RESULT_HDR
    for buf in buffers:
        n = len(buf.data)
        chunk = raw[cursor : cursor + n] if cursor + n <= len(raw) else b""
        if len(chunk) < n:
            chunk = chunk + b"\x00" * (n - len(chunk))
        out[buf.ident] = chunk[:n]
        cursor += n
    _ = need
    return rax, xmm0, regs, out


def _decode_return(ret: CType, rax: int, xmm0: bytes) -> int | float | None:
    if ret.kind == "void":
        return None
    if ret.kind == "float":
        if ret.size == 4:
            return struct.unpack("<f", xmm0[:4])[0]
        return struct.unpack("<d", xmm0[:8])[0]
    return unpack_int(rax, ret.size if ret.size else 8, ret.signed)
