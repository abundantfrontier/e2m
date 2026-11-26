"""Run one SysV x86-64 function under Unicorn. Never executes on the host CPU.

The emulator maps RX code, a RW stack with unmapped guard gaps, and a private
RW page per pointer argument. The bytes of a short array are right-aligned at
the end of that page. The following page is readable and not writable: AVX
tail loads may read up to 64 bytes past the logical end (a full page is mapped
because protection is per page), and a store past the end still faults. An
unmapped guard page follows the pad. Execution stops when ``ret`` pops a
sentinel return address. Syscall/int/sysenter, unmapped accesses, an
instruction cap, and a wall-clock timeout all fail closed. Any exception is
converted into a structured :class:`EmuResult`; the caller always gets a value.
"""

from __future__ import annotations

import os
import struct
import traceback
from dataclasses import dataclass, field
from typing import Any

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from unicorn import (
    UC_ARCH_X86,
    UC_HOOK_CODE,
    UC_HOOK_INSN,
    UC_HOOK_INSN_INVALID,
    UC_HOOK_INTR,
    UC_HOOK_MEM_INVALID,
    UC_MEM_FETCH_PROT,
    UC_MEM_FETCH_UNMAPPED,
    UC_MEM_READ_PROT,
    UC_MEM_READ_UNMAPPED,
    UC_MEM_WRITE_PROT,
    UC_MEM_WRITE_UNMAPPED,
    UC_MODE_64,
    UC_PROT_EXEC,
    UC_PROT_READ,
    UC_PROT_WRITE,
    UC_QUERY_TIMEOUT,
    Uc,
    UcError,
)
from unicorn.unicorn_const import (
    UC_ERR_EXCEPTION,
    UC_ERR_FETCH_PROT,
    UC_ERR_FETCH_UNALIGNED,
    UC_ERR_FETCH_UNMAPPED,
    UC_ERR_INSN_INVALID,
    UC_ERR_READ_PROT,
    UC_ERR_READ_UNALIGNED,
    UC_ERR_READ_UNMAPPED,
    UC_ERR_WRITE_PROT,
    UC_ERR_WRITE_UNALIGNED,
    UC_ERR_WRITE_UNMAPPED,
)
from unicorn.x86_const import (
    UC_X86_INS_IN,
    UC_X86_INS_OUT,
    UC_X86_INS_SYSCALL,
    UC_X86_INS_SYSENTER,
    UC_X86_REG_CR0,
    UC_X86_REG_CR4,
    UC_X86_REG_MXCSR,
    UC_X86_REG_R8,
    UC_X86_REG_R9,
    UC_X86_REG_R10,
    UC_X86_REG_R11,
    UC_X86_REG_R12,
    UC_X86_REG_R13,
    UC_X86_REG_R14,
    UC_X86_REG_R15,
    UC_X86_REG_RAX,
    UC_X86_REG_RBP,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDI,
    UC_X86_REG_RDX,
    UC_X86_REG_RFLAGS,
    UC_X86_REG_RIP,
    UC_X86_REG_RSI,
    UC_X86_REG_RSP,
    UC_X86_REG_XMM0,
    UC_X86_REG_XMM1,
    UC_X86_REG_XMM2,
    UC_X86_REG_XMM3,
    UC_X86_REG_XMM4,
    UC_X86_REG_XMM5,
    UC_X86_REG_XMM6,
    UC_X86_REG_XMM7,
)

from e2m.harness.abi import CType, Signature, pack_data, parse_signature, unpack_int
from e2m.harness.assemble import CODE_BASE, PAGE, Assembled, assemble

# Memory map (low → high), with unmapped gaps between regions:
#   SENTINEL  0x00001000  RX  1 page   (ret target; emu_start until=)
#   CODE      0x00400000  RX  up to 1 MiB
#   RODATA    next page   R
#   STACK     0x00800000  RW  1 MiB, grows down; red zone sits inside
#   BUFFERS   0x10000000  RW data page, then a read-only tail pad, then a guard
SENTINEL = 0x1000
# AVX/AVX-512 tail loads read past a short array (ymm=32, zmm=64). One
# read-only page is at least 64 bytes and is not a place a store can land.
TAIL_READ_PAD = 64
STACK_BASE = 0x00800000
STACK_SIZE = 0x00100000  # 1 MiB
STACK_TOP = STACK_BASE + STACK_SIZE  # 16-byte aligned
BUFFER_BASE = 0x10000000
CODE_LIMIT = 1 << 20

INT_ARG_REGS = (
    UC_X86_REG_RDI,
    UC_X86_REG_RSI,
    UC_X86_REG_RDX,
    UC_X86_REG_RCX,
    UC_X86_REG_R8,
    UC_X86_REG_R9,
)
XMM_REGS = (
    UC_X86_REG_XMM0,
    UC_X86_REG_XMM1,
    UC_X86_REG_XMM2,
    UC_X86_REG_XMM3,
    UC_X86_REG_XMM4,
    UC_X86_REG_XMM5,
    UC_X86_REG_XMM6,
    UC_X86_REG_XMM7,
)

# Distinctive callee-saved canaries (SysV: rbx, rbp, r12-r15, rsp).
CANARY = {
    "rbx": 0x1111111111111111,
    "rbp": 0x2222222222222222,
    "r12": 0x1212121212121212,
    "r13": 0x1313131313131313,
    "r14": 0x1414141414141414,
    "r15": 0x1515151515151515,
}
_CANARY_REGS = {
    "rbx": UC_X86_REG_RBX,
    "rbp": UC_X86_REG_RBP,
    "r12": UC_X86_REG_R12,
    "r13": UC_X86_REG_R13,
    "r14": UC_X86_REG_R14,
    "r15": UC_X86_REG_R15,
}

_GPR_NAMES = {
    "rax": UC_X86_REG_RAX,
    "rbx": UC_X86_REG_RBX,
    "rcx": UC_X86_REG_RCX,
    "rdx": UC_X86_REG_RDX,
    "rsi": UC_X86_REG_RSI,
    "rdi": UC_X86_REG_RDI,
    "rbp": UC_X86_REG_RBP,
    "rsp": UC_X86_REG_RSP,
    "r8": UC_X86_REG_R8,
    "r9": UC_X86_REG_R9,
    "r10": UC_X86_REG_R10,
    "r11": UC_X86_REG_R11,
    "r12": UC_X86_REG_R12,
    "r13": UC_X86_REG_R13,
    "r14": UC_X86_REG_R14,
    "r15": UC_X86_REG_R15,
    "rip": UC_X86_REG_RIP,
    "rflags": UC_X86_REG_RFLAGS,
}

_ERR_STATUS = {
    UC_ERR_READ_UNMAPPED: "unmapped",
    UC_ERR_WRITE_UNMAPPED: "unmapped",
    UC_ERR_FETCH_UNMAPPED: "unmapped",
    UC_ERR_READ_PROT: "prot",
    UC_ERR_WRITE_PROT: "prot",
    UC_ERR_FETCH_PROT: "prot",
    UC_ERR_INSN_INVALID: "invalid_insn",
    UC_ERR_READ_UNALIGNED: "unaligned",
    UC_ERR_WRITE_UNALIGNED: "unaligned",
    UC_ERR_FETCH_UNALIGNED: "unaligned",
    UC_ERR_EXCEPTION: "exception",
}

_CS = Cs(CS_ARCH_X86, CS_MODE_64)
_CS.detail = False


@dataclass
class CrashReport:
    fault_type: str
    fault_address: int | None
    rip: int
    disasm_window: str
    registers: dict[str, str]


@dataclass
class EmuResult:
    status: str
    ok: bool
    return_value: int | float | None = None
    buffers: dict[str, bytes] = field(default_factory=dict)
    insn_count: int = 0
    abi_ok: bool = True
    abi_errors: list[str] = field(default_factory=list)
    crash: CrashReport | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "ok": self.ok,
            "return_value": self.return_value,
            "buffers": {k: v.hex() for k, v in self.buffers.items()},
            "insn_count": self.insn_count,
            "abi_ok": self.abi_ok,
            "abi_errors": list(self.abi_errors),
            "message": self.message,
        }
        if self.crash is not None:
            d["crash"] = {
                "fault_type": self.crash.fault_type,
                "fault_address": self.crash.fault_address,
                "rip": self.crash.rip,
                "disasm_window": self.crash.disasm_window,
                "registers": self.crash.registers,
            }
        return d


def emulate(
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
    backend: str | None = None,
    entry_noise: int | None = None,
) -> EmuResult:
    """Run one function. Always returns; never raises to the caller.

    ``entry_noise`` (round 7, unicorn only): when an int seed is given, the
    entry state is made adversarial within what the SysV ABI allows: random
    arithmetic flags, random caller-saved scratch registers (rax, r10, r11 and
    unused argument registers), and random upper bits above narrow int args.

    ``backend`` is ``unicorn`` (default) or ``qemu``. ``E2M_EMU_BACKEND``
    overrides the default when ``backend`` is omitted.
    """
    chosen = (backend or os.environ.get("E2M_EMU_BACKEND") or "unicorn").strip().lower()
    if chosen in {"qemu", "qemu-user", "qemu_user"}:
        from e2m.harness.qemu_backend import emulate_qemu

        return emulate_qemu(
            source,
            signature=signature,
            args=args,
            entry=entry,
            insn_limit=insn_limit,
            timeout_s=timeout_s,
            check_abi=check_abi,
            syntax=syntax,
            count_instructions=count_instructions,
        )
    try:
        return _emulate_inner(
            source,
            signature=signature,
            args=args or [],
            entry=entry,
            insn_limit=insn_limit,
            timeout_s=timeout_s,
            check_abi=check_abi,
            syntax=syntax,
            count_instructions=count_instructions,
            entry_noise=entry_noise,
        )
    except Exception as exc:  # noqa: BLE001 — harness must survive any input
        return EmuResult(
            status="harness_error",
            ok=False,
            message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _emulate_inner(
    source: Assembled | str | bytes,
    *,
    signature: str | Signature,
    args: list[Any],
    entry: str | int | None,
    insn_limit: int,
    timeout_s: float,
    check_abi: bool,
    syntax: str,
    count_instructions: bool,
    entry_noise: int | None = None,
) -> EmuResult:
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

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    _select_cpu(uc)
    _enable_sse(uc)
    _map_code(uc, image)
    _map_stack(uc)
    _map_page(uc, SENTINEL, PAGE, UC_PROT_READ | UC_PROT_EXEC)
    try:
        uc.mem_write(SENTINEL, b"\xf4")  # hlt if we ever execute it
    except UcError:
        pass

    buffers = _place_buffers(uc, sig, args)
    _load_args(uc, sig, args, buffers)

    # Function-entry ABI: (%rsp + 8) % 16 == 0, i.e. rsp % 16 == 8 after the
    # return address is pushed. STACK_TOP is 16-aligned, so STACK_TOP-8 works.
    rsp_entry = STACK_TOP - 8
    uc.mem_write(rsp_entry, SENTINEL.to_bytes(8, "little"))
    uc.reg_write(UC_X86_REG_RSP, rsp_entry)
    uc.reg_write(UC_X86_REG_RFLAGS, 0x202)
    for name, reg in _CANARY_REGS.items():
        uc.reg_write(reg, CANARY[name])
    if entry_noise is not None:
        _apply_entry_noise(uc, sig, entry_noise)

    ctx: dict[str, Any] = {
        "status": None,
        "insns": 0,
        "fault_addr": None,
        "fault_access": None,
        "abi_ok": True,
        "abi_errors": [],
        "intno": None,
    }

    def on_code(_uc: Uc, address: int, size: int, _user: Any) -> None:
        ctx["insns"] += 1
        if ctx["insns"] > insn_limit:
            ctx["status"] = "insn_limit"
            _uc.emu_stop()
            return
        if not check_abi and not count_instructions:
            return
        try:
            op = bytes(_uc.mem_read(address, 1))[0]
        except Exception:
            return
        if op == 0xF4:
            ctx["status"] = "hlt"
            _uc.emu_stop()
            return
        if check_abi and op in (0xE8, 0x9A, 0xFF):
            _check_call_align(_uc, address, size, ctx)

    def on_intr(_uc: Uc, intno: int, _user: Any) -> None:
        ctx["status"] = "int"
        ctx["intno"] = intno
        ctx["fault_addr"] = _uc.reg_read(UC_X86_REG_RIP)
        _uc.emu_stop()

    def on_syscall(_uc: Uc, _user: Any) -> None:
        ctx["status"] = "syscall"
        ctx["fault_addr"] = _uc.reg_read(UC_X86_REG_RIP)
        _uc.emu_stop()

    def on_sysenter(_uc: Uc, _user: Any) -> None:
        ctx["status"] = "sysenter"
        ctx["fault_addr"] = _uc.reg_read(UC_X86_REG_RIP)
        _uc.emu_stop()

    def on_in(_uc: Uc, port: int, size: int, _user: Any) -> int:
        ctx["status"] = "io"
        ctx["fault_addr"] = port
        _uc.emu_stop()
        return 0

    def on_out(_uc: Uc, port: int, size: int, value: int, _user: Any) -> None:
        ctx["status"] = "io"
        ctx["fault_addr"] = port
        _uc.emu_stop()

    def on_mem(_uc: Uc, access: int, address: int, size: int, value: int, _user: Any) -> bool:
        ctx["fault_addr"] = address
        ctx["fault_access"] = access
        if access in (UC_MEM_READ_UNMAPPED, UC_MEM_WRITE_UNMAPPED, UC_MEM_FETCH_UNMAPPED):
            ctx["status"] = "unmapped"
        elif access in (UC_MEM_READ_PROT, UC_MEM_WRITE_PROT, UC_MEM_FETCH_PROT):
            ctx["status"] = "prot"
        else:
            ctx["status"] = "mem"
        return False

    def on_invalid_insn(_uc: Uc, _user: Any) -> bool:
        ctx["status"] = "invalid_insn"
        return False

    uc.hook_add(UC_HOOK_INTR, on_intr)
    uc.hook_add(UC_HOOK_INSN, on_syscall, None, 1, 0, UC_X86_INS_SYSCALL)
    uc.hook_add(UC_HOOK_INSN, on_sysenter, None, 1, 0, UC_X86_INS_SYSENTER)
    uc.hook_add(UC_HOOK_INSN, on_in, None, 1, 0, UC_X86_INS_IN)
    uc.hook_add(UC_HOOK_INSN, on_out, None, 1, 0, UC_X86_INS_OUT)
    uc.hook_add(UC_HOOK_MEM_INVALID, on_mem)
    uc.hook_add(UC_HOOK_INSN_INVALID, on_invalid_insn)
    if count_instructions or check_abi:
        uc.hook_add(UC_HOOK_CODE, on_code)

    timeout_us = max(1, int(timeout_s * 1_000_000))
    uc_error: UcError | None = None
    try:
        uc.emu_start(start, SENTINEL, timeout=timeout_us, count=insn_limit)
    except UcError as exc:
        uc_error = exc

    insn_count = int(ctx["insns"])
    rip = _safe_reg(uc, UC_X86_REG_RIP)
    timed_out = False
    try:
        timed_out = bool(uc.query(UC_QUERY_TIMEOUT))
    except Exception:
        timed_out = False

    status = ctx["status"]
    if status is None:
        if uc_error is not None:
            errno = uc_error.errno if hasattr(uc_error, "errno") else None
            status = _ERR_STATUS.get(errno, "fault") if errno is not None else "fault"
            if ctx["status"]:
                status = ctx["status"]
        elif timed_out:
            status = "timeout"
        elif rip == SENTINEL:
            status = "ok"
        elif insn_count >= insn_limit:
            status = "insn_limit"
        else:
            status = "no_return"

    abi_ok = bool(ctx["abi_ok"])
    abi_errors: list[str] = list(ctx["abi_errors"])
    if check_abi and status == "ok":
        _check_callee_saved(uc, rsp_entry, abi_errors)
        if abi_errors:
            abi_ok = False

    ret_val: int | float | None = None
    out_bufs: dict[str, bytes] = {}
    if status == "ok":
        ret_val = _read_return(uc, sig.ret)
        for spec in buffers:
            try:
                n = len(spec.data)
                out_bufs[spec.ident] = bytes(uc.mem_read(spec.addr, n)) if n else b""
            except Exception:
                out_bufs[spec.ident] = b""

    crash = None
    if status != "ok":
        crash = CrashReport(
            fault_type=status,
            fault_address=ctx.get("fault_addr"),
            rip=rip,
            disasm_window=_disasm_window(uc, rip),
            registers=_snapshot_regs(uc),
        )

    ok = status == "ok" and abi_ok
    msg = ""
    if status == "int" and ctx.get("intno") is not None:
        msg = f"interrupt {ctx['intno']}"
    elif status != "ok":
        msg = status
    if abi_errors and status == "ok":
        msg = "; ".join(abi_errors)

    return EmuResult(
        status="abi" if (status == "ok" and not abi_ok) else status,
        ok=ok,
        return_value=ret_val,
        buffers=out_bufs,
        insn_count=insn_count,
        abi_ok=abi_ok,
        abi_errors=abi_errors,
        crash=crash,
        message=msg,
    )


def _select_cpu(uc: Uc) -> None:
    """Pick a model that implements SSE4.2, POPCNT, BMI2, and ADX.

    Unicorn 2.1.4 still rejects VEX.L (256-bit YMM) encodings. The sweep
    drops those bytes instead of pretending they ran.
    """
    try:
        from unicorn.x86_const import UC_CPU_X86_ICELAKE_CLIENT

        uc.ctl_set_cpu_model(UC_CPU_X86_ICELAKE_CLIENT)
    except Exception:
        return


def _enable_sse(uc: Uc) -> None:
    # CR0.EM=0, CR0.MP=1, CR4.OSFXSR=1, CR4.OSXMMEXCPT=1 so SSE ops run.
    cr0 = uc.reg_read(UC_X86_REG_CR0)
    cr0 = (cr0 & ~(1 << 2)) | (1 << 1)
    uc.reg_write(UC_X86_REG_CR0, cr0)
    cr4 = uc.reg_read(UC_X86_REG_CR4)
    cr4 |= (1 << 9) | (1 << 10)
    uc.reg_write(UC_X86_REG_CR4, cr4)
    uc.reg_write(UC_X86_REG_MXCSR, 0x1F80)


def _align_up(n: int, page: int = PAGE) -> int:
    return (n + page - 1) & ~(page - 1)


def _map_page(uc: Uc, addr: int, size: int, prot: int) -> None:
    uc.mem_map(addr, _align_up(max(size, 1)), prot)


def _map_code(uc: Uc, image: Assembled) -> None:
    text_addr = image.text_addr or CODE_BASE
    text_size = _align_up(max(len(image.text), 1))
    _map_page(uc, text_addr, text_size, UC_PROT_READ | UC_PROT_EXEC)
    uc.mem_write(text_addr, image.text)
    if image.rodata and image.rodata_addr is not None:
        ro_addr = image.rodata_addr
        ro_size = _align_up(max(len(image.rodata), 1))
        # Same page as .text: keep RX (ld script page-aligns .rodata, but be safe).
        already = text_addr <= ro_addr < text_addr + text_size
        if not already:
            try:
                _map_page(uc, ro_addr, ro_size, UC_PROT_READ)
            except UcError:
                # Overlap with an existing map: write into it.
                pass
        try:
            uc.mem_write(ro_addr, image.rodata)
        except UcError:
            pass


def _map_stack(uc: Uc) -> None:
    _map_page(uc, STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)


@dataclass
class _PlacedBuf:
    ident: str
    addr: int
    data: bytes
    writable: bool
    elem: CType | None


def _place_buffers(uc: Uc, sig: Signature, args: list[Any]) -> list[_PlacedBuf]:
    placed: list[_PlacedBuf] = []
    cursor = BUFFER_BASE
    n = max(len(sig.params), len(args))
    for i in range(n):
        ctype = sig.params[i].ctype if i < len(sig.params) else CType("int", 8, True)
        raw = args[i] if i < len(args) else 0
        if ctype.kind != "ptr":
            continue
        ident, data, writable = _coerce_ptr_arg(i, ctype, raw)
        # Right-align in a whole number of pages so the first byte past the
        # logical end is outside the RW span. The next page stays readable
        # (AVX tail loads, and QEMU probes that end exactly on the boundary)
        # and not writable, so a store past the end is still a fault.
        need = max(len(data), 1)
        span = _align_up(need)
        pad = _align_up(TAIL_READ_PAD)
        try:
            _map_page(uc, cursor, span, UC_PROT_READ | UC_PROT_WRITE)
        except UcError:
            cursor += span + pad + PAGE
            _map_page(uc, cursor, span, UC_PROT_READ | UC_PROT_WRITE)
        _map_page(uc, cursor + span, pad, UC_PROT_READ)
        # Empty buffers (n=0) point at the page start and must not sit on the pad.
        offset = (span - len(data)) if data else 0
        addr = cursor + offset
        if data:
            uc.mem_write(addr, data)
        placed.append(_PlacedBuf(ident=ident, addr=addr, data=data, writable=writable, elem=ctype.pointee))
        cursor += span + pad + PAGE  # read-only tail, then an unmapped guard
    return placed


def _coerce_ptr_arg(index: int, ctype: CType, raw: Any) -> tuple[str, bytes, bool]:
    ident = f"arg{index}"
    writable = True
    data: bytes
    if isinstance(raw, dict):
        ident = str(raw.get("id", raw.get("name", ident)))
        writable = bool(raw.get("out", raw.get("writable", True)))
        elem = ctype.pointee or CType("int", 1, False)
        if ctype.cstr or raw.get("ptr") in ("cstr", "str", "char"):
            elem = CType("int", 1, False, cstr=True, spelling="char")
        elif "elem" in raw or "ptr" in raw:
            elem = _elem_from_tag(str(raw.get("elem", raw.get("ptr"))))
        data = pack_data(elem, raw.get("data", raw.get("value", b"")))
        return ident, data, writable
    if isinstance(raw, str):
        return ident, raw.encode("utf-8") + b"\x00", writable
    if isinstance(raw, (bytes, bytearray)):
        return ident, bytes(raw), writable
    if isinstance(raw, list):
        elem = ctype.pointee or CType("int", 4, True)
        return ident, pack_data(elem, raw), writable
    # Scalar given for a pointer: treat as a one-element buffer.
    elem = ctype.pointee or CType("int", 8, True)
    return ident, pack_data(elem, raw), writable


def _elem_from_tag(tag: str) -> CType:
    from e2m.harness.abi import parse_type

    tag = tag.strip().rstrip("*")
    aliases = {
        "i8": "int8_t",
        "u8": "uint8_t",
        "i16": "int16_t",
        "u16": "uint16_t",
        "i32": "int32_t",
        "u32": "uint32_t",
        "i64": "int64_t",
        "u64": "uint64_t",
        "f32": "float",
        "f64": "double",
        "cstr": "char",
        "str": "char",
        "char": "char",
    }
    spelling = aliases.get(tag, tag)
    t = parse_type(spelling)
    if spelling in ("char",) or tag in ("cstr", "str"):
        t = CType(kind="int", size=1, signed=True, cstr=True, spelling="char")
    return t


def _apply_entry_noise(uc: Uc, sig: Signature, seed: int) -> None:
    import random as _random

    from unicorn.x86_const import UC_X86_REG_R10, UC_X86_REG_R11, UC_X86_REG_RAX

    rng = _random.Random(seed)
    # CF PF AF ZF SF OF random; IF set; DF must be clear at entry per the ABI.
    flags = 0x202
    for bit in (0, 2, 4, 6, 7, 11):
        if rng.random() < 0.5:
            flags |= 1 << bit
    uc.reg_write(UC_X86_REG_RFLAGS, flags)
    for reg in (UC_X86_REG_RAX, UC_X86_REG_R10, UC_X86_REG_R11):
        uc.reg_write(reg, rng.getrandbits(64))
    n_int = 0
    for param in sig.params:
        ctype = param.ctype
        if ctype.kind == "float":
            continue
        reg = INT_ARG_REGS[n_int]
        n_int += 1
        if ctype.kind != "ptr" and ctype.size < 8:
            # gcc and clang both extend char/short/bool args to 32 bits (de facto ABI);
            # only bits 32..63 are truly undefined.
            bits = ctype.size * 8
            v = uc.reg_read(reg) & ((1 << bits) - 1)
            if ctype.signed and v >> (bits - 1):
                v -= 1 << bits
            low = v & 0xFFFFFFFF
            uc.reg_write(reg, (rng.getrandbits(32) << 32) | low)
    for reg in INT_ARG_REGS[n_int:]:
        uc.reg_write(reg, rng.getrandbits(64))


def _load_args(uc: Uc, sig: Signature, args: list[Any], buffers: list[_PlacedBuf]) -> None:
    int_i = 0
    xmm_i = 0
    buf_i = 0
    for i, param in enumerate(sig.params):
        ctype = param.ctype
        raw = args[i] if i < len(args) else 0
        if ctype.kind == "float":
            if xmm_i >= len(XMM_REGS):
                raise ValueError("too many floating-point arguments (max 8 in registers)")
            val = float(raw["value"] if isinstance(raw, dict) and "value" in raw else raw)
            uc.reg_write(XMM_REGS[xmm_i], _pack_xmm(val, ctype.size))
            xmm_i += 1
            continue
        # INTEGER class (ints and pointers).
        if int_i >= len(INT_ARG_REGS):
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
        uc.reg_write(INT_ARG_REGS[int_i], value & ((1 << 64) - 1))
        int_i += 1


def _pack_xmm(value: float, size: int) -> int:
    if size == 4:
        payload = struct.pack("<f", value) + b"\x00" * 12
    else:
        payload = struct.pack("<d", value) + b"\x00" * 8
    return int.from_bytes(payload, "little")


def _unpack_xmm(raw: int, size: int) -> float:
    blob = raw.to_bytes(16, "little")
    if size == 4:
        return struct.unpack("<f", blob[:4])[0]
    return struct.unpack("<d", blob[:8])[0]


def _read_return(uc: Uc, ret: CType) -> int | float | None:
    if ret.kind == "void":
        return None
    if ret.kind == "float":
        return _unpack_xmm(uc.reg_read(UC_X86_REG_XMM0), ret.size)
    rax = uc.reg_read(UC_X86_REG_RAX)
    return unpack_int(rax, ret.size if ret.size else 8, ret.signed)


def _check_call_align(uc: Uc, address: int, size: int, ctx: dict[str, Any]) -> None:
    try:
        raw = bytes(uc.mem_read(address, max(size, 16)))
        insns = list(_CS.disasm(raw, address, count=1))
    except Exception:
        return
    if not insns:
        return
    mnem = insns[0].mnemonic
    if mnem not in ("call", "lcall"):
        return
    rsp = uc.reg_read(UC_X86_REG_RSP)
    if rsp & 0xF != 0:
        ctx["abi_ok"] = False
        ctx["abi_errors"].append(f"stack misaligned at call {address:#x} (rsp={rsp:#x})")


def _check_callee_saved(uc: Uc, rsp_entry: int, errors: list[str]) -> None:
    for name, reg in _CANARY_REGS.items():
        got = uc.reg_read(reg)
        want = CANARY[name]
        if got != want:
            errors.append(f"callee-saved {name} clobbered: got {got:#x} want {want:#x}")
    rsp = uc.reg_read(UC_X86_REG_RSP)
    want_rsp = rsp_entry + 8  # ret popped the sentinel
    if rsp != want_rsp:
        errors.append(f"rsp not restored: got {rsp:#x} want {want_rsp:#x}")


def _safe_reg(uc: Uc, reg: int) -> int:
    try:
        return int(uc.reg_read(reg))
    except Exception:
        return 0


def _snapshot_regs(uc: Uc) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, reg in _GPR_NAMES.items():
        try:
            out[name] = f"{uc.reg_read(reg):#018x}"
        except Exception:
            out[name] = "<unreadable>"
    try:
        out["xmm0"] = f"{uc.reg_read(UC_X86_REG_XMM0):#034x}"
    except Exception:
        pass
    return out


def _disasm_window(uc: Uc, rip: int, before: int = 16, after: int = 16) -> str:
    start = rip - before if rip >= before else 0
    length = before + after
    try:
        data = bytes(uc.mem_read(start, length))
    except Exception:
        return f"<unmapped at rip={rip:#x}>"
    lines: list[str] = []
    try:
        for ins in _CS.disasm(data, start):
            mark = "=>" if ins.address == rip else "  "
            lines.append(f"{mark} {ins.address:016x}: {ins.mnemonic} {ins.op_str}".rstrip())
            if ins.address > rip + after:
                break
    except Exception:
        lines.append(data.hex())
    if not lines:
        return f"{rip:#x}: {data.hex()}"
    return "\n".join(lines)
