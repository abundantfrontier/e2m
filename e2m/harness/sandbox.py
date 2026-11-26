"""Whole-program runner: child process + rlimits + seccomp (stretch goal).

Generated code is mapped in a forked child after rlimits and a seccomp
allowlist of write / exit / exit_group. The child never execs a host binary,
so the guest cannot spawn /bin/sh. Network syscalls are not on the allowlist.

Requires libseccomp.so.2 (present on this box as a shared library; no sudo).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import select
import signal
import struct
import traceback
from dataclasses import dataclass

# x86-64 Linux syscall numbers.
SYS_WRITE = 1
SYS_EXIT = 60
SYS_EXIT_GROUP = 231

SCMP_ACT_KILL_PROCESS = 0x80000000
SCMP_ACT_ALLOW = 0x7FFF0000
PR_SET_NO_NEW_PRIVS = 38

CLONE_NEWNET = 0x40000000
CLONE_NEWUSER = 0x10000000


@dataclass
class SandboxResult:
    status: str
    exit_code: int | None
    stdout: bytes
    stderr: str
    timed_out: bool
    message: str = ""


def _libseccomp():
    path = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    lib = ctypes.CDLL(path, use_errno=True)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    return lib


def apply_seccomp_allowlist(syscalls: tuple[int, ...] = (SYS_WRITE, SYS_EXIT, SYS_EXIT_GROUP)) -> None:
    """Install a kill-by-default seccomp filter. Must run after no_new_privs."""
    lib = _libseccomp()
    ctx = lib.seccomp_init(SCMP_ACT_KILL_PROCESS)
    if not ctx:
        raise OSError("seccomp_init failed")
    try:
        for nr in syscalls:
            rc = lib.seccomp_rule_add(ctx, SCMP_ACT_ALLOW, int(nr), 0)
            if rc < 0:
                raise OSError(f"seccomp_rule_add({nr}) failed: {rc}")
        rc = lib.seccomp_load(ctx)
        if rc < 0:
            raise OSError(f"seccomp_load failed: {rc}")
    finally:
        lib.seccomp_release(ctx)


def _no_new_privs() -> None:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    rc = libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _try_unshare_net() -> None:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    # Best-effort: a new user+net namespace isolates the network without sudo.
    libc.unshare(CLONE_NEWUSER | CLONE_NEWNET)


def _set_rlimits(cpu_s: int, as_bytes: int, nofile: int, nproc: int) -> None:
    import resource

    def lim(kind: int, value: int) -> None:
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, resource.error, OSError):
            pass

    lim(resource.RLIMIT_CPU, cpu_s)
    lim(resource.RLIMIT_AS, as_bytes)
    lim(resource.RLIMIT_NOFILE, nofile)
    lim(resource.RLIMIT_NPROC, nproc)
    lim(resource.RLIMIT_CORE, 0)
    lim(resource.RLIMIT_FSIZE, 1 << 20)


def _close_extra_fds(keep: set[int]) -> None:
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        names = [str(i) for i in range(256)]
    for name in names:
        try:
            fd = int(name)
        except ValueError:
            continue
        if fd in keep:
            continue
        try:
            os.close(fd)
        except OSError:
            pass


def run_shellcode(
    code: bytes,
    *,
    cpu_s: int = 1,
    as_bytes: int = 512 * 1024 * 1024,
    nofile: int = 16,
    nproc: int = 8,
    timeout_s: float = 2.0,
    stdin: bytes = b"",
) -> SandboxResult:
    """Run raw x86-64 bytes as a whole program in a seccomp-jailed child.

    The bytes are entered as ``_start``: they should use syscall write/exit
    only. Python libc remains mapped in the child, but seccomp blocks every
    syscall except the allowlist, so jumping into libc cannot exec a shell.
    """
    if not code:
        return SandboxResult(status="empty", exit_code=None, stdout=b"", stderr="", timed_out=False, message="empty code")

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    in_r, in_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child. Do not return.
        try:
            os.close(out_r)
            os.close(err_r)
            os.close(in_w)
            os.dup2(in_r, 0)
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            if in_r not in (0,):
                os.close(in_r)
            if out_w not in (1,):
                os.close(out_w)
            if err_w not in (2,):
                os.close(err_w)
            _close_extra_fds({0, 1, 2})
            try:
                _try_unshare_net()
            except Exception:
                pass
            # Map RW, copy, mprotect RX (never RWX — Ubuntu MDWE / W^X).
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            libc.mmap.restype = ctypes.c_void_p
            libc.mmap.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_long,
            ]
            size = max(4096, (len(code) + 4095) & ~4095)
            PROT_READ, PROT_WRITE, PROT_EXEC = 1, 2, 4
            MAP_PRIVATE, MAP_ANONYMOUS = 0x02, 0x20
            addr = libc.mmap(None, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
            if not addr or addr == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), "mmap failed")
            ctypes.memmove(addr, code, len(code))
            if libc.mprotect(ctypes.c_void_p(addr), size, PROT_READ | PROT_EXEC) != 0:
                raise OSError(ctypes.get_errno(), "mprotect RX failed")
            # rlimits after the guest mapping so RLIMIT_AS cannot starve the mmap.
            _set_rlimits(cpu_s, as_bytes, nofile, nproc)
            _no_new_privs()
            apply_seccomp_allowlist()
            fn = ctypes.CFUNCTYPE(None)(addr)
            fn()
            os._exit(0)
        except Exception:
            try:
                os.write(2, traceback.format_exc().encode("utf-8", "replace")[:4000])
            except OSError:
                pass
            os._exit(99)

    os.close(out_w)
    os.close(err_w)
    os.close(in_r)
    try:
        if stdin:
            os.write(in_w, stdin)
    except OSError:
        pass
    os.close(in_w)

    stdout = b""
    stderr_b = b""
    timed_out = False
    deadline = timeout_s
    end = os.times()  # just to have a clock; use time.monotonic
    import time

    t_end = time.monotonic() + deadline
    while True:
        remaining = t_end - time.monotonic()
        if remaining <= 0:
            timed_out = True
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            break
        ready, _, _ = select.select([out_r, err_r], [], [], min(remaining, 0.05))
        if out_r in ready:
            chunk = os.read(out_r, 4096)
            if chunk:
                stdout += chunk
                if len(stdout) > 1 << 20:
                    stdout = stdout[: 1 << 20]
            else:
                # keep draining stderr
                pass
        if err_r in ready:
            chunk = os.read(err_r, 4096)
            if chunk:
                stderr_b += chunk
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            wpid, status = pid, 0
            break
        if wpid == pid:
            break

    if timed_out:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        status = 0
        wpid = pid

    # Drain leftovers.
    for fd, acc in ((out_r, "out"), (err_r, "err")):
        try:
            os.set_blocking(fd, False)
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                if acc == "out":
                    stdout += chunk
                else:
                    stderr_b += chunk
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    exit_code: int | None = None
    st = "ok"
    if timed_out:
        st = "timeout"
    elif os.WIFEXITED(status):
        exit_code = os.WEXITSTATUS(status)
        if exit_code == 99:
            st = "child_error"
        elif exit_code != 0:
            st = "exit"
    elif os.WIFSIGNALED(status):
        exit_code = -os.WTERMSIG(status)
        sig = os.WTERMSIG(status)
        st = "sigsys" if sig == 31 else f"signal_{sig}"  # SIGSYS=31 on Linux
    stderr = stderr_b.decode("utf-8", "replace")
    return SandboxResult(
        status=st,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        message=st,
    )


def make_write_exit_program(message: bytes, exit_code: int = 0) -> bytes:
    """Tiny SysV _start: write(1, msg, n); exit(exit_code). msg is inlined after the code."""
    # We build:
    #   lea rsi, [rip+msg]
    #   mov rdi, 1
    #   mov rdx, n
    #   mov rax, 1          ; sys_write
    #   syscall
    #   mov rdi, exit_code
    #   mov rax, 60         ; sys_exit
    #   syscall
    n = len(message)
    # lea rsi, [rip + disp32]; disp is from next insn.
    # We'll place msg immediately after the last syscall, compute disp.
    code_wo_disp = bytearray()
    # lea rsi, [rip+disp32]  = 48 8d 35 xx xx xx xx
    code_wo_disp += b"\x48\x8d\x35\x00\x00\x00\x00"
    code_wo_disp += b"\x48\xc7\xc7" + struct.pack("<i", 1)  # mov rdi, 1  (sign-extended imm32)
    code_wo_disp += b"\x48\xc7\xc2" + struct.pack("<i", n)  # mov rdx, n
    code_wo_disp += b"\x48\xc7\xc0" + struct.pack("<i", 1)  # mov rax, 1
    code_wo_disp += b"\x0f\x05"
    code_wo_disp += b"\x48\xc7\xc7" + struct.pack("<i", exit_code)
    code_wo_disp += b"\x48\xc7\xc0" + struct.pack("<i", 60)
    code_wo_disp += b"\x0f\x05"
    # disp = len(rest of code after the lea) = len(code) - 7
    disp = len(code_wo_disp) - 7
    code_wo_disp[3:7] = struct.pack("<i", disp)
    return bytes(code_wo_disp) + message
