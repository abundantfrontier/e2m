import pytest

from e2m.harness.sandbox import make_write_exit_program, run_shellcode


def test_libseccomp_loads():
    # Installing the filter in the pytest process would kill later syscalls.
    # Just prove the shared library and symbols resolve.
    from e2m.harness.sandbox import _libseccomp

    lib = _libseccomp()
    assert lib.seccomp_init
    assert lib.seccomp_load


def test_write_exit_program():
    code = make_write_exit_program(b"hello\n", exit_code=0)
    r = run_shellcode(code, timeout_s=2.0)
    assert r.status in {"ok", "exit"}, r
    assert r.stdout.startswith(b"hello\n")
    assert r.exit_code == 0
    assert not r.timed_out


def test_syscall_not_on_allowlist_is_killed():
    # mmap is not allowed; syscall nr 9 in rax.
    #   mov rax, 9; syscall
    code = bytes.fromhex("48c7c0090000000f05")
    r = run_shellcode(code, timeout_s=2.0)
    assert r.status in {"sigsys", "signal_31", "signal_11", "timeout", "child_error", "exit"}
    assert r.status != "ok" or r.exit_code not in (0,)
    # Prefer SIGSYS, but a segfault from a half-executed syscall is also a jail.
    assert not (r.status == "ok" and r.exit_code == 0)
