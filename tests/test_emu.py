from e2m.harness.emu import emulate


def test_add_hex():
    r = emulate(
        "4889f84801f0c3",
        signature="int64_t add_i64(int64_t a, int64_t b)",
        args=[40, 2],
        syntax="hex",
    )
    assert r.ok, r.message
    assert r.return_value == 42
    assert r.insn_count >= 3


def test_add_att():
    src = """\
.text
.globl add_i64
add_i64:
    movq %rdi, %rax
    addq %rsi, %rax
    ret
"""
    r = emulate(src, signature="int64_t add_i64(int64_t a, int64_t b)", args=[-7, 10])
    assert r.ok, r.as_dict()
    assert r.return_value == 3


def test_syscall_blocked():
    r = emulate(bytes.fromhex("0f05"), signature="void f(void)", args=[])
    assert not r.ok
    assert r.status == "syscall"
    assert r.crash is not None
    assert r.crash.disasm_window


def test_int80_blocked():
    r = emulate(bytes.fromhex("cd80"), signature="void f(void)", args=[])
    assert r.status == "int"
    assert r.crash is not None


def test_sysenter_blocked():
    r = emulate(bytes.fromhex("0f34"), signature="void f(void)", args=[])
    assert r.status in {"sysenter", "invalid_insn", "unmapped", "fault"}
    assert not r.ok


def test_infinite_loop_hits_limit():
    r = emulate(bytes.fromhex("ebfe"), signature="void f(void)", args=[], insn_limit=200, timeout_s=0.2)
    assert r.status in {"insn_limit", "timeout", "no_return"}
    assert not r.ok


def test_unmapped_fetch():
    # jmp to a high unmapped address: e9 rel32
    # jmp 0x12340000 from 0x400000: rel = 0x12340000 - (0x400000+5)
    r = emulate(bytes.fromhex("ff2500000000"), signature="void f(void)", args=[])  # jmp *(%rip) -> 0
    assert not r.ok
    assert r.status in {"unmapped", "prot", "fault"}


def test_callee_saved_clobber():
    # pop rbx into nowhere: mov $1, %rbx; ret  — clobbers rbx
    r = emulate(
        bytes.fromhex("48c7c301000000c3"),
        signature="void f(void)",
        args=[],
        check_abi=True,
    )
    assert not r.ok
    assert r.status == "abi"
    assert any("rbx" in e for e in r.abi_errors)


def test_callee_saved_preserved():
    # push rbx; pop rbx; ret
    r = emulate(bytes.fromhex("53 5b c3"), signature="void f(void)", args=[], check_abi=True)
    assert r.ok, r.as_dict()


def test_call_alignment_ok():
    # Entry has rsp % 16 == 8. Push 8 bytes so the call sees rsp % 16 == 0.
    #   0: 50                  push rax
    #   1: e8 02 00 00 00      call 8
    #   6: 58                  pop rax
    #   7: c3                  ret
    #   8: c3                  ret (callee)
    r = emulate(bytes.fromhex("50e80200000058c3c3"), signature="void f(void)", args=[], check_abi=True)
    assert r.ok, r.as_dict()
    assert r.abi_ok


def test_call_alignment_flagged():
    # Call at function entry without aligning: rsp % 16 == 8.
    #   0: e8 01 00 00 00      call 6
    #   5: c3
    #   6: c3
    r = emulate(bytes.fromhex("e801000000c3c3"), signature="void f(void)", args=[], check_abi=True)
    assert not r.abi_ok
    assert any("misaligned" in e for e in r.abi_errors)


def test_never_raises_on_garbage():
    for blob in [b"", b"\x00" * 64, b"\xff" * 17, bytes(range(256)), b"\x0f\x05" * 8]:
        r = emulate(blob, signature="int64_t f(int64_t a)", args=[1], insn_limit=500, timeout_s=0.05)
        assert r.status != "harness_error", r.message


def test_short_array_tail_readable_write_faults():
    """Wide reads past a short buffer succeed; a store past the end still faults.

    AVX tail code reads up to 64 bytes past a short array. That padding is
    mapped read-only so the read works and an out-of-bounds store is caught.
    """
    wide = emulate(
        """\
.text
.globl f
f:
    movups (%rdi), %xmm0
    movb 60(%rdi), %cl
    movl (%rdi), %eax
    ret
""",
        signature="int32_t f(int32_t *p)",
        args=[{"id": "a", "ptr": "i32", "data": [0x12345678]}],
    )
    assert wide.ok, wide.as_dict()
    assert wide.return_value == 0x12345678

    store = emulate(
        """\
.text
.globl f
f:
    movb $0x5a, 1(%rdi)
    ret
""",
        signature="void f(uint8_t *p)",
        args=[{"id": "a", "ptr": "u8", "data": [0]}],
    )
    assert not store.ok
    assert store.status in {"prot", "unmapped", "fault"}

    inside = emulate(
        """\
.text
.globl f
f:
    movb $0x5a, (%rdi)
    ret
""",
        signature="void f(uint8_t *p)",
        args=[{"id": "a", "ptr": "u8", "data": [0]}],
    )
    assert inside.ok, inside.as_dict()
    assert inside.buffers["a"] == bytes([0x5A])


def test_double_add():
    r = emulate(
        bytes.fromhex("f20f58c1c3"),
        signature="double add_f64(double a, double b)",
        args=[1.5, 2.25],
    )
    assert r.ok, r.as_dict()
    assert r.return_value == 3.75


def test_pointer_roundtrip_reverse():
    src = """\
.text
.globl reverse_i32
reverse_i32:
    # rdi=arr, rsi=n; reverse two-pointer
    testq %rsi, %rsi
    jle .Ldone
    leaq -4(%rdi,%rsi,4), %rdx
.Lloop:
    cmpq %rdi, %rdx
    jbe .Ldone
    movl (%rdi), %eax
    movl (%rdx), %ecx
    movl %ecx, (%rdi)
    movl %eax, (%rdx)
    addq $4, %rdi
    subq $4, %rdx
    jmp .Lloop
.Ldone:
    ret
"""
    r = emulate(
        src,
        signature="void reverse_i32(int32_t *arr, int64_t n)",
        args=[{"id": "a", "ptr": "i32", "data": [1, 2, 3, 4]}, 4],
    )
    assert r.ok, r.as_dict()
    import struct

    got = struct.unpack("<4i", r.buffers["a"])
    assert got == (4, 3, 2, 1)
