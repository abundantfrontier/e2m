from e2m.harness.assemble import AssembleError, assemble, detect_syntax
import pytest

ATT_ADD = """\
.text
.globl add_i64
add_i64:
    movq %rdi, %rax
    addq %rsi, %rax
    ret
"""

INTEL_ADD = """\
.text
.globl add_i64
add_i64:
    mov rax, rdi
    add rax, rsi
    ret
"""

HEX_ADD = "4889f84801f0c3"


def test_att_add():
    img = assemble(ATT_ADD, syntax="att")
    assert img.code_size >= 5
    assert "add_i64" in img.symbols
    assert img.symbols["add_i64"] == img.text_addr
    assert img.text.endswith(b"\xc3")


def test_intel_add():
    img = assemble(INTEL_ADD, syntax="intel")
    assert img.text == bytes.fromhex(HEX_ADD)


def test_auto_detect_intel():
    assert detect_syntax(INTEL_ADD) == "intel"
    img = assemble(INTEL_ADD, syntax="auto")
    assert img.text == bytes.fromhex(HEX_ADD)


def test_hex_and_bytes():
    a = assemble(HEX_ADD, syntax="hex")
    b = assemble(bytes.fromhex(HEX_ADD))
    assert a.text == b.text == bytes.fromhex(HEX_ADD)


def test_reject_external_symbol():
    src = """\
.text
.globl bar
bar:
    call foo
    ret
"""
    with pytest.raises(AssembleError) as ei:
        assemble(src, syntax="att")
    msg = str(ei.value).lower()
    assert "external" in msg or "undefined" in msg
    assert "foo" in str(ei.value)


def test_reject_data_section():
    src = """\
.data
.globl x
x:
    .quad 1
.text
.globl f
f:
    movq x(%rip), %rax
    ret
"""
    with pytest.raises(AssembleError) as ei:
        assemble(src, syntax="att")
    msg = str(ei.value).lower()
    assert "section" in msg or "external" in msg or "got" in msg or "reloc" in msg


def test_rodata_string_literal():
    src = """\
.section .rodata
.LC0:
    .string "hi"
.text
.globl use
use:
    leaq .LC0(%rip), %rax
    ret
"""
    img = assemble(src, syntax="att")
    assert img.rodata.startswith(b"hi\x00")
    assert img.rodata_addr is not None
    assert img.rodata_addr != img.text_addr


def test_empty_rejected():
    with pytest.raises(AssembleError):
        assemble("   ")
    with pytest.raises(AssembleError):
        assemble(b"")
