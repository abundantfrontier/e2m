"""QEMU user-mode backend: starters, AVX2 that Unicorn rejects, throughput."""

from __future__ import annotations

import time

import pytest

from e2m.data.compile import compile_c_asm, find_tool
from e2m.data.extract import extract_function_asm, unsupported_instruction
from e2m.harness.assemble import assemble
from e2m.harness.emu import emulate
from e2m.harness.qemu_backend import find_qemu
from e2m.harness.starter import STARTER_TASKS
from e2m.harness.tasks import assemble_reference, run_gcc_baseline, run_task

qemu_bin = find_qemu()
pytestmark = pytest.mark.skipif(not qemu_bin, reason="qemu-x86_64 not installed under tools/qemu")


def test_qemu_found():
    assert qemu_bin
    assert "qemu" in qemu_bin


def test_qemu_add_hex():
    r = emulate(
        "4889f84801f0c3",
        signature="int64_t add_i64(int64_t a, int64_t b)",
        args=[40, 2],
        syntax="hex",
        backend="qemu",
        timeout_s=1.0,
    )
    assert r.ok, r.message
    assert r.return_value == 42


def test_qemu_syscall_prescan():
    r = emulate(bytes.fromhex("0f05"), signature="void f(void)", args=[], backend="qemu")
    assert not r.ok
    assert r.status == "syscall"


def test_default_backend_is_unicorn():
    r = emulate("4889f84801f0c3", signature="int64_t add_i64(int64_t a, int64_t b)", args=[1, 2], syntax="hex")
    assert r.ok
    assert r.insn_count >= 3  # Unicorn counts; QEMU reports 0


@pytest.mark.parametrize("task", STARTER_TASKS, ids=lambda t: t.id)
def test_starter_o2_on_qemu(task):
    run = run_gcc_baseline(task, opt="-O2", backend="qemu")
    assert run.assemble_error is None, run.assemble_error
    assert run.passed, f"{task.id} qemu -O2: {run.fail_reason}"


def test_avx2_rejected_by_unicorn_passes_qemu():
    task = next(t for t in STARTER_TASKS if t.id == "array_sum_i32")
    gcc = find_tool("gcc")
    assert gcc
    asm = compile_c_asm(task.reference_c, gcc, ["-O3", "-march=x86-64-v3", "-mavx2"])
    extracted = extract_function_asm(asm, "array_sum_i32")
    image = assemble(extracted, syntax="att")
    assert unsupported_instruction(image.text, image.text_addr)
    args = [{"id": "a", "ptr": "i32", "data": list(range(16))}, 16]
    ru = emulate(image, signature=task.sig(), args=args, entry=task.sig().name, backend="unicorn")
    assert not ru.ok
    assert ru.status in {"invalid_insn", "fault"}
    rq = emulate(
        image,
        signature=task.sig(),
        args=args,
        entry=task.sig().name,
        backend="qemu",
        timeout_s=1.0,
    )
    assert rq.ok, rq.message
    assert rq.return_value == sum(range(16))


def test_dot_f64_avx_short_array_on_qemu():
    """n=3 used to SIGSEGV: the tail load ends on the guard page."""
    task = next(t for t in STARTER_TASKS if t.id == "dot_f64")
    gcc = find_tool("gcc")
    assert gcc
    asm = compile_c_asm(task.reference_c, gcc, ["-O3", "-march=x86-64-v3", "-mavx2"])
    extracted = extract_function_asm(asm, "dot_f64")
    image = assemble(extracted, syntax="att")
    run = run_task(task, image, backend="qemu", timeout_s=2.0, count_instructions=False)
    assert run.passed, run.fail_reason


def test_qemu_oob_store_faults_tail_read_ok():
    wide = emulate(
        """\
.text
.globl f
f:
    vmovdqu (%rdi), %ymm0
    movl (%rdi), %eax
    ret
""",
        signature="int32_t f(int32_t *p)",
        args=[{"id": "a", "ptr": "i32", "data": [7]}],
        backend="qemu",
        timeout_s=2.0,
        count_instructions=False,
    )
    assert wide.ok, wide.message
    assert wide.return_value == 7
    store = emulate(
        """\
.text
.globl f
f:
    movb $1, 1(%rdi)
    ret
""",
        signature="void f(uint8_t *p)",
        args=[{"id": "a", "ptr": "u8", "data": [0]}],
        backend="qemu",
        timeout_s=2.0,
        count_instructions=False,
    )
    assert not store.ok
    assert store.status in {"fault", "prot", "unmapped"}


def test_qemu_insn_count_matches_unicorn():
    blob = "4889f84801f0c3"
    sig = "int64_t add_i64(int64_t a, int64_t b)"
    uni = emulate(blob, signature=sig, args=[40, 2], syntax="hex", backend="unicorn")
    qem = emulate(
        blob,
        signature=sig,
        args=[40, 2],
        syntax="hex",
        backend="qemu",
        timeout_s=2.0,
        count_instructions=True,
    )
    assert uni.ok and qem.ok
    assert qem.insn_count == uni.insn_count
    assert qem.insn_count >= 3


def test_qemu_throughput_vs_unicorn():
    task = next(t for t in STARTER_TASKS if t.id == "add_i64")
    image = assemble_reference(task, opt="-O2")
    n = 40
    t0 = time.perf_counter()
    for _ in range(n):
        r = emulate(image, signature=task.sig(), args=[3, 4], entry=task.sig().name, backend="unicorn")
        assert r.ok
    uni = n / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(n):
        r = emulate(
            image,
            signature=task.sig(),
            args=[3, 4],
            entry=task.sig().name,
            backend="qemu",
            timeout_s=1.0,
            count_instructions=False,
        )
        assert r.ok, r.message
    q = n / (time.perf_counter() - t0)
    print(f"\nthroughput add_i64: unicorn={uni:.1f} runs/s qemu={q:.1f} runs/s")
    assert q > 1.0
