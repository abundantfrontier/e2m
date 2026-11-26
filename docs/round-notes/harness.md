# Round notes: E2M execution harness v0 (CPU only)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Context: E2M is research on a model that turns English into machine code. Read docs/concept-note.md (sections on evaluation and the prototype). Before any model training, we need a safe harness that assembles and runs model-generated x86-64 code and scores it, so a crashing or malicious output can never hurt this machine.

## Deliverables
1. Python project (system python is 3.14; use uv to create a 3.12 venv at .venv if unicorn wheels need it). Package `e2m` with `pyproject.toml`, pytest tests, README section "Harness".
2. `e2m/harness/assemble.py`: take model output as either GNU AT&T or Intel-syntax assembly text (auto-detect or flag) or raw hex bytes; assemble with GNU `as` to an object, extract .text bytes plus symbol offsets. Reject anything with relocations to external symbols or sections other than .text/.rodata (report clearly).
3. `e2m/harness/emu.py`: run one function under Unicorn x86-64 with SysV ABI: map code (RX), a stack (RW, with guard gaps), and input buffers for pointer args; set args in rdi/rsi/rdx/rcx/r8/r9 (and xmm0-7 for doubles); push a sentinel return address and stop when ret hits it. Enforce: instruction count limit, wall timeout, no `syscall`/`int`/`sysenter` (hook and fail), unmapped/permission faults caught. Check callee-saved registers (rbx, rbp, r12-r15, rsp) are preserved and stack is 16-byte aligned at calls. Return a structured result: ok/fault type, return value, output buffers, instruction count, and on fault a crash report (fault address, rip, disassembly window via capstone, all registers). Never let an exception escape; the harness must survive any byte sequence.
4. `e2m/harness/tasks.py` + a task format (JSONL): id, English spec, C signature, test cases (args incl. arrays/strings, expected return and expected output buffers), and a C reference solution. Include ~20 hand-written starter tasks (ints, loops, arrays, strings, a few doubles), each verified by compiling the reference with gcc at -O0/-O2/-Os, extracting the function's asm, and running it through the harness (this is the self-test and the "compiler baseline").
5. `e2m/harness/score.py` + CLI `python -m e2m.score --tasks tasks.jsonl --candidates cands.jsonl`: pass@k, per-task pass/fail with fault reasons, code size in bytes, instruction counts vs the gcc -O2 baseline. Output JSON + a short Markdown summary.
6. Fuzz/robustness test: feed thousands of random byte strings and mutated real functions; harness must never crash, hang, or leak memory; report throughput (runs/sec) using multiprocessing with per-worker timeouts.

## Done means
- `pytest` passes; all 20 starter tasks pass via the gcc baseline at -O0/-O2/-Os; fuzz run of >=10k inputs completes with zero harness crashes.
