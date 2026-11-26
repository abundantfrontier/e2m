# Round notes: E2M compiler parameter sweep for dataset generation (CPU only)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Prereq: the harness from BRIEF.md is done and STATUS.md says tests pass. Read STATUS.md, docs/concept-note.md, and the harness code first.

## Goal
For each task (English spec + tests + C reference), produce many verified machine-code variants by sweeping compiler settings, then dedupe and rank them. This becomes E2M training data.

## Deliverables
1. Add a Rust reference to each starter task: `#[no_mangle] pub extern "C" fn` with the same SysV signature, built with `panic=abort`, `--crate-type=staticlib` or `--emit=obj`, no std where practical (`#![no_std]` + panic handler) so the function has no external calls. Verify each Rust reference passes the task tests through the harness.
2. `e2m/data/sweep.py`: for each task and each compiler (gcc, clang if available, rustc), sweep a configurable grid:
   - opt levels (gcc/clang -O0..3,-Os,-Oz; rustc opt-level 0,1,2,3,s,z)
   - target cpu/features (x86-64, x86-64-v2, x86-64-v3, znver3; +/-avx2 etc.)
   - codegen toggles (unroll, vectorize, inline thresholds via -mllvm / -C llvm-args, overflow-checks for rust)
   - optional random LLVM pass orderings on emitted IR via `opt` + `llc` (N random seeds), if LLVM tools are available
   Extract only the target function's .text bytes (and asm text), reject anything with external relocations/calls.
3. Verify every variant with the harness against the task's tests; keep only passing ones. Record compiler, flags, seed, byte size, instruction count per test, asm text, bytes (hex), sha256.
4. Dedupe by sha256 of bytes; keep the flag sets that produced each unique variant.
5. Rank per task: smallest bytes, fewest instructions (proxy for speed). Emit:
   - `data/variants.jsonl` (all unique verified variants)
   - `data/sft.jsonl` (spec + signature -> best variant asm, with a `goal` field: size|speed|default)
   - `data/pairs.jsonl` (preference pairs: better vs worse variant for the same spec and goal)
6. Stats report `data/SWEEP_REPORT.md`: variants generated vs passing vs unique per task and per compiler; size/insn-count spread; which knobs produced the most diversity; failures and why.
7. Must be parallel (multiprocessing, 16 cores available) and resumable (skip already-done task/config combos).

## Done means
