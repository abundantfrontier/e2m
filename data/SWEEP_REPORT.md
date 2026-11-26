# Compiler sweep report

Every kept variant was assembled and executed under Unicorn by the E2M harness. No generated code ran on the host CPU.

## Tools

- `rustc`: rustc 1.99.1 (58cf8f9c3 2026-10-13) (~/.cargo/bin/rustc)
- `clang`: clang version 23.1.5 (https://github.com/llvm/llvm-project 392dfff97d29b59d77e5cac6d4b6ef2585fe7117) (tools/LLVM-23.1.5-Linux-X64/bin/clang)
- `gcc`: gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0 (/usr/bin/gcc)
- `opt`: LLVM (http://llvm.org/): (tools/LLVM-23.1.5-Linux-X64/bin/opt)
- `llc`: LLVM (http://llvm.org/): (tools/LLVM-23.1.5-Linux-X64/bin/llc)

## Grid

Cartesian grids (see `e2m/data/grid.py`):

- **gcc / clang**: `-O0,-O1,-O2,-O3,-Os,-Oz` × march `x86-64`, `x86-64-v2`, `x86-64-v3`, `znver3` × feature bundles (default, `-mno-avx -mno-avx2`, 128-bit AVX without FMA, plain `-mavx2`, SSE2-only) × toggles (unroll on/off, vectorize on/off, inline-insns 1 and 1000, no alignment).
- **rustc**: `opt-level` `0,1,2,3,s,z` × the same four CPUs × feature bundles (`+avx2`, `-avx,-avx2`, `+avx2,-fma`, SSE2-only, default) × `overflow-checks` off/on × LLVM args (unroll threshold, inline threshold, vectorize-loops off). Built as `#![no_std]` + panic handler, `-C panic=abort`, `--emit=asm`.
- **llvm-opt** (when `clang`, `opt`, and `llc` exist): unoptimized IR (`-disable-O0-optnone`), then `default<O0>`…`default<O3>` and N random new-PM pass pipelines (`--llvm-seeds`). LLVM 23 has no `-Os`/`-Oz` and no `default<Os>`; the size rows are `default<O2>` plus `optsize` / `minsize` function attributes. `llc` gets `-O0`…`-O3` and `-mcpu`.

Fixed C flags on every gcc/clang compile: `-fno-asynchronous-unwind-tables -fno-unwind-tables -fcf-protection=none -fno-stack-protector -fno-pic -ffreestanding -m64`.

Compilers in this run: `gcc`, `clang`, `rustc`, `llvm-opt`.

## Totals

| | count |
|---|---:|
| configs generated | 67640 |
| passing | 61743 |
| failing | 5897 |
| unique variants (task, sha256) | 721 |

## Per task and compiler

| task | gcc gen/pass/uniq | clang gen/pass/uniq | rustc gen/pass/uniq | llvm-opt gen/pass/uniq | size min/med/max | insn min/med/max |
|---|---:|---:|---:|---:|---:|---:|
| abs_i64 | 960/960/2 | 960/960/2 | 1440/1440/2 | 22/22/3 | 11 / 24 / 44 | 20 / 42 / 54 |
| add_f64 | 960/960/4 | 960/960/4 | 1440/1440/2 | 22/22/2 | 5 / 10 / 26 | 8 / 20 / 32 |
| add_i64 | 960/960/2 | 960/960/2 | 1440/1440/2 | 22/22/3 | 5 / 12 / 25 | 10 / 30 / 45 |
| array_max_i32 | 960/830/32 | 960/740/16 | 1440/1240/12 | 22/22/12 | 31 / 148 / 781 | 73 / 91 / 235 |
| array_sum_i32 | 960/834/26 | 960/780/18 | 1440/1240/12 | 22/22/10 | 24 / 114 / 852 | 84 / 105 / 297 |
| clamp_i64 | 960/960/5 | 960/960/3 | 1440/1440/3 | 22/22/5 | 17 / 23 / 77 | 30 / 45 / 79 |
| count_char | 960/960/13 | 960/960/7 | 1440/720/7 | 22/22/13 | 22 / 59 / 103 | 116 / 136 / 349 |
| count_eq_i32 | 960/858/29 | 960/768/18 | 1440/1240/13 | 22/22/12 | 27 / 150 / 996 | 96 / 110 / 308 |
| dot_f64 | 960/720/30 | 960/528/14 | 1440/1332/19 | 22/22/9 | 30 / 102 / 746 | 44 / 57 / 161 |
| dot_i32 | 960/834/24 | 960/768/16 | 1440/1240/12 | 22/22/10 | 33 / 114 / 569 | 49 / 61 / 161 |
| factorial | 960/960/11 | 960/828/10 | 1440/1240/3 | 22/22/11 | 26 / 73 / 209 | 109 / 155 / 529 |
| fib | 960/960/13 | 960/960/8 | 1440/1440/7 | 22/22/13 | 39 / 59 / 257 | 101 / 255 / 710 |
| gcd | 960/960/6 | 960/960/6 | 1440/1440/6 | 22/22/14 | 30 / 62 / 115 | 82 / 106 / 183 |
| is_prime | 960/960/10 | 960/960/5 | 1440/1440/5 | 22/22/14 | 67 / 94 / 235 | 111 / 136 / 228 |
| max3_i64 | 960/960/4 | 960/960/3 | 1440/1440/3 | 22/22/3 | 17 / 23 / 66 | 30 / 45 / 86 |
| memcpy_u8 | 960/906/31 | 960/885/20 | 1440/1440/7 | 22/22/11 | 19 / 182 / 790 | 46 / 60 / 142 |
| reverse_i32 | 960/906/36 | 960/960/13 | 1440/600/10 | 22/22/10 | 45 / 136 / 1310 | 70 / 92 / 238 |
| strcmp_u8 | 960/960/10 | 960/960/3 | 1440/720/4 | 22/22/6 | 27 / 66 / 118 | 104 / 125 / 322 |
| strlen_u8 | 960/960/9 | 960/960/4 | 1440/1440/4 | 22/22/8 | 14 / 32 / 54 | 91 / 103 / 272 |
| sum_to | 960/906/13 | 960/960/7 | 1440/1240/6 | 22/22/10 | 22 / 47 / 188 | 32 / 467 / 1874 |

## Per compiler

| compiler | generated | passing | failing | unique |
|---|---:|---:|---:|---:|
| gcc | 19200 | 18314 | 886 | 310 |
| clang | 19200 | 17777 | 1423 | 179 |
| rustc | 28800 | 25212 | 3588 | 139 |
| llvm-opt | 440 | 440 | 0 | 179 |

Size and instruction-count columns above are min / median / max over unique passing variants of that task. Instruction count is the sum of Unicorn retired instructions across the task's tests, the same proxy the harness scorer uses.

## Which knobs moved the bytes

For each knob, configs are grouped by task, compiler, and every other knob. `fraction` is how often a group that varies that knob produced more than one unique passing sha256. Groups where every config failed do not count.

| knob | groups | groups with >1 sha | fraction | mean unique | max unique |
|---|---:|---:|---:|---:|---:|
| opt | 11098 | 10565 | 0.952 | 3.75 | 6 |
| cpu | 16011 | 6329 | 0.395 | 1.48 | 3 |
| toggle | 9578 | 1388 | 0.145 | 1.21 | 5 |
| feat | 13424 | 1625 | 0.121 | 1.12 | 3 |
| overflow | 13626 | 0 | 0.000 | 1.00 | 1 |

## Failures

| reason | count |
|---|---:|
| unsupported_insn | 3563 |
| external_call | 2280 |
| external_reloc | 54 |

Reason keys:

- `compile_error`: the compiler rejected the flags or crashed.
- `external_call`: the function contains a `call`, so it is not a leaf.
- `external_reloc`: a relocation targets a symbol outside the extracted function and its rodata.
- `unsupported_insn`: YMM, FMA, `vpbroadcast*`, or a VEX instruction whose `vvvv` source is not the destination. Unicorn 2.1.4 cannot run those faithfully, so those variants are not kept.
- `test_fail`: assembled and ran, but a harness test failed.
- `extract` / `assemble_error` / `worker_error`: other drops.

Examples:

- `external_reloc` on `sum_to` via `gcc`: reference to undefined label .LC2
- `unsupported_insn` on `array_sum_i32` via `gcc`: ymm/zmm not emulated: vmovdqu ymm0, ymmword ptr [rax]
- `external_call` on `reverse_i32` via `rustc`: external call in: callq	_RNvNtNtCskADBYZwJmSU_4core9panicking11panic_const24panic_const_div_overflow

## Notes

- Dedup key is sha256 of the linked `.text` bytes of the extracted function.
- `goal=size` prefers fewer bytes, then fewer instructions. `goal=speed` prefers fewer instructions, then fewer bytes. `goal=default` is the plain gcc `-O2 -march=x86-64` variant when it passed (else clang, else rustc `opt-level=2`), which is the usual imitation target.
- Preference pairs sample the ranked list (best against a spread of worse unique variants), capped at 8 per task per goal.
- Rust references use wrapping arithmetic and an inline `idiv` for remainder so the object does not call `core::panicking`. `-C overflow-checks=on` therefore often matches `off` on these tasks. Byte loops load through `read_volatile` so LLVM does not emit a `strlen`/`memcpy`/`memcmp` call.
- LLVM 23 has no `opt`/`llc` `-Os` or `-Oz`, and `default<Os>` is rejected. Size rows run `default<O2>` with `optsize` or `minsize optsize` on the functions. IR is clang `-O0 -disable-O0-optnone` so later pipelines are not blocked by `optnone`.
- `+avx2` and `-march=x86-64-v3`/`znver3` stay in the grid. Outputs that use YMM or FMA fail `unsupported_insn` and are dropped. `-mno-avx` and `-mavx2 -mno-fma -mprefer-vector-width=128` are the executable vector points.
- Unicorn 2.1.4 ignores `VEX.vvvv` and computes `dest = dest OP rm`. A 128-bit VEX instruction is kept only when `vvvv` is unused or names the destination (the destructive form). Non-destructive AVX is `unsupported_insn`, not a passing variant. SSE without a VEX prefix is unaffected.
- `sum_to` under gcc `-mavx2` sometimes emits `.set .LC2, .LC0+24` inside a 32-byte constant pool. The extractor does not follow `.set`, so those configs fail `external_reloc`. The pool is YMM-sized and would be dropped as `unsupported_insn` anyway.
- A unique sha256 that two compilers both emit counts once in the total and once in each compiler's unique column.
- llvm-opt random seeds 6–9 were first recorded with a different `llc` `-O` (the old 6-level cycle) and then again after the cycle was limited to `-O0..-O3`. Both pairings passed and stay in the cache, so llvm-opt's generated count is 80 above the current grid (440 recorded, 360 in the grid).
- Resumable cache: `sweep-cache.jsonl` next to this report. A rerun skips `config_id`s already recorded. That cache is gitignored; the jsonl datasets and this report are not.

