# Round notes: E2M task scale-up (ExeBench/AnghaBench) + AVX-capable emulator backend

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Prereq: read STATUS.md, docs/concept-note.md, BRIEF.md, BRIEF-sweep.md, data/SWEEP_REPORT.md and the code in e2m/ first.

## Part A: scale the task set
1. Get ExeBench (HF: jordiae/exebench) and AnghaBench (GitHub brenocfg/AnghaBench). Prefer ExeBench first since it ships IO examples/harness hints. Download only what's needed (a few splits/shards), log sizes.
2. Filter to functions the harness can handle: SysV, int/pointer/fixed-size args, no external calls after -O2 (check relocations), no globals or with easily stubbed globals, no floats for now unless trivially supported, under ~300 lines. Record rejection reasons/counts.
3. For each kept function, use qwen3:14b via Ollama (http://localhost:11434, think off / concise prompts, JSON output) to write: an English spec (what the function does, plain language, no code), and if the source has no IO examples, propose test inputs. Expected outputs ALWAYS come from running the reference C (gcc -O0) through the harness, never from the LLM. Also have the model write a second, short "user-style" English prompt (how a person would ask for it).
4. Filter: tests must run cleanly at -O0 and -O2 with matching outputs, and must distinguish the function from trivial stubs (return 0 / return arg0 / identity) - drop tasks where a stub passes. Optional consistency check: ask qwen to judge spec vs code, drop low scores.
5. Target: ~3,000 verified tasks (stop at 5,000 or ~6 hours of LLM time, whichever first). Must be resumable and parallel where CPU-bound. Write tasks/scaled.jsonl in the same schema as tasks/starter.jsonl.
6. Re-run the sweep (data/sweep.py) over the scaled tasks with gcc + clang + rustc-free (C sources only; skip rust for these) and a reduced grid if needed to fit ~4-6h on 16 cores. Output to data/scaled/ (variants, sft, pairs, report). Don't overwrite the starter outputs.

## Part B: AVX-capable emulator backend (investigate, then prototype)
Unicorn 2.1.4 can't run AVX2/FMA/VEX 3-operand, so ~3.5k vectorized variants were dropped. Evaluate options and pick one:
- QEMU user-mode (qemu-x86_64 >= 7.2 TCG supports AVX2; build user-level or static binary) driving a tiny ELF wrapper per test, with gdbstub or a result-dump convention, syscalls limited
- Intel SDE (check license terms; note if it forbids redistribution, user-level use is OK to evaluate)
- Newer Unicorn / other emulators (check if any release supports AVX2)
- Native execution in a subprocess under seccomp-strict + rlimits + ptrace single-step is NOT allowed as the default backend (rule above); you may only note it as an option in the report.
Prototype the chosen backend behind the existing harness API (backend switch), keep Unicorn the default, and add tests: the starter tasks must pass on the new backend, plus a few -march=x86-64-v3 AVX2 variants that Unicorn rejected. Measure runs/sec vs Unicorn. If it works, re-verify the previously dropped AVX variants and report how many now pass.

## Done means
