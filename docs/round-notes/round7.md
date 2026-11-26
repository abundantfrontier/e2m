# E2M round 7 brief: beat the compiler on programs that already pass (approved Thu 26 Nov 2026, 10:53 AM PT)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Source proposal: Report 7 next-job section (copy at ~/e2m-proposals/NEXT-JOB-round7-proposal.md).
No training. Pipeline: `e2m/loop/round7_run.sh cpu` (stages 0, 1, 3; CPU only) and
`e2m/loop/round7_run.sh gpu` (stages 2, 4 under the GPU lock, then the stage-5 summary).
Code: `e2m/harness/diff.py`, `e2m/loop/round7_common.py`, `round7_cpu.py`, `round7_gpu.py`, `round7_summary.py`.

## Why
Round 6: a correct C program exists for 70/78 held-out tasks, but that code is only gcc -O2-sized at the
median and 1.51x gcc -Os. The program's goal is machine code better than a compiler's. This round measures
whether the untrained 7B, a blind search, or both, beat the best of five compiler settings on size or speed
without losing correctness.

## Rules
  atomically (bash noclobber). A trap releases it on EXIT/INT/TERM. After taking it, it waits for GPU memory
  to drop below 3 GB (up to 10 min, else releases and waits again).
  resume-safe). The search stage pauses its workers while the file exists.
- Budget: 4.0 GPU hours hard cap (`runs/round7/gpu_hours.jsonl`, rows with gpu=true). Stage caps: 2 = 2.5 h,
  4 = 0.5 h; each gets min(stage cap, 4.0 - used).
- No courtesy time gate.

## Definitions
- Pass set: held-out and val tasks with a passing C program from the round 6 C loop's best pass, the untrained
  7B's C (greedy and passing samples), or the round 4 hybrid C (up to 6 distinct programs per task; the first
  in that order is "the pass-set C" used in prompts).
- Oracle: gcc -O2 of the task's reference C (stricter than the proposal's "gcc -O2 of the passing C").
- Acceptance: all tests pass (visible and hidden) AND the variant agrees with the oracle on N=500 generated
  inputs (`diff.py`): type-preserving mutations of the task's tests (ints over edges, ranges, wide 33-62-bit
  and k*2^32+small values; buffer lengths kept; strings re-drawn from the tests' alphabet with varied lengths).
  The candidate runs from an adversarial ABI-legal entry state (random flags, rax/r10/r11 and unused argument
  registers, random bits 32-63 above narrow int args; `emulate(..., entry_noise=seed)`, off by default).
  Inputs on which the oracle crashes or times out are skipped.
- Size = .text bytes. Speed = emulator instructions executed, summed over the task's tests.
- Bars: best_size = min bytes over gcc -O2/-O3/-Os and clang -O2/-Oz, over the reference C and every
  pass-set C program (stronger than the proposal, which used the pass-set C only; `ps_best_*` keeps that bar
  too). best_speed = min instructions over gcc -O2/-O3 and clang -O2, same sources. A bar must itself pass
  the acceptance check. clang's `.prefalign` becomes `.p2align 4` and `.addrsig` is dropped so GNU as accepts it.
- Win: size strictly below best_size; speed at most 0.9x best_speed. Every claimed win is re-checked on 5,000
  fresh inputs in the summary.

## Stages
0. `s0-diff-selftest` (CPU): 10 equivalent pairs (other compiler settings of the reference C) must agree; 10
   known-different pairs (round 6 C-loop programs that passed visible tests and failed hidden ones; the checker
   is seeded from the visible tests only and never sees the hidden ones) must be caught; plus hand-kept
   adversarial cases (32-bit truncation, reliance on entry flags) and a control.
1. `s1-portfolio` (CPU, 0.5 h cap, 8 workers): `runs/round7/portfolio.jsonl`, `portfolio_summary.json`.
2. `s2-model-edits` (GPU, 2.5 h): untrained Qwen2.5-Coder-7B-Instruct 4-bit, no adapter. Prompt = spec, the
   pass-set C, the best compiler listing for the goal, the bar. k=8 at T=0.8, then up to R=2 refinement rounds
   with feedback (test failure or diff counterexample, bytes/instructions against the bar). Both goals per task,
   held-out first (round 6 interleaved MBPP/HumanEval order), then val. `edits.jsonl`.
3. `s3-search` (CPU, 2.0 h wall, 8 workers while the other project may be using CPU): STOKE-lite hill climb on
   the same starting listing with the same acceptance (delete / swap / rewrite one instruction from a small
   rule set, delete alignment directives, copy an instruction). `search.jsonl`.
4. `s4-faster-c` (GPU, 0.5 h): tasks whose pass-set C runs > 1.5x gcc -O2's instructions on the reference;
   k=8 faster C programs, compiled with gcc -O2/-O3 and clang -O2, same acceptance. `fasterc.jsonl`.
5. Summary (CPU): `runs/round7/{SUMMARY.md, RESULTS.json, failure_analysis.md, recheck.json}`, then
   `logs/round7.done`. The public auto-sync watcher publishes on it. Round 7 per-task jsonl rows and
   `diff_selftest.json` are published as pointers only (raw assembly, hex buffers, ids like he78_hex_key).

## Decision rule (fixed before the run)
- Win: >= 10 held-out size wins or >= 10 held-out speed wins, the model beats search on at least one goal,
  and zero re-check failures.
- Partial: wins only from search, or fewer than 10.
- No win: no strict wins on either goal.

## Logs
`logs/round7-cpu.log`, `logs/round7-pipeline.log`, per stage `runs/round7/stages/<stage>/log`.
