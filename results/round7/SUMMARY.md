# Round 7: beating the compiler on programs that already pass

Generated 2026-11-26T14:22:25.667723-08:00. GPU hours used: 2.546. No training.

**Decision: Partial** (held-out size wins 29, speed wins 7, model beats search on some goal: True, 5,000-input re-check failures: 3).

Acceptance for every variant: all tests (visible and hidden) pass, and the variant agrees with gcc -O2 of the task's reference C on 500 generated inputs, run from an adversarial ABI-legal entry state (random flags, scratch registers and upper halves of narrow int args). Every claimed win was re-checked on 5,000 fresh inputs.

## Differential checker self-test

Equivalent pairs agreed: 10/10; known-different pairs (round 6 visible-only passes) caught: 10/10; hand-kept adversarial cases as expected: 3/3.

## Compiler portfolio

- heldout: 73 tasks; best-size winner {'clang -Oz': 34, 'gcc -Os': 28, 'gcc -O2': 9, 'clang -O2': 2}; median best_size / gcc -O2 = 0.544, / gcc -Os = 0.957; median best_speed / gcc -O2 = 1.0; bar set by a model-written C program for size on 18, speed on 20.
- val: 77 tasks; best-size winner {'gcc -O2': 31, 'gcc -Os': 20, 'clang -Oz': 19, 'clang -O2': 7}; median best_size / gcc -O2 = 0.818, / gcc -Os = 1.0; median best_speed / gcc -O2 = 1.0; bar set by a model-written C program for size on 14, speed on 14.

## Beating the best compiler

| Set | Goal | Tasks | Model wins | Search wins | Either | Mean ratio (model / search / either) | Median ratio (either) |
|---|---|---:|---:|---:|---:|---|---:|
| heldout | size (strictly smaller) | 73 | 4 | 26 | 29 | 0.9951 / 0.9701 / 0.9654 | 1.0 |
| heldout | speed (<= 0.9x insns) | 73 | 2 | 6 | 7 | 0.9816 / 0.9716 / 0.9612 | 1.0 |
| val | size (strictly smaller) | 77 | 0 | 11 | 11 | 1.0 / 0.9535 / 0.9535 | 1.0 |
| val | speed (<= 0.9x insns) | 77 | 1 | 11 | 12 | 0.9986 / 0.9344 / 0.9331 | 1.0 |

Ratios are against the best compiler, over all pass-set tasks, with a task without a verified improvement counted as 1.0. Model coverage (task-goals attempted before the GPU cap): heldout/size: 73, heldout/speed: 73, val/size: 8, val/speed: 8.

## Faster C (stage 4)

8 tasks tried; 0 verified wins (<= 0.9x the best compiler's instruction count on the best C).
- he75_is_multiply_prime: ratio vs reference gcc -O2 34.08 -> None (win: False)
- mbpp78_odd_setbit_count: ratio vs reference gcc -O2 5.373 -> None (win: False)
- mbpp76_count_squares: ratio vs reference gcc -O2 2.576 -> None (win: False)
- he139_special_factorial: ratio vs reference gcc -O2 2.022 -> 1.0 (win: False)
- exebench_bar_c3147f1d23f5: ratio vs reference gcc -O2 2.0 -> None (win: False)
- he25_factorize: ratio vs reference gcc -O2 1.611 -> None (win: False)
- he114_min_subarray_sum: ratio vs reference gcc -O2 1.535 -> 0.599 (win: False)
- he120_k_largest: ratio vs reference gcc -O2 1.52 -> None (win: False)

## Files

portfolio.jsonl, edits.jsonl, search.jsonl, fasterc.jsonl and diff_selftest.json hold the per-task rows (kept with the run; not published because they carry raw assembly and counterexample buffers). recheck.json lists every re-check.

## Post-run audit (added after the run)

The checker used in the run missed whole input families. Every one of the 61 wins that passed the 5,000-input re-check was checked again on 5,000 inputs from wider families (equal values, resized buffers with their length argument, newline and other characters in strings, large float magnitudes). **16 of them are wrong** and are removed below. The run's own table above is kept for the record.

| Set | Goal | Model wins | Search wins | Either | Model-only tasks | Median ratio of the wins |
|---|---|---:|---:|---:|---|---:|
| heldout | size | 3 | 25 | 27 | he131_odd_digit_product, mbpp234_cube_volume | 0.923 |
| heldout | speed | 2 | 2 | 3 | mbpp224_count_set_bits | 0.883 |
| val | size | 0 | 7 | 7 | - | 0.897 |
| val | speed | 0 | 6 | 6 | - | 0.817 |

Wrong wins found by the audit:

- model he3_below_zero (size): mismatch: ref ret=1 cand ret=513 cand status=ok
- model exebench_CmpDoubles_1153267d175b (speed): mismatch: ref ret=0 cand ret=-1 cand status=ok
- search he85_sum_even_odd_idx (speed): mismatch: ref ret=-62 cand ret=None cand status=unmapped unmapped
- search he100_make_a_pile (speed): mismatch: ref ret=None cand ret=None cand status=prot prot
- search he114_min_subarray_sum (size): mismatch: ref ret=0 cand ret=2 cand status=ok
- search he120_k_largest (speed): mismatch: ref ret=3 cand ret=None cand status=prot prot
- search he152_compare_scores (speed): mismatch: ref ret=None cand ret=None cand status=prot prot
- search exebench_az_paragraph_num_lines_23afebeee5fd (size): mismatch: ref ret=2 cand ret=1 cand status=ok
- search exebench_az_paragraph_num_lines_23afebeee5fd (speed): mismatch: ref ret=2 cand ret=1 cand status=ok
- search exebench_checkmd5_f5e6fad22803 (size): mismatch: ref ret=0 cand ret=-1 cand status=ok
- search exebench_checkmd5_f5e6fad22803 (speed): mismatch: ref ret=0 cand ret=-1 cand status=ok
- search exebench_delta_AT_b6ca07e07493 (size): mismatch: ref ret=35.0 cand ret=10.0 cand status=ok
- search exebench_delta_AT_b6ca07e07493 (speed): mismatch: ref ret=35.0 cand ret=10.0 cand status=ok
- search exebench_ehnlc_strncmp_9353c4ce09ba (speed): mismatch: ref ret=0 cand ret=None cand status=insn_limit insn_limit
- search exebench_ehstrnlen_ad3e8426bbc9 (speed): mismatch: ref ret=1 cand ret=27 cand status=ok
- search exebench_ehstrnlen_ad3e8426bbc9 (size): mismatch: ref ret=1 cand ret=27 cand status=ok

Questionable but kept: he24_largest_divisor (size): drops the d*d<=n bound and tests the remainder with testl: same results on every reachable input, but correct only while divisors stay below 2^32 and asymptotically slower.

Decision after the audit: still **Partial** (held-out size wins 27, speed wins 3; model-only wins on held-out: 3).
