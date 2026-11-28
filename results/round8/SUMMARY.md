# Round 8 summary

Generated 2026-11-27T17:34:07.695488-08:00 (PT). Checker v3; GPU hours 1.01.

**Decision: Partial.** Held-out verified wins (task-goals, either source): 42; held-out claims that failed the recheck or held-back audit: 2; held-out speed wins that also improve llvm-mca on znver3 and skylake: 2; model-unique wins: 3.

## Checker gate (stage A)

Known-wrong round-7 wins rejected: 19/19; equivalent pairs agreed: 10/10; audited round-7 wins still accepted: 43/45. Gate pass: True.

Audited round-7 wins v3 now rejects: search he24_largest_divisor size (mismatch: ref ret=1 cand ret=None cand status=insn_limit); search mbpp100_next_palindrome size (mismatch: ref ret=1000000000001 cand ret=None cand status=insn_limit)

## Round 7 erratum (stage B)

See erratum_round7.md.

## heldout: beating the best compiler bar

| Goal | Tasks | Model wins | Search wins | Either | Median ratio | llvm-mca better (both CPUs) |
|---|---:|---:|---:|---:|---:|---:|
| size | 73 | 2 | 36 | 38 | 0.915 | 3 |
| speed | 73 | 1 | 3 | 4 | 0.877 | 2 |

## val: beating the best compiler bar

| Goal | Tasks | Model wins | Search wins | Either | Median ratio | llvm-mca better (both CPUs) |
|---|---:|---:|---:|---:|---:|---:|
| size | 77 | 0 | 12 | 12 | 0.911 | 3 |
| speed | 77 | 0 | 1 | 1 | 0.705 | 0 |

## ISA extension (reported separately)

Flags -mpopcnt -mlzcnt -mbmi: claimed 11, verified 11: he100_make_a_pile:size, he122_add_elements:size, he146_special_filter:size, he152_compare_scores:size, he26_remove_duplicates:size, he72_will_it_fly:size, he8_sum_product:size, he9_rolling_max:size, mbpp6_differ_one_bit:size, mbpp6_differ_one_bit:speed, mbpp8_square_nums:size.

## Held-back audit

68 claimed wins checked; 2 failed.

- search heldout he114_min_subarray_sum size: holdout mismatch: ref ret=7 cand ret=-9223372036854775808 status=ok
- search heldout mbpp256_count_primes size: holdout mismatch: ref ret=172 cand ret=None status=timeout
