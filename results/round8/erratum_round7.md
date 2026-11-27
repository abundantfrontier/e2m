# Round 7 erratum (round 8 stage B, checker v3)

Every round-7 bar (best of five compiler settings) and every claimed win was re-checked with checker v3 (tests + 2000 coverage-guided, UB-filtered inputs). Tasks checked: 150.

## Bars replaced

| Set | Task | Goal | Old bar | New bar | Why the old one failed |
|---|---|---|---|---|---|
| heldout | mbpp47_fact_div_last_digit | size | 49 (clang -Oz, r6-loop) | 65 (clang -Oz, reference) | mismatch: ref ret=0 cand ret=4 cand status=ok |
| heldout | mbpp47_fact_div_last_digit | speed | 119 (clang -O2, r6-loop) | 178 (clang -O2, r4-hybrid) | mismatch: ref ret=0 cand ret=4 cand status=ok |
| heldout | mbpp45_array_gcd | size | 52 (clang -Oz, r6-loop) | 66 (clang -Oz, reference) | mismatch: ref ret=0 cand ret=1 cand status=ok |
| heldout | mbpp45_array_gcd | speed | 280 (gcc -O2, r6-loop) | 325 (gcc -O2, reference) | mismatch: ref ret=0 cand ret=1 cand status=ok |
| heldout | he114_min_subarray_sum | size | 46 (gcc -Os, r6-loop) | 51 (clang -Oz, base7b-greedy) | mismatch: ref ret=0 cand ret=5 cand status=ok |
| val | exebench_delta_AT_b6ca07e07493 | size | 279 (gcc -Os, r6-loop) | 542 (gcc -Os, reference) | mismatch: ref ret=35.0 cand ret=-1.0 cand status=ok |
| val | exebench_delta_AT_b6ca07e07493 | speed | 48 (clang -O2, r6-loop) | 48 (gcc -O2, reference) | mismatch: ref ret=35.0 cand ret=-1.0 cand status=ok |
| val | exebench_ehnlc_strncmp_9353c4ce09ba | size | 34 (clang -Oz, base7b-sample) | 40 (clang -Oz, base7b-greedy) | mismatch: ref ret=0 cand ret=None cand status=insn_limit |
| val | exebench_ehnlc_strncmp_9353c4ce09ba | speed | 80 (clang -O2, base7b-sample) | 96 (clang -O2, base7b-greedy) | mismatch: ref ret=0 cand ret=None cand status=insn_limit |

## Round-7 wins under v3

| Set/goal | Claimed task-goals | Stand under v3 | Rejected by v3 |
|---|---:|---:|---|
| heldout/size | 31 | 25 | model:he3_below_zero, search:he114_min_subarray_sum, search:he131_odd_digit_product, search:he24_largest_divisor, search:he75_is_multiply_prime, search:mbpp100_next_palindrome, search:mbpp151_is_coprime |
| heldout/speed | 7 | 3 | search:he100_make_a_pile, search:he120_k_largest, search:he152_compare_scores, search:he85_sum_even_odd_idx |
| val/size | 11 | 7 | search:exebench_az_paragraph_num_lines_23afebeee5fd, search:exebench_checkmd5_f5e6fad22803, search:exebench_delta_AT_b6ca07e07493, search:exebench_ehstrnlen_ad3e8426bbc9 |
| val/speed | 12 | 6 | model:exebench_CmpDoubles_1153267d175b, search:exebench_az_paragraph_num_lines_23afebeee5fd, search:exebench_checkmd5_f5e6fad22803, search:exebench_delta_AT_b6ca07e07493, search:exebench_ehnlc_strncmp_9353c4ce09ba, search:exebench_ehstrnlen_ad3e8426bbc9 |
