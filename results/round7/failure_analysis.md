# Round 7 failure analysis

## Model candidates by outcome

| Outcome | Count | Share |
|---|---:|---:|
| assembled, failed a test (or did not assemble) | 2351 | 61.9% |
| accepted (tests + diff) | 1308 | 34.4% |
| passed all tests, failed the diff check (overfit) | 141 | 3.7% |

## Re-check failures (5,000 inputs)

- search|he75_is_multiply_prime|size: {'accepted': False, 'stage': 'diff', 'detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok', 'diff_cases': 2311, 'size': 87, 'insn': 2399}
- search|mbpp151_is_coprime|size: {'accepted': False, 'stage': 'diff', 'detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok', 'diff_cases': 64, 'size': 47, 'insn': 163}
- search|he131_odd_digit_product|size: {'accepted': False, 'stage': 'diff', 'detail': 'mismatch: ref ret=4557106575 cand ret=262139279 cand status=ok', 'diff_cases': 1183, 'size': 61, 'insn': 237}

## Search

300 task-goals; 850672 candidates tried; 301512 passed the tests; 91 improved on their start; 57 claimed wins before re-check.

## Post-run audit: wins the run's checker accepted wrongly

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
