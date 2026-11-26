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
