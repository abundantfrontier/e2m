# Round 5b failure analysis

Buckets use `e2m.train.failure_analysis.classify_asm_fail` (`wrong_answer` is `logic_mismatch`). The 48 hybrid-pass tasks are copied from round 4.

## Stage A direct asm (asm-distill-7b, held-out)

Scored 78. Passes 5. Hybrid overlap (of 48 round-4 hybrid passes, not re-scored): 5.

| bucket | n |
|---|---:|
| doesnt_assemble | 6 |
| abi | 1 |
| logic_mismatch | 49 |
| crash | 9 |
| timeout | 8 |
| other | 0 |

## Stage B direct asm (asm-distill-7b-v2, held-out)

Scored 78. Passes 4. Hybrid overlap (of 48 round-4 hybrid passes, not re-scored): 4.

| bucket | n |
|---|---:|
| doesnt_assemble | 11 |
| abi | 0 |
| logic_mismatch | 46 |
| crash | 11 |
| timeout | 6 |
| other | 0 |


