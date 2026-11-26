# E2M round 4

No calendar deadline. GPU-hour budget: 30 hours. Each GPU stage capped at 10 h.
GPU hours used: 7.81/30.
Order: A then B then C then D. Matched 7B C control first, then CPU hybrid gcc lowering, then resume the incomplete C-then-asm held-out loop, then CPU failure analysis.

Prior numbers for lora2, qlora7b-v2, c-then-asm, and the 1.5B C control are copied from `runs/round3/RESULTS.json`. They were not re-scored.

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| c-only-train | ok | 0 |  |
| c-only-eval-val | ok | 0 |  |
| c-only-eval-heldout | ok | 0 |  |
| hybrid-score | ok | 0 |  |
| loop-c-then-asm-heldout | ok | 0 |  |
| failure-analysis | ok | 0 |  |

## Greedy pass@1 / pass@5

c-then-asm pass@k scores the assembly half only. c-only-7b has no direct assembly.

| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |
|---|---|---:|---|---|---|---|
| lora2 | val | 90 | 0.422 (38/90) | 0.500 (45/90) | 1.111 (n=38) | 1.044 (n=38) |
| lora2 | heldout | 78 | 0.051 (4/78) | 0.064 (5/78) | 1.100 (n=4) | 1.310 (n=4) |
| qlora7b-v2 | val | 90 | 0.456 (41/90) | 0.578 (52/90) | 1.019 (n=41) | 0.989 (n=41) |
| qlora7b-v2 | heldout | 78 | 0.064 (5/78) | 0.154 (12/78) | 0.960 (n=5) | 1.011 (n=5) |
| c-then-asm asm | val | 90 | 0.444 (40/90) | 0.600 (54/90) | 1.055 (n=40) | 0.935 (n=40) |
| c-then-asm asm | heldout | 78 | 0.090 (7/78) | 0.115 (9/78) | 1.132 (n=7) | 3.584 (n=7) |
| c-only-7b | val | 90 | 0.689 (62/90) | 0.800 (72/90) | 1.089 (n=62) | 1.200 (n=62) |
| c-only-7b | heldout | 78 | 0.590 (46/78) | 0.718 (56/78) | 1.110 (n=46) | 1.863 (n=46) |
| base-c 1.5B | heldout | 78 | 0.397 (31/78) | 0.628 (49/78) | 1.003 (n=31) | 0.989 (n=31) |

## C half and hybrid (model C, gcc lowers it)

Hybrid compiles the model's C with gcc and scores the image in the same harness. c-only-7b has no direct assembly.

| model | set | path | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |
|---|---|---|---:|---|---|---|---|
| c-then-asm | val | C half gcc -O2 (copied) | 90 | 63/90 | — | — | — |
| c-then-asm | val | direct asm | 90 | 0.444 (40/90) | 0.600 (54/90) | 1.055 (n=40) | 0.935 (n=40) |
| c-then-asm | val | C gcc -O2 | 90 | 0.700 (63/90) | 0.778 (70/90) | 1.049 (n=63) | 1.177 (n=63) |
| c-then-asm | val | C gcc -Os | 90 | 0.700 (63/90) | 0.778 (70/90) | 0.858 (n=63) | 1.462 (n=63) |
| c-only-7b | val | C gcc -O2 | 90 | 0.689 (62/90) | 0.800 (72/90) | 1.089 (n=62) | 1.200 (n=62) |
| c-only-7b | val | C gcc -Os | 90 | 0.689 (62/90) | 0.800 (72/90) | 0.902 (n=62) | 1.513 (n=62) |
| c-only-7b | val | direct asm | 90 | no direct asm | — | — | — |
| c-then-asm | heldout | C half gcc -O2 (copied) | 78 | 48/78 | — | — | — |
| c-then-asm | heldout | direct asm | 78 | 0.090 (7/78) | 0.115 (9/78) | 1.132 (n=7) | 3.584 (n=7) |
| c-then-asm | heldout | C gcc -O2 | 78 | 0.615 (48/78) | 0.782 (61/78) | 1.031 (n=48) | 1.877 (n=48) |
| c-then-asm | heldout | C gcc -Os | 78 | 0.615 (48/78) | 0.782 (61/78) | 0.691 (n=48) | 2.019 (n=48) |
| c-only-7b | heldout | C gcc -O2 | 78 | 0.590 (46/78) | 0.718 (56/78) | 1.110 (n=46) | 1.863 (n=46) |
| c-only-7b | heldout | C gcc -Os | 78 | 0.590 (46/78) | 0.718 (56/78) | 0.670 (n=46) | 2.028 (n=46) |
| c-only-7b | heldout | direct asm | 78 | no direct asm | — | — | — |

## C-then-asm C half and agreement (copied)

- val: C pass 63/90, C/asm agree 63/90, both 38, C-only 25, asm-only 2
- heldout: C pass 48/78, C/asm agree 35/78, both 6, C-only 42, asm-only 1

## Repair loops

| run | n | pass_full | pass_visible |
|---|---:|---|---|
| loop-lora2-val | 90 | 60/90 | 60/90 |
| loop-lora2-heldout | 78 | 10/78 | 15/78 |
| loop-qlora7b-v2-heldout | 78 | 12/78 | 17/78 |
| loop-base-c-heldout | 78 | 44/78 | 48/78 |
| loop-c-then-asm-heldout | 78 | 19/78 | 27/78 |

loop-c-then-asm-heldout combined coverage: 78/78 of held-out. Unfinished (0): none.

## C-only 7B QLoRA train

train examples 6778/6781 tasks, dropped_over_p99 3, token_p50 166, token_p99 458, max_len 2048, steps 1272, train_loss 0.16586638312587668, target c.

## Part D: C-pass / asm-fail held-out

n=78, C-only 42, both 6, asm-only 1, agree 35/78.

| bucket | n |
|---|---:|
| doesnt_assemble | 1 |
| abi | 0 |
| logic_mismatch | 23 |
| crash | 9 |
| timeout | 9 |
| other | 0 |

See `runs/round4/failure_analysis.md` for truncated C vs asm examples.

## GPU hours

| stage | seconds | hours | gpu | ok |
|---|---:|---:|---|---|
| c-only-train | 7950.19 | 2.21 | True | True |
| c-only-eval-val | 751.03 | 0.21 | True | True |
| c-only-eval-heldout | 800.51 | 0.22 | True | True |
| hybrid-score | 163.89 | 0.05 | False | True |
| loop-c-then-asm-heldout | 18602.35 | 5.17 | True | True |
| failure-analysis | 0.03 | 0.00 | False | True |
| **GPU total** | | **7.81** | | |

