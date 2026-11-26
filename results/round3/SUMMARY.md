# E2M round 3

No calendar deadline. GPU-hour budget: 30 hours.
GPU hours used: 18.60/30.
Order: A then B. DPO collapse guard should stop within a few hundred steps if generation breaks again; C-then-asm is the held-out-gap experiment and gets the remaining budget.

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| dpo-pairs | ok | 0 |  |
| dpo2-train | ok | 0 |  |
| dpo2-eval-val | ok | 0 |  |
| dpo2-eval-heldout | ok | 0 |  |
| loop-dpo2-heldout | skipped |  | val pass@1 26/90 < lora2 38/90 |
| c-then-asm-data | ok | 0 |  |
| c-then-asm-train | ok | 0 |  |
| c-then-asm-eval-val | ok | 0 |  |
| c-then-asm-eval-heldout | ok | 0 |  |
| loop-c-then-asm-heldout | stopped | 3 | max-hours |

## A1 pair-goal choice

Choice: `pair_goal_in_prompt`. user_asm already takes Goal: default|size|speed; each pair keeps its real goal
Pairs loaded: 46472. Goal counts: {'size': 15104, 'speed': 15065, 'default': 16303}.

## Greedy pass@1 / pass@5

Compared with lora2, qlora7b-v2, round-2 dpo, and the C control. c-then-asm pass@k scores the assembly half only.

| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |
|---|---|---:|---|---|---|---|
| lora2 | val | 90 | 0.422 (38/90) | 0.500 (45/90) | 1.111 (n=38) | 1.044 (n=38) |
| lora2 | heldout | 78 | 0.051 (4/78) | 0.064 (5/78) | 1.100 (n=4) | 1.310 (n=4) |
| qlora7b-v2 | val | 90 | 0.456 (41/90) | 0.578 (52/90) | 1.019 (n=41) | 0.989 (n=41) |
| qlora7b-v2 | heldout | 78 | 0.064 (5/78) | 0.154 (12/78) | 0.960 (n=5) | 1.011 (n=5) |
| dpo | val | 90 | 0.033 (3/90) | 0.044 (4/90) | 1.000 (n=3) | 1.000 (n=3) |
| dpo | heldout | 78 | 0.000 (0/78) | 0.000 (0/78) | — | — |
| dpo2 | val | 90 | 0.289 (26/90) | 0.478 (43/90) | 1.121 (n=26) | 1.366 (n=26) |
| dpo2 | heldout | 78 | 0.026 (2/78) | 0.026 (2/78) | 0.727 (n=2) | 1.079 (n=2) |
| c-then-asm | val | 90 | 0.444 (40/90) | 0.600 (54/90) | 1.055 (n=40) | 0.935 (n=40) |
| c-then-asm | heldout | 78 | 0.090 (7/78) | 0.115 (9/78) | 1.132 (n=7) | 3.584 (n=7) |
| base-c | heldout | 78 | 0.397 (31/78) | 0.628 (49/78) | 1.003 (n=31) | 0.989 (n=31) |

## C-then-asm C half and agreement

- val: C pass 63/90, C/asm agree 63/90, both 38, C-only 25, asm-only 2
- heldout: C pass 48/78, C/asm agree 35/78, both 6, C-only 42, asm-only 1

## Repair loops

| run | n | pass_full | pass_visible |
|---|---:|---|---|
| loop-lora2-val | 90 | 60/90 | 60/90 |
| loop-lora2-heldout | 78 | 10/78 | 15/78 |
| loop-qlora7b-v2-heldout | 78 | 12/78 | 17/78 |
| loop-base-c-heldout | 78 | 44/78 | 48/78 |
| loop-dpo2-heldout | — | — | — |
| loop-c-then-asm-heldout | 59 | 15/59 | 21/59 |

## DPO2 train

pairs loaded 46472, train 44848, eval-holdout pairs 256, beta 0.2, lr 5e-06, rpo_weight 1.0, prompt_goal pair, steps 1500, last_good_step 1500, collapsed False, train_loss 0.4841766357421875.
stop_reason: completed

Collapse assemble-rate curve:

- step 0: 29/30 = 0.9666666666666667 (threshold 0.7733333333333334, keep True)
- step 250: 25/30 = 0.8333333333333334 (threshold 0.7733333333333334, keep True)
- step 500: 25/30 = 0.8333333333333334 (threshold 0.7733333333333334, keep True)
- step 750: 28/30 = 0.9333333333333333 (threshold 0.7733333333333334, keep True)
- step 1000: 27/30 = 0.9 (threshold 0.7733333333333334, keep True)
- step 1250: 26/30 = 0.8666666666666667 (threshold 0.7733333333333334, keep True)
- step 1500: 26/30 = 0.8666666666666667 (threshold 0.7733333333333334, keep True)

## C-then-asm QLoRA train

train examples 6725/6781 tasks, steps 1263, train_loss 0.1263779882144003, max_len 2048, target c-then-asm.

C-then-asm rows 6781/6781 (missing C 0).

## GPU hours

| stage | seconds | hours | ok |
|---|---:|---:|---|
| dpo-pairs | 5.19 | 0.00 | True |
| dpo2-train | 8321.18 | 2.31 | True |
| dpo2-eval-val | 895.61 | 0.25 | True |
| dpo2-eval-heldout | 1445.80 | 0.40 | True |
| c-then-asm-data | 0.19 | 0.00 | True |
| c-then-asm-train | 12713.01 | 3.53 | True |
| c-then-asm-eval-val | 2898.80 | 0.81 | True |
| c-then-asm-eval-heldout | 4445.09 | 1.23 | True |
| loop-c-then-asm-heldout | 36246.91 | 10.07 | False |
| **total** | | **18.60** | |

