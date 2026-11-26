# E2M round 2

Deadline: 2026-11-18T18:00:00-08:00 (America/Los_Angeles). GPU-hour budget: 60 additional hours.
GPU hours used: 23.35/60.

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| loop-lora2-val | ok | 0 |  |
| dpo-train | ok | 0 |  |
| dpo-eval-val | ok | 0 |  |
| dpo-eval-heldout | ok | 0 |  |
| loop-dpo-heldout | stopped | 3 | deadline or max-hours |
| 7b-base-asm | skipped |  | pipeline already halted |
| qlora7b-v2-train | skipped |  | halted before stage 6 |
| qlora7b-v2-eval-val | skipped |  | halted before stage 6 |
| qlora7b-v2-eval-heldout | skipped |  | halted before stage 6 |
| loop-qlora7b-v2-heldout | skipped |  | halted before stage 6 |

## Greedy pass@1 / pass@5

| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |
|---|---|---:|---|---|---|---|
| lora2 | val | 90 | 0.422 (38/90) | 0.500 (45/90) | 1.111 (n=38) | 1.044 (n=38) |
| lora2 | heldout | 78 | 0.051 (4/78) | 0.064 (5/78) | 1.100 (n=4) | 1.310 (n=4) |
| dpo | val | 90 | 0.033 (3/90) | 0.044 (4/90) | 1.000 (n=3) | 1.000 (n=3) |
| dpo | heldout | 78 | 0.000 (0/78) | 0.000 (0/78) | — | — |
| base7b | heldout | 28 | 0.000 (0/28) | 0.071 (2/28) | — | — |
| qlora7b-v2 | val | — | — | — | — | — |
| qlora7b-v2 | heldout | — | — | — | — | — |

## Repair loops

| run | n | pass_full | pass_visible |
|---|---:|---|---|
| loop-lora2-val | 90 | 60/90 | 60/90 |
| loop-lora2-heldout | 78 | 10/78 | 15/78 |
| loop-dpo-heldout | 62 | 0/62 | 0/62 |
| loop-qlora7b-v2-heldout | — | — | — |

## DPO train

pairs loaded 46472, train 44592, eval-holdout pairs 512, beta 0.1, steps 5574, train_loss 0.18589138984680176.
pair eval_loss 0.2093 on 512 batches (accuracy 0.90234375).

## GPU hours

| stage | seconds | hours | ok |
|---|---:|---:|---|
| loop-lora2-val | 12349.46 | 3.43 | True |
| dpo-train | 27026.52 | 7.51 | True |
| dpo-eval-val | 4210.26 | 1.17 | True |
| dpo-eval-heldout | 4363.42 | 1.21 | True |
| loop-dpo-heldout | 36106.05 | 10.03 | False |
| **total** | | **23.35** | |

