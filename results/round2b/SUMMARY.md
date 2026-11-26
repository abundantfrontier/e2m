# E2M round 2b

Deadline: 2026-11-18T18:00:00-08:00 (America/Los_Angeles). GPU-hour budget: 60 additional hours.
GPU hours used: 40.68/60 (17.33 this round, 23.35 carried).

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| loop-lora2-val | skipped | 0 | already recorded ok |
| dpo-train | skipped | 0 | already recorded ok |
| dpo-eval-val | skipped | 0 | already recorded ok |
| dpo-eval-heldout | skipped | 0 | already recorded ok |
| loop-dpo-heldout | skipped |  | resume does not rerun or continue loop-dpo-heldout |
| 7b-base-asm | ok | 0 |  |
| qlora7b-v2-train | ok | 0 |  |
| qlora7b-v2-eval-val | ok | 0 |  |
| qlora7b-v2-eval-heldout | ok | 0 |  |
| loop-qlora7b-v2-heldout | ok | 0 |  |

## Greedy pass@1 / pass@5

Compared with lora1, lora2, qlora7b, and dpo.

| model | set | n | pass@1 | pass@5 | size/gcc -O2 | insn/gcc -O2 |
|---|---|---:|---|---|---|---|
| lora1 | val | 90 | 0.211 (19/90) | 0.367 (33/90) | 1.033 (n=19) | 1.008 (n=19) |
| lora1 | heldout | 78 | 0.026 (2/78) | 0.064 (5/78) | 1.030 (n=2) | 1.026 (n=2) |
| qlora7b | val | 90 | 0.367 (33/90) | 0.500 (45/90) | 1.037 (n=33) | 1.018 (n=33) |
| qlora7b | heldout | 78 | 0.064 (5/78) | 0.103 (8/78) | 1.008 (n=5) | 1.151 (n=5) |
| lora2 | val | 90 | 0.422 (38/90) | 0.500 (45/90) | 1.111 (n=38) | 1.044 (n=38) |
| lora2 | heldout | 78 | 0.051 (4/78) | 0.064 (5/78) | 1.100 (n=4) | 1.310 (n=4) |
| dpo | val | 90 | 0.033 (3/90) | 0.044 (4/90) | 1.000 (n=3) | 1.000 (n=3) |
| dpo | heldout | 78 | 0.000 (0/78) | 0.000 (0/78) | — | — |
| base7b | heldout | 78 | 0.000 (0/78) | 0.038 (3/78) | — | — |
| qlora7b-v2 | val | 90 | 0.456 (41/90) | 0.578 (52/90) | 1.019 (n=41) | 0.989 (n=41) |
| qlora7b-v2 | heldout | 78 | 0.064 (5/78) | 0.154 (12/78) | 0.960 (n=5) | 1.011 (n=5) |

## Repair loops

| run | n | pass_full | pass_visible |
|---|---:|---|---|
| loop-lora1-val | 90 | 49/90 | 51/90 |
| loop-lora1-heldout | 78 | 6/78 | 10/78 |
| loop-lora2-val | 90 | 60/90 | 60/90 |
| loop-lora2-heldout | 78 | 10/78 | 15/78 |
| loop-dpo-heldout | 62 | 0/62 | 0/62 |
| loop-qlora7b-v2-heldout | 78 | 12/78 | 17/78 |

## DPO train

pairs loaded 46472, train 44592, eval-holdout pairs 512, beta 0.1, steps 5574, train_loss 0.18589138984680176.
pair eval_loss 0.2093 on 512 batches (accuracy 0.90234375).

## QLoRA 7B v2 train

train examples 6713/6781 tasks, steps 1260, train_loss 0.11852527523915919.

## GPU hours

| stage | seconds | hours | ok |
|---|---:|---:|---|
| loop-lora2-val | 12349.46 | 3.43 | True |
| dpo-train | 27026.52 | 7.51 | True |
| dpo-eval-val | 4210.26 | 1.17 | True |
| dpo-eval-heldout | 4363.42 | 1.21 | True |
| loop-dpo-heldout | 36106.05 | 10.03 | False |
| 7b-base-asm | 1105.95 | 0.31 | True |
| qlora7b-v2-train | 10711.71 | 2.98 | True |
| qlora7b-v2-eval-val | 1983.07 | 0.55 | True |
| qlora7b-v2-eval-heldout | 4241.10 | 1.18 | True |
| loop-qlora7b-v2-heldout | 44337.50 | 12.32 | True |
| **total** | | **40.68** | |

