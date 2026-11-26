# Round 5b: finish round 5, then distill-heavy retrain

Stage A finishes `runs/asm-distill-7b`. Stage B trains `runs/asm-distill-7b-v2` with distill rows in the majority and `sft_lora2` capped at about 25%. Teacher samples resume `runs/round5/sample_c.jsonl` (copied, not rewritten) and add temperature completions up to k=8. Every distinct passing asm is kept.

GPU hours: 8.0957 / 8.0.

## Stage A (round-5 adapter, completed here)

Held-out direct asm n=78. pass@1 5/78 (0.0641025641025641). pass@5 0.10256410256410256.
C-then-asm loop pass 8 / n=20 (partial=True).
Full writeup: `runs/round5/SUMMARY_full.md`.

## Stage B held-out direct asm

Scored n=78/78. pass@1 4/78 (0.05128205128205128). pass@5 0.08974358974358974.
Size ratio vs gcc -O2: 0.817516194554238. Insn ratio: 1.1812865497076024.
Greedy categories: `{'wrong_answer': 46, 'crash': 11, 'pass': 4, 'timeout': 6, 'doesnt_assemble': 11}`.
Target >= 20/78: False.

C-then-asm repair loop (k=8, R=3): pass 4 / n=10 partial=True stopped=max-hours.
Loop size ratio: 1.0588888888888888. Insn ratio: 0.8839531680440771.

## Stage B val direct asm

Scored n=90/90. pass@1 22/90 (0.24444444444444444). pass@5 0.3111111111111111.
Size ratio vs gcc -O2: 0.961838591009877. Insn ratio: 1.080208270002539.

## Comparison

Round 4 direct asm 7/78, C-then-asm loop 19/78, hybrid 48/78 (copied, not re-scored).
Stage A direct asm 5/78 on n=78.
Stage B direct asm 4/78 on n=78.

## Data mix

Rows 1614. Proportions `{'mbpp': 0.2751, 'synth': 0.0849, 'train': 0.3903, 'sft_lora2': 0.2497}`. By origin `{'mbpp': 444, 'synth': 137, 'train': 630, 'sft_lora2': 403}`. Distill majority: True.
sft cap: `{'sft_fraction_cap': 0.25, 'sft_origin': 'sft_lora2', 'sft_in': 5097, 'distill_in': 1211, 'sft_kept': 403, 'capped': True}`.

Epochs 2, same hyperparameters as round 5 (r=32, alpha=64, dropout 0.05, lr 2e-4, batch 1, grad accum 16, 4-bit, max len 2048, goals default,size). The train pool was not enlarged past the round-5 screened 346 tasks.

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| a-eval-heldout | ok | 0 |  |
| a-eval-val | ok | 0 |  |
| a-results | ok | 0 |  |
| b-synth | ok | 3 | max-hours |
| b-sample | ok | 0 |  |
| b-lower | ok | 0 |  |
| b-leakage-text | ok | 0 |  |
| b-leakage-embed | ok | 0 |  |
| b-mix | ok | 0 |  |
| b-train | ok | 0 |  |
| b-eval-heldout | ok | 0 |  |
| a-loop-c-then-asm | ok | 3 | max-hours |
| b-eval-val | ok | 0 |  |
| b-loop-c-then-asm | ok | 3 | max-hours |
| a-loop-asm | skipped | 3 | gpu-hour budget |
| b-loop-val | skipped |  | gpu-hour budget |
| a-loop-val | skipped |  | gpu-hour budget |
| leftover-eval | skipped |  | gpu-hour budget |
| failure-analysis | ok | 0 |  |

Halt: gpu-hour budget.

