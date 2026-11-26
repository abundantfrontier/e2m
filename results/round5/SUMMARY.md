# Round 5: direct English to asm

Teacher is `runs/c-only-7b` (held-out C pass@1 46/78, pass@5 56/78 in round 4). Training targets are gcc `-O2` / `-Os` assembly of C that the teacher passed, mixed with the existing `sft_lora2` English-to-asm pairs. No C-to-asm auxiliary: the eval prompt is English-to-asm, and a row that contains both `c` and `asm` is trained as C-then-asm text.

GPU hours: 4.1684 / 8.0.

## Held-out direct asm

Scored n=52. pass@1 4/52 (0.07692307692307693). pass@5 0.1346153846153846.
Size ratio vs gcc -O2: 1.0151515151515151. Insn ratio: 1.010983981693364.
Greedy categories: `{'doesnt_assemble': 4, 'wrong_answer': 35, 'pass': 4, 'crash': 4, 'timeout': 5}`.

Repair loop (k=8, R=3): pass 0 / n=0.

## Val direct asm

Scored n=0. pass@1 0/0 (None). pass@5 None.
Size ratio vs gcc -O2: None. Insn ratio: None.
Val loop: pass 0 / n=0.

## Round 4 comparison (copied, not re-scored)

Direct asm pass@1 7/78, pass@5 9/78, C-then-asm loop 19/78, hybrid gcc -O2 48/78. C-only pass@1 46/78, pass@5 56/78.

Of the 42 round-4 C-pass/asm-fail tasks: logic_mismatch 23, crash 9, timeout 9, doesnt_assemble 1, abi 0.

## Data mix

Rows 5886. Proportions `{'sft_lora2': 0.866, 'mbpp': 0.0352, 'synth': 0.0065, 'train': 0.0924}`. By origin `{'sft_lora2': 5097, 'mbpp': 207, 'synth': 38, 'train': 544}`. By goal `{'default': 5576, 'size': 310}`.

Auxiliary C-to-asm: False. Eval uses the English-to-asm prompt (Goal: default). A C-to-asm auxiliary would train a different prompt than inference, and a row that contains both c and asm is treated as C-then-asm text by the trainer. Distill targets stay plain gcc AT&T of the teacher's passing C.

## Stages

| stage | status | code | reason |
|---|---|---:|---|
| prep-mbpp | ok | 0 |  |
| prep-train-manifest | ok | 0 |  |
| screen-tasks | ok | 0 |  |
| synth | error | 1 | stage failed; continuing |
| sample-c | ok | 0 |  |
| lower-asm | ok | 0 |  |
| leakage-text | ok | 0 |  |
| leakage-embed | ok | 0 |  |
| mix | ok | 0 |  |
| train | ok | 0 |  |
| eval-heldout | stopped | 3 | max-hours |
| eval-val | skipped | 3 | gpu-hour budget |
| loop-heldout | skipped |  | gpu-hour budget |
| loop-val | skipped |  | gpu-hour budget |
| failure-analysis | ok | 0 |  |

Halt: gpu-hour budget.

