# Round 5 full eval (stage A of round 5b)

Status: partial. Model `asm-distill-7b`. The original partial `runs/round5/RESULTS.json` and `SUMMARY.md` are unchanged.

Direct asm resumes runs/asm-distill-7b/eval.jsonl (append; the first 52 rows are snapshotted under runs/round5b). The C-then-asm loop is a new directory. Round 4 numbers are copied, not re-scored.

## Held-out direct asm

Scored n=78/78. pass@1 5/78 (0.0641025641025641). pass@5 0.10256410256410256.
Size ratio vs gcc -O2: 0.9371212121212121 (n=5). Insn ratio: 0.9421205186880244 (n=5).
Greedy categories: `{'doesnt_assemble': 6, 'wrong_answer': 49, 'pass': 5, 'crash': 9, 'timeout': 8, 'abi': 1}`.
Failure buckets: `{'doesnt_assemble': 6, 'logic_mismatch': 49, 'crash': 9, 'timeout': 8, 'abi': 1}`.
Overlap with the 48 round-4 hybrid-pass tasks: 5 `['he138_sum_of_four_evens', 'he23_strlen', 'mbpp14_prism_volume', 'mbpp234_cube_volume', 'mbpp8_square_nums']`.

C-then-asm repair loop (k=8, R=3): pass 8 / n=20 of 78. partial=True. stopped=max-hours.
Loop size ratio: 0.7801092049864841. Insn ratio: 0.960959315443011.
Loop categories: `{'wrong_answer': 7, 'pass': 8, 'timeout': 2, 'doesnt_assemble': 2, 'crash': 1}`.

Direct-asm repair loop (only if budget remained): pass 0 / n=0. note=not started.

## Val direct asm

Scored n=90/90. pass@1 29/90 (0.32222222222222224). pass@5 0.5.
Size ratio vs gcc -O2: 1.0678896917699636. Insn ratio: 1.2781692621739646.
Greedy categories: `{'pass': 29, 'timeout': 4, 'doesnt_assemble': 5, 'wrong_answer': 51, 'crash': 1}`.

## Round 4 comparison (copied, not re-scored)

Direct asm pass@1 7/78, pass@5 9/78, C-then-asm loop 19/78, hybrid gcc -O2 48/78.

