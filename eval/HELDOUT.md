# Held-out evaluation set

This file is never a training source. Do not add these tasks to `data/scaled/sft.jsonl` or any other train split.

- Tasks: 78 (55 HumanEval, 23 MBPP).
- Each task has an English spec, a SysV signature, a leaf C reference, and tests.
- Every task passed the harness at `gcc -O0` and `gcc -O2`.
- Near-duplicate check against `tasks/scaled.jsonl`: function name, task id, exact normalized body, and difflib ratio >= 0.92.
- Dropped by that check: 1.

Dropped:

- `he41_car_race_collision` (HumanEval/41): body

## Visible / hidden tests (repair loop)

Repair feedback may use **visible** tests only. Scoring always uses the full test
set. The split is frozen in `eval/heldout_visible.json`.

Rule: the first `ceil(n / 2)` tests in list order are visible; the rest are
hidden. A task with one test has no hidden split (hidden pass equals full pass).

- 78 tasks, 352 tests total.
- Visible: 199. Hidden: 153.
- Every task has at least one hidden test. The only n=2 task is
  `he152_compare_scores` (1 visible, 1 hidden). Per-task counts are in
  `eval/heldout_visible.json`.

The scaled val split uses the same rule at runtime. Val task ids stay those in
`eval/val_split.json`.
