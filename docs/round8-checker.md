# Round 8 equivalence checker (v3)

`e2m/harness/diff3.py` decides whether an assembly listing computes the same function as a task's reference C. The
oracle is gcc -O2 of the reference C. A candidate is accepted only if both hold:
- it passes every test of the task;
- it agrees with the oracle on 2,000 generated inputs (5,000 for final checks). Agreement covers the return value
  and every buffer. The candidate runs from an adversarial entry state (random flags and scratch registers).

How the inputs are built:
- the round-7 mutation families (`diff.py` v1 and v2);
- wide-int and digit-rich integers (31–62 bits);
- "late event" arrays: 257–700 elements whose first 256+ are tame, which catches byte-width index bugs;
- coverage-guided growth: any input that reaches a new basic-block edge of the oracle (Unicorn block hook) joins the
  corpus and gets mutated further;
- argument-dependence pairs: inputs that differ in one argument and change the oracle's output.

The acceptance set always includes one input per discovered edge and all the dependence pairs.

Inputs are dropped in two cases:
- the oracle faults or times out;
- gcc -O3/-Os or clang -O2/-Oz of the same reference C disagree with the oracle. That signals undefined behaviour
  (signed overflow, INT_MIN negation and the like), where no single answer is right.

Length arguments stay within their array.

Usage:

```python
from e2m.harness.diff3 import Checker3
from e2m.loop.round7_common import oracle_image
from e2m.loop.round8_common import consensus_images
c = Checker3(task, oracle_image(task), seed=0, consensus=consensus_images(task))
r = c.check(candidate_asm, n=2000, seed=1)   # r.ok, r.detail, r.counterexample, r.edge_coverage
```

## Test cases

`e2m/harness/diff3_cases.py` lists the 19 known-wrong round-7 "wins" (with their assembly, what caught them, and
v3's counterexample) and the 10 equivalent pairs. Run every checker on them with `python -m e2m.harness.diff3_cases`,
or pick one with `--only v1|v2|v3`. In round 8 stage A, v3 rejected 19/19 and accepted 10/10. v1, the round-7
default, misses the 16 cases that the round-7 audit found.

Held-out tasks are in `eval/heldout.jsonl`. The val tasks (`exebench_*`) need `tasks/scaled.jsonl`, which is not in
this repository; rebuild it as described in docs/DATA.md. The runner skips cases whose task is missing.

The held-back generator used for round 8's final audit (`e2m/harness/diff_holdout.py`) is published after round 8 ends.
