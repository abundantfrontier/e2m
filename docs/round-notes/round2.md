# Round notes: E2M round 2

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

(success, failure, or budget stop) — use try/finally / trap.

New budget: hard deadline 2026-11-18 18:00 America/Los_Angeles, or 60 additional GPU hours, whichever first.
Update e2m/loop/guard.py (or add a round-2 guard) so the deadline/budget apply; stop cleanly at a stage boundary.

pipeline (extend e2m.loop.pipeline or add e2m.loop.round2), smoke-test each stage on 1-2 tasks, then launch

Stages, in order (skip a stage only if it errors; log why and continue):
1. loop-lora2-val: repair loop (k=8, R=3, visible-test feedback) with lora2 on the 90-task val set.
2. dpo-train: DPO on the 46,472 preference pairs starting from lora2 (1.5B). Sensible defaults (beta ~0.1,
   1 epoch or a cap so it fits budget). Hold out a small pair slice for eval loss.
3. dpo-eval: greedy pass@1 / pass@5 on held-out (78) and val (90), same harness as lora1/lora2 evals.
4. loop-dpo-heldout: repair loop with the DPO model on held-out (78).
5. 7b-base-asm: rerun the interrupted 7B base-asm baseline on held-out (78).
6. qlora7b-v2: QLoRA 7B on the sft_lora2 mix (rejection-sampled data), then greedy eval held-out + val,
   then loop on held-out if budget remains.
Order 1-5 first; 6 only if budget remains after 5.
