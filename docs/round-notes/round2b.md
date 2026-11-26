# Round notes: E2M round 2b — finish the 7B stages

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Round 2 (e2m.loop.round2) exited 0 on Nov 15 10:39 PT with 23.35 GPU h used. loop-dpo-heldout hit its
10 h stage cap (DPO model collapsed: 0/62, all doesnt_assemble), which halted the pipeline, so stages 5-6 were skipped.

Budget: remaining 36.65 GPU h and hard deadline 2026-11-18 18:00 America/Los_Angeles; stop at stage boundaries.

(e.g. --only / --resume that skips stages already recorded ok, and does NOT rerun or continue loop-dpo-heldout),

Stages:
1. 7b-base-asm: full 78-task held-out baseline for base Qwen2.5-Coder-7B (reuse the 28 seeded rows if valid).
2. qlora7b-v2-train on the sft_lora2 mix, then qlora7b-v2-eval-val and -eval-heldout (greedy pass@1/pass@5).
3. loop-qlora7b-v2-heldout if budget remains (keep a per-stage cap so it can't starve the deadline).
A stage failure must not stop later independent stages.

At the end write runs/round2b/RESULTS.json, runs/round2b/SUMMARY.md (numbers with denominators, compared
