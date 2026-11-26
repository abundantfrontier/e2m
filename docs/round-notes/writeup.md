# Round notes: write up repair loop + rejection-sampling round 1 (no GPU)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

The pipeline (python -m e2m.loop.pipeline) ended Nov 13 22:41 PT with exit 3 after loop-lora2-heldout,
at 46.45 GPU hours (over the ~40 h budget). loop-lora2-val did not run.

Tasks:
1. Read runs/loop-pipeline/ (stage.txt, exit, logs), runs/loop-*/, runs/lora2*/ summaries, and BRIEF-loop.md.
   Confirm why exit was 3 by reading the gate in e2m/loop/ and state it.
2. Collect exact numbers (with denominators) for: rewrite stats; loop-lora1-heldout, loop-lora1-val,
   loop-base-c-heldout, loop-lora2-heldout (pass_full, visible pass, pass-by-budget at 1/4/8/16/32 calls);
   loop-reject-train / loop-reject-rewritten pass rates; rejection-sampled kept count and sft_lora2 set size;
   lora2 training stats; lora2-eval-val and lora2-eval-heldout pass@1/pass@5 vs lora1 and qlora7b
   (Report 1: lora1 held-out 0.026/0.064, val 0.211/0.367; qlora7b held-out 0.064/0.103, val 0.367/0.500;
   NL->C->gcc -O2 held-out 0.397/0.628). Note failure categories if logged.
3. Write runs/loop-pipeline/RESULTS.json (machine-readable, all of the above) and append to STATUS.md a
   section "Repair loop + rejection-sampling round 1 (2026-11-12)" with tables and a short list of caveats
   (e.g. lora2-val loop not run, budget overrun). Keep prose short and factual; no paper edits.
4. If git is available, make a local commit; if not, note "git not installed, not committed" in STATUS.md.
5. Finally: echo "done $(date -Is)" > logs/loop.done
