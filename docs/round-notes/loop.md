# Round notes: test-and-retry loop, train on passing programs, held-out-style data, then next logical steps

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

## Context (from STATUS.md, 2026-11-11)
- 1.5B LoRA (runs/lora1): held-out pass@1 0.026 (2/78), pass@5 0.064; val pass@1 0.211.
- 7B QLoRA (runs/qlora7b): held-out pass@1 0.064 (5/78), pass@5 0.103; val pass@1 0.367.
- Base 1.5B NL->C->gcc -O2 control: held-out pass@1 0.397 (31/78), pass@5 0.628. This is the bar to beat.
- Dominant held-out failures: wrong_answer, then doesnt_assemble. Training specs come from ExeBench and read differently from HumanEval/MBPP.

## Stage 1: test-and-retry (repair) loop, no retraining
1. Build e2m/loop (or similar): for each task, sample k candidates (start k=8, T=0.8), assemble and run via the existing harness (Unicorn default, QEMU for AVX). For any failure, give the model the assembler error, crash/timeout info, or the first failing test (input, expected, got) and ask for a fix; up to R rounds (start R=3). Stop a task at its first pass.
2. Only the visible tests may be fed back. Score with the full test set. If the held-out tasks have no visible/hidden split, make one (e.g. first half visible) and record it in eval/HELDOUT.md; report both "with feedback on visible tests" and final hidden-test pass.
3. Run it on the 1.5B LoRA (runs/lora1) for held-out and val. Report pass@1 at fixed sample budgets (e.g. total model calls 1, 4, 8, 16, 32), plus size/insn ratio vs gcc -O2 on passes.
4. If cheap, also run the same loop on the base-1.5B C route (C -> gcc -O2 with compile/test feedback) so the comparison stays fair.

## Stage 2: held-out-style specs (local qwen3:14b via Ollama)
1. Do not run Ollama and training on the GPU at the same time.
2. Rewrite the specs of the training tasks (tasks/scaled.jsonl and the starter tasks) into HumanEval/MBPP-style prompts: a short docstring, a function signature in C terms, and 1-2 examples. Keep the original spec too. Target several thousand rewritten tasks.
3. Check for leakage: drop any rewritten task whose name or text closely matches a held-out task (n-gram or embedding similarity); log the counts.

## Stage 3: train on passing programs (rejection sampling / expert iteration)
1. Run the Stage-1 loop over several thousand training tasks (original plus rewritten specs) with the 1.5B LoRA. Keep only programs that pass the full tests. Also keep the existing verified gcc-derived SFT rows.
2. Retrain the 1.5B (same LoRA config as runs/lora1 unless there's a clear reason) on that mix. Name it runs/lora2.
3. Evaluate lora2 on held-out and val: greedy pass@1, pass@5, and with the Stage-1 loop. Compare in one table with lora1, qlora7b, base asm, and the NL->C->gcc -O2 control.
4. If lora2 clearly helps, a second iteration (sample with lora2, retrain) is worthwhile if time allows.

## Stage 4 (only if time remains): next logical steps, in order
1. Preference training (DPO or similar) for size/speed on data/scaled/pairs.jsonl (46,472 pairs) on top of the best 1.5B, gated on correctness not dropping. Report size/insn ratios vs gcc -O2.
2. Apply the best recipe (rejection-sampled data plus loop) to the 7B QLoRA, and rerun the 7B base-asm baseline that was interrupted last time.
3. Anything else clearly logical from the results; explain the choice in STATUS.md.

## Done means
- A new STATUS.md section "Repair loop + rejection-sampling round 1 (2026-11-12)" with: what was built, data counts (rewritten specs, leakage drops, passing programs kept), training config and loss, a results table (model x eval set x pass@1/pass@5/loop pass at budgets/size/insn ratio) including all previous rows and the C control, failure breakdowns, 5 example outputs, and the recommended next step.
- Tests pass, local commit (no push).
- Write DONE to logs/loop.done. If you stop early (budget, error), write STOPPED: <reason> to the same file.
