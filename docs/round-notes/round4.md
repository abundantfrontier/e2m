# E2M round 4 brief

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Budget: 30 GPU hours total for round 4, no calendar deadline. Each stage is capped at 10 h. Log GPU hours (runs/round4/gpu_hours.jsonl and the summary). Write runs/round4/RESULTS.json, runs/round4/SUMMARY.md, and finally logs/round4.done (also write the done file on budget stop, with the reason). Stage/resume state should live under runs/round4/ so a restart can continue.

## Why this round

Round 3's C-then-asm 7B writes strong C (held-out 48/78 when that C is compiled with gcc -O2) but its own assembly half is only 7/78 greedy / 9/78 pass@5. The published C control is the untrained 1.5B (31/78 greedy, 49/78 pass@5, loop 44/78), so model size is confounded with fine-tuning. Round 4 unconfounds that, then measures a gcc-lowered hybrid against the model's direct assembly, finishes the incomplete loop, and categorizes the C-vs-asm failures.

Compare everything against c-then-asm (C half and asm half), qlora7b-v2, lora2, and the 1.5B C control, same denominators (val 90, held-out 78). Do not invent or re-score old numbers; copy them from runs/round3 and earlier summaries.

## Part A (priority 1): matched 7B C control

QLoRA-fine-tune the same 7B base on the same training tasks, but C output only.

- Base: `runs/models/Qwen2.5-Coder-7B-Instruct`
- Training tasks: the sft_lora2 mix (`data/scaled/sft_lora2.jsonl`, 6,781 tasks). C sources are already in `data/scaled/sft_c_then_asm.jsonl` (6,781 rows with a `c` field) and can be joined the same way `e2m.train.c_then_asm` does.
- Response: C only, using the existing `c_messages` prompt in `e2m/train/format.py`. `e2m/train/sft_lora.py` today only accepts `--target {asm,c-then-asm}`; add a `c` target (or an equivalent small trainer change) rather than changing the C-then-asm recipe.
- Hyperparameters, match c-then-asm where possible (`runs/c-then-asm/train_config.json`): r=32, alpha=64, dropout 0.05, lr 2e-4, 3 epochs, batch 1, grad_accum 16, load_4bit, max_len as needed (c-then-asm used 2048; C-only sequences should be shorter, so you may drop fewer over-length rows — record the counts). Write the adapter and train_config under `runs/c-only-7b/` (or a clearly named sibling).
- Evaluate with the existing harness: `python -m e2m.train.eval_generate` already has `--mode c`, which compiles the model's C with gcc -O2 and scores it in Unicorn/QEMU. Run val (`tasks/scaled.jsonl` + `eval/val_split.json`, 90 tasks) and held-out (`eval/heldout.jsonl`, 78 tasks), greedy pass@1 and pass@5. Record size and instruction-count ratios vs the gcc -O2 reference on the passing tasks.

## Part B (priority 2): hybrid (model C, gcc lowers it)

Mostly CPU. For both the existing C-then-asm 7B (`runs/c-then-asm/adapter`) and the new 7B C model from A:

- Take each model's C (reuse already-generated C from `runs/c-then-asm/eval.jsonl` where it exists; generate C for the new model from A, or from c-then-asm if a field is missing).
- Compile that C to x86-64 with gcc -O2 and with gcc -Os.
- Score the resulting images in the same Unicorn/QEMU harness used for asm outputs (`assemble_reference` / `run_gcc_baseline` in `e2m/harness/tasks.py` already take `opt=`; `_score_c` in `eval_generate.py` currently hardcodes `-O2`, so extend it or add a small scorer rather than going around the harness).
- Report pass@1, pass@5, and instruction-count/size ratios against the reference, side by side with the model's direct assembly (c-then-asm assembly half only; the C-only model has no direct asm — say so). Include val and held-out.
- This is the test of "the model writes C, gcc turns it into machine code" vs "the model writes assembly itself."

## Part C (priority 3): finish the C-then-asm held-out repair loop

The round-3 loop (`runs/loop-c-then-asm-heldout/`) stopped at the 10 h cap after 59 of 78 tasks (15/59 full, 21/59 visible). Resume it; do not redo the 59 finished tasks. Combine the cached 59 with the remaining 19 so the summary has a 78/78 number.

Same settings: k=8, R=3, temperature 0.8, mode c-then-asm, adapter `runs/c-then-asm/adapter`, 4bit, max-new 2048, stage cap 10 h. The existing `python -m e2m.loop run` path should skip task_ids already in `loop.jsonl`.

The 19 remaining held-out task ids are:

he102_choose_num, he104_unique_digits, he106_f_factorial_or_sum, he114_min_subarray_sum, he120_k_largest, he121_sum_odd_even_idx, he122_add_elements, he131_odd_digit_product, he138_sum_of_four_evens, he139_special_factorial, he142_indexed_sum_squares, he146_special_filter, he150_x_or_y, he151_double_the_difference, he152_compare_scores, he155_even_odd_count, he157_right_angle_triangle, he159_eat, he163_generate_integers

If the stage cap hits again, still write a combined 59+n / 78 figure and list what is unfinished.

## Part D (priority 4): failure analysis of the 42 C-pass / asm-fail held-out tasks

From the c-then-asm held-out eval: 42 tasks pass in the C half and fail in the assembly half (plus 6 both, 1 asm-only; agreement 35/78). Classify those 42 (doesn't assemble, wrong ABI / calling convention, logic mismatch versus the model's own C, crash, timeout, and any other category the outputs support). Include a handful of short examples (truncated C vs asm, and why it failed). This can use already-written outputs in `runs/c-then-asm/eval.jsonl`; it should not need much GPU. Write the analysis under `runs/round4/` (e.g. `failure_analysis.md` plus a JSON tally) and summarize it in SUMMARY.md.
