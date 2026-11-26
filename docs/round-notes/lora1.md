# Round notes: QEMU re-check of scaled set + first E2M LoRA run (1.5B)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Prereq: read STATUS.md (all sections), docs/concept-note.md, BRIEF*.md, data/scaled/SWEEP_REPORT.md, and the harness code.

## Part A: fix + QEMU re-check (CPU)
1. Fix the harness buffer issue that made 330 AVX variants crash on QEMU: short arrays must sit in buffers with enough readable padding (e.g. 64 bytes past the end, or round allocations up) while out-of-bounds WRITES are still detected. Keep all existing tests green; add a test.
2. Re-verify every scaled-sweep config that failed with unsupported_insn (~34.5k) on the QEMU backend, in parallel (16 cores), resumable. Merge the new passing variants into data/scaled/ (re-dedupe, re-rank, regenerate sft.jsonl and pairs.jsonl). Report how many recovered.

## Part B: held-out eval set
3. Build eval/heldout.jsonl: at least 60 problems from HumanEval and MBPP (e.g. via MultiPL-E prompts/tests) that fit the harness (int/pointer/fixed-size args, no libc). Each gets an English spec, SysV signature, C reference, tests, and verification through the harness at gcc -O0 and -O2. Confirm none appear in tasks/scaled.jsonl (name + normalized-body near-dup check). This set is NEVER trained on.

## Part C: first LoRA
4. Base model: Qwen2.5-Coder-1.5B-Instruct (download from HF). If unavailable, use the cached Qwen3-1.7B. Train LoRA (bf16, PEFT/TRL or plain HF Trainer) on data/scaled/sft.jsonl goal=default: input = English spec + signature + goal, output = AT&T or Intel asm text (pick one, be consistent, document it) for the function body. Hold out 3% of scaled tasks as a val split by task. Reasonable defaults: r=16-64, lr ~2e-4, 2-3 epochs, seq len to fit the longest ~99th percentile. Log loss curves to runs/lora1/.
5. Assembly pipeline for model output: asm text -> assemble (GNU as / llvm-mc already available) -> extract bytes -> harness. Anything that doesn't assemble counts as a failure.
6. Evaluate pass@1 (greedy) and pass@5 (sampling, T=0.8) on: the scaled val split, and eval/heldout.jsonl. Also report bytes and dynamic instruction count against gcc -O2 for passing solutions, plus failure categories (doesn't assemble, crash, wrong answer, timeout, ABI violation).
7. Baselines on eval/heldout.jsonl, same metrics:
   - base model without LoRA (same prompt)
   - NL -> C -> gcc -O2 control: the same base model writes C from the spec, compiled with gcc -O2, run through the harness
8. If time and VRAM allow after the 1.5B results, optionally run QLoRA 7B (Qwen2.5-Coder-7B-Instruct, 4-bit) with the same eval. Budget: stop the whole session at ~10 hours of GPU time.

## Done means
New STATUS.md section: recovered AVX count, eval set details, training config, loss, and a results table (model x eval set x pass@1/pass@5/size/insn ratio) with the baselines, failure breakdown, 5 example outputs (good and bad), and the recommended next step. Tests pass, local commit. Then write DONE to logs/lora1.done.
