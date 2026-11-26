# E2M round 5b brief: finish round 5 eval, then distill-heavy retrain (Nov 25 2026)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

## What round 5 already did
Teacher: runs/c-only-7b. Adapter: runs/asm-distill-7b (trained 6619 s, 2 epochs, same
hyperparameters as below). GPU hours 4.1684 / 8.0. Halt: gpu-hour budget.

- eval-heldout stopped at 52/78 on a 0.7 h max-hours cap (2534.21 s). pass@1 4/52, pass@5
  ~0.135. Greedy buckets: wrong_answer 35, timeout 5, crash 4, doesnt_assemble 4, pass 4.
  Results are in runs/asm-distill-7b/eval.jsonl (model=asm-distill-7b, mode=asm, split=heldout).
  The 26 unfinished held-out ids are: he82_prime_length, he83_starts_one_ends,
  he85_sum_even_odd_idx, he89_encrypt, he92_any_int, he97_multiply_units, he100_make_a_pile,
  he102_choose_num, he104_unique_digits, he106_f_factorial_or_sum, he114_min_subarray_sum,
  he120_k_largest, he121_sum_odd_even_idx, he122_add_elements, he131_odd_digit_product,
  he138_sum_of_four_evens, he139_special_factorial, he142_indexed_sum_squares,
  he146_special_filter, he150_x_or_y, he151_double_the_difference, he152_compare_scores,
  he155_even_odd_count, he157_right_angle_triangle, he159_eat, he163_generate_integers.
- eval-val and loop-heldout / loop-val were skipped for budget (n=0).
- Data mix 5886 rows: sft_lora2 5097 (87%), train 544, mbpp 207, synth 38. Distill was diluted.
- sample-c used k=4 on 550 tasks (346 train + 182 mbpp + 22 synth); 481 tasks had at least one
  passing C; distill_pairs currently keeps at most 2 distinct asms per task.
- synth: cap 80, kept 22, then exit 1 after `ollama: HTTP Error 500: Internal Server Error`
  and `synth ollama unreachable` (see runs/round5/stages/synth/log). The abort is a logic bug:
  cmd_synth treats "ollama:" in the exception AND fail-streak >= 3 as unreachable, but the
  streak counts harness rejects too. The 500 was reject 135 with streak=4 after unrelated
  rejects (wrong -O0 returns, TypeError int() on a dict, 'int' object is not iterable). A
  single 500 should not abort the stage. As of this brief, `ollama serve` is active and
  /api/tags lists qwen3:14b, gemma3:12b, bge-m3, and qwen3:8b. Diagnose first; it is likely
  fixable (retry 500s, count only consecutive ollama errors, resume from the 22 kept tasks).
  If it cannot be fixed safely, skip synth and write the reason into SUMMARY.md.

Measured eval rate: 2534.21 s / 52 tasks = 48.73 s/task. Use this, plus a margin, to size
Stage A (and Stage B) eval max-hours. Do not reuse the 0.7 h eval cap — that is why held-out
stopped early. The repair loop is more expensive than direct eval: round 4's C-then-asm
held-out loop (k=8, R=3) took 18602 s (~5.17 h) for 78 tasks. Do not size the loop from the
48.73 s/task eval rate.

## Milestone
Stage A produces a complete number for the round 5 model (all 78 held-out + 90 val, plus the
held-out repair loop or a clearly marked partial). Stage B retrains with distill rows in the
majority and evaluates the new adapter. Stage B success target is still >= 20/78 held-out
direct asm pass@1 (no compiler at inference), compared against the completed Stage A number
and against round 4 (7/78 direct, 19/78 C-then-asm loop, 48/78 hybrid; copy those, don't
re-score).

### 1. Stage A — finish scoring runs/asm-distill-7b
Do this first, after the gate opens, before any Stage B GPU work.
- Direct asm (eval_generate --mode asm --name asm-distill-7b --4bit) on all 78 held-out tasks
  and the 90 val tasks. Resume/skip the 52 already in runs/asm-distill-7b/eval.jsonl:
  eval_generate._done_ids already skips rows matching model/mode/split in the --out file, and
  appends. Point --out at that same path. If a flag or version of eval_generate would rewrite
  the file, copy it and resume from the copy rather than losing the 52. Do not rerun the 52
  unless resume is genuinely broken; if you must rerun, say so in the summary.
- Then the C-then-asm repair loop (e2m.loop run, --mode c-then-asm, k=8, R=3) on held-out.
  Write it under a new path such as runs/loop-asm-distill-heldout-c-then-asm (do not clobber
  older loop runs). Resume if the loop runner supports it. (Round 5's skipped stage was a
  direct-asm loop; if budget remains after the C-then-asm loop, also run the direct-asm loop
  k=8 R=3 that was skipped, under a distinct path.)
- Size max-hours from the measured 48.73 s/task with margin for eval (26 remaining held-out
  ≈ 0.35 h raw → ~0.55 h; 90 val ≈ 1.22 h raw → ~1.6 h). Size the loop from the round 4
  loop time (~5.17 h for 78) with margin, then shrink the cap so Stage B still fits in the
  8 h total (see Budget). Partial loop coverage is acceptable if the cap is hit; report n
  scored, as round 4/5 do. Val loop only if budget remains after Stage B's required work;
  Stage A should not spend the whole budget on a val loop.
- Write runs/round5/RESULTS_full.json and runs/round5/SUMMARY_full.md (leave the original
  partial RESULTS.json / SUMMARY.md in place). Include pass@1 / pass@5, size and insn ratios
  vs gcc -O2 on passing tasks, failure-category breakdown, and overlap with the 48 hybrid-pass
  tasks. Then continue to Stage B in the same pipeline (--resume must be able to skip a
  finished Stage A).

### 2. Stage B — distill-heavy retrain as asm-distill-7b-v2
- More teacher C: sample k=5 to 8 (temperature ~0.8 plus greedy) from runs/c-only-7b on the
  train + MBPP task pools (reuse runs/round5 screened/sample task lists; you may enlarge the
  train-pool cap if time allows). Resume existing runs/round5/sample_c.jsonl rather than
  discarding the k=4 samples; add completions up to the new k. Keep every distinct passing C
  (lift distill_pairs' "keep up to two distinct asms" cap). Lower with gcc -O2 and -Os, same
  cleaning and asm re-verification as round 5.
- Fix or diagnose the synth/ollama failure as described above. Check that ollama is running
  and qwen3:14b is present at gate time (it was, when this brief was written). Resume the 22
  kept tasks in runs/round5/synth_tasks.jsonl. If it cannot be fixed safely, skip synth, keep
- Mix: cap old sft_lora2 at about 25% of rows (add a flag to write_mix / the mixer). Distill
  rows (train + mbpp + synth, gcc of teacher C) must be the majority. Report the mix.
- Strict leakage de-dup (text + embed) against held-out (78) and val (90), same checks and
  thresholds as round 5 (id, name, signature, C body, difflib >= 0.85, 5-gram, bge-m3 cosine
  >= 0.85). Prefer the CPU embed path. Write leakage reports under runs/round5b/.
- Train QLoRA of the 7B (runs/models/Qwen2.5-Coder-7B-Instruct), English->asm, same
  hyperparameters as round 5 unless the new data size calls for a different epoch count:
  r=32, alpha=64, dropout 0.05, lr 2e-4, batch 1, grad_accum 16, load_4bit, max_len 2048,
  goals default,size, save-steps 100, hold-file round5b.hold. Output:
  runs/asm-distill-7b-v2/{adapter,train_config.json}. Do not overwrite runs/asm-distill-7b.
- Evaluate the new adapter: held-out + val direct asm, and the C-then-asm repair loop (k=8,
  R=3) on held-out, with correctly sized budgets (48.73 s/task plus margin for eval; loop
  sized from round 4 / Stage A measurements). Failure-category breakdown and size/insn
  ratios as in Stage A. Compare to the completed Stage A number and to round 4.

### 3. Budget and outputs
Total GPU budget is 8 h or less, hard cap, with Stage A first. Count only time holding the
lock. Log runs/round5b/gpu_hours.jsonl. Suggested split when 8 h is tight (adjust from
measured times, but keep this priority order):
1. Stage A held-out remainder + val direct asm (non-negotiable, ~2.15 h with margin).
2. Stage B teacher sampling + mix + train (non-negotiable once A eval is done).
3. Stage B held-out direct asm eval of v2 (non-negotiable once trained).
4. Stage A held-out C-then-asm loop, then Stage B val eval, then Stage B loop, then any
   leftover loop/val work. Shrink 4 so 1-3 still finish. Partial loops are OK.

Per-stage caps must be computed (do not hard-code 0.7 h for eval). State under runs/round5b/
so a restart resumes. Write runs/round5b/{SUMMARY.md, RESULTS.json, failure_analysis.md,
leakage.json, data_stats.json} and finally logs/round5b.done (also on budget stop or fatal
error, with the reason). Stage A full results also go to runs/round5/RESULTS_full.json and
SUMMARY_full.md as above.

### 4. Smoke test (before the gate, no GPU)
`--smoke` and/or a tiny CPU-only subset: parse flags, dry-run the stage plan, exercise
resume-of-eval on a copied eval.jsonl if cheap, and compile/mix path on a couple of existing
or ollama on the GPU. If the smoke would need a GPU, skip that part and say so.

## After the round (public repo)
A separate CPU-only watcher runs `scripts/publish_public.sh` once `logs/round5b.done` appears
(it already keys on logs/round*.done) and syncs a scrubbed copy to
https://github.com/abundantfrontier/e2m. This pipeline itself must not push.
