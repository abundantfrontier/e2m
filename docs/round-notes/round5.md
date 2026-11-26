# E2M round 5 brief: distill the hybrid into direct English-to-asm (Nov 25 2026)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

## Milestone
Round 4 held-out (78): 7B C-only 46/78 pass@1 (56 pass@5); hybrid (model C -> gcc -O2) 48/78;
model's direct asm 7/78; C-then-asm loop 19/78. Of 42 C-pass/asm-fail tasks: logic_mismatch 23,
crash 9, timeout 9, doesnt_assemble 1, abi 0. The model's asm is well-formed but doesn't implement
its own C. Note also the distribution shift: asm models score ~40-44/90 on val (exebench-style)
but 5-7/78 on held-out (HumanEval/MBPP-style), and the existing train set (data/scaled/sft_lora2.jsonl)
ALREADY is English -> gcc asm of the reference C. So the new data's value must come from
(a) asm of C the model itself writes (closer to what it can express), and (b) more
HumanEval/MBPP-style English task prompts. Weight effort accordingly.

Round 5 goal: distill the hybrid into direct English->asm. Success target: >= 20/78 held-out
direct asm pass@1 (no compiler at inference).

### 1. Data (execution-verified (English, asm) pairs)
- Teacher: runs/c-only-7b/adapter on Qwen2.5-Coder-7B-Instruct, --4bit, C mode (round 4 SUMMARY's
  best C model; c-then-asm's C half is equivalent at 48 vs 46 — use c-only-7b unless SUMMARY says otherwise).
- Prompt pools, in priority order:
  a. MBPP tasks in datasets/heldout/mbpp.jsonl that are NOT in eval/heldout.jsonl and survive the
     strict leakage filter (step 2). Convert to the task format (spec, signature, tests) using the
     same converter/rules used to build eval/heldout.jsonl (leaf SysV signature, tests from the
     MBPP asserts; tests must pass the reference via gcc -O0/-O2 in the harness). NO HumanEval
     tasks at all (held-out has 55 of them; exclude the whole benchmark).
  b. Synthetic HumanEval/MBPP-style English tasks cheaply generated with local ollama qwen3:14b
     — short integer/array/string leaf functions, with reference C
     and tests verified in the harness. Cap generation time (<= 1 GPU h) and count.
  c. The TRAIN task pool (tasks/scaled.jsonl minus eval/val_split.json ids; sft_lora2 tasks) and
     data/scaled/rewritten.jsonl specs. Sample a capped subset if sampling time is tight.
- Sample k=4 C completions per task (temperature ~0.8, plus greedy). Keep only C that passes the
  task's full tests in the harness. Compile kept C with gcc -O2 (and -Os as a second variant,
  tagged goal=size), emit asm in the existing target format/syntax (same cleaning as the
  data/scaled/sft.jsonl asm: AT&T, .text/.rodata only, no externals). Re-verify the asm itself
  in the harness (Unicorn, QEMU for AVX). Dedupe identical asm per task (keep up to 2 distinct).
- Always include the existing verified sft_lora2 pairs as the base mix (so we don't regress on val);
  report the mix proportions.

### 2. Strict de-dup / leakage check
No held-out (eval/heldout.jsonl, 78) or val (eval/val_split.json, 90) task — or near-duplicate
prompt — may enter training. Checks, per training row vs every held-out and val task:
task id, function name, normalized signature+name, exact normalized C body, difflib ratio >= 0.85
on spec text and on reference C, word 5-gram overlap, and bge-m3 embedding cosine >= 0.85
(the round-3 rewrite used 0.85ish; be at least that strict). Drop on any hit.
Write runs/round5/leakage.json with counts per check and per source, and the dropped ids.

### 3. Fine-tune
QLoRA on the 7B (runs/models/Qwen2.5-Coder-7B-Instruct), English->asm using the existing asm
prompt format in e2m/train/format.py (same as qlora7b-v2), hyperparameters as round 4
(r=32, alpha=64, dropout 0.05, lr 2e-4, batch 1, grad_accum 16, load_4bit, max_len 2048).
Pick epochs to fit the budget (≈2.5 GPU h for training). Optionally add a C->asm auxiliary
format if you judge it helps; record the decision. Save checkpoints often.
Output: runs/asm-distill-7b/{adapter,train_config.json}.

### 4. Eval (held-out 78, val 90)
- direct asm greedy pass@1 and pass@5 (eval_generate --mode asm);
- repair loop on direct asm (e2m.loop run, k=8, R=3) on held-out; on val only if budget remains;
  cap the loop at ~2 GPU h — if it can't finish, report partial coverage (as round 4 did);
- failure-category breakdown as round 4 (e2m.train.failure_analysis buckets: doesnt_assemble,
  abi, logic_mismatch, crash, timeout, other), plus the overlap with the 48 hybrid-pass tasks;
- size and insn ratios vs reference gcc -O2 on passing tasks.
Compare against round 4: 7/78 direct, 19/78 loop, 48/78 hybrid (copy old numbers; don't re-score).

### 5. Budget and outputs
Total GPU budget ≈ 8 GPU h hard cap (suggested: sampling+synthetic ≤ 3 h, training ≤ 2.5 h,
eval ≤ 0.7 h, loop ≤ 1.8 h). Per-stage caps; log runs/round5/gpu_hours.jsonl. Count only time
leakage.json, data_stats.json} and finally logs/round5.done (also on budget stop or fatal error,
with the reason). State under runs/round5/ so a restart resumes.

## After the round (public repo)
A separate CPU-only watcher runs `scripts/publish_public.sh` once `logs/round5.done` appears; it syncs
results/reports to https://github.com/abundantfrontier/e2m (scrubbed). This pipeline itself must not push.
Future rounds: call `bash scripts/publish_public.sh` at the end, after writing `logs/roundN.done`
(or rely on the watcher, which picks up any newer `logs/round*.done`).
