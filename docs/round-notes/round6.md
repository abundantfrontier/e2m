# E2M round 6 brief: the C path's ceiling (approved Thu 26 Nov 2026, 7:48 AM PT)

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Source proposal: Report 6 §10 (copy at ~/e2m-proposals/NEXT-JOB-round6-proposal.md).
No training. Existing CLIs only. Pipeline: `e2m/loop/round6_run.sh` (bash) + `e2m/loop/round6_summary.py` (CPU).

## Why
Seven direct-asm recipes (rounds 1-5b, incl. two distill mixes) sit at 4-7/78 held-out. The practical
system is "model writes C, gcc -O2 lowers it" (46-48/78 greedy). Its repair-loop ceiling with the
fine-tuned 7B has never been measured. Validation (ExeBench, mostly <=40-byte functions) is not a
useful selection signal (Report 6 §6).

## Rules
  a bash `trap ... EXIT` releases it on every exit (and on INT/TERM after stopping the child).
  If a live non-e2m holder is present, the pipeline waits; a dead holder is treated as stale.
  loops and evals skip tasks already scored).
- Budget: 5.0 GPU hours hard cap, accounted in `runs/round6/gpu_hours.jsonl`. Each stage's
  `--max-hours` is min(stage cap, 5.0 - used). A stage with < 0.05 h left is skipped.
- Held-out loop order: MBPP and HumanEval interleaved (`runs/round6/heldout_interleaved.jsonl`,
  order in `heldout_order.json`) so a cap-stopped partial is not MBPP-only.
- Visible-test feedback only in loops; scoring on the full tests.

## Stages (order; caps)
1. `loop-c-only-7b-heldout` (2.5 h): `python -m e2m.loop run --adapter runs/c-only-7b/adapter
   --name c-only-7b --mode c --split heldout --tasks runs/round6/heldout_interleaved.jsonl --k 8
   --rounds 3 --4bit --out runs/loop-c-only-7b-heldout`. Headline.
2. `base7b-c-eval-heldout` (0.35 h) and 3. `base7b-c-eval-val` (0.40 h): untrained
   Qwen2.5-Coder-7B-Instruct (4-bit, no adapter) in C mode -> `runs/base7b-c/eval.jsonl`.
   Closes the size vs fine-tune confound (Report 5 §8).
4. `loop-c-only-7b-val` (1.25 h): C loop on val, as budget allows.
5. CPU summary (`python -m e2m.loop.round6_summary`): size-bucketed re-score (reference gcc -O2
   <= 40 bytes vs larger) of c-only-7b, asm-distill-7b(-v2), c-then-asm and base7b-c; gcc -Os on
   the reference C and the hybrid -Os size vs reference -Os; failure analysis of the C loop.

Deferred (CPU, not in this pipeline): a frozen "hard val" split (~60 non-ExeBench / >40-byte tasks,
de-duplicated against held-out and train) under eval/.

## Outputs
- `runs/round6/{SUMMARY.md, RESULTS.json, failure_analysis.md, os_reference.json, gpu_hours.jsonl}`
- `logs/round6.done` (written last by the summary step). The public auto-sync watcher publishes on it.
- RESULTS.json uses the "greedy"/"loops" layout the public sync reads.
- Logs: `logs/round6-pipeline.log`, per stage `runs/round6/stages/<stage>/log`.

## Decision rule
- C loop held-out >= 55/78: the C path is the system; next round superoptimizes the hybrid's gcc
  output (floor = hybrid pass rate; accept only test-passing, differential-equivalent variants).
- < 50/78: C quality is the bottleneck; next round works on the C side (algorithms, bigger/better
  C model), still lowered by gcc.
