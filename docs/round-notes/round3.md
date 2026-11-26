# E2M round 3 brief

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Budget: 30 GPU hours total for round 3, no calendar deadline. Per-stage caps as in round2 (loops 10 h max). Write runs/round3/RESULTS.json, runs/round3/SUMMARY.md, and finally logs/round3.done (also write it on budget stop, with the reason).

## Part A: DPO repair (decide go/no-go on preference training)
Round 2 DPO from lora2 (beta 0.1, lr 5e-5, 1 epoch, 5,574 steps, all goals mixed but prompts said Goal: default) collapsed: 83/90 val outputs don't assemble.
A1. Build a goal-consistent pair set from data/scaled/pairs.jsonl: either put each pair's real goal (default/size/speed) in the prompt, or keep only default-goal pairs. Prefer the real goal in the prompt if the prompt format already supports goals; record which.
A2. Train DPO from lora2 with an SFT/NLL term on the chosen response (RPO-style, weight ~1.0), beta 0.1-0.3, lr ~5e-6, cap at ~1,500 steps or 0.5 epoch.
A3. Collapse guard: every ~250 steps, greedy-generate a fixed 30-task val subset and measure the fraction that assembles; if it drops below 80% of lora2's rate on the same subset, stop early and keep the last good checkpoint. Log the curve.
A4. Eval the kept checkpoint greedy pass@1/pass@5 on val (90) and held-out (78) with the standard harness and failure categories. Run a repair loop (k=8, R=3) on held-out only if val pass@1 >= lora2's 38/90.

## Part B: C-then-assembly route (does going through C close the held-out gap?)
The C control (NL -> C -> gcc -O2) gets 31/78 greedy and 44/78 looped on held-out; the asm models get 5-12/78.
B1. Find the C sources behind the SFT data (the sweep compiled C to produce asm). Build a training set where the response is the C program followed by the assembly (one response, C first, clearly delimited). Use the sft_lora2 task mix.
B2. Train QLoRA on Qwen2.5-Coder-7B-Instruct with the qlora7b-v2 settings (r=32, alpha=64, lr 2e-4, up to 3 epochs; reduce epochs if time is long; raise max_len as needed).
B3. Eval greedy pass@1/pass@5 on val and held-out, scoring only the assembly part. Also report: fraction where the model's own C passes the tests (compile with gcc, run harness), and agreement between the C and asm halves.
B4. If budget remains, repair loop (k=8, R=3) on held-out.

Order: A then B, unless you judge B is more valuable to finish first; record the choice. Compare everything against lora2, qlora7b-v2 and the C control in SUMMARY.md, same denominators. Do not rerun older stages.
