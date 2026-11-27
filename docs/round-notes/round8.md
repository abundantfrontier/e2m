# Round 8 brief: a stronger checker, then beat the compiler again

> Cleaned planning note for this round. Machine-specific operating rules, launch commands and
> GPU-scheduling details were removed. Results are in `results/`.

Approved by Boyd on Thu Nov 26, 2026. No training. GPU is used only by stage D, capped at 1.0 GPU hour.

## Why
The post-run audit of round 7 removed 16 of 61 claimed wins, and the 5,000-input recheck had already removed 3.
That makes 19 known-wrong wins in total. Round 8 first builds a checker that rejects all 19. It then re-checks
the round-7 bars and wins with it, and searches again under that checker.

## Stages (e2m/loop/round8_run.sh)
- **A. Checker v3 gate** (CPU). The checker is e2m/harness/diff3.py. On top of the v2 families it adds:
  - coverage-guided input generation, using basic-block edges of the oracle
  - long "late event" arrays
  - wide-int and digit-rich stress
  - argument-dependence pairs
  - an output-diversity guard
  - a UB filter: inputs on which gcc -O3/-Os or clang -O2/-Oz of the reference C disagree with the gcc -O2 oracle are dropped

  Acceptance is all tests plus 2,000 inputs. The selection covers 100% of the edges the input pool discovered.
  The gate passes only if all 19 known-wrong wins are rejected and all 10 equivalent pairs are accepted.
  If the gate fails, the run stops after B.
- **B. Re-validate round 7** (CPU). Every round-7 bar and claimed win goes through v3. A failing bar is replaced
  by the next-best compile that v3 accepts. The results go to erratum_round7.md and erratum_round7.json.
- **C. Search** (CPU, 6 h wall cap, 12 workers).
  - Starts: all five compiler listings of the bar's C source, the bar itself, and the round-7 wins that survived B.
  - New moves: push/pop constants, lea folding, register/constant reuse, one-operand mul, dead-store and
    dead-flag elimination.
  - Restarts and annealing.
  - Improvements need v3 at 2,000 inputs; the final listing needs 5,000.
  - ISA-extension jobs (-mpopcnt -mlzcnt -mbmi, plus tzcnt rewrites) are reported separately and never count as baseline wins.
- **D. Model** (GPU, at most 1.0 h). Runs only if search leaves at least 15 held-out near-misses.
  - Model: the untrained Qwen2.5-Coder-7B-Instruct in 4-bit, k=8, one refinement round.
  - Prompts include trick hints and the rejected candidate with its counterexample.
  - Stops after 0.5 h if it has no unique win.
- **E. Audit and summary** (CPU). Every claimed win is re-checked:
  - on 5,000 fresh v3 inputs
  - on 5,000 inputs from the held-back generator e2m/harness/diff_holdout.py, written before any round-8 candidate existed
  - with llvm-mca on znver3 and skylake

  Outputs: RESULTS.json, SUMMARY.md, failure_analysis.md, audit.json, then logs/round8.done.

## Decision
- **Win:** at least 30 held-out verified wins, no held-out claim failing the final checks, and at least 5 held-out
  speed wins that llvm-mca also rates better on both CPUs.
- **No win:** fewer than 10.
- **Partial:** anything else.

The model is worth keeping if it adds at least 3 unique wins per GPU hour.

## Pause
