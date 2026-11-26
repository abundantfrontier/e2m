# E2M: English to machine code

E2M is an open research project from the [Abundant Frontier Institute](https://abundantfrontier.org/)
toward a model that takes an English description of a function and produces
machine code (x86-64 assembly that a plain assembler turns into bytes), with
no human-visible source language in the loop.

Every result here comes from one 24 GB RTX 3090 workstation. The write-ups are
living papers on the AFI Lab (links below); this repository holds the code,
configs, small data and the results behind them.

## Approach

1. **A safe harness.** Model output is assembled with GNU `as`/`ld` and run
   inside the [Unicorn](https://www.unicorn-engine.org/) CPU emulator under the
   SysV AMD64 ABI, never on the host CPU. AVX2/FMA code runs on an opt-in
   QEMU user-mode backend. Crashes, hangs, ABI violations (callee-saved
   registers, stack alignment) and syscalls become structured failures.
   Scoring reports pass@k plus `.text` size and dynamic instruction counts
   against gcc `-O2`.
2. **Tasks.** 20 hand-written starter tasks; a scaled pool built from ExeBench
   (and a small AnghaBench sample) with English specs written by a local
   Qwen3 14B; a frozen 78-task held-out set rewritten from HumanEval (55) and
   MBPP (23) that never enters training; and a 90-task validation split.
3. **Data.** A compiler sweep (gcc/clang flag grids) turns every task's C
   reference into verified assembly variants for SFT and preference pairs.
4. **Models.** LoRA / QLoRA fine-tunes of Qwen2.5-Coder 1.5B and 7B Instruct
   that write assembly directly, write C, or write C and then assembly; DPO
   variants; and a test-and-retry repair loop (k = 8 samples, R = 3 rounds,
   feedback from visible tests only).
5. **Controls.** Every assembly result is compared with the obvious baseline:
   the model writes C and gcc compiles it.

## Current status

- Round 1: done ([results](results/round1/RESULTS.json)).
- Round 2: done ([results](results/round2/SUMMARY.md)).
- Round 2b: done ([results](results/round2b/SUMMARY.md)).
- Round 3: done ([results](results/round3/SUMMARY.md)).
- Round 4: done ([results](results/round4/SUMMARY.md)).
- Round 5: done ([results](results/round5/SUMMARY.md)).
- Round 5b: done ([results](results/round5b/SUMMARY.md)).
- Round 6: done ([results](results/round6/SUMMARY.md)).

Headline (round 4, held-out, 78 tasks, greedy pass@1 unless noted):

- 7B C-only model, C compiled by gcc `-O2`: **46/78** (56/78 pass@5).
- Hybrid, the C-then-asm model's C compiled by gcc `-O2`: **48/78** (61/78 pass@5).
- The same model's own assembly (direct, no compiler): **7/78** (9/78 pass@5).
- C-then-asm repair loop (k = 8, R = 3): **19/78** full tests (27/78 visible).
- Of the 42 held-out tasks whose C passes but whose assembly fails:
  logic mismatch 23, crash 9, timeout 9, does not assemble 1, ABI 0.

So far the practical system is "model writes C, gcc lowers it"; the model's own
assembly is usually well-formed and ABI-correct but does not implement its own
C. Rounds 5 and 5b trained the 7B to write assembly directly from gcc lowerings
of C it wrote itself (target: at least 20/78 direct assembly with no compiler).
The target was missed: `asm-distill-7b` scores **5/78** held-out (29/90 val) and
the distill-heavy retrain `asm-distill-7b-v2` **4/78** (22/90 val), both below
round 4's 7/78 for direct assembly. See `results/round5b/SUMMARY.md`.

Round 6 (no training, 2.0 GPU h) measured the C path's ceiling. A repair loop on
`c-only-7b`'s C (k = 8, R = 3, visible-test feedback, gcc `-O2`) passes **64/78**
held-out on the full tests (69/78 visible) and 76/90 val. The untrained
Qwen2.5-Coder-7B-Instruct, writing C greedily with no loop, passes **63/78**
(66/78 pass@5): fine-tuning on the training mix cost the C model about 17
held-out tasks (46/78 greedy). See `results/round6/SUMMARY.md`.

## Results

Held-out set (78 tasks). Regenerated from `results/*/RESULTS.json` after each
run; each row shows the latest round that reported it.
"Own assembly" rows score the model's assembly with no compiler; "its C" and
"hybrid" rows compile the model's C with gcc and score the result in the same
harness.

| Model | Output scored | Round | n | pass@1 | pass@5 |
|---|---|---|---:|---:|---:|
| qlora7b | own assembly | 2b | 78 | 5/78 | 8/78 |
| qlora7b-v2 | own assembly | 4 | 78 | 5/78 | 12/78 |
| asm-distill-7b | own assembly | 5b | 78 | 5/78 | 8/78 |
| lora2 | own assembly | 4 | 78 | 4/78 | 5/78 |
| asm-distill-7b-v2 | own assembly | 5b | 78 | 4/78 | 7/78 |
| lora1 | own assembly | 2b | 78 | 2/78 | 5/78 |
| dpo2 | own assembly | 3 | 78 | 2/78 | 2/78 |
| base 1.5B (untrained) | own assembly | 1 | 78 | 0/78 | 0/78 |
| base 7B (untrained) | own assembly | 2b | 78 | 0/78 | 3/78 |
| dpo | own assembly | 3 | 78 | 0/78 | 0/78 |
| c-then-asm | own assembly (after its C) | 4 | 78 | 7/78 | 9/78 |
| base 7B (untrained) | its C, gcc -O2 | 6 | 78 | 63/78 | 66/78 |
| c-only-7b | its C, gcc -O2 | 4 | 78 | 46/78 | 56/78 |
| base 1.5B (untrained) | its C, gcc -O2 | 4 | 78 | 31/78 | 49/78 |
| c-then-asm | hybrid: its C, gcc -O2 | 4 | 78 | 48/78 | 61/78 |
| c-only-7b | hybrid: its C, gcc -O2 | 4 | 78 | 46/78 | 56/78 |
| c-then-asm | hybrid: its C, gcc -Os | 4 | 78 | 48/78 | 61/78 |
| c-only-7b | hybrid: its C, gcc -Os | 4 | 78 | 46/78 | 56/78 |

Repair loops (k = 8, R = 3) on the held-out set:

| Loop | Model | Round | n | pass (all tests) | pass (visible tests) |
|---|---|---|---:|---:|---:|
| loop-lora1-heldout | lora1 | 2b | 78 | 6/78 | 10/78 |
| loop-base-c-heldout | base 1.5B (untrained) | 4 | 78 | 44/78 | 48/78 |
| loop-lora2-heldout | lora2 | 4 | 78 | 10/78 | 15/78 |
| loop-dpo-heldout | dpo | 2b | 62 | 0/62 | 0/62 |
| loop-qlora7b-v2-heldout | qlora7b-v2 | 4 | 78 | 12/78 | 17/78 |
| loop-c-then-asm-heldout | c-then-asm | 4 | 78 | 19/78 | 27/78 |
| loop-asm-distill-7b-heldout-c-then-asm (partial, of 78) | asm-distill-7b | 5b | 20 | 8/20 | — |
| loop-asm-distill-7b-v2-heldout-c-then-asm (partial, of 78) | asm-distill-7b-v2 | 5b | 10 | 4/10 | — |
| loop-c-only-7b-heldout | c-only-7b | 6 | 78 | 64/78 | 69/78 |

Validation (90 tasks) numbers, size and instruction ratios, and GPU hours are in
each round's `results/<round>/SUMMARY.md` and `RESULTS.json`.

## Reports

| Report | Live (AFI Lab) | Repo copy |
|---|---|---|
| Concept paper | [abundantfrontier.org/lab/e2m/concept](https://abundantfrontier.org/lab/e2m/concept/) | [reports/concept](reports/concept/index.html) |
| 1. Harness, compiler-sweep data, scale-up | [report-1](https://abundantfrontier.org/lab/e2m/report-1/) | [reports/report-1](reports/report-1/index.html) |
| 2. Repair loop and rejection sampling | [report-2](https://abundantfrontier.org/lab/e2m/report-2/) | [reports/report-2](reports/report-2/index.html) |
| 3. Preference optimization and a second 7B (rounds 2 and 2b) | [report-3](https://abundantfrontier.org/lab/e2m/report-3/) | [reports/report-3](reports/report-3/index.html) |
| 4. Preference training with a supervised term, and C then assembly (round 3) | [report-4](https://abundantfrontier.org/lab/e2m/report-4/) | [reports/report-4](reports/report-4/index.html) |
| 5. A matched C control, a gcc hybrid, and where the model's assembly goes wrong (round 4) | [report-5](https://abundantfrontier.org/lab/e2m/report-5/) | [reports/report-5](reports/report-5/index.html) |
| 6. Distilling gcc's lowering into direct assembly (rounds 5 and 5b) | [report-6](https://abundantfrontier.org/lab/e2m/report-6/) | [reports/report-6](reports/report-6/index.html) |
| 7. The C path's ceiling: a matched repair loop, and an untrained 7B that writes better C than the fine-tuned one (round 6) | [report-7](https://abundantfrontier.org/lab/e2m/report-7/) | [reports/report-7](reports/report-7/index.html) |

The repo copies are the HTML sources (open them locally or through a static
file server; `reports/style.css` is shared). `reports/CHANGELOG.md` lists
revisions.

## Model weights

Adapters are not stored in git. Weights are on Hugging Face: [AbundantFrontierInstitute/e2m](https://huggingface.co/AbundantFrontierInstitute/e2m).

- **lora1** (round 1, base `Qwen2.5-Coder-1.5B-Instruct`): First LoRA: English to AT&T assembly, 2,910 scaled tasks. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/lora1)
- **qlora7b** (round 1, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA: English to assembly, same 2,910 tasks as lora1. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/qlora7b)
- **lora2** (round 1, base `Qwen2.5-Coder-1.5B-Instruct`): LoRA on the 6,781-task mix (rejection-sampled loop programs plus rewritten specs). [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/lora2)
- **dpo** (round 2, base `Qwen2.5-Coder-1.5B-Instruct`): DPO from lora2 (beta 0.1); generation collapsed. Kept for the record. Weights not published (collapsed run).
- **qlora7b-v2** (round 2b, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA on the 6,781-task mix, English to assembly. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/qlora7b-v2)
- **dpo2** (round 3, base `Qwen2.5-Coder-1.5B-Instruct`): DPO from lora2 with an RPO-style supervised term (beta 0.2, 1,500 steps). [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/dpo2)
- **c-then-asm** (round 3, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA that writes C, then assembly for that C. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/c-then-asm)
- **c-only-7b** (round 4, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA that writes C only; the C model of the gcc hybrid. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/c-only-7b)
- **asm-distill-7b** (round 5, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA, English to assembly distilled from gcc lowerings of the model's own C. [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/asm-distill-7b)
- **asm-distill-7b-v2** (round 5b, base `Qwen2.5-Coder-7B-Instruct`): 4-bit QLoRA, distill-heavy retrain (distilled rows in the majority, earlier pairs capped at 25%). [weights](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/asm-distill-7b-v2)

See [docs/MODELS.md](docs/MODELS.md) for base models, rounds and settings.

## Reproduce

### Environment

- Linux x86-64, Python 3.12, gcc and GNU binutils (`as`, `ld`) on `PATH`.
- Optional: `qemu-x86_64` user-mode 8.2 for AVX2/FMA tasks (set
  `E2M_EMU_BACKEND=qemu` or pass `backend="qemu"`). The code looks for it in
  `$E2M_QEMU`, under `tools/qemu/`, or on `PATH`.
- Training: one NVIDIA GPU with 24 GB (7B runs use 4-bit QLoRA). The runs
  used torch 2.11 (CUDA 12.8), transformers 5.22, peft 0.21,
  bitsandbytes 0.50, accelerate 1.15 and datasets 5.2.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch transformers peft bitsandbytes accelerate datasets   # training only
python -m pytest                      # harness, tasks, gcc baselines, fuzz
```

### Base models

```bash
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir runs/models/Qwen2.5-Coder-1.5B-Instruct
hf download Qwen/Qwen2.5-Coder-7B-Instruct   --local-dir runs/models/Qwen2.5-Coder-7B-Instruct
```

Spec synthesis and rewrites call a local [Ollama](https://ollama.com/) server
(`qwen3:14b`, embeddings with `bge-m3`).

### Harness and data

```bash
python -m e2m.fuzz --n 10000
python -m e2m.score --tasks tasks/starter.jsonl --candidates tasks/gcc-O2-cands.jsonl --baseline O2
python -m e2m.data.sweep --tasks tasks/starter.jsonl --out data --jobs 16
python -m e2m.scale --target 3000 --jobs 16          # needs ExeBench + AnghaBench, see docs/DATA.md
python -m e2m.data.sweep --tasks tasks/scaled.jsonl --out data/scaled \
    --compilers gcc,clang --grid reduced --jobs 16 --llvm-seeds 0
```

### Rounds

Each round is a resumable pipeline that writes `runs/<round>/SUMMARY.md` and
`RESULTS.json` (copied here under `results/`). Planning notes for each round
are in [docs/round-notes](docs/round-notes/).

| Round | What it ran | Command |
|---|---|---|
| 1 | spec rewrite, lora1 / qlora7b / lora2 SFT, repair loops, rejection sampling | `python -m e2m.loop.pipeline` |
| 2 | lora2 val loop, DPO, 7B base | `python -m e2m.loop round2` |
| 2b | 7B QLoRA v2 and its held-out loop | `python -m e2m.loop.round2 --round2b` |
| 3 | DPO with a supervised term, C-then-asm 7B | `python -m e2m.loop.round3` |
| 4 | matched 7B C control, gcc hybrid, loop resume, failure analysis | `python -m e2m.loop.round4` |

Single steps can be run on their own, for example
`python -m e2m.train.sft_lora --help`, `python -m e2m.train.eval_generate --help`
and `python -m e2m.loop run --help` (repair loop).

The pipelines take a simple lock file (`~/gpu.lock`) so they share one GPU
politely with other jobs; delete nothing you did not create.

## Layout

```
e2m/            Python package: harness/, data/, scale/, heldout/, loop/, train/
tests/          pytest suite
eval/           frozen held-out set (78) and validation split ids (90)
tasks/          starter tasks and gcc -O2 candidates
data/           small sweep outputs and stats (large files: docs/DATA.md)
configs/        training configs of each run
results/        per-round SUMMARY.md, RESULTS.json, failure analyses, run summaries
reports/        HTML sources of the concept paper and technical reports
docs/           concept note, data pointers, model list, round notes
```

## License

Apache-2.0, copyright 2026 Abundant Frontier Institute. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). Held-out tasks adapted from HumanEval (MIT) and MBPP
(CC BY 4.0) and other third-party material are credited in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Each report has a "How to cite" section with BibTeX. For the project as a whole:

```bibtex
@misc{wilson2026e2m,
  author       = {Wilson, Boyd},
  title        = {{E2M}: English to machine code},
  year         = {2026},
  institution  = {Abundant Frontier Institute},
  howpublished = {\url{https://github.com/abundantfrontier/e2m}}
}
```

<sub>Last synced results as of 2026-11-26.</sub>
