# E2M: English to machine code. First concept and research note (v0.1)

Date: 2026-11-10 (America/Los_Angeles). Status: first draft for Boyd to mark up.

E2M here means a model that takes English and produces machine code (executable bytes, or assembly that a plain assembler turns into bytes) for a target ISA such as x86-64 or aarch64, without a human-visible source language and compiler step in the loop.

Reading guide: every number below comes from a paper abstract or project page I opened on 2026-11-10 (links in section 11). Where I could not verify something, I say so. Things marked "my estimate" or "my measurement" are mine.

## 1. Problem statement

1.1. Goal: given an English specification of a function (plus its signature and calling convention), emit machine code that passes tests, is memory safe, and is competitive in size or speed.

1.2. Why it is hard:
   1.2.1. English is ambiguous. Whatever E2M outputs, someone has to decide it matches intent.
   1.2.2. Assembly has low information density and is much longer than the source (the Nova paper names this as a core challenge).
   1.2.3. One wrong instruction can break everything, and the failure is often silent (wrong register, wrong stack alignment, clobbered callee-saved register).
   1.2.4. Outputs depend on the ISA, ABI, and optimization level.

1.3. What would count as success: a measured win over the obvious baseline (English to C to gcc/clang with the same base model) on at least one axis, such as .text size, cycles, or "works with no toolchain on the target".

## 2. Direct versus via source: the honest comparison

2.1. Case for going through source (NL -> C -> compiler). This is the strong default:
   2.1.1. Code LLMs are trained on huge amounts of source (The Stack is 3 TB of permissively licensed code) and very little assembly.
   2.1.2. A human can read C. Intent review happens at the C level, which is the cheapest place to catch an ambiguous spec.
   2.1.3. Mature compilers give semantics-preserving optimization for free, and CompCert shows it can be proved for a C subset.
   2.1.4. Retargeting is a compiler flag. Direct generation needs training data per ISA.
   2.1.5. Output length: in my 5-function smoke test (section 8), the C source was 449 characters while gcc assembly was 1,429 to 3,588 characters depending on -O level. Tokens are a rough proxy, so this is only indicative.
   2.1.6. Published neural compilation results are mixed. Armengol-Estapé and O'Boyle (2021) call their C-to-x86 results "relatively weak". The 2025 LaaC paper (CompilerEval) reports basic ability but low compilation success rates. NeuComBack reports 44% x86-64 correctness for a baseline workflow. LLM Compiler reports 45% disassembly round trip and 14% exact match.

2.2. Case for going direct:
   2.2.1. No toolchain needed at generation time or on the target. This matters for new or odd ISAs, firmware, and embedded targets.
   2.2.2. Search can beat the compiler on small kernels. STOKE (2013) matched or beat gcc -O3 and icc -O3 on small loop-free x86-64 kernels. AlphaDev found sorting routines for 3 to 5 elements that were merged into LLVM libc++. SuperCoder (2025) reports an RL-tuned 7B model reaching 95.0% correctness and 1.46x average speedup over gcc -O3 on a benchmark of 8,072 assembly programs.
   2.2.3. A single model could trade size, speed, and spec together instead of via flags.
   2.2.4. It is an interesting research question, and a self-hosted one.

2.3. Verdict:
   2.3.1. For general "write me a program" work, NL -> C -> compiler should dominate for the foreseeable future. E2M should treat it as the baseline to beat, not as a strawman.
   2.3.2. E2M's defensible niches: (a) small hot kernels where verified search beats the compiler, (b) targets with no compiler, (c) spec-to-binary pipelines where the spec carries tests and size or speed targets, (d) research on "compiler as a learned component".
   2.3.3. The most promising shape is hybrid: let the model write code, let tools (assembler, sandbox, equivalence checker) judge it, and loop.

## 3. What prior work says

3.1. NL to code, and what it implies:
   3.1.1. Codex / HumanEval: 28.8% pass@1, 70.2% with 100 samples per problem (Chen et al., 2021). Repeated sampling plus tests is a strong lever, and it transfers directly to E2M because assembly can be tested by running it.
   3.1.2. MBPP: 974 entry-level Python tasks. The largest few-shot model (137B) solved 59.6% (Austin et al., 2021).
   3.1.3. AlphaCode: top 54.3% on average in Codeforces contests with 5,000+ participants, using massive sampling, filtering, and clustering (Li et al., 2022). Same lesson: sample many, filter with tests.
   3.1.4. StarCoder (open, trained on The Stack): about 40% pass@1 on HumanEval when prompted (Li et al., 2023). Newer open models are much stronger; I did not verify current leaderboard numbers, so I do not quote them.
   3.1.5. Benchmarks have false positives. EvalPlus (Liu et al., 2023) adds many more tests to HumanEval-style tasks. E2M needs the same care, since weak tests will reward wrong assembly.
   3.1.6. Multilingual benchmark ports exist (MultiPL-E, MBXP, HumanEval-X incl. C++). These give NL specs with tests in C-family languages.
   3.1.7. Implication: NL to a high-level language is solved well enough at small scale that skipping it needs a stated reason (section 2).

3.2. Neural compilation (C or IR to assembly):
   3.2.1. Armengol-Estapé and O'Boyle (2021): first Transformer experiment on C to x86, "relatively weak" results, data and code released.
   3.2.2. LLM Compiler (Meta, 2024): Code Llama continued-trained on 546B tokens of LLVM-IR and assembly, 7B and 13B, bespoke commercial license. Fine-tuned variants get 77% of an autotuner's code-size gains and 45% disassembly round trip (14% exact).
   3.2.3. Large Language Models for Compiler Optimization (2023) and Compiler Generated Feedback (2024): models predict optimization passes and are trained against compiler feedback.
   3.2.4. LEGO-Compiler (2025): decomposes programs into blocks, checks each with external tests, uses feedback. Reports over 99% on ExeBench and 97.9% on AnsiBench. Note the heavy scaffolding.
   3.2.5. QiMeng-NeuComBack (NeurIPS 2025): IR-to-assembly benchmark (200 ExeBench cases plus 151 TSVC cases). A self-evolving prompt method raises x86-64 correctness from 44% to 64% and aarch64 from 36% to 58%. 14 of 16 correct x86-64 programs beat clang -O3.
   3.2.6. LaaC / CompilerEval (2025): evaluates mainstream LLMs on assembly generation, finds basic ability with low success rates, and argues targeted training and prompting help.
   3.2.7. Guaranteed Guess (2025): LLM translates assembly between ISAs (CISC to RISC) inside a testing framework. 99% correct on HumanEval programs, 49% on BringupBench.
   3.2.8. IR as an intermediate: IRCoder and TransCoder-IR show that compiler IR training helps multilingual code models and translation.
   3.2.9. Implication: models can produce assembly, but reliable results come from scaffolding (decompose, test, retry), not from one-shot generation.

3.3. Neural decompilation (the inverse, and a data source):
   3.3.1. LLM4Decompile (2024): open 1.3B to 33B models. Reported over 100% relative gain in re-executability over GPT-4o and Ghidra on HumanEval and ExeBench.
   3.3.2. Nova (2023): assembly-specialized LLM with hierarchical attention, motivated by assembly's low information density and optimization diversity.
   3.3.3. SLaDe (2023): small Transformer decompiler, evaluated on 4,000+ ExeBench functions on two ISAs and two -O levels, up to 6x more accurate than Ghidra.
   3.3.4. Guess & Sketch (2023): LM-guided transpilation with a symbolic repair step.
   3.3.5. Caveat: LLM4Decompile and Nova report different numbers for the same model family because their evaluation setups differ. Compare only within one paper.
   3.3.6. Use for E2M: decompiler corpora show how to build (source, binary) pairs at scale, and the asm tokenization tricks transfer.

3.4. Superoptimization and tuning for size or speed:
   3.4.1. STOKE (ASPLOS 2013): MCMC search over loop-free x86-64, correctness plus speed in a cost function, equivalence verified, matches or beats gcc -O3 and icc -O3, sometimes expert asm.
   3.4.2. Souper (2017): synthesizing superoptimizer for LLVM IR, finds missed peephole optimizations.
   3.4.3. Learning to Superoptimize Programs (2016): neural proposal distribution for the search.
   3.4.4. AlphaDev (Nature, 2023): RL on an "AssemblyGame"; sort 3, 4, 5 routines adopted into LLVM libc++; up to 70% faster for short sequences and about 1.7% for sequences over 250,000 elements.
   3.4.5. SuperCoder (2025): LLMs as assembly superoptimizers; RL with a correctness-plus-speed reward; Qwen2.5-Coder-7B-Instruct went from 61.4% correctness and 1.10x speedup to 95.0% and 1.46x (average over gcc -O3). Strongest zero-shot baseline in that paper: 51.5% test pass, 1.43x.
   3.4.6. LLM-Vectorizer (2024): LLM-written SIMD code with test feedback; 1.1x to 9.4x speedups; Alive2 verified 38.2% of vectorizations as correct on TSVC.
   3.4.7. Source-level variants: PIE (Learning Performance-Improving Code Edits) and Supersonic.
   3.4.8. Implication: the strongest evidence for E2M's value is "model proposes, tools verify", on small kernels.

3.5. Program synthesis and verified compilation as correctness tools:
   3.5.1. CompCert: machine-checked proof that compiled code preserves C semantics (Leroy, CACM 2009). The gold standard, but only for a compiler, not a neural model.
   3.5.2. Translation validation (Pnueli, Siegel, Singerman, 1998): do not prove the compiler, validate each run. This is the right pattern for a neural compiler: check each output.
   3.5.3. Alive2 (PLDI 2021): bounded translation validation for LLVM IR with an SMT solver, found 47 bugs in LLVM's own tests (28 fixed). Works on IR, so it points at an NL -> IR route.
   3.5.4. STOKE includes x86-64 equivalence checking from tests to bounded to full verification, usable on loop-free code.
   3.5.5. Implication: tests give confidence, translation validation gives guarantees for a subset (loop-free or bounded loops), and E2M should report which tier each result reached.

3.6. Datasets usable now:
   3.6.1. ExeBench: 4.5M compilable and 700k executable real C functions with IO examples. v1.01 splits include train_real_simple_io (43.8k), train_synth_simple_io (550k), train_synth_rich_io (97k), and test_real (2.1k). On Hugging Face.
   3.6.2. AnghaBench: over 1 million compilable C functions mined from public repos.
   3.6.3. ComPile: a 182B-token LLVM-IR dataset built from production Rust, Swift, Julia, and C/C++ code (per its abstract).
   3.6.4. The Stack (3 TB) and The Stack v2 (StarCoder2): permissively licensed source for C and other languages.
   3.6.5. HumanEval-X (C++), HumanEval-Decompile (C ports from LLM4Decompile), MBPP and MBXP, MultiPL-E, EvalPlus: specs plus tests for evaluation.
   3.6.6. SuperCoder's benchmark of 8,072 assembly programs (C plus gcc assembly) is the nearest thing to a ready-made asm optimization set; a copy of its data is on Hugging Face, so check its provenance and license.
   3.6.7. Gap: none of these pair English specs with machine code. We must synthesize the English side.

## 4. Candidate architectures

4.1. A. Direct assembly generation: spec (+ signature) -> x86-64 assembly text -> `as` -> test.
   4.1.1. Pros: simplest, matches the question, assembler gives cheap syntax feedback.
   4.1.2. Cons: long outputs, no types, easy to violate the ABI.

4.2. B. NL -> LLVM IR -> `llc`:
   4.2.1. Pros: typed SSA, target independent, retargeting by `llc`, Alive2 can check IR transformations, and IR training helps code models (IRCoder).
   4.2.2. Cons: IR is also long and fiddly; and it starts to look like a compiler front end, so the argument for C gets stronger.

4.3. C. Hybrid with compiler and execution in the loop:
   4.3.1. Generate N candidates (A or B), assemble, run tests in a sandbox, feed errors and failing cases back (Self-Debug, RLEF, Compiler Generated Feedback, NeuComBack).
   4.3.2. Optional second stage: a superoptimization pass with equivalence checking (STOKE or Alive2 style) against the first verified candidate.
   4.3.3. Train with RL using reward = correctness + size or speed (SuperCoder style).

4.4. D. Control baseline: same base model, NL -> C -> gcc/clang -O2. Every claim about A to C must be compared with D.

4.5. Recommendation: build D and A first on the same harness, then add the C loop. B only if A struggles with ABI or retargeting.

## 5. Dataset construction plan

5.1. Seed functions:
   5.1.1. ExeBench executable splits (real and synthetic IO).
   5.1.2. AnghaBench and a C subset of The Stack for scale (needs generated tests, see 5.3).
   5.1.3. Hold HumanEval-C and MBPP-C aside as evaluation sets (164 and 974 tasks is too small to train on anyway).

5.2. Spec generation:
   5.2.1. Use a self-hosted open code model to write an English spec from each function, keeping the C signature.
   5.2.2. Adequacy filter: a different model implements the spec in C, and the original tests must pass. Drop specs that fail. This removes specs that omit key behavior.
   5.2.3. Vary style (terse, docstring, step-by-step) to avoid one phrasing.

5.3. Tests: ExeBench IO pairs where available. Otherwise generate inputs from the signature (ExeBench's approach), run the reference binary built with sanitizers, and record outputs.

5.4. Compilation sweep:
   5.4.1. gcc and clang, -O0 -O1 -O2 -O3 -Os, x86-64 first. aarch64 later via cross compilers.
   5.4.2. Normalize assembly: `-fno-asynchronous-unwind-tables -fcf-protection=none`, strip `.file` and CFI noise, canonicalize labels.
   5.4.3. Record .text bytes and measured cycles per variant so the target can be chosen by size or speed, not just "gcc -O2".

5.5. Split by project or file, not by function, and decontaminate against HumanEval and MBPP.

5.6. Size to start: 20k to 50k verified (spec, tests, asm) triples. Scale only if the prototype shows signal.

5.7. Licensing: ExeBench and AnghaBench come from GitHub code. Check licenses before any release. The Stack is filtered for permissive licenses.

## 6. Evaluation

6.1. Functional correctness: pass@k with the unbiased estimator from the Codex paper (sample n >= k, count correct c). Report pass@1 and pass@10.

6.2. Sandbox:
   6.2.1. Unicorn for function-level runs: map code, stack, and data pages, run with instruction and time limits, hook memory and interrupts. I confirmed `pip install unicorn` (2.1.4) works in a venv on the box and executed a 4-byte x86-64 function.
   6.2.2. QEMU user-mode for whole programs and for aarch64 (docs: qemu.org). Use inside namespaces plus seccomp for any native run.
   6.2.3. Allowlist: no syscalls unless the task needs them (default: none for pure functions).

6.3. Metrics, reported per -O level and per function-length bin:
   6.3.1. Assembles, links, passes tests.
   6.3.2. Code size (.text bytes), relative to gcc -O2 and -Os.
   6.3.3. Dynamic instruction count in the emulator, and cycles via `perf` on native x86-64 for final numbers.
   6.3.4. ABI conformance: callee-saved registers restored, stack aligned, return value in the right register (a small checker harness).

6.4. Safety:
   6.4.1. Memory: flag out-of-bounds reads and writes via emulator memory hooks, and compare against the sanitizer-built reference on the same inputs.
   6.4.2. Control flow: no jumps outside the function, no self-modifying code, no indirect calls to unmapped addresses.
   6.4.3. Differential fuzzing against the reference, with more inputs than training tests (EvalPlus lesson).

6.5. Strongest tier (loop-free or bounded code only): equivalence checking with STOKE-style verification or Alive2 on IR. Report which tier each pass reached: tests only, fuzzed, or proved.

6.6. Spec adequacy: sample 50 to 100 specs for a human check, since a passing test on a bad spec means nothing.

## 7. First prototype that fits one RTX 3090 (24 GB)

7.1. Task: English spec + C signature -> x86-64 assembly for one pure function, O2-style, small functions (HumanEval and MBPP style).

7.2. Model: LoRA or QLoRA fine-tune of a small open code model.
   7.2.1. Start with a 1.5B model for fast iteration (e.g. Qwen2.5-Coder-1.5B).
   7.2.2. Then a 7B model with QLoRA. SuperCoder used Qwen2.5-Coder-7B-Instruct as its RL starting point, which is evidence it handles assembly tasks reasonably. Check each model card's license.
   7.2.3. My estimate, not tested: a 7B QLoRA run fits in 24 GB at sequence length 2k to 4k with gradient checkpointing. Assembly is long, so sequence length is the binding constraint.

7.3. Self-hosted stack, no subscriptions: Hugging Face PEFT/TRL or Unsloth or Axolotl for training, llama.cpp or vLLM for inference, gcc/clang/as/objdump, Unicorn and QEMU for tests. Spec writing and test generation use a local open model.

7.4. Experiments:
   7.4.1. E0: zero-shot base model, NL -> asm. Gives the floor.
   7.4.2. E1: LoRA SFT on the 20k to 50k triples.
   7.4.3. E2: E1 plus best-of-n sampling and test filtering, then an assemble-error and test-failure repair loop.
   7.4.4. Control D: same base, NL -> C -> gcc -O2.
   7.4.5. Success criterion: E1 or E2 within a few points of D on pass@1 on HumanEval-C and MBPP-C, or ahead of D on size at equal pass rate.

7.5. Hardware note: the box this note was written on has no GPU, so training must run on your 3090 machine. The dataset builder and the evaluation harness can be built and tested here.

## 8. What I measured (toy smoke test)

8.1. Setup: 5 small C functions (sum_to, is_prime, max_of_three, count_vowels, dot), 449 characters of C. For each of -O0, -O1, -O2, -O3, -Os I ran gcc -S (14.2, x86-64, no CFI, no CET), assembled with `as`, linked with a C test harness, and ran it. All five variants passed.

8.2. Results (my measurement, tiny sample):
   8.2.1. -O0: 2,537 chars of asm, 352 bytes text.
   8.2.2. -O1: 1,751 chars, 316 bytes.
   8.2.3. -O2: 2,101 chars, 381 bytes.
   8.2.4. -O3: 3,588 chars, 655 bytes.
   8.2.5. -Os: 1,429 chars, 146 bytes.

8.3. Takeaways:
   8.3.1. The target depends heavily on the -O level. Machine code size varied 4.5x across levels for the same spec. We must pick the target on purpose.
   8.3.2. Assembly text is 3x to 8x longer than the C. Expect long sequences.
   8.3.3. The assemble-link-run loop works with only gcc and binutils. Unicorn also works, so the harness needs no paid tools.

## 9. Risks and open questions

9.1. Risks:
   9.1.1. Imitation ceiling: SFT on compiler output teaches the model to imitate gcc. Beating gcc needs search or RL with a speed reward.
   9.1.2. Spec ambiguity and weak tests produce false "passes".
   9.1.3. Undefined behavior in C seeds: the compiler may exploit it, so the "right" asm is not unique. Prefer UB-free seeds (sanitizers).
   9.1.4. ABI and calling-convention errors that tests on a single harness miss.
   9.1.5. Long-context cost for larger functions. LEGO-Compiler's block decomposition is one answer.
   9.1.6. Generated machine code is executable content. Keep every run sandboxed, never run it natively without isolation.
   9.1.7. Data licensing, and contamination of HumanEval and MBPP in base-model pretraining.
   9.1.8. Floating point, SIMD, libc calls, and global data are outside the first prototype.

9.2. Open questions for Boyd:
   9.2.1. Is E2M a research probe or a tool you want to use? That changes whether beating the C baseline matters.
   9.2.2. Which ISA first? I suggest x86-64, then aarch64.
   9.2.3. Function-level or whole-program? I suggest function-level first.
   9.2.4. What should "better than the compiler" mean to you: size, speed, or no toolchain?
   9.2.5. Should the model ask clarifying questions when a spec is ambiguous?
   9.2.6. Which machine has the 3090, and can the box reach it?

## 10. Next steps

10.1. You: answer 9.2.1 to 9.2.4 and mark up this note.
10.2. Build the harness on the box: compile sweep, normalization, Unicorn runner, pass@k scorer.
10.3. Build baseline D (NL -> C -> gcc) with a local open model and score it on HumanEval-C and MBPP-C.
10.4. Dataset v0: ExeBench executable subset with generated specs and the adequacy filter.
10.5. E0 and E1 on the 3090, starting with the 1.5B model.
10.6. Decide on RL or the compiler-in-the-loop stage, using the SuperCoder recipe as the reference.

## 11. References

Code generation and benchmarks
- Codex / HumanEval: https://arxiv.org/abs/2107.03374
- MBPP: https://arxiv.org/abs/2108.07732
- AlphaCode: https://arxiv.org/abs/2203.07814
- StarCoder: https://arxiv.org/abs/2305.06161
- StarCoder 2 and The Stack v2: https://arxiv.org/abs/2402.19173
- The Stack: https://arxiv.org/abs/2211.15533
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- EvalPlus: https://arxiv.org/abs/2305.01210
- HumanEval-X (CodeGeeX): https://arxiv.org/abs/2303.17568
- MultiPL-E: https://arxiv.org/abs/2208.08227
- Self-Debug: https://arxiv.org/abs/2304.05128
- RLEF: https://arxiv.org/abs/2410.02089

Neural compilation
- Learning C to x86 Translation: https://arxiv.org/abs/2108.07639
- Meta LLM Compiler: https://arxiv.org/abs/2407.02524
- LLMs for Compiler Optimization: https://arxiv.org/abs/2309.07062
- Compiler Generated Feedback for LLMs: https://arxiv.org/abs/2403.14714
- LEGO-Compiler: https://arxiv.org/abs/2505.20356
- QiMeng-NeuComBack: https://arxiv.org/abs/2511.01183
- LLM as a Compiler (CompilerEval): https://arxiv.org/abs/2511.04132
- Guaranteed Guess: https://arxiv.org/abs/2506.14606
- IRCoder: https://arxiv.org/abs/2403.03894
- TransCoder-IR: https://arxiv.org/abs/2207.03578

Neural decompilation
- LLM4Decompile: https://arxiv.org/abs/2403.05286 (code: https://github.com/albertan017/LLM4Decompile)
- Nova: https://arxiv.org/abs/2311.13721
- SLaDe: https://arxiv.org/abs/2305.12520
- Guess & Sketch: https://arxiv.org/abs/2309.14396

Superoptimization and performance
- STOKE (ASPLOS 2013): https://dl.acm.org/doi/10.1145/2451116.2451150 (code: https://github.com/StanfordPL/stoke)
- Souper: https://arxiv.org/abs/1711.04422 (code: https://github.com/google/souper)
- Learning to Superoptimize Programs: https://arxiv.org/abs/1611.01787
- AlphaDev: https://www.nature.com/articles/s41586-023-06004-9
- SuperCoder: https://arxiv.org/abs/2505.11480
- LLM-Vectorizer: https://arxiv.org/abs/2406.04693
- PIE: https://arxiv.org/abs/2302.07867
- Supersonic: https://arxiv.org/abs/2309.14846

Verified compilation and validation
- CompCert (Leroy, CACM 2009): https://cacm.acm.org/research/formal-verification-of-a-realistic-compiler/ (project: https://compcert.org/)
- Translation Validation (Pnueli et al., 1998): https://doi.org/10.1007/BFb0054170
- Alive2: https://dl.acm.org/doi/10.1145/3453483.3454030 (code: https://github.com/AliveToolkit/alive2)

Datasets
- ExeBench: https://dl.acm.org/doi/10.1145/3520312.3534867 (data: https://huggingface.co/datasets/jordiae/exebench)
- AnghaBench: https://github.com/brenocfg/AnghaBench
- ComPile: https://arxiv.org/abs/2309.15432

Tooling
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Unicorn: https://www.unicorn-engine.org/
- QEMU user mode: https://www.qemu.org/docs/master/user/main.html
- llc: https://llvm.org/docs/CommandGuide/llc.html
- PEFT: https://github.com/huggingface/peft
- Unsloth: https://github.com/unslothai/unsloth
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- llama.cpp: https://github.com/ggml-org/llama.cpp
