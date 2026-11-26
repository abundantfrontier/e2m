# Round 6: the C path's ceiling

GPU hours: 1.999 / 5.0.

## C repair loops (c-only-7b, mode c, k=8, R=3, visible-test feedback; gcc -O2)

- loop-c-only-7b-heldout: pass 64/78 full, 69/78 visible, mean calls 8.2; partial=False (of 78). Budget curve {'1': 0.5769230769230769, '4': 0.6153846153846154, '8': 0.6923076923076923, '16': 0.782051282051282, '32': 0.8205128205128205}. By source {'MBPP_n': 23, 'MBPP_pass': 18, 'HumanEval_n': 55, 'HumanEval_pass': 46}.
- loop-c-only-7b-val: pass 76/90 full, 76/90 visible, mean calls 6.8; partial=False (of 90). Budget curve {'1': 0.7222222222222222, '4': 0.7333333333333333, '8': 0.7777777777777778, '16': 0.8333333333333334, '32': 0.8444444444444444}. By source {'ExeBench_n': 90, 'ExeBench_pass': 76}.

## Greedy C (pass@1 / pass@5)

- c-only-7b/c/heldout/round4: n=78 pass@1 46/78 pass@5 56/78 categories {'wrong_answer': 23, 'pass': 46, 'doesnt_assemble': 7, 'crash': 1, 'timeout': 1}
- c-only-7b/c/val/round4: n=90 pass@1 62/90 pass@5 72/90 categories {'pass': 62, 'wrong_answer': 27, 'doesnt_assemble': 1}
- base7b-c/c/heldout: n=78 pass@1 63/78 pass@5 66/78 categories {'wrong_answer': 7, 'pass': 63, 'doesnt_assemble': 8}
- base7b-c/c/val: n=90 pass@1 61/90 pass@5 65/90 categories {'pass': 61, 'wrong_answer': 22, 'crash': 3, 'doesnt_assemble': 4}

Comparison (copied, not re-scored): round 4 hybrid gcc -O2 48/78 (c-then-asm C), c-only-7b C 46/78, 1.5B untrained C loop 44/78 (k=4, R=2), best direct asm 7/78.

## Size buckets (reference gcc -O2 <= 40 bytes = small)

```
{
 "c-only-7b/c": {
  "heldout": {
   "large_n": 64,
   "large_pass": 34,
   "small_n": 14,
   "small_pass": 12
  },
  "val": {
   "small_n": 59,
   "small_pass": 49,
   "large_n": 31,
   "large_pass": 13
  }
 },
 "asm-distill-7b/asm": {
  "heldout": {
   "large_n": 64,
   "large_pass": 1,
   "small_n": 14,
   "small_pass": 4
  },
  "val": {
   "small_n": 59,
   "small_pass": 27,
   "large_n": 31,
   "large_pass": 2
  }
 },
 "asm-distill-7b-v2/asm": {
  "heldout": {
   "large_n": 64,
   "large_pass": 0,
   "small_n": 14,
   "small_pass": 4
  },
  "val": {
   "small_n": 59,
   "small_pass": 19,
   "large_n": 31,
   "large_pass": 3
  }
 },
 "c-then-asm/c-then-asm": {
  "heldout": {
   "large_n": 64,
   "large_pass": 1,
   "small_n": 14,
   "small_pass": 6
  },
  "val": {
   "small_n": 59,
   "small_pass": 37,
   "large_n": 31,
   "large_pass": 3
  }
 },
 "base7b-c/c": {
  "heldout": {
   "large_n": 64,
   "large_pass": 50,
   "small_n": 14,
   "small_pass": 13
  },
  "val": {
   "small_n": 59,
   "small_pass": 48,
   "large_n": 31,
   "large_pass": 13
  }
 }
}
```

## gcc -Os on the reference C

```
{
 "heldout": {
  "n": 78,
  "mean_os_over_o2_size": 0.6711291334001076
 },
 "val": {
  "n": 90,
  "mean_os_over_o2_size": 0.8630047165123952
 },
 "hybrid-Os/c-only-7b/heldout": {
  "n": 46,
  "mean_size_vs_ref_Os": 0.9989771676589517
 },
 "hybrid-Os/c-only-7b/val": {
  "n": 62,
  "mean_size_vs_ref_Os": 1.0416039788797518
 },
 "hybrid-Os/c-then-asm/heldout": {
  "n": 48,
  "mean_size_vs_ref_Os": 0.9749812319262691
 },
 "hybrid-Os/c-then-asm/val": {
  "n": 63,
  "mean_size_vs_ref_Os": 0.9933263349270407
 }
}
```

## Stages

| stage | status |
|---|---|
| base7b-c-eval-heldout | ok |
| base7b-c-eval-val | ok |
| loop-c-only-7b-heldout | ok |
| loop-c-only-7b-val | ok |

Decision rule: C loop held-out >= 55/78: next round superoptimizes the hybrid's gcc output. < 50/78: next round improves the C side.
