"""Round-8 checker test cases: the 19 known-wrong round-7 "wins" and the 10 equivalent pairs.

The 19 known-wrong wins are assembly listings that passed every test of their task (and, in round 7, a
500-input differential check) but compute a different result from the reference C on some inputs:
3 were caught by the round-7 5,000-input recheck (diff.py v1 families) and 16 by the post-run audit
(v2 families). Checker v3 (e2m/harness/diff3.py) must reject all 19 at 2,000 inputs. The 10 equivalent
pairs are other compilations of the reference C (gcc -Os/-O3, clang -Oz/-O2); every checker must accept them.
Round 8 stage A (e2m/loop/round8.py gate) is this test; it passed 19/19 and 10/10 on Thu Nov 26, 2026.

Run (from the repo root, after `pip install -e .`; needs gcc and the pinned clang for consensus/equivalent cases):
    python -m e2m.harness.diff3_cases            # all three checkers (v1, v2, v3) on every case
    python -m e2m.harness.diff3_cases --only v3  # one checker
Held-out tasks come from eval/heldout.jsonl. Val tasks (exebench_*) live in tasks/scaled.jsonl, which is not
in this repository (rebuild it with `python -m e2m.scale`, see docs/DATA.md); cases on missing tasks are skipped.
"""
from __future__ import annotations

import argparse
import sys

KNOWN_WRONG = [{'src': 'model',
  'split': 'heldout',
  'task_id': 'he3_below_zero',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=1 cand ret=513 cand status=ok',
  'v3_detail': 'mismatch: ref ret=1 cand ret=257 cand status=ok',
  'asm': '.text\n'
         '\t.globl\the_below_zero\n'
         'he_below_zero:\n'
         '\txorl\t%eax, %eax\n'
         '\txorl\t%edx, %edx\n'
         '.L2:\n'
         '\tcmpq\t%rsi, %rax\n'
         '\tjge\t.L7\n'
         '\taddq\t(%rdi,%rax,8), %rdx\n'
         '\tjs\t.L5\n'
         '\tincq\t%rax\n'
         '\tjmp\t.L2\n'
         '.L7:\n'
         '\txorl\t%eax, %eax\n'
         '\tret\n'
         '.L5:\n'
         '\tmovb\t$1, %al\n'
         '\tret'},
 {'src': 'model',
  'split': 'val',
  'task_id': 'exebench_CmpDoubles_1153267d175b',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'asm': '.text\n'
         '\t.p2align 4\n'
         '\t.globl\tCmpDoubles\n'
         'CmpDoubles:\n'
         '\tmovsd\t(%rdi), %xmm1\n'
         '\tmovsd\t(%rsi), %xmm0\n'
         '\tmovl\t$-1, %eax\n'
         '\tcomisd\t%xmm1, %xmm0\n'
         '\tjae\t.L2\n'
         '\tmovl\t$1, %eax\n'
         '.L2:\n'
         '\tret'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he75_is_multiply_prime',
  'goal': 'size',
  'caught_by': 'round-7 5,000-input recheck (v1 families)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok',
  'asm': '\t.att_syntax\n'
         '\t.text\n'
         '\t.globl\the_is_multiply_prime\n'
         'he_is_multiply_prime:\n'
         '\tcmpq\t$2, %rdi\n'
         '\tjge\t.LBB0_2\n'
         '.LBB0_1:\n'
         '\txorl\t%ecx, %ecx\n'
         '\tjmp\t.LBB0_13\n'
         '.LBB0_2:\n'
         '\txorl\t%ecx, %ecx\n'
         '\txorl\t%esi, %esi\n'
         '.LBB0_3:\n'
         '\ttestb\t$1, %dil\n'
         '\tjne\t.LBB0_4\n'
         '\tincq\t%rsi\n'
         '\tshrq\t%rdi\n'
         '\tjne\t.LBB0_3\n'
         '.LBB0_13:\n'
         '\tmovl\t%ecx, %eax\n'
         '\tretq\n'
         '.LBB0_4:\n'
         '\tpushq\t$3\n'
         '\tpopq\t%r8\n'
         '.LBB0_5:\n'
         '\tcmpq\t$100, %r8\n'
         '\tja\t.LBB0_11\n'
         '.LBB0_7:\n'
         '\tmovl\t%edi, %eax\n'
         '\txorl\t%edx, %edx\n'
         '\tdivq\t%r8\n'
         '\ttestl\t%edx, %edx\n'
         '\tjne\t.LBB0_10\n'
         '\tmovl\t%eax, %edi\n'
         '\tcmpq\t$2, %rsi\n'
         '\tleaq\t1(%rsi), %rsi\n'
         '\tjle\t.LBB0_7\n'
         '.LBB0_10:\n'
         '\taddq\t$2, %r8\n'
         '\tjmp\t.LBB0_5\n'
         '.LBB0_11:\n'
         '\tcmpq\t$1, %rdi\n'
         '\tjne\t.LBB0_1\n'
         '\txorl\t%ecx, %ecx\n'
         '\tcmpq\t$3, %rsi\n'
         '\tsete\t%cl\n'
         '\tjmp\t.LBB0_13\n'
         '.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he85_sum_even_odd_idx',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=-62 cand ret=None cand status=unmapped unmapped',
  'v3_detail': 'mismatch: ref ret=0 cand ret=None cand status=unmapped',
  'asm': '\t.text\n'
         '\t.globl\the_sum_even_odd_idx\n'
         'he_sum_even_odd_idx:\n'
         '\tcmpq\t$1, %rsi\n'
         '\tjle\t.L14\n'
         '\tleaq\t-2(%rsi), %rax\n'
         '\tcmpq\t$11, %rax\n'
         '\tjbe\t.L15\n'
         '.L4:\n'
         '\tmovdqa\t%xmm0, %xmm1\n'
         '.L3:\n'
         '\tmovq\t(%rdi,%rdx,8), %rcx\n'
         '\tleaq\t(%rax,%rcx), %r8\n'
         '\tandl\t$1, %ecx\n'
         '\tcmove\t%r8, %rax\n'
         '\tmovq\t16(%rdi,%rdx,8), %rcx\n'
         '\tleaq\t(%rax,%rcx), %r8\n'
         '\tandl\t$1, %ecx\n'
         '\tcmove\t%r8, %rax\n'
         '\tret\n'
         '\tcmpq\t$11, %rax\n'
         '\tret\n'
         '.L14:\n'
         '\txorl\t%eax, %eax\n'
         '.L1:\n'
         '\tret\n'
         '\tret\n'
         '\tret\n'
         '.L15:\n'
         '\txorl\t%eax, %eax\n'
         '\tmovl\t$1, %edx\n'
         '\tjmp\t.L3\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he100_make_a_pile',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=None cand ret=None cand status=prot prot',
  'v3_detail': 'mismatch: ref ret=None cand ret=None cand status=prot',
  'asm': '\t.text\n'
         '\t.globl\the_make_a_pile\n'
         'he_make_a_pile:\n'
         '\tleaq\t(%rdi,%rdi,2), %rcx\n'
         '\tmovl\t%edi, %eax\n'
         '\tnegq\t%rax\n'
         '\tleaq\t(%rsi,%rax,4), %rdx\n'
         '\ttestb\t$1, %dil\n'
         '\tje\t.L3\n'
         '\tmovq\t%rdi, (%rdx,%rdi,4)\n'
         '\taddq\t$2, %rdi\n'
         '\tcmpq\t%rcx, %rdi\n'
         '\tje\t.L11\n'
         '.L3:\n'
         '\tmovq\t%rdi, (%rdx,%rdi,4)\n'
         '\tleaq\t2(%rdi), %rax\n'
         '\taddq\t$4, %rdi\n'
         '\tmovq\t%rax, (%rdx,%rax,4)\n'
         '\tcmpq\t%rcx, %rdi\n'
         '\tjne\t.L3\n'
         '.L1:\n'
         '\tret\n'
         '.L11:\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'mbpp151_is_coprime',
  'goal': 'size',
  'caught_by': 'round-7 5,000-input recheck (v1 families)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=1 cand ret=0 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\tmbpp_is_coprime\n'
         'mbpp_is_coprime:\n'
         '\tmovq\t%rsi, %rcx\n'
         '\tmovq\t%rdi, %rax\n'
         '\tnegq\t%rax\n'
         '\tcmovs\t%rdi, %rax\n'
         '\tnegq\t%rcx\n'
         '\tcmovs\t%rsi, %rcx\n'
         '.L2:\n'
         '\tcqto\n'
         '\ttestq\t%rcx, %rcx\n'
         '\tje\t.L5\n'
         '\tidivq\t%rcx\n'
         '\tmovq\t%rcx, %rax\n'
         '\tmovl\t%edx, %ecx\n'
         '\tjmp\t.L2\n'
         '.L5:\n'
         '\tdecq\t%rax\n'
         '\tsete\t%al\n'
         '\tmovzbl\t%al, %eax\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he114_min_subarray_sum',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=2 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=5 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\the_min_subarray_sum\n'
         'he_min_subarray_sum:\n'
         '\tmovl\t%edi, %r8d\n'
         '\txorl\t%edx, %edx\n'
         '\tmovq\t(%r8), %rax\n'
         '\tmovl\t%esi, %edi\n'
         '.L2:\n'
         '\tcmpq\t%rdi, %rdx\n'
         '\tjge\t.L7\n'
         '\tmovl\t%edx, %ecx\n'
         '\txorl\t%esi, %esi\n'
         '.L3:\n'
         '\taddq\t(%r8,%rcx,8), %rsi\n'
         '\tcmpq\t%rsi, %rax\n'
         '\tcmovg\t%rsi, %rax\n'
         '\tincq\t%rcx\n'
         '\tcmpq\t%rcx, %rdi\n'
         '\tjne\t.L3\n'
         '\tincq\t%rdx\n'
         '\tjmp\t.L2\n'
         '.L7:\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he120_k_largest',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=3 cand ret=None cand status=prot prot',
  'v3_detail': 'mismatch: ref ret=1 cand ret=None cand status=prot',
  'asm': '\t.text\n'
         '\t.globl\the_k_largest\n'
         'he_k_largest:\n'
         '\tmovl\t%edx, %eax\n'
         '\ttestl\t%eax, %eax\n'
         '\tjle\t.L9\n'
         '\tmovl\t%ecx, %edx\n'
         '\tmovl\t%esi, %r10d\n'
         '\tmovl\t$1, %r11d\n'
         '\tmovl\t%edi, %r9d\n'
         '\tmovl\t%edi, %r8d\n'
         '.L5:\n'
         '\tmovl\t%r11d, %ecx\n'
         '.L7:\n'
         '\tmovq\t(%r8), %rsi\n'
         '\tmovq\t(%r9,%rcx,8), %rdi\n'
         '\tcmpq\t%rsi, %rdi\n'
         '\tjge\t.L6\n'
         '\tmovq\t%rdi, (%r8)\n'
         '\tmovq\t%rsi, (%r9,%rcx,8)\n'
         '.L6:\n'
         '\tincq\t%rcx\n'
         '\tcmpq\t%rcx, %r10\n'
         '\tjg\t.L7\n'
         '\taddq\t$8, %r8\n'
         '\tincq\t%r11\n'
         '\tcmpq\t%r11, %r10\n'
         '\tjne\t.L5\n'
         '.L3:\n'
         '\tsubq\t%rax, %r10\n'
         '\tleaq\t(%r9,%r10,8), %rdi\n'
         '\txorl\t%ecx, %ecx\n'
         '.L8:\n'
         '\tmovq\t(%rdi,%rcx,8), %rsi\n'
         '\tmovq\t%rsi, (%rdx,%rcx,8)\n'
         '\tincq\t%rcx\n'
         '\tcmpq\t%rcx, %rax\n'
         '\tjne\t.L8\n'
         '\tret\n'
         '.L9:\n'
         '\tret\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he131_odd_digit_product',
  'goal': 'size',
  'caught_by': 'round-7 5,000-input recheck (v1 families)',
  'round7_detail': 'mismatch: ref ret=4557106575 cand ret=262139279 cand status=ok',
  'v3_detail': 'mismatch: ref ret=232630513987207 cand ret=2200333959 cand status=ok',
  'asm': '\t.att_syntax\n'
         '\t.text\n'
         '\t.globl\the_odd_digit_product\n'
         'he_odd_digit_product:\n'
         '\tmovq\t%rdi, %rax\n'
         '\tnegq\t%rax\n'
         '\tcmovsq\t%rdi, %rax\n'
         '\tpushq\t$1\n'
         '\txorl\t%esi, %esi\n'
         '\tpopq\t%rdi\n'
         '\tpushq\t$10\n'
         '\tpopq\t%r8\n'
         '\tpushq\t$1\n'
         '\tpopq\t%r9\n'
         '\tmovl\t%edi, %ecx\n'
         '.LBB0_1:\n'
         '\ttestq\t%rax, %rax\n'
         '\tje\t.LBB0_3\n'
         '\txorl\t%edx, %edx\n'
         '\tdivq\t%r8\n'
         '\ttestb\t$1, %dl\n'
         '\tcmoveq\t%rdi, %rdx\n'
         '\tcmovnel\t%r9d, %esi\n'
         '\timulq\t%rdx, %rcx\n'
         '\tjmp\t.LBB0_1\n'
         '.LBB0_3:\n'
         '\ttestl\t%esi, %esi\n'
         '\tcmoveq\t%rax, %rcx\n'
         '\tmovl\t%ecx, %eax\n'
         '\tretq\n'
         '.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'heldout',
  'task_id': 'he152_compare_scores',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=None cand ret=None cand status=prot prot',
  'v3_detail': 'mismatch: ref ret=None cand ret=None cand status=prot',
  'asm': '\t.text\n'
         '\t.globl\the_compare_scores\n'
         'he_compare_scores:\n'
         '\tmovl\t%esi, %r8d\n'
         '\txorl\t%eax, %eax\n'
         '\tmovl\t%edx, %esi\n'
         '\tmovl\t%ecx, %r9d\n'
         '.L3:\n'
         '\tmovq\t(%rdi,%rax,8), %rdx\n'
         '\tsubq\t(%r8,%rax,8), %rdx\n'
         '\tmovq\t%rdx, %rcx\n'
         '\tnegq\t%rcx\n'
         '\tcmovns\t%rcx, %rdx\n'
         '\tmovq\t%rdx, (%r9,%rax,8)\n'
         '\tincq\t%rax\n'
         '\tcmpq\t%rax, %rsi\n'
         '\tjne\t.L3\n'
         '.L1:\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_az_paragraph_num_lines_23afebeee5fd',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=2 cand ret=1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=2 cand ret=1 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\taz_paragraph_num_lines\n'
         'az_paragraph_num_lines:\n'
         '\txorl\t%edx, %edx\n'
         '.L2:\n'
         '\tleal\t1(%rdx), %eax\n'
         '.L8:\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_az_paragraph_num_lines_23afebeee5fd',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=2 cand ret=1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=2 cand ret=1 cand status=ok',
  'asm': '\t.att_syntax\n'
         '\t.text\n'
         '\t.globl\taz_paragraph_num_lines\n'
         'az_paragraph_num_lines:\n'
         '\tmovl\t$1, %eax\n'
         '.LBB0_4:\n'
         '\tretq\n'
         '\tincq\t%rdi\n'
         '.LBB0_5:\n'
         '.LBB0_1:\n'
         '\tincq\t%rdi\n'
         '\tretq\n'
         '\tincq\t%rdi\n'
         '.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_checkmd5_f5e6fad22803',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'asm': '\t.text\n\t.globl\tcheckmd5\ncheckmd5:\n.L3:\n\tmovl\t$-1, %eax\n\tret\n.L4:\n\tret\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_checkmd5_f5e6fad22803',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=-1 cand status=ok',
  'asm': '\t.att_syntax\n\t.text\n\t.globl\tcheckmd5\ncheckmd5:\n\tmovl\t$-1, %eax\n.LBB0_16:\n\tretq\n.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_delta_AT_b6ca07e07493',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=35.0 cand ret=10.0 cand status=ok',
  'v3_detail': 'mismatch: ref ret=35.0 cand ret=10.0 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\tdelta_AT\n'
         'delta_AT:\n'
         '.L3:\n'
         '\tmovsd\t.LC0(%rip), %xmm1\n'
         '.L1:\n'
         '\tmovaps\t%xmm1, %xmm0\n'
         '\tret\n'
         '\t.section\t.rodata.cst8,"aM",@progbits,8\n'
         '.LC0:\n'
         '\t.long\t0\n'
         '\t.long\t1076101120\n'
         '.LC1:\n'
         '\t.long\t0\n'
         '\t.long\t1075970048\n'
         '.LC2:\n'
         '\t.long\t0\n'
         '\t.long\t1075838976\n'
         '.LC3:\n'
         '\t.long\t0\n'
         '\t.long\t1075576832\n'
         '.LC4:\n'
         '\t.long\t0\n'
         '\t.long\t1075314688\n'
         '.LC5:\n'
         '\t.long\t0\n'
         '\t.long\t1075052544\n'
         '.LC6:\n'
         '\t.long\t0\n'
         '\t.long\t1074790400\n'
         '.LC7:\n'
         '\t.long\t0\n'
         '\t.long\t1074266112\n'
         '.LC8:\n'
         '\t.long\t0\n'
         '\t.long\t1073741824\n'
         '.LC9:\n'
         '\t.long\t0\n'
         '\t.long\t1072693248\n'
         '.LC11:\n'
         '\t.long\t0\n'
         '\t.long\t-1074790400\n'
         '.LC12:\n'
         '\t.long\t1073741824\n'
         '\t.long\t1094864768\n'
         '.LC13:\n'
         '\t.long\t-858993459\n'
         '\t.long\t-1058084804\n'
         '.LC14:\n'
         '\t.long\t-858993459\n'
         '\t.long\t-1058557412\n'
         '.LC15:\n'
         '\t.long\t-1717986918\n'
         '\t.long\t-1059056103\n'
         '.LC16:\n'
         '\t.long\t858993459\n'
         '\t.long\t-1060053261\n'
         '.LC17:\n'
         '\t.long\t0\n'
         '\t.long\t1082160128\n'
         '.LC18:\n'
         '\t.long\t0\n'
         '\t.long\t1084246272\n'
         '.LC19:\n'
         '\t.long\t0\n'
         '\t.long\t1085049088\n'
         '.LC20:\n'
         '\t.long\t0\n'
         '\t.long\t1085554048\n'
         '.LC21:\n'
         '\t.long\t0\n'
         '\t.long\t1085919616\n'
         '.LC22:\n'
         '\t.long\t0\n'
         '\t.long\t1086285184\n'
         '.LC23:\n'
         '\t.long\t0\n'
         '\t.long\t1086487872\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_delta_AT_b6ca07e07493',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=35.0 cand ret=10.0 cand status=ok',
  'v3_detail': 'mismatch: ref ret=35.0 cand ret=10.0 cand status=ok',
  'asm': '\t.att_syntax\n'
         '\t.section\t.rodata.cst8,"aM",@progbits,8\n'
         '.LCPI0_0:\n'
         '\t.quad\t0xc1424f8040000000\n'
         '.LCPI0_1:\n'
         '\t.quad\t0x4024000000000000\n'
         '.LCPI0_2:\n'
         '\t.quad\t0xc0eee83ccccccccd\n'
         '.LCPI0_3:\n'
         '\t.quad\t0x4022000000000000\n'
         '.LCPI0_4:\n'
         '\t.quad\t0xc0e7b21ccccccccd\n'
         '.LCPI0_5:\n'
         '\t.quad\t0x4020000000000000\n'
         '.LCPI0_6:\n'
         '\t.quad\t0xc0e016199999999a\n'
         '.LCPI0_7:\n'
         '\t.quad\t0x401c000000000000\n'
         '.LCPI0_8:\n'
         '\t.quad\t0xc0d0def333333333\n'
         '.LCPI0_9:\n'
         '\t.quad\t0x4018000000000000\n'
         '.LCPI0_10:\n'
         '\t.quad\t0x4080740000000000\n'
         '.LCPI0_11:\n'
         '\t.quad\t0x4014000000000000\n'
         '.LCPI0_12:\n'
         '\t.quad\t0x40a0490000000000\n'
         '.LCPI0_13:\n'
         '\t.quad\t0x4010000000000000\n'
         '.LCPI0_14:\n'
         '\t.quad\t0x40ac890000000000\n'
         '.LCPI0_15:\n'
         '\t.quad\t0x4008000000000000\n'
         '.LCPI0_16:\n'
         '\t.quad\t0x40b43d8000000000\n'
         '.LCPI0_17:\n'
         '\t.quad\t0x4000000000000000\n'
         '.LCPI0_18:\n'
         '\t.quad\t0x40b9d18000000000\n'
         '.LCPI0_19:\n'
         '\t.quad\t0x3ff0000000000000\n'
         '.LCPI0_20:\n'
         '\t.quad\t0x40bf658000000000\n'
         '.LCPI0_21:\n'
         '\t.quad\t0x40c27d4000000000\n'
         '.LCPI0_22:\n'
         '\t.quad\t0xbff0000000000000\n'
         '\t.text\n'
         '\t.globl\tdelta_AT\n'
         'delta_AT:\n'
         '\tmovsd\t.LCPI0_1(%rip), %xmm0\n'
         '\tretq\n'
         '\tucomisd\t%xmm0, %xmm1\n'
         '\tretq\n'
         '\tjae\t.LBB0_9\n'
         '\tretq\n'
         '.LBB0_1:\n'
         '\tmovsd\t.LCPI0_1(%rip), %xmm0\n'
         '\tretq\n'
         '.LBB0_3:\n'
         '\tretq\n'
         '\tretq\n'
         '\tretq\n'
         '.LBB0_5:\n'
         '\tretq\n'
         '\tretq\n'
         '.LBB0_7:\n'
         '\tretq\n'
         '.LBB0_9:\n'
         '\tretq\n'
         '\tretq\n'
         '.LBB0_11:\n'
         '\tretq\n'
         '.LBB0_13:\n'
         '\tretq\n'
         '.LBB0_15:\n'
         '\tretq\n'
         '.LBB0_17:\n'
         '\tretq\n'
         '.LBB0_19:\n'
         '\tretq\n'
         '\tretq\n'
         '.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_ehnlc_strncmp_9353c4ce09ba',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=0 cand ret=None cand status=insn_limit insn_limit',
  'v3_detail': 'mismatch: ref ret=0 cand ret=62 cand status=ok',
  'asm': '\t.att_syntax\n'
         '\t.text\n'
         '\t.globl\tehnlc_strncmp\n'
         'ehnlc_strncmp:\n'
         '\txorl\t%ecx, %ecx\n'
         '.LBB0_2:\n'
         '\tmovzbl\t(%rdi,%rcx), %eax\n'
         '\tmovzbl\t(%rsi,%rcx), %r8d\n'
         '\tcmpb\t%r8b, %al\n'
         '\tjne\t.LBB0_6\n'
         '\tincq\t%rcx\n'
         '\tcmpq\t%rcx, %rdx\n'
         '\tjne\t.LBB0_2\n'
         '.LBB0_5:\n'
         '\txorl\t%eax, %eax\n'
         '\tretq\n'
         '.LBB0_6:\n'
         '\tsubl\t%r8d, %eax\n'
         '\tretq\n'
         '.Lfunc_end0:\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_ehstrnlen_ad3e8426bbc9',
  'goal': 'speed',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=1 cand ret=27 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\tehstrnlen\n'
         'ehstrnlen:\n'
         '\tmovl\t%esi, %eax\n'
         '\tret\n'
         '\tjne\t.L2\n'
         '.L4:\n'
         '.L2:\n'
         '\tcmpb\t$0, (%rdi,%rdx)\n'
         '\tret\n'
         '\tret\n'
         '.L3:\n'
         '\tret\n'},
 {'src': 'search',
  'split': 'val',
  'task_id': 'exebench_ehstrnlen_ad3e8426bbc9',
  'goal': 'size',
  'caught_by': 'round-7 post-run audit (v2 families, 5,000 inputs)',
  'round7_detail': 'mismatch: ref ret=1 cand ret=27 cand status=ok',
  'v3_detail': 'mismatch: ref ret=0 cand ret=1 cand status=ok',
  'asm': '\t.text\n'
         '\t.globl\tehstrnlen\n'
         'ehstrnlen:\n'
         '\txorl\t%eax, %eax\n'
         '.L2:\n'
         '\tcmpq\t%rsi, %rax\n'
         '\tje\t.L1\n'
         '\tincq\t%rax\n'
         '\tjmp\t.L2\n'
         '.L1:\n'
         '\tret\n'}]

EQUIVALENT = [{'task_id': 'mbpp3_is_not_prime', 'candidate': 'gcc -Os of reference C'},
 {'task_id': 'he3_below_zero', 'candidate': 'clang -Oz of reference C'},
 {'task_id': 'he8_sum_product', 'candidate': 'gcc -O3 of reference C'},
 {'task_id': 'he9_rolling_max', 'candidate': 'clang -O2 of reference C'},
 {'task_id': 'mbpp6_differ_one_bit', 'candidate': 'gcc -Os of reference C'},
 {'task_id': 'he11_string_xor', 'candidate': 'clang -Oz of reference C'},
 {'task_id': 'he13_gcd', 'candidate': 'gcc -O3 of reference C'},
 {'task_id': 'mbpp8_square_nums', 'candidate': 'clang -O2 of reference C'},
 {'task_id': 'he16_count_distinct_chars', 'candidate': 'gcc -Os of reference C'},
 {'task_id': 'he18_how_many_times', 'candidate': 'clang -Oz of reference C'}]


def _tasks():
    from e2m.loop.round7_common import interleaved_tasks
    try:
        return {t.id: t for _, t in interleaved_tasks()}
    except FileNotFoundError:
        from e2m.loop.round7_common import load_split
        return {t.id: t for t in load_split("heldout")}


def run_case(task, asm: str, checker: str):
    """Return (accepted, detail) for one checker: v1/v2 = diff.diff_check at 5,000 inputs, v3 = Checker3 at 2,000."""
    from e2m.loop.round7_common import measure, oracle_image
    m = measure(task, asm)
    if not m["passed"]:
        return False, "fails the task's tests: " + m["fail"]
    if checker in ("v1", "v2"):
        from e2m.harness.diff import diff_check
        d = diff_check(task, asm, oracle_image(task), n=5000, seed=8080, families=checker, time_budget_s=300)
        return d.ok, d.detail
    from e2m.harness.diff3 import Checker3
    from e2m.loop.round8_common import consensus_images
    c = Checker3(task, oracle_image(task), seed=0, consensus=consensus_images(task))
    r = c.check(asm, n=2000, seed=1)
    return r.ok, r.detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["v1", "v2", "v3"])
    a = ap.parse_args(argv)
    from e2m.loop.round7_common import compile_c
    tasks = _tasks()
    checkers = [a.only] if a.only else ["v1", "v2", "v3"]
    for kind, cases in (("known-wrong (expect reject)", KNOWN_WRONG), ("equivalent (expect accept)", EQUIVALENT)):
        print(f"== {kind}")
        for c in cases:
            t = tasks.get(c["task_id"])
            if t is None:
                print(f"  {c['task_id']}: task not available here; skipped")
                continue
            if "asm" in c:
                asm = c["asm"]
            else:
                comp, opt = c["candidate"].split()[:2]
                asm = compile_c(t.reference_c, comp, opt)
            res = []
            for ch in checkers:
                ok, detail = run_case(t, asm, ch)
                res.append(f"{ch}={'accept' if ok else 'reject'}")
            print(f"  {c['task_id']} {c.get('goal', '')} {' '.join(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
