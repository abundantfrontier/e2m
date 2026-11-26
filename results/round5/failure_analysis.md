# Round 5 direct-asm held-out failures

Model `asm-distill-7b`, split `heldout`, greedy. Scored tasks: 52. Passes: 4.

Buckets use `e2m.train.failure_analysis.classify_asm_fail` (`wrong_answer` is `logic_mismatch`).

| bucket | n |
|---|---:|
| doesnt_assemble | 4 |
| abi | 0 |
| logic_mismatch | 35 |
| crash | 4 |
| timeout | 5 |
| other | 0 |

Round 4 hybrid pass set (c-then-asm, model C compiled gcc -O2, held-out), copied from `runs/round4/hybrid.jsonl`: 48 tasks. Not re-scored.

Direct-asm passes that are also in that hybrid set: 4.

Failures among those hybrid-pass tasks:

| bucket | n |
|---|---:|
| doesnt_assemble | 3 |
| abi | 0 |
| logic_mismatch | 23 |
| crash | 4 |
| timeout | 2 |
| other | 0 |

## Examples

### `mbpp3_is_not_prime` — doesnt_assemble

Detail: `assembler rejected the input as both AT&T and Intel syntax
AT&T:
/tmp/e2m-as-vy1zm68x/in.s: Assembler messages:
/tmp/e2m-as-vy1zm68x/in.s:26: Error: register type mismatch for `cmove'

Intel:
/tmp/e2m-as-vy1zm68x/in.s: Assembler messages:
/`

### `mbpp6_differ_one_bit` — logic_mismatch

Detail: `return 4294967295 != expected 1`

### `mbpp45_array_gcd` — crash

Detail: `int: interrupt 0`

### `mbpp76_count_squares` — timeout

Detail: `timeout: timeout`

