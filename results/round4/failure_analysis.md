# C-then-asm held-out failure analysis

Source: `runs/c-then-asm/eval.jsonl`, greedy held-out, C half vs assembly half.
Agreement and C-only counts are copied from the eval rows; categories use the harness labels already stored on the assembly half.

Tasks: 78. C and asm both pass: 6. C-only: 42. Asm-only: 1. Neither: 29. Agree: 35/78.

## Tally of C-pass / asm-fail

| bucket | n | harness category |
|---|---:|---|
| doesnt_assemble | 1 | doesnt_assemble |
| abi | 0 | abi |
| logic_mismatch | 23 | wrong_answer vs the model's own C |
| crash | 9 | crash |
| timeout | 9 | timeout / insn_limit |
| other | 0 | other |
| **total** | **42** | |

`logic_mismatch` means the C half passes the tests after gcc -O2, and the assembly half assembles and runs but returns the wrong value. That is a semantic gap between the two halves, not a compile failure.

## Examples

### `he11_string_xor` — crash (crash)

Detail: `unmapped: unmapped`

C (truncated):

```c
void he_string_xor(const char *a, const char *b, char *out) {
    while (*a && *b) {
        if (*a == *b)
            *out = '0';
        else
            *out = '1';

        a++;
        b++;
        out++;
    }

    *out = 0;
}
```

Asm (truncated):

```asm
.text
	.globl	he_string_xor
	.type	he_string_xor, @function
he_string_xor:
	movzbl	(%rdi), %edx
	xorl	%eax, %eax
	testb	%dl, %dl
	jne	.L2
	jmp	.L5
	.p2align 5
	.p2align 4,,10
	.p2align 3
.L11:
	addq	$1, %rax
	movzbl	(%rdi,%rax), %edx
	testb	%dl, %dl
…
```

### `mbpp45_array_gcd` — doesnt_assemble (doesnt_assemble)

Detail: `assembler rejected the input as both AT&T and Intel syntax
AT&T:
/tmp/e2m-as-93ukqo8b/in.s: Assembler messages:
/tmp/e2m-as-93ukqo8b/in.s:37: Error: symbol `.L3' is already defined

Intel:
/tmp/e2m-as-93ukqo8b/in.s: Assembler messages:
/tmp/e2m-as-93ukqo8b/in.s:5: Warning: mnemonic suffix used with `mov'
/tmp/e2m-as-93ukqo8b/in.s:5: Warning: NOTE: Such forms are deprecated and will be rejected by `

C (truncated):

```c
int64_t mbpp_array_gcd(const int64_t *vals, int64_t n) {
  if (n == 1) return vals[0];

  int64_t a = vals[0];
  int64_t b = vals[1];
  while (b) {
    int64_t t = b;
    b = a % b;
    a = t;
  }

  for (int64_t i = 2; i < n; i++) {
    int64_t v = vals[i];
    while (v) {
      int64_t t = v;
      v = a % v;
…
```

Asm (truncated):

```asm
.text
	.globl	mbpp_array_gcd
	.type	mbpp_array_gcd, @function
mbpp_array_gcd:
	movq	(%rdi), %rax
	cmpq	$1, %rsi
	je	.L1
	movq	8(%rdi), %rcx
	xorl	%edx, %edx
	.p2align 5
	.p2align 4
	.p2align 3
.L3:
	movq	%rdx, %r8
	movq	%rax, %rdx
	movq	%rcx, %rax
…
```

### `he102_choose_num` — logic_mismatch (wrong_answer)

Detail: `return 4294967295 != expected 14`

C (truncated):

```c
int64_t he_choose_num(int64_t x, int64_t y) {
  if (x > y) return -1;
  if ((y & 1)) {
    if (y - 1 < x) return -1;
    else return y - 1;
  } else {
    if (y < x) return -1;
    else return y;
  }
}
```

Asm (truncated):

```asm
.text
	.globl	he_choose_num
	.type	he_choose_num, @function
he_choose_num:
	movq	%rsi, %rax
	cmpq	%rdi, %rsi
	jl	.L2
	andl	$1, %eax
	je	.L3
	subq	$1, %rax
	cmpq	%rdi, %rax
	jl	.L2
.L3:
	ret
	.p2align 4,,10
	.p2align 3
…
```

### `he25_factorize` — timeout (timeout)

Detail: `timeout: timeout`

C (truncated):

```c
int64_t he_factorize(int64_t n, int64_t *out) {
  int64_t d = 2;
  int64_t k = 0;
  while (n > 1 && d * d <= n) {
    while (n % d == 0) {
      n /= d;
      out[k++] = d;
    }
    d++;
  }
  if (n > 1) {
    out[k] = n;
    k++;
  }
  return k;
}
```

Asm (truncated):

```asm
.text
	.globl	he_factorize
	.type	he_factorize, @function
he_factorize:
	movq	%rdi, %r9
	movq	%rsi, %rax
	cmpq	$1, %rdi
	jle	.L7
	movl	$2, %ecx
	xorl	%edi, %edi
	imulq	%rcx, %rdx
	cmpq	%rdx, %r9
	jl	.L7
	.p2align 5
	.p2align 4
	.p2align 3
…
```

### `he146_special_filter` — crash (crash)

Detail: `unmapped: unmapped`

C (truncated):

```c
int64_t he_special_filter(const int64_t *nums, int64_t n) {
  int64_t c = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t v = nums[i];
    if (v <= 10) continue;
    int64_t vs = v % 100;
    if (vs / 10 % 2 == 0) continue;
    if (vs % 2 == 0) continue;
    c++;
  }
  return c;
}
```

Asm (truncated):

```asm
.text
	.globl	he_special_filter
	.type	he_special_filter, @function
he_special_filter:
	movq	%rdi, %r9
	xorl	%ecx, %ecx
	testq	%rsi, %rsi
	jle	.L1
	xorl	%eax, %eax
	xorl	%edi, %edi
	.p2align 6
	.p2align 4
	.p2align 3
.L5:
	movq	(%r9,%rax,8), %rdx
	cmpq	$10, %rdx
…
```

### `he151_double_the_difference` — crash (crash)

Detail: `unmapped: unmapped`

C (truncated):

```c
int64_t he_double_the_difference(const int64_t *lst, int64_t n) {
  int64_t sum = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t x = lst[i];
    if (x > 0 && x % 2 == 1) {
      sum += x * x;
    }
  }
  return sum;
}
```

Asm (truncated):

```asm
.text
	.globl	he_double_the_difference
	.type	he_double_the_difference, @function
he_double_the_difference:
	testq	%rsi, %rsi
	jle	.L5
	movq	%rsi, %r8
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	salq	$3, %r8
	.p2align 5
	.p2align 4
	.p2align 3
.L4:
	movq	(%rdi,%rax), %rdx
	addq	$1, %rax
…
```

