"""Leaf-C adaptations of HumanEval and MBPP problems.

Each task is an English spec, a SysV signature, a C reference, and tests
whose expected values come from a Python oracle checked against the
benchmark's published examples. The C is verified separately by the harness.
"""

from __future__ import annotations

from typing import Any, Callable


def _fn(ret: str, name: str, args: str, body: str) -> tuple[str, str]:
    sig = f"{ret} {name}({args})"
    return sig, sig + " {\n" + body.strip("\n") + "\n}\n"


def _task(
    id_: str,
    source: str,
    spec: str,
    sig: str,
    c: str,
    tests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": id_,
        "source": source,
        "spec": spec,
        "signature": sig,
        "reference_c": c,
        "tests": tests,
    }


def _buf(ident: str, data: list[int] | str, *, out: bool = False, ptr: str = "i64") -> dict[str, Any]:
    raw: dict[str, Any] = {"id": ident, "ptr": ptr, "data": data}
    if out:
        raw["out"] = True
    return raw


def _ret(args: list[Any], ret: int | float, bufs: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"args": args, "ret": ret}
    if bufs:
        row["bufs"] = bufs
    return row


def _void(args: list[Any], bufs: dict[str, Any]) -> dict[str, Any]:
    return {"args": args, "bufs": bufs}


def tasks() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(id_: str, source: str, spec: str, sig: str, c: str, tests: list[dict[str, Any]]) -> None:
        out.append(_task(id_, source, spec, sig, c, tests))

    # --- MBPP scalar / array -------------------------------------------------
    sig, c = _fn(
        "int64_t",
        "mbpp_is_not_prime",
        "int64_t n",
        """
  int64_t i;
  if (n < 2) return 0;
  for (i = 2; i * i <= n; i++) {
    if (n % i == 0) return 1;
  }
  return 0;
""",
    )
    add(
        "mbpp3_is_not_prime",
        "MBPP/3",
        "Return 1 if n is composite, otherwise 0. n = 2 is prime and returns 0. "
        "A composite number has a divisor d with 2 <= d <= sqrt(n).",
        sig,
        c,
        [_ret([2], 0), _ret([10], 1), _ret([35], 1), _ret([17], 0), _ret([1], 0), _ret([9], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_differ_one_bit",
        "int64_t a, int64_t b",
        """
  uint64_t x = (uint64_t)a ^ (uint64_t)b;
  if (x == 0) return 0;
  return (x & (x - 1)) == 0 ? 1 : 0;
""",
    )
    # uint64_t needs the preamble; the body uses uint64_t. preamble is added
    # because the source has no #include. Good.
    add(
        "mbpp6_differ_one_bit",
        "MBPP/6",
        "Return 1 if the two's-complement bit patterns of a and b differ in exactly one bit, otherwise 0.",
        sig,
        c,
        [_ret([13, 9], 1), _ret([15, 8], 0), _ret([2, 4], 0), _ret([7, 7], 0), _ret([0, 1], 1)],
    )

    sig, c = _fn(
        "void",
        "mbpp_square_nums",
        "const int64_t *nums, int64_t n, int64_t *out",
        """
  int64_t i;
  for (i = 0; i < n; i++) out[i] = nums[i] * nums[i];
""",
    )
    add(
        "mbpp8_square_nums",
        "MBPP/8",
        "Write the square of each of the n integers in nums into out. n may be 0.",
        sig,
        c,
        [
            _void(
                [_buf("nums", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10, _buf("out", [0] * 10, out=True)],
                {"out": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]},
            ),
            _void(
                [_buf("nums", [10, 20, 30]), 3, _buf("out", [0, 0, 0], out=True)],
                {"out": [100, 400, 900]},
            ),
            _void(
                [_buf("nums", [12, 15]), 2, _buf("out", [0, 0], out=True)],
                {"out": [144, 225]},
            ),
            _void([_buf("nums", []), 0, _buf("out", [], out=True)], {"out": []}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_prism_volume",
        "int64_t length, int64_t base, int64_t height",
        """
  return (length * base * height) / 2;
""",
    )
    add(
        "mbpp14_prism_volume",
        "MBPP/14",
        "Return the integer volume of a triangular prism, (length * base * height) / 2. "
        "The product is even for every test.",
        sig,
        c,
        [_ret([10, 8, 6], 240), _ret([3, 2, 2], 6), _ret([1, 2, 1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_square_perimeter",
        "int64_t side",
        "  return 4 * side;\n",
    )
    add(
        "mbpp17_square_perimeter",
        "MBPP/17",
        "Return the perimeter of a square with the given side length.",
        sig,
        c,
        [_ret([10], 40), _ret([5], 20), _ret([4], 16), _ret([0], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_binary_to_decimal",
        "int64_t binary",
        """
  int64_t decimal = 0;
  int64_t place = 1;
  if (binary < 0) return 0;
  while (binary != 0) {
    int64_t bit = binary % 10;
    decimal += bit * place;
    binary /= 10;
    place *= 2;
  }
  return decimal;
""",
    )
    add(
        "mbpp24_binary_to_decimal",
        "MBPP/24",
        "Interpret the decimal digits of binary as a binary numeral and return its integer value. "
        "binary is non-negative and contains only digits 0 and 1. Example: 1011 is eleven.",
        sig,
        c,
        [_ret([100], 4), _ret([1011], 11), _ret([1101101], 109), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_max_prime_factor",
        "int64_t n",
        """
  int64_t max_prime = -1;
  int64_t i;
  if (n < 2) return -1;
  while ((n % 2) == 0) {
    max_prime = 2;
    n /= 2;
  }
  for (i = 3; i * i <= n; i += 2) {
    while ((n % i) == 0) {
      max_prime = i;
      n /= i;
    }
  }
  if (n > 2) max_prime = n;
  return max_prime;
""",
    )
    add(
        "mbpp32_max_prime_factor",
        "MBPP/32",
        "Return the largest prime factor of n. n is an integer >= 2.",
        sig,
        c,
        [_ret([15], 5), _ret([6], 3), _ret([2], 2), _ret([13195], 29), _ret([2048], 2)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_decimal_to_binary",
        "int64_t n",
        """
  int64_t bits = 0;
  int64_t place = 1;
  if (n < 0) return 0;
  if (n == 0) return 0;
  while (n != 0) {
    bits += (n % 2) * place;
    n /= 2;
    place *= 10;
  }
  return bits;
""",
    )
    add(
        "mbpp33_decimal_to_binary",
        "MBPP/33",
        "Return the binary representation of non-negative n, encoded as a decimal integer "
        "whose digits are 0 and 1. Example: 10 becomes 1010. n = 0 returns 0.",
        sig,
        c,
        [_ret([10], 1010), _ret([1], 1), _ret([20], 10100), _ret([0], 0), _ret([7], 111)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_array_gcd",
        "const int64_t *vals, int64_t n",
        """
  int64_t i, g, t;
  if (n <= 0) return 0;
  g = vals[0];
  if (g < 0) g = -g;
  for (i = 1; i < n; i++) {
    int64_t b = vals[i];
    if (b < 0) b = -b;
    while (b != 0) {
      t = g % b;
      g = b;
      b = t;
    }
  }
  return g;
""",
    )
    add(
        "mbpp45_array_gcd",
        "MBPP/45",
        "Return the greatest common divisor of the n integers in vals. n >= 1. "
        "The result is non-negative.",
        sig,
        c,
        [
            _ret([_buf("vals", [2, 4, 6, 8, 16]), 5], 2),
            _ret([_buf("vals", [1, 2, 3]), 3], 1),
            _ret([_buf("vals", [2, 4, 6, 8]), 4], 2),
            _ret([_buf("vals", [15, 25, 10]), 3], 5),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_fact_div_last_digit",
        "int64_t a, int64_t b",
        """
  int64_t variable = 1;
  int64_t i;
  if (a == b) return 1;
  if ((b - a) >= 5) return 0;
  for (i = a + 1; i <= b; i++) variable = (variable * (i % 10)) % 10;
  return variable % 10;
""",
    )
    add(
        "mbpp47_fact_div_last_digit",
        "MBPP/47",
        "Return the last decimal digit of (b!) / (a!), with 0 <= a <= b. "
        "If b - a >= 5 the product contains a factor of 10, so the digit is 0. "
        "If a == b the value is 1.",
        sig,
        c,
        [_ret([2, 4], 2), _ret([6, 8], 6), _ret([1, 2], 2), _ret([3, 3], 1), _ret([1, 8], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_set_odd_bits",
        "int64_t n",
        """
  int64_t count = 0;
  int64_t res = 0;
  int64_t temp = n;
  if (temp < 0) return n;
  while (temp > 0) {
    if ((count % 2) == 0) res |= (1LL << count);
    count += 1;
    temp >>= 1;
  }
  return n | res;
""",
    )
    add(
        "mbpp48_set_odd_bits",
        "MBPP/48",
        "Return n with every other bit set, starting at bit 0, across the significant width of n. "
        "n is non-negative. Bits past the highest set bit of n are left clear. "
        "Examples: 10 becomes 15, 20 becomes 21, 30 becomes 31.",
        sig,
        c,
        [_ret([10], 15), _ret([20], 21), _ret([30], 31), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_diff_of_squares",
        "int64_t n",
        """
  if ((n % 4) != 2) return 1;
  return 0;
""",
    )
    add(
        "mbpp72_diff_of_squares",
        "MBPP/72",
        "Return 1 if n can be written as the difference of two integer squares, otherwise 0. "
        "That is true exactly when n is not congruent to 2 modulo 4.",
        sig,
        c,
        [_ret([5], 1), _ret([10], 0), _ret([15], 1), _ret([2], 0), _ret([0], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_count_squares",
        "int64_t width, int64_t height",
        """
  int64_t m = width;
  int64_t n = height;
  int64_t t;
  if (n < m) {
    t = m;
    m = n;
    n = t;
  }
  return (m * (m + 1) * (2 * m + 1)) / 6 + ((n - m) * m * (m + 1)) / 2;
""",
    )
    add(
        "mbpp76_count_squares",
        "MBPP/76",
        "Return how many squares (of every size, axis-aligned) are contained in a width-by-height grid of unit cells. "
        "width and height are positive.",
        sig,
        c,
        [_ret([4, 3], 20), _ret([2, 2], 5), _ret([1, 1], 1), _ret([3, 4], 20)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_odd_setbit_count",
        "int64_t n",
        """
  int64_t count = 0;
  int64_t tmp;
  int64_t ans;
  if (n < 0) return 0;
  if ((n % 2) != 0) return (n + 1) / 2;
  tmp = n;
  while (tmp) {
    count += tmp & 1;
    tmp >>= 1;
  }
  ans = n / 2;
  if ((count % 2) != 0) ans += 1;
  return ans;
""",
    )
    add(
        "mbpp78_odd_setbit_count",
        "MBPP/78",
        "Return how many integers in 1..n inclusive have an odd number of set bits. n is non-negative. "
        "The value for n = 0 is 0.",
        sig,
        c,
        [_ret([5], 3), _ret([10], 5), _ret([15], 8), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_next_palindrome",
        "int64_t num",
        """
  int64_t i, x, rev, t;
  for (i = num + 1; i < num + 1000000; i++) {
    x = i;
    rev = 0;
    t = x < 0 ? -x : x;
    while (t > 0) {
      rev = rev * 10 + (t % 10);
      t /= 10;
    }
    if (rev == (i < 0 ? -i : i)) return i;
  }
  return -1;
""",
    )
    add(
        "mbpp100_next_palindrome",
        "MBPP/100",
        "Return the smallest integer strictly greater than num whose decimal representation is a palindrome. "
        "num is non-negative and the answer is within one million of num.",
        sig,
        c,
        [_ret([99], 101), _ret([1221], 1331), _ret([120], 121), _ret([0], 1), _ret([7], 8)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_is_coprime",
        "int64_t x, int64_t y",
        """
  int64_t t;
  if (x < 0) x = -x;
  if (y < 0) y = -y;
  while (y != 0) {
    t = x % y;
    x = y;
    y = t;
  }
  return x == 1 ? 1 : 0;
""",
    )
    add(
        "mbpp151_is_coprime",
        "MBPP/151",
        "Return 1 if x and y are coprime (their greatest common divisor is 1), otherwise 0.",
        sig,
        c,
        [_ret([17, 13], 1), _ret([15, 21], 0), _ret([25, 45], 0), _ret([1, 8], 1), _ret([8, 1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_toggle_even_bits",
        "int64_t n",
        """
  int64_t res = 0;
  int64_t count = 0;
  int64_t temp = n;
  if (temp < 0) return n;
  while (temp > 0) {
    if ((count % 2) == 1) res |= (1LL << count);
    count += 1;
    temp >>= 1;
  }
  return n ^ res;
""",
    )
    add(
        "mbpp155_toggle_even_bits",
        "MBPP/155",
        "Toggle every other bit of non-negative n, starting at bit 1 (the bit worth 2), "
        "and only within the significant width of n. Examples: 10 becomes 0, 20 becomes 30, 30 becomes 20.",
        sig,
        c,
        [_ret([10], 0), _ret([20], 30), _ret([30], 20), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_count_set_bits",
        "int64_t n",
        """
  int64_t count = 0;
  if (n < 0) return 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
""",
    )
    add(
        "mbpp224_count_set_bits",
        "MBPP/224",
        "Return the number of set bits in non-negative n (the population count).",
        sig,
        c,
        [_ret([2], 1), _ret([4], 1), _ret([6], 2), _ret([0], 0), _ret([255], 8)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_cube_volume",
        "int64_t side",
        "  return side * side * side;\n",
    )
    add(
        "mbpp234_cube_volume",
        "MBPP/234",
        "Return the volume of a cube with the given integer side length.",
        sig,
        c,
        [_ret([3], 27), _ret([2], 8), _ret([5], 125), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_next_square",
        "int64_t n",
        """
  int64_t root = 0;
  if (n < 0) n = 0;
  while ((root + 1) * (root + 1) <= n) root++;
  return (root + 1) * (root + 1);
""",
    )
    add(
        "mbpp244_next_square",
        "MBPP/244",
        "Return the smallest perfect square strictly greater than n. n is non-negative and at most 1000000.",
        sig,
        c,
        [_ret([35], 36), _ret([6], 9), _ret([9], 16), _ret([0], 1), _ret([1], 4)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_count_primes",
        "int64_t n",
        """
  int64_t ctr = 0;
  int64_t num, i, prime;
  for (num = 2; num < n; num++) {
    prime = 1;
    for (i = 2; i * i <= num; i++) {
      if ((num % i) == 0) {
        prime = 0;
        break;
      }
    }
    if (prime) ctr++;
  }
  return ctr;
""",
    )
    add(
        "mbpp256_count_primes",
        "MBPP/256",
        "Return how many prime numbers are strictly less than n. n is non-negative.",
        sig,
        c,
        [_ret([5], 2), _ret([10], 4), _ret([100], 25), _ret([2], 0), _ret([1], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_sum_odd_squares",
        "int64_t n",
        """
  if (n < 0) return 0;
  return (n * (4 * n * n - 1)) / 3;
""",
    )
    add(
        "mbpp267_sum_odd_squares",
        "MBPP/267",
        "Return the sum of the squares of the first n odd natural numbers: 1^2 + 3^2 + ... + (2n-1)^2. "
        "n is non-negative. The closed form is n * (4n^2 - 1) / 3.",
        sig,
        c,
        [_ret([2], 10), _ret([3], 35), _ret([4], 84), _ret([0], 0), _ret([1], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "mbpp_sum_even_squares",
        "int64_t n",
        """
  if (n < 0) return 0;
  return (2 * n * (n + 1) * (2 * n + 1)) / 3;
""",
    )
    add(
        "mbpp287_sum_even_squares",
        "MBPP/287",
        "Return the sum of the squares of the first n even natural numbers: 2^2 + 4^2 + ... + (2n)^2. "
        "n is non-negative. The closed form is 2 * n * (n + 1) * (2n + 1) / 3.",
        sig,
        c,
        [_ret([2], 20), _ret([3], 56), _ret([4], 120), _ret([0], 0), _ret([1], 4)],
    )

    # --- HumanEval -----------------------------------------------------------
    sig, c = _fn(
        "int64_t",
        "he_below_zero",
        "const int64_t *ops, int64_t n",
        """
  int64_t bal = 0;
  int64_t i;
  for (i = 0; i < n; i++) {
    bal += ops[i];
    if (bal < 0) return 1;
  }
  return 0;
""",
    )
    add(
        "he3_below_zero",
        "HumanEval/3",
        "ops is a sequence of n deposits (positive) and withdrawals (negative) applied to a balance that starts at 0. "
        "Return 1 if the balance ever becomes negative, otherwise 0.",
        sig,
        c,
        [
            _ret([_buf("ops", [1, 2, 3]), 3], 0),
            _ret([_buf("ops", [1, 2, -4, 5]), 4], 1),
            _ret([_buf("ops", []), 0], 0),
            _ret([_buf("ops", [-1]), 1], 1),
            _ret([_buf("ops", [5, -5, 1]), 3], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_sum_product",
        "const int64_t *nums, int64_t n, int64_t *product",
        """
  int64_t i, s = 0, p = 1;
  for (i = 0; i < n; i++) {
    s += nums[i];
    p *= nums[i];
  }
  *product = p;
  return s;
""",
    )
    add(
        "he8_sum_product",
        "HumanEval/8",
        "Return the sum of the n integers in nums, and write their product into *product. "
        "The empty sequence has sum 0 and product 1.",
        sig,
        c,
        [
            _ret([_buf("nums", []), 0, _buf("product", [0], out=True)], 0, {"product": [1]}),
            _ret([_buf("nums", [1, 2, 3, 4]), 4, _buf("product", [0], out=True)], 10, {"product": [24]}),
            _ret([_buf("nums", [5]), 1, _buf("product", [0], out=True)], 5, {"product": [5]}),
            _ret([_buf("nums", [-2, 3]), 2, _buf("product", [0], out=True)], 1, {"product": [-6]}),
        ],
    )

    sig, c = _fn(
        "void",
        "he_rolling_max",
        "const int64_t *nums, int64_t n, int64_t *out",
        """
  int64_t i, m;
  if (n <= 0) return;
  m = nums[0];
  out[0] = m;
  for (i = 1; i < n; i++) {
    if (nums[i] > m) m = nums[i];
    out[i] = m;
  }
""",
    )
    add(
        "he9_rolling_max",
        "HumanEval/9",
        "Write into out the rolling maximum of nums: out[i] is the maximum of nums[0..i]. n may be 0.",
        sig,
        c,
        [
            _void(
                [_buf("nums", [1, 2, 3, 2, 3, 4, 2]), 7, _buf("out", [0] * 7, out=True)],
                {"out": [1, 2, 3, 3, 3, 4, 4]},
            ),
            _void(
                [_buf("nums", [-2, -1, -3]), 3, _buf("out", [0, 0, 0], out=True)],
                {"out": [-2, -1, -1]},
            ),
            _void([_buf("nums", []), 0, _buf("out", [], out=True)], {"out": []}),
        ],
    )

    sig, c = _fn(
        "void",
        "he_string_xor",
        "const char *a, const char *b, char *out",
        """
  int64_t i = 0;
  for (; a[i] && b[i]; i++) {
    out[i] = (a[i] == b[i]) ? '0' : '1';
  }
  out[i] = 0;
""",
    )
    add(
        "he11_string_xor",
        "HumanEval/11",
        "a and b are equally long strings of the characters '0' and '1'. "
        "Write their bitwise XOR, as a string of '0' and '1' of the same length, into out, and terminate it with a NUL.",
        sig,
        c,
        [
            _void(
                [_buf("a", "010", ptr="cstr"), _buf("b", "110", ptr="cstr"), _buf("out", [0, 0, 0, 0], out=True, ptr="cstr")],
                {"out": "100"},
            ),
            _void(
                [_buf("a", "000", ptr="cstr"), _buf("b", "000", ptr="cstr"), _buf("out", [0, 0, 0, 0], out=True, ptr="cstr")],
                {"out": "000"},
            ),
            _void(
                [_buf("a", "1", ptr="cstr"), _buf("b", "1", ptr="cstr"), _buf("out", [0, 0], out=True, ptr="cstr")],
                {"out": "0"},
            ),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_gcd",
        "int64_t a, int64_t b",
        """
  int64_t t;
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    t = a % b;
    a = b;
    b = t;
  }
  return a;
""",
    )
    add(
        "he13_gcd",
        "HumanEval/13",
        "Return the greatest common divisor of the integers a and b. The result is non-negative. gcd(0, 0) is 0.",
        sig,
        c,
        [_ret([3, 5], 1), _ret([25, 15], 5), _ret([0, 5], 5), _ret([0, 0], 0), _ret([-4, 6], 2)],
    )

    sig, c = _fn(
        "int64_t",
        "he_count_distinct_chars",
        "const char *s",
        """
  unsigned char seen[256];
  int64_t i, n = 0;
  for (i = 0; i < 256; i++) seen[i] = 0;
  for (i = 0; s[i]; i++) {
    unsigned char c = (unsigned char)s[i];
    if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
    if (!seen[c]) {
      seen[c] = 1;
      n++;
    }
  }
  return n;
""",
    )
    add(
        "he16_count_distinct_chars",
        "HumanEval/16",
        "Return how many distinct characters are in s, ignoring ASCII case. 'a' and 'A' count as one character.",
        sig,
        c,
        [
            _ret([_buf("s", "xyzXYZ", ptr="cstr")], 3),
            _ret([_buf("s", "Jerry", ptr="cstr")], 4),
            _ret([_buf("s", "", ptr="cstr")], 0),
            _ret([_buf("s", "aaa", ptr="cstr")], 1),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_how_many_times",
        "const char *string, const char *sub",
        """
  int64_t n = 0, i, j, ls = 0, lsub = 0;
  if (!sub[0]) return 0;
  while (string[ls]) ls++;
  while (sub[lsub]) lsub++;
  for (i = 0; i + lsub <= ls; i++) {
    for (j = 0; j < lsub; j++) {
      if (string[i + j] != sub[j]) break;
    }
    if (j == lsub) n++;
  }
  return n;
""",
    )
    add(
        "he18_how_many_times",
        "HumanEval/18",
        "Return how many times sub occurs in string, counting overlapping occurrences. "
        "An empty sub occurs 0 times.",
        sig,
        c,
        [
            _ret([_buf("s", "", ptr="cstr"), _buf("sub", "a", ptr="cstr")], 0),
            _ret([_buf("s", "aaa", ptr="cstr"), _buf("sub", "a", ptr="cstr")], 3),
            _ret([_buf("s", "aaaa", ptr="cstr"), _buf("sub", "aa", ptr="cstr")], 3),
            _ret([_buf("s", "abc", ptr="cstr"), _buf("sub", "", ptr="cstr")], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_strlen",
        "const char *string",
        """
  int64_t n = 0;
  while (string[n]) n++;
  return n;
""",
    )
    add(
        "he23_strlen",
        "HumanEval/23",
        "Return the number of bytes in the NUL-terminated string, not counting the NUL.",
        sig,
        c,
        [
            _ret([_buf("s", "", ptr="cstr")], 0),
            _ret([_buf("s", "abc", ptr="cstr")], 3),
            _ret([_buf("s", "hello", ptr="cstr")], 5),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_largest_divisor",
        "int64_t n",
        """
  int64_t d;
  if (n <= 1) return 0;
  for (d = 2; d * d <= n; d++) {
    if ((n % d) == 0) return n / d;
  }
  return 1;
""",
    )
    add(
        "he24_largest_divisor",
        "HumanEval/24",
        "Return the largest positive divisor of n that is strictly smaller than n. n is an integer >= 2.",
        sig,
        c,
        [_ret([15], 5), _ret([2], 1), _ret([49], 7), _ret([12], 6), _ret([9], 3)],
    )

    sig, c = _fn(
        "int64_t",
        "he_factorize",
        "int64_t n, int64_t *out",
        """
  int64_t m = 0;
  int64_t p;
  if (n < 2) return 0;
  while ((n % 2) == 0) {
    out[m++] = 2;
    n /= 2;
  }
  for (p = 3; p * p <= n; p += 2) {
    while ((n % p) == 0) {
      out[m++] = p;
      n /= p;
    }
  }
  if (n > 1) out[m++] = n;
  return m;
""",
    )
    add(
        "he25_factorize",
        "HumanEval/25",
        "Write the prime factors of n into out, smallest first, repeating a factor as many times as it divides n. "
        "Return how many factors were written. n >= 2. out has room for at least 16 factors. Unused slots stay 0.",
        sig,
        c,
        [
            _ret([8, _buf("out", [0] * 8, out=True)], 3, {"out": [2, 2, 2, 0, 0, 0, 0, 0]}),
            _ret([25, _buf("out", [0] * 8, out=True)], 2, {"out": [5, 5, 0, 0, 0, 0, 0, 0]}),
            _ret([70, _buf("out", [0] * 8, out=True)], 3, {"out": [2, 5, 7, 0, 0, 0, 0, 0]}),
            _ret([13, _buf("out", [0] * 4, out=True)], 1, {"out": [13, 0, 0, 0]}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_remove_duplicates",
        "const int64_t *nums, int64_t n, int64_t *out",
        """
  int64_t i, j, m = 0, c;
  for (i = 0; i < n; i++) {
    c = 0;
    for (j = 0; j < n; j++) if (nums[j] == nums[i]) c++;
    if (c == 1) out[m++] = nums[i];
  }
  return m;
""",
    )
    add(
        "he26_remove_duplicates",
        "HumanEval/26",
        "Copy into out the values of nums that occur exactly once, preserving order. "
        "Return how many values were copied. Slots past that count are left unchanged.",
        sig,
        c,
        [
            _ret(
                [_buf("nums", [1, 2, 3, 2, 4]), 5, _buf("out", [-1, -1, -1, -1, -1], out=True)],
                3,
                {"out": [1, 3, 4, -1, -1]},
            ),
            _ret(
                [_buf("nums", [1, 1, 1]), 3, _buf("out", [7, 7, 7], out=True)],
                0,
                {"out": [7, 7, 7]},
            ),
            _ret(
                [_buf("nums", [5]), 1, _buf("out", [0], out=True)],
                1,
                {"out": [5]},
            ),
        ],
    )

    sig, c = _fn(
        "void",
        "he_flip_case",
        "const char *s, char *out",
        """
  int64_t i;
  for (i = 0; s[i]; i++) {
    char ch = s[i];
    if (ch >= 'a' && ch <= 'z') out[i] = (char)(ch - 'a' + 'A');
    else if (ch >= 'A' && ch <= 'Z') out[i] = (char)(ch - 'A' + 'a');
    else out[i] = ch;
  }
  out[i] = 0;
""",
    )
    add(
        "he27_flip_case",
        "HumanEval/27",
        "Write into out a copy of s with ASCII lowercase and uppercase letters swapped. "
        "Other bytes are copied unchanged. out is NUL-terminated and has room for the whole string.",
        sig,
        c,
        [
            _void(
                [_buf("s", "Hello", ptr="cstr"), _buf("out", [0] * 6, out=True, ptr="cstr")],
                {"out": "hELLO"},
            ),
            _void(
                [_buf("s", "", ptr="cstr"), _buf("out", [0], out=True, ptr="cstr")],
                {"out": ""},
            ),
            _void(
                [_buf("s", "a1B", ptr="cstr"), _buf("out", [0, 0, 0, 0], out=True, ptr="cstr")],
                {"out": "A1b"},
            ),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_fizz_buzz",
        "int64_t n",
        """
  int64_t i, count = 0, x;
  for (i = 1; i < n; i++) {
    if ((i % 11) != 0 && (i % 13) != 0) continue;
    x = i;
    while (x > 0) {
      if ((x % 10) == 7) count++;
      x /= 10;
    }
  }
  return count;
""",
    )
    add(
        "he36_fizz_buzz",
        "HumanEval/36",
        "Return how many times the digit 7 appears among the positive integers strictly less than n "
        "that are divisible by 11 or by 13.",
        sig,
        c,
        [_ret([50], 0), _ret([78], 2), _ret([79], 3), _ret([1], 0), _ret([0], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_car_race_collision",
        "int64_t n",
        "  return n * n;\n",
    )
    add(
        "he41_car_race_collision",
        "HumanEval/41",
        "n cars drive left to right and n cars drive right to left on a straight line, at the same speed. "
        "Return how many pairs of cars collide. Cars that have already collided continue, so each left-going "
        "car meets every right-going car. The answer is n * n.",
        sig,
        c,
        [_ret([0], 0), _ret([1], 1), _ret([2], 4), _ret([5], 25), _ret([10], 100)],
    )

    sig, c = _fn(
        "int64_t",
        "he_pairs_sum_to_zero",
        "const int64_t *vals, int64_t n",
        """
  int64_t i, j;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      if (vals[i] + vals[j] == 0) return 1;
    }
  }
  return 0;
""",
    )
    add(
        "he43_pairs_sum_to_zero",
        "HumanEval/43",
        "Return 1 if two distinct elements of vals sum to 0, otherwise 0.",
        sig,
        c,
        [
            _ret([_buf("v", [1, 3, 5, 0]), 4], 0),
            _ret([_buf("v", [1, 3, -2, 1]), 4], 0),
            _ret([_buf("v", [2, 4, -5, 3, 5, 7]), 6], 1),
            _ret([_buf("v", [1]), 1], 0),
            _ret([_buf("v", [2, -2]), 2], 1),
        ],
    )

    sig, c = _fn(
        "double",
        "he_triangle_area",
        "double base, double height",
        "  return base * height / 2.0;\n",
    )
    add(
        "he45_triangle_area",
        "HumanEval/45",
        "Return the area of a triangle with the given base and height, base * height / 2.",
        sig,
        c,
        [_ret([5.0, 3.0], 7.5), _ret([2.0, 2.0], 2.0), _ret([0.0, 10.0], 0.0), _ret([8.0, 1.0], 4.0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_is_palindrome",
        "const char *text",
        """
  int64_t i = 0, j = 0;
  while (text[j]) j++;
  if (j == 0) return 1;
  j--;
  while (i < j) {
    if (text[i] != text[j]) return 0;
    i++;
    j--;
  }
  return 1;
""",
    )
    add(
        "he48_is_palindrome",
        "HumanEval/48",
        "Return 1 if text is a palindrome (it equals its reverse), otherwise 0. The empty string is a palindrome.",
        sig,
        c,
        [
            _ret([_buf("t", "", ptr="cstr")], 1),
            _ret([_buf("t", "aba", ptr="cstr")], 1),
            _ret([_buf("t", "aaaaa", ptr="cstr")], 1),
            _ret([_buf("t", "zbcd", ptr="cstr")], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_modp",
        "int64_t n, int64_t p",
        """
  int64_t r = 1;
  int64_t i;
  if (p <= 0) return 0;
  for (i = 0; i < n; i++) r = (r * 2) % p;
  return r;
""",
    )
    add(
        "he49_modp",
        "HumanEval/49",
        "Return 2^n modulo p. n is non-negative and p is positive. Compute the power by repeated doubling modulo p.",
        sig,
        c,
        [_ret([3, 5], 3), _ret([1101, 101], 2), _ret([0, 101], 1), _ret([3, 11], 8), _ret([100, 101], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "he_correct_bracketing",
        "const char *brackets",
        """
  int64_t depth = 0, i;
  for (i = 0; brackets[i]; i++) {
    if (brackets[i] == '<') depth++;
    else if (brackets[i] == '>') {
      depth--;
      if (depth < 0) return 0;
    }
  }
  return depth == 0 ? 1 : 0;
""",
    )
    add(
        "he56_correct_bracketing",
        "HumanEval/56",
        "brackets is a string of '<' and '>'. Return 1 if it is a correctly matched bracket sequence, otherwise 0. "
        "Every '>' closes a previous unmatched '<', and every '<' is closed.",
        sig,
        c,
        [
            _ret([_buf("b", "<", ptr="cstr")], 0),
            _ret([_buf("b", "<>", ptr="cstr")], 1),
            _ret([_buf("b", "<<><>>", ptr="cstr")], 1),
            _ret([_buf("b", "><<>", ptr="cstr")], 0),
            _ret([_buf("b", "", ptr="cstr")], 1),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_largest_prime_factor",
        "int64_t n",
        """
  int64_t factor = 2;
  int64_t last = 2;
  if (n < 2) return 0;
  while (n > 1) {
    if ((n % factor) == 0) {
      last = factor;
      n /= factor;
    } else {
      factor++;
    }
  }
  return last;
""",
    )
    add(
        "he59_largest_prime_factor",
        "HumanEval/59",
        "Return the largest prime factor of n. n is an integer >= 2.",
        sig,
        c,
        [_ret([13195], 29), _ret([2048], 2), _ret([15], 5), _ret([49], 7)],
    )

    sig, c = _fn(
        "int64_t",
        "he_sum_to_n",
        "int64_t n",
        """
  if (n < 1) return 0;
  return n * (n + 1) / 2;
""",
    )
    add(
        "he60_sum_to_n",
        "HumanEval/60",
        "Return the sum of the integers from 1 through n inclusive. If n < 1, return 0.",
        sig,
        c,
        [_ret([30], 465), _ret([100], 5050), _ret([5], 15), _ret([10], 55), _ret([1], 1), _ret([0], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_fibfib",
        "int64_t n",
        """
  int64_t a = 0, b = 0, c = 1, i, nxt;
  if (n == 0 || n == 1) return 0;
  if (n == 2) return 1;
  for (i = 3; i <= n; i++) {
    nxt = a + b + c;
    a = b;
    b = c;
    c = nxt;
  }
  return c;
""",
    )
    add(
        "he63_fibfib",
        "HumanEval/63",
        "Return the n-th FibFib number. FibFib(0) = 0, FibFib(1) = 0, FibFib(2) = 1, and "
        "FibFib(k) = FibFib(k-1) + FibFib(k-2) + FibFib(k-3) for k > 2. n is non-negative and small enough that the value fits in int64.",
        sig,
        c,
        [_ret([1], 0), _ret([5], 4), _ret([8], 24), _ret([0], 0), _ret([2], 1), _ret([3], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "he_vowels_count",
        "const char *s",
        """
  int64_t n = 0, i, last = -1;
  for (i = 0; s[i]; i++) {
    char ch = s[i];
    if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
        ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') n++;
    last = i;
  }
  if (last >= 0 && (s[last] == 'y' || s[last] == 'Y')) n++;
  return n;
""",
    )
    add(
        "he64_vowels_count",
        "HumanEval/64",
        "Return the number of vowels in s. Vowels are a, e, i, o, u in either case. "
        "A trailing y or Y also counts as a vowel. s is non-empty.",
        sig,
        c,
        [
            _ret([_buf("s", "abcde", ptr="cstr")], 2),
            _ret([_buf("s", "ACEDY", ptr="cstr")], 3),
            _ret([_buf("s", "y", ptr="cstr")], 1),
            _ret([_buf("s", "by", ptr="cstr")], 1),
            _ret([_buf("s", "rhythm", ptr="cstr")], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_digit_sum_upper",
        "const char *s",
        """
  int64_t sum = 0, i;
  for (i = 0; s[i]; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z') sum += (unsigned char)s[i];
  }
  return sum;
""",
    )
    add(
        "he66_digit_sum_upper",
        "HumanEval/66",
        "Return the sum of the ASCII codes of the uppercase letters in s. Other characters contribute 0. "
        "The empty string returns 0.",
        sig,
        c,
        [
            _ret([_buf("s", "", ptr="cstr")], 0),
            _ret([_buf("s", "abAB", ptr="cstr")], 65 + 66),
            _ret([_buf("s", "abc", ptr="cstr")], 0),
            _ret([_buf("s", "Z", ptr="cstr")], 90),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_will_it_fly",
        "const int64_t *q, int64_t n, int64_t w",
        """
  int64_t i, j, sum = 0;
  for (i = 0; i < n; i++) sum += q[i];
  if (sum > w) return 0;
  i = 0;
  j = n - 1;
  while (i < j) {
    if (q[i] != q[j]) return 0;
    i++;
    j--;
  }
  return 1;
""",
    )
    add(
        "he72_will_it_fly",
        "HumanEval/72",
        "Return 1 if q is a palindrome and the sum of its n elements is at most w, otherwise 0. "
        "An empty sequence is a palindrome with sum 0.",
        sig,
        c,
        [
            _ret([_buf("q", [1, 2, 3, 2, 1]), 5, 9], 1),
            _ret([_buf("q", [1, 2, 3, 2, 1]), 5, 8], 0),
            _ret([_buf("q", [1, 2, 3]), 3, 100], 0),
            _ret([_buf("q", []), 0, 0], 1),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_smallest_change",
        "const int64_t *arr, int64_t n",
        """
  int64_t ans = 0, i;
  for (i = 0; i < n / 2; i++) {
    if (arr[i] != arr[n - 1 - i]) ans++;
  }
  return ans;
""",
    )
    add(
        "he73_smallest_change",
        "HumanEval/73",
        "Return the minimum number of element replacements needed to make arr a palindrome. "
        "That is the number of index pairs (i, n-1-i) with i < n/2 whose values differ.",
        sig,
        c,
        [
            _ret([_buf("a", [1, 2, 3, 5, 4, 7, 9, 6]), 8], 4),
            _ret([_buf("a", [1, 2, 3, 4, 3, 2, 2]), 7], 1),
            _ret([_buf("a", [1, 2, 1]), 3], 0),
            _ret([_buf("a", []), 0], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_is_multiply_prime",
        "int64_t a",
        """
  int64_t count = 0;
  int64_t p;
  if (a <= 1) return 0;
  /* Same predicate as searching primes <= 100 whose product is a:
     exactly three prime factors, counting multiplicity, each <= 100. */
  while ((a % 2) == 0) {
    count++;
    a /= 2;
    if (count > 3) return 0;
  }
  for (p = 3; p <= a && p <= 100; p += 2) {
    while ((a % p) == 0) {
      count++;
      a /= p;
      if (count > 3) return 0;
    }
  }
  if (a != 1) return 0;
  return count == 3 ? 1 : 0;
""",
    )
    add(
        "he75_is_multiply_prime",
        "HumanEval/75",
        "Return 1 if a is the product of three primes (repetition allowed), each prime at most 100, otherwise 0. "
        "Order does not matter. Examples: 30 = 2*3*5 returns 1, 8 = 2*2*2 returns 1, 6 returns 0, 1 returns 0.",
        sig,
        c,
        [_ret([30], 1), _ret([8], 1), _ret([6], 0), _ret([1], 0), _ret([12], 1), _ret([105], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "he_is_simple_power",
        "int64_t x, int64_t n",
        """
  int64_t power = 1;
  if (n == 1) return x == 1 ? 1 : 0;
  if (n == 0) return x == 1 ? 1 : 0;
  if (n == -1) return (x == 1 || x == -1) ? 1 : 0;
  if (n < -1 || x < 0) return 0;
  while (power < x) {
    if (power > (9223372036854775807LL / n)) return 0;
    power *= n;
  }
  return power == x ? 1 : 0;
""",
    )
    add(
        "he76_is_simple_power",
        "HumanEval/76",
        "Return 1 if x is some non-negative integer power of n, including n^0 = 1. "
        "If n is 1, return 1 only when x is 1. x and n are small enough that the power fits in int64 when it is equal to x.",
        sig,
        c,
        [_ret([1, 4], 1), _ret([8, 2], 1), _ret([3, 2], 0), _ret([1, 1], 1), _ret([4, 1], 0), _ret([16, 2], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "he_iscube",
        "int64_t a",
        """
  int64_t r = 0;
  if (a < 0) a = -a;
  while ((r + 1) <= 2642245 && (r + 1) * (r + 1) * (r + 1) <= a) r++;
  return (r * r * r == a) ? 1 : 0;
""",
    )
    add(
        "he77_iscube",
        "HumanEval/77",
        "Return 1 if a is a perfect cube of an integer (negative bases count: -8 is a cube), otherwise 0.",
        sig,
        c,
        [_ret([1], 1), _ret([8], 1), _ret([0], 1), _ret([2], 0), _ret([-8], 1), _ret([27], 1), _ret([9], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_hex_key",
        "const char *num",
        """
  int64_t total = 0, i;
  for (i = 0; num[i]; i++) {
    char c = num[i];
    if (c == '2' || c == '3' || c == '5' || c == '7' || c == 'B' || c == 'D') total++;
  }
  return total;
""",
    )
    add(
        "he78_hex_key",
        "HumanEval/78",
        "num is a hexadecimal string. Return how many of its characters are prime hexadecimal digits. "
        "Those digits are 2, 3, 5, 7, B, and D.",
        sig,
        c,
        [
            _ret([_buf("n", "AB", ptr="cstr")], 1),
            _ret([_buf("n", "1077E", ptr="cstr")], 2),
            _ret([_buf("n", "ABED1A33", ptr="cstr")], 4),
            _ret([_buf("n", "", ptr="cstr")], 0),
            _ret([_buf("n", "123456789ABCDEF", ptr="cstr")], 6),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_prime_length",
        "const char *string",
        """
  int64_t n = 0, i;
  while (string[n]) n++;
  if (n <= 1) return 0;
  for (i = 2; i * i <= n; i++) if ((n % i) == 0) return 0;
  return 1;
""",
    )
    add(
        "he82_prime_length",
        "HumanEval/82",
        "Return 1 if the length of string is a prime number, otherwise 0. Length 0 and 1 are not prime.",
        sig,
        c,
        [
            _ret([_buf("s", "Hello", ptr="cstr")], 1),
            _ret([_buf("s", "a", ptr="cstr")], 0),
            _ret([_buf("s", "", ptr="cstr")], 0),
            _ret([_buf("s", "ab", ptr="cstr")], 1),
            _ret([_buf("s", "abc", ptr="cstr")], 1),
            _ret([_buf("s", "orange", ptr="cstr")], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_starts_one_ends",
        "int64_t n",
        """
  int64_t p = 1, i;
  if (n <= 1) return n == 1 ? 1 : 0;
  for (i = 0; i < n - 2; i++) p *= 10;
  return 18 * p;
""",
    )
    add(
        "he83_starts_one_ends",
        "HumanEval/83",
        "Return how many n-digit positive integers start with 1 or end with 1 (or both). "
        "n = 1 returns 1. n >= 2 returns 18 * 10^(n-2). n is between 1 and 12.",
        sig,
        c,
        [_ret([1], 1), _ret([2], 18), _ret([3], 180), _ret([4], 1800)],
    )

    sig, c = _fn(
        "int64_t",
        "he_sum_even_odd_idx",
        "const int64_t *lst, int64_t n",
        """
  int64_t s = 0, i;
  for (i = 1; i < n; i += 2) {
    if ((lst[i] % 2) == 0) s += lst[i];
  }
  return s;
""",
    )
    add(
        "he85_sum_even_odd_idx",
        "HumanEval/85",
        "Return the sum of the even elements of lst that sit at odd indexes. Indexes start at 0. "
        "Negative even numbers are even.",
        sig,
        c,
        [
            _ret([_buf("a", [4, 2, 6, 7]), 4], 2),
            _ret([_buf("a", [1, 2, 3, 4, 5]), 5], 6),
            _ret([_buf("a", [2]), 1], 0),
            _ret([_buf("a", []), 0], 0),
            _ret([_buf("a", [1, -2, 3, -4]), 4], -6),
        ],
    )

    sig, c = _fn(
        "void",
        "he_encrypt",
        "const char *s, char *out",
        """
  int64_t i;
  for (i = 0; s[i]; i++) {
    char ch = s[i];
    if (ch >= 'a' && ch <= 'z') out[i] = (char)('a' + (ch - 'a' + 4) % 26);
    else out[i] = ch;
  }
  out[i] = 0;
""",
    )
    add(
        "he89_encrypt",
        "HumanEval/89",
        "Write into out the string s with each lowercase letter shifted forward by 4 in the alphabet, wrapping from z to a. "
        "Other characters are copied unchanged. out is NUL-terminated.",
        sig,
        c,
        [
            _void(
                [_buf("s", "hi", ptr="cstr"), _buf("out", [0, 0, 0], out=True, ptr="cstr")],
                {"out": "lm"},
            ),
            _void(
                [_buf("s", "asdfghjkl", ptr="cstr"), _buf("out", [0] * 10, out=True, ptr="cstr")],
                {"out": "ewhjklnop"},
            ),
            _void(
                [_buf("s", "gf", ptr="cstr"), _buf("out", [0, 0, 0], out=True, ptr="cstr")],
                {"out": "kj"},
            ),
            _void(
                [_buf("s", "A z", ptr="cstr"), _buf("out", [0, 0, 0, 0], out=True, ptr="cstr")],
                {"out": "A d"},
            ),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_any_int",
        "int64_t x, int64_t y, int64_t z",
        """
  if ((x + y == z) || (x + z == y) || (y + z == x)) return 1;
  return 0;
""",
    )
    add(
        "he92_any_int",
        "HumanEval/92",
        "Return 1 if one of the three integers is the sum of the other two, otherwise 0. All three arguments are integers.",
        sig,
        c,
        [_ret([2, 3, 1], 1), _ret([3, 2, 2], 0), _ret([3, -2, 1], 1), _ret([5, 5, 10], 1), _ret([1, 1, 1], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_multiply_units",
        "int64_t a, int64_t b",
        """
  int64_t da = a % 10;
  int64_t db = b % 10;
  if (da < 0) da = -da;
  if (db < 0) db = -db;
  return da * db;
""",
    )
    add(
        "he97_multiply_units",
        "HumanEval/97",
        "Return the product of the last decimal digits of a and b. Both arguments are non-negative in the tests, "
        "and the last digit is a % 10.",
        sig,
        c,
        [_ret([148, 412], 16), _ret([19, 28], 72), _ret([2020, 1851], 0), _ret([14, 15], 20), _ret([0, 9], 0)],
    )

    sig, c = _fn(
        "void",
        "he_make_a_pile",
        "int64_t n, int64_t *out",
        """
  int64_t i;
  for (i = 0; i < n; i++) out[i] = n + 2 * i;
""",
    )
    add(
        "he100_make_a_pile",
        "HumanEval/100",
        "Write n levels of a pile into out. Level 0 has n stones, and each next level has two more stones than the previous one. "
        "n is positive.",
        sig,
        c,
        [
            _void([3, _buf("out", [0, 0, 0], out=True)], {"out": [3, 5, 7]}),
            _void([1, _buf("out", [0], out=True)], {"out": [1]}),
            _void([4, _buf("out", [0, 0, 0, 0], out=True)], {"out": [4, 6, 8, 10]}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_choose_num",
        "int64_t x, int64_t y",
        """
  if (x > y) return -1;
  if ((y % 2) == 0) return y;
  if (x == y) return -1;
  return y - 1;
""",
    )
    add(
        "he102_choose_num",
        "HumanEval/102",
        "Return the largest even integer in the closed range from x to y. If x > y, or the range contains no even integer, return -1.",
        sig,
        c,
        [_ret([12, 15], 14), _ret([13, 12], -1), _ret([2, 2], 2), _ret([3, 3], -1), _ret([1, 4], 4)],
    )

    sig, c = _fn(
        "int64_t",
        "he_unique_digits",
        "const int64_t *xs, int64_t n, int64_t *out",
        """
  int64_t i, m = 0, x, odd, d;
  for (i = 0; i < n; i++) {
    x = xs[i];
    if (x < 0) continue;
    odd = 1;
    if (x == 0) odd = 0;
    while (x > 0) {
      d = x % 10;
      if ((d % 2) == 0) odd = 0;
      x /= 10;
    }
    if (odd) out[m++] = xs[i];
  }
  return m;
""",
    )
    add(
        "he104_unique_digits",
        "HumanEval/104",
        "Copy into out, in order, the non-negative values of xs whose decimal digits are all odd. "
        "Return how many were copied. Slots past that count stay at the initial value -1. Zero is not all-odd.",
        sig,
        c,
        [
            _ret(
                [_buf("xs", [15, 33, 1422, 1]), 4, _buf("out", [-1, -1, -1, -1], out=True)],
                3,
                {"out": [15, 33, 1, -1]},
            ),
            _ret(
                [_buf("xs", [152, 323, 1422, 10]), 4, _buf("out", [-1, -1, -1, -1], out=True)],
                0,
                {"out": [-1, -1, -1, -1]},
            ),
            _ret(
                [_buf("xs", [0, 7]), 2, _buf("out", [-1, -1], out=True)],
                1,
                {"out": [7, -1]},
            ),
        ],
    )

    sig, c = _fn(
        "void",
        "he_f_factorial_or_sum",
        "int64_t n, int64_t *out",
        """
  int64_t i, j, x;
  for (i = 1; i <= n; i++) {
    if ((i % 2) == 0) {
      x = 1;
      for (j = 1; j <= i; j++) x *= j;
    } else {
      x = 0;
      for (j = 1; j <= i; j++) x += j;
    }
    out[i - 1] = x;
  }
""",
    )
    add(
        "he106_f_factorial_or_sum",
        "HumanEval/106",
        "Write n numbers into out. For i from 1 to n, out[i-1] is i! when i is even, and the sum 1+...+i when i is odd. "
        "n is between 0 and 12 so the factorials fit in int64.",
        sig,
        c,
        [
            _void([1, _buf("out", [0], out=True)], {"out": [1]}),
            _void([5, _buf("out", [0, 0, 0, 0, 0], out=True)], {"out": [1, 2, 6, 24, 15]}),
            _void([0, _buf("out", [], out=True)], {"out": []}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_min_subarray_sum",
        "const int64_t *nums, int64_t n",
        """
  int64_t max_sum = 0, s = 0, i, best_neg;
  if (n <= 0) return 0;
  for (i = 0; i < n; i++) {
    s += -nums[i];
    if (s < 0) s = 0;
    if (s > max_sum) max_sum = s;
  }
  if (max_sum == 0) {
    best_neg = -nums[0];
    for (i = 1; i < n; i++) if (-nums[i] > best_neg) best_neg = -nums[i];
    max_sum = best_neg;
  }
  return -max_sum;
""",
    )
    add(
        "he114_min_subarray_sum",
        "HumanEval/114",
        "Return the minimum sum of any non-empty contiguous subarray of nums.",
        sig,
        c,
        [
            _ret([_buf("a", [2, 3, 4, 1, 2, 4]), 6], 1),
            _ret([_buf("a", [-1, -2, -3]), 3], -6),
            _ret([_buf("a", [1]), 1], 1),
            _ret([_buf("a", [5, -2, 3]), 3], -2),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_k_largest",
        "int64_t *arr, int64_t n, int64_t k, int64_t *out",
        """
  int64_t i, j, t;
  if (k <= 0) return 0;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      if (arr[j] < arr[i]) {
        t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }
  for (i = 0; i < k; i++) out[i] = arr[n - k + i];
  return k;
""",
    )
    add(
        "he120_k_largest",
        "HumanEval/120",
        "Write the k largest values of arr into out, in ascending order. Return k. "
        "If k is 0, write nothing and return 0. 0 <= k <= n. arr may be reordered.",
        sig,
        c,
        [
            _ret(
                [_buf("arr", [4, 1, 3, 2], out=True), 4, 2, _buf("out", [0, 0], out=True)],
                2,
                {"out": [3, 4]},
            ),
            _ret(
                [_buf("arr", [7], out=True), 1, 0, _buf("out", [], out=True)],
                0,
                {"out": []},
            ),
            _ret(
                [_buf("arr", [5, 5, 1], out=True), 3, 3, _buf("out", [0, 0, 0], out=True)],
                3,
                {"out": [1, 5, 5]},
            ),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_sum_odd_even_idx",
        "const int64_t *lst, int64_t n",
        """
  int64_t s = 0, i;
  for (i = 0; i < n; i += 2) {
    if ((lst[i] % 2) != 0) s += lst[i];
  }
  return s;
""",
    )
    add(
        "he121_sum_odd_even_idx",
        "HumanEval/121",
        "Return the sum of the odd elements of lst that sit at even indexes. Indexes start at 0.",
        sig,
        c,
        [
            _ret([_buf("a", [5, 8, 7, 1]), 4], 12),
            _ret([_buf("a", [3, 3, 3, 3, 3]), 5], 9),
            _ret([_buf("a", [30, 13, 24, 321]), 4], 0),
            _ret([_buf("a", []), 0], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_add_elements",
        "const int64_t *arr, int64_t n, int64_t k",
        """
  int64_t s = 0, i, x, digits, neg;
  if (k > n) k = n;
  if (k < 0) k = 0;
  for (i = 0; i < k; i++) {
    x = arr[i];
    neg = 0;
    if (x < 0) {
      neg = 1;
      x = -x;
    }
    digits = (x == 0) ? 1 : 0;
    while (x > 0) {
      digits++;
      x /= 10;
    }
    if (neg) digits++;
    if (digits <= 2) s += arr[i];
  }
  return s;
""",
    )
    add(
        "he122_add_elements",
        "HumanEval/122",
        "Return the sum of the first k elements of arr whose decimal representation, including a leading minus sign, "
        "has at most two characters. If k exceeds n, only n elements are considered.",
        sig,
        c,
        [
            _ret([_buf("a", [1, -2, 100, 3]), 4, 3], -1),
            _ret([_buf("a", [111, 21, 3, 15]), 4, 2], 21),
            _ret([_buf("a", [1, 2, 3]), 3, 0], 0),
            _ret([_buf("a", [-5, 10, 99]), 3, 3], 104),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_odd_digit_product",
        "int64_t n",
        """
  int64_t product = 1;
  int64_t odd_count = 0;
  int64_t d;
  if (n < 0) n = -n;
  if (n == 0) return 0;
  while (n > 0) {
    d = n % 10;
    if ((d % 2) == 1) {
      product *= d;
      odd_count++;
    }
    n /= 10;
  }
  if (odd_count == 0) return 0;
  return product;
""",
    )
    add(
        "he131_odd_digit_product",
        "HumanEval/131",
        "Return the product of the odd decimal digits of n. If n has no odd digit, return 0. The sign of n is ignored.",
        sig,
        c,
        [_ret([12345], 15), _ret([2468], 0), _ret([1], 1), _ret([135], 15), _ret([0], 0), _ret([-35], 15)],
    )

    sig, c = _fn(
        "int64_t",
        "he_sum_of_four_evens",
        "int64_t n",
        """
  if ((n % 2) == 0 && n >= 8) return 1;
  return 0;
""",
    )
    add(
        "he138_sum_of_four_evens",
        "HumanEval/138",
        "Return 1 if n can be written as the sum of four positive even integers, otherwise 0. "
        "That is exactly the even integers n >= 8.",
        sig,
        c,
        [_ret([4], 0), _ret([6], 0), _ret([8], 1), _ret([7], 0), _ret([10], 1), _ret([0], 0)],
    )

    sig, c = _fn(
        "int64_t",
        "he_special_factorial",
        "int64_t n",
        """
  int64_t result = 1;
  int64_t fact = 1;
  int64_t i;
  for (i = 1; i <= n; i++) {
    fact *= i;
    result *= fact;
  }
  return result;
""",
    )
    add(
        "he139_special_factorial",
        "HumanEval/139",
        "Return the Brazilian factorial of n: the product 1! * 2! * ... * n!. n is between 0 and 8. The empty product (n = 0) is 1.",
        sig,
        c,
        [_ret([4], 288), _ret([1], 1), _ret([0], 1), _ret([2], 2), _ret([3], 12)],
    )

    sig, c = _fn(
        "int64_t",
        "he_indexed_sum_squares",
        "const int64_t *lst, int64_t n",
        """
  int64_t s = 0, i, v;
  for (i = 0; i < n; i++) {
    v = lst[i];
    if ((i % 3) == 0) s += v * v;
    else if ((i % 4) == 0) s += v * v * v;
    else s += v;
  }
  return s;
""",
    )
    add(
        "he142_indexed_sum_squares",
        "HumanEval/142",
        "Return a mixed sum of lst. At an index divisible by 3, add the square of the element. "
        "At an index divisible by 4 but not by 3, add the cube. Otherwise add the element itself.",
        sig,
        c,
        [
            _ret([_buf("a", [1, 2, 3]), 3], 6),
            _ret([_buf("a", [1, 2, 3, 4, 5]), 5], 1 + 2 + 3 + 16 + 125),
            _ret([_buf("a", []), 0], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_special_filter",
        "const int64_t *nums, int64_t n",
        """
  int64_t count = 0, i, x, last, first;
  for (i = 0; i < n; i++) {
    x = nums[i];
    if (x <= 10) continue;
    last = x % 10;
    first = x;
    while (first >= 10) first /= 10;
    if ((first % 2) == 1 && (last % 2) == 1) count++;
  }
  return count;
""",
    )
    add(
        "he146_special_filter",
        "HumanEval/146",
        "Return how many values in nums are greater than 10 and have an odd first decimal digit and an odd last decimal digit. "
        "The values in the tests are positive.",
        sig,
        c,
        [
            _ret([_buf("a", [15, 33, 1422, 1]), 4], 2),
            _ret([_buf("a", [11, 13, 15]), 3], 3),
            _ret([_buf("a", [10, 20, 22]), 3], 0),
            _ret([_buf("a", []), 0], 0),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_x_or_y",
        "int64_t n, int64_t x, int64_t y",
        """
  int64_t i;
  if (n <= 1) return y;
  for (i = 2; i * i <= n; i++) {
    if ((n % i) == 0) return y;
  }
  return x;
""",
    )
    add(
        "he150_x_or_y",
        "HumanEval/150",
        "Return x if n is prime, and y otherwise. n = 1 is not prime.",
        sig,
        c,
        [_ret([7, 34, 12], 34), _ret([15, 8, 5], 5), _ret([1, 9, 4], 4), _ret([2, 3, 4], 3), _ret([9, 1, 2], 2)],
    )

    sig, c = _fn(
        "int64_t",
        "he_double_the_difference",
        "const int64_t *lst, int64_t n",
        """
  int64_t s = 0, i, v;
  for (i = 0; i < n; i++) {
    v = lst[i];
    if (v > 0 && (v % 2) != 0) s += v * v;
  }
  return s;
""",
    )
    add(
        "he151_double_the_difference",
        "HumanEval/151",
        "Return the sum of the squares of the positive odd integers in lst. Non-positive values and even values contribute nothing.",
        sig,
        c,
        [
            _ret([_buf("a", [1, 3, 2, 0]), 4], 10),
            _ret([_buf("a", [-1, -2, 0]), 3], 0),
            _ret([_buf("a", [9, -2]), 2], 81),
            _ret([_buf("a", [5, 4]), 2], 25),
            _ret([_buf("a", []), 0], 0),
        ],
    )

    sig, c = _fn(
        "void",
        "he_compare_scores",
        "const int64_t *game, const int64_t *guess, int64_t n, int64_t *out",
        """
  int64_t i, d;
  for (i = 0; i < n; i++) {
    d = game[i] - guess[i];
    if (d < 0) d = -d;
    out[i] = d;
  }
""",
    )
    add(
        "he152_compare_scores",
        "HumanEval/152",
        "Write into out the absolute difference of game[i] and guess[i] for each i in 0..n-1.",
        sig,
        c,
        [
            _void(
                [_buf("g", [1, 2, 3, 4]), _buf("h", [2, 2, 1, 6]), 4, _buf("out", [0, 0, 0, 0], out=True)],
                {"out": [1, 0, 2, 2]},
            ),
            _void(
                [_buf("g", [0]), _buf("h", [0]), 1, _buf("out", [9], out=True)],
                {"out": [0]},
            ),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_even_odd_count",
        "int64_t num, int64_t *odd_count",
        """
  int64_t even_count = 0;
  int64_t odd = 0;
  int64_t d;
  if (num < 0) num = -num;
  if (num == 0) {
    *odd_count = 0;
    return 1;
  }
  while (num > 0) {
    d = num % 10;
    if ((d % 2) == 0) even_count++;
    else odd++;
    num /= 10;
  }
  *odd_count = odd;
  return even_count;
""",
    )
    add(
        "he155_even_odd_count",
        "HumanEval/155",
        "Count the even and odd decimal digits of num, ignoring its sign. "
        "Return the even-digit count and write the odd-digit count into *odd_count. "
        "The number 0 has one even digit.",
        sig,
        c,
        [
            _ret([123, _buf("odd", [0], out=True)], 1, {"odd": [2]}),
            _ret([-12, _buf("odd", [0], out=True)], 1, {"odd": [1]}),
            _ret([0, _buf("odd", [0], out=True)], 1, {"odd": [0]}),
            _ret([2468, _buf("odd", [0], out=True)], 4, {"odd": [0]}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_right_angle_triangle",
        "int64_t a, int64_t b, int64_t c",
        """
  if (a * a == b * b + c * c) return 1;
  if (b * b == a * a + c * c) return 1;
  if (c * c == a * a + b * b) return 1;
  return 0;
""",
    )
    add(
        "he157_right_angle_triangle",
        "HumanEval/157",
        "Return 1 if a, b, and c are the side lengths of a right triangle, otherwise 0. "
        "Any side may be the hypotenuse. The check is a^2 + b^2 = c^2 for some ordering.",
        sig,
        c,
        [_ret([3, 4, 5], 1), _ret([5, 12, 13], 1), _ret([1, 2, 3], 0), _ret([5, 3, 4], 1), _ret([0, 0, 0], 1)],
    )

    sig, c = _fn(
        "int64_t",
        "he_eat",
        "int64_t number, int64_t need, int64_t remaining, int64_t *left",
        """
  if (need <= remaining) {
    *left = remaining - need;
    return number + need;
  }
  *left = 0;
  return number + remaining;
""",
    )
    add(
        "he159_eat",
        "HumanEval/159",
        "A rabbit has already eaten `number` carrots and needs to eat `need` more. "
        "`remaining` carrots are left. It eats as many as it can. "
        "Return the total eaten and write how many carrots remain into *left.",
        sig,
        c,
        [
            _ret([5, 6, 10, _buf("left", [0], out=True)], 11, {"left": [4]}),
            _ret([4, 8, 9, _buf("left", [0], out=True)], 12, {"left": [1]}),
            _ret([1, 10, 10, _buf("left", [0], out=True)], 11, {"left": [0]}),
            _ret([2, 11, 5, _buf("left", [0], out=True)], 7, {"left": [0]}),
        ],
    )

    sig, c = _fn(
        "int64_t",
        "he_generate_integers",
        "int64_t a, int64_t b, int64_t *out",
        """
  int64_t lower = a < b ? a : b;
  int64_t upper = a > b ? a : b;
  int64_t i, m = 0;
  if (lower < 2) lower = 2;
  if (upper > 8) upper = 8;
  for (i = lower; i <= upper; i++) {
    if ((i % 2) == 0) out[m++] = i;
  }
  return m;
""",
    )
    add(
        "he163_generate_integers",
        "HumanEval/163",
        "Write into out the even digits from 2, 4, 6, and 8 that lie in the inclusive range between a and b "
        "(the endpoints are ordered, so a may be larger than b). The range is clipped to 2..8. "
        "Return how many values were written. Unused slots stay -1.",
        sig,
        c,
        [
            _ret([2, 8, _buf("out", [-1, -1, -1, -1], out=True)], 4, {"out": [2, 4, 6, 8]}),
            _ret([8, 2, _buf("out", [-1, -1, -1, -1], out=True)], 4, {"out": [2, 4, 6, 8]}),
            _ret([10, 14, _buf("out", [-1, -1, -1, -1], out=True)], 0, {"out": [-1, -1, -1, -1]}),
            _ret([3, 6, _buf("out", [-1, -1, -1, -1], out=True)], 2, {"out": [4, 6, -1, -1]}),
        ],
    )

    _self_check(out)
    return out


def _self_check(rows: list[dict[str, Any]]) -> None:
    by_id = {r["id"]: r for r in rows}
    assert len(rows) >= 60, len(rows)
    sources = {r["source"].split("/")[0] for r in rows}
    assert sources == {"HumanEval", "MBPP"}
    assert by_id["he36_fizz_buzz"]["tests"][2]["ret"] == 3
    assert by_id["he13_gcd"]["tests"][1]["ret"] == 5
    assert by_id["mbpp6_differ_one_bit"]["tests"][0]["ret"] == 1
    assert by_id["he139_special_factorial"]["tests"][0]["ret"] == 288
    assert by_id["he45_triangle_area"]["tests"][0]["ret"] == 7.5
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


_ORACLE_NOTES: Callable[[], None] | None = None
