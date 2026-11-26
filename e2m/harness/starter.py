"""Hand-written starter tasks (~20). Source of truth; also dumped to tasks/starter.jsonl."""

from __future__ import annotations

from e2m.harness.rust_refs import RUST_REFS
from e2m.harness.tasks import Task, dump_jsonl, starter_jsonl_path, task_from_dict

_RAW: list[dict] = [
    {
        "id": "add_i64",
        "spec": "Return the sum of two 64-bit signed integers a and b.",
        "signature": "int64_t add_i64(int64_t a, int64_t b)",
        "reference_c": "int64_t add_i64(int64_t a, int64_t b) { return a + b; }\n",
        "tests": [
            {"args": [1, 2], "ret": 3},
            {"args": [0, 0], "ret": 0},
            {"args": [-5, 5], "ret": 0},
            {"args": [-100, -23], "ret": -123},
            {"args": [2**62, 1], "ret": 2**62 + 1},
        ],
    },
    {
        "id": "max3_i64",
        "spec": "Return the maximum of three 64-bit signed integers a, b, and c.",
        "signature": "int64_t max3_i64(int64_t a, int64_t b, int64_t c)",
        "reference_c": (
            "int64_t max3_i64(int64_t a, int64_t b, int64_t c) {\n"
            "  int64_t m = a;\n"
            "  if (b > m) m = b;\n"
            "  if (c > m) m = c;\n"
            "  return m;\n"
            "}\n"
        ),
        "tests": [
            {"args": [1, 2, 3], "ret": 3},
            {"args": [3, 2, 1], "ret": 3},
            {"args": [-5, -1, -9], "ret": -1},
            {"args": [0, 0, 0], "ret": 0},
            {"args": [7, 7, -3], "ret": 7},
        ],
    },
    {
        "id": "abs_i64",
        "spec": "Return the absolute value of a 64-bit signed integer. x is never INT64_MIN.",
        "signature": "int64_t abs_i64(int64_t x)",
        "reference_c": "int64_t abs_i64(int64_t x) { return x < 0 ? -x : x; }\n",
        "tests": [
            {"args": [0], "ret": 0},
            {"args": [5], "ret": 5},
            {"args": [-5], "ret": 5},
            {"args": [-123456789], "ret": 123456789},
            {"args": [2**62], "ret": 2**62},
        ],
    },
    {
        "id": "clamp_i64",
        "spec": "Clamp x into the closed interval [lo, hi]. Assume lo <= hi.",
        "signature": "int64_t clamp_i64(int64_t x, int64_t lo, int64_t hi)",
        "reference_c": (
            "int64_t clamp_i64(int64_t x, int64_t lo, int64_t hi) {\n"
            "  if (x < lo) return lo;\n"
            "  if (x > hi) return hi;\n"
            "  return x;\n"
            "}\n"
        ),
        "tests": [
            {"args": [5, 0, 10], "ret": 5},
            {"args": [-1, 0, 10], "ret": 0},
            {"args": [99, 0, 10], "ret": 10},
            {"args": [0, 0, 0], "ret": 0},
            {"args": [-50, -100, -10], "ret": -50},
        ],
    },
    {
        "id": "sum_to",
        "spec": "Return the sum of integers from 1 to n inclusive. If n < 1, return 0.",
        "signature": "int64_t sum_to(int64_t n)",
        "reference_c": (
            "int64_t sum_to(int64_t n) {\n"
            "  if (n < 1) return 0;\n"
            "  int64_t s = 0;\n"
            "  for (int64_t i = 1; i <= n; i++) s += i;\n"
            "  return s;\n"
            "}\n"
        ),
        "tests": [
            {"args": [0], "ret": 0},
            {"args": [-3], "ret": 0},
            {"args": [1], "ret": 1},
            {"args": [10], "ret": 55},
            {"args": [100], "ret": 5050},
        ],
    },
    {
        "id": "is_prime",
        "spec": "Return 1 if n is a prime number, else 0. n is a small non-negative integer.",
        "signature": "int64_t is_prime(int64_t n)",
        "reference_c": (
            "int64_t is_prime(int64_t n) {\n"
            "  if (n < 2) return 0;\n"
            "  if (n % 2 == 0) return n == 2;\n"
            "  for (int64_t i = 3; i * i <= n; i += 2)\n"
            "    if (n % i == 0) return 0;\n"
            "  return 1;\n"
            "}\n"
        ),
        "tests": [
            {"args": [0], "ret": 0},
            {"args": [1], "ret": 0},
            {"args": [2], "ret": 1},
            {"args": [4], "ret": 0},
            {"args": [9], "ret": 0},
            {"args": [97], "ret": 1},
            {"args": [100], "ret": 0},
        ],
    },
    {
        "id": "gcd",
        "spec": "Return the greatest common divisor of two non-negative 64-bit integers using Euclid's algorithm. gcd(0, 0) is 0.",
        "signature": "int64_t gcd(int64_t a, int64_t b)",
        "reference_c": (
            "int64_t gcd(int64_t a, int64_t b) {\n"
            "  while (b != 0) {\n"
            "    int64_t t = a % b;\n"
            "    a = b;\n"
            "    b = t;\n"
            "  }\n"
            "  return a < 0 ? -a : a;\n"
            "}\n"
        ),
        "tests": [
            {"args": [0, 0], "ret": 0},
            {"args": [0, 5], "ret": 5},
            {"args": [54, 24], "ret": 6},
            {"args": [17, 13], "ret": 1},
            {"args": [100, 10], "ret": 10},
        ],
    },
    {
        "id": "factorial",
        "spec": "Return n! for n in 0..12 inclusive. For n < 0 return 1.",
        "signature": "int64_t factorial(int64_t n)",
        "reference_c": (
            "int64_t factorial(int64_t n) {\n"
            "  if (n < 0) return 1;\n"
            "  int64_t r = 1;\n"
            "  for (int64_t i = 2; i <= n; i++) r *= i;\n"
            "  return r;\n"
            "}\n"
        ),
        "tests": [
            {"args": [0], "ret": 1},
            {"args": [1], "ret": 1},
            {"args": [5], "ret": 120},
            {"args": [10], "ret": 3628800},
            {"args": [12], "ret": 479001600},
        ],
    },
    {
        "id": "fib",
        "spec": "Return the n-th Fibonacci number with fib(0)=0, fib(1)=1, iteratively. n is in 0..30.",
        "signature": "int64_t fib(int64_t n)",
        "reference_c": (
            "int64_t fib(int64_t n) {\n"
            "  if (n <= 0) return 0;\n"
            "  if (n == 1) return 1;\n"
            "  int64_t a = 0, b = 1;\n"
            "  for (int64_t i = 2; i <= n; i++) {\n"
            "    int64_t c = a + b;\n"
            "    a = b;\n"
            "    b = c;\n"
            "  }\n"
            "  return b;\n"
            "}\n"
        ),
        "tests": [
            {"args": [0], "ret": 0},
            {"args": [1], "ret": 1},
            {"args": [2], "ret": 1},
            {"args": [10], "ret": 55},
            {"args": [20], "ret": 6765},
        ],
    },
    {
        "id": "array_sum_i32",
        "spec": "Return the sum of n 32-bit signed integers in arr, as a 64-bit integer. If n <= 0, return 0.",
        "signature": "int64_t array_sum_i32(const int32_t *arr, int64_t n)",
        "reference_c": (
            "int64_t array_sum_i32(const int32_t *arr, int64_t n) {\n"
            "  int64_t s = 0;\n"
            "  for (int64_t i = 0; i < n; i++) s += arr[i];\n"
            "  return s;\n"
            "}\n"
        ),
        "tests": [
            {"args": [{"id": "a", "ptr": "i32", "data": []}, 0], "ret": 0},
            {"args": [{"id": "a", "ptr": "i32", "data": [1, 2, 3, 4]}, 4], "ret": 10},
            {"args": [{"id": "a", "ptr": "i32", "data": [-5, 5, -5, 5]}, 4], "ret": 0},
            {"args": [{"id": "a", "ptr": "i32", "data": [1000000, 1000000, 1000000]}, 3], "ret": 3000000},
            {"args": [{"id": "a", "ptr": "i32", "data": [42]}, 1], "ret": 42},
        ],
    },
    {
        "id": "array_max_i32",
        "spec": "Return the maximum of n 32-bit signed integers in arr. If n <= 0, return 0.",
        "signature": "int32_t array_max_i32(const int32_t *arr, int64_t n)",
        "reference_c": (
            "int32_t array_max_i32(const int32_t *arr, int64_t n) {\n"
            "  if (n <= 0) return 0;\n"
            "  int32_t m = arr[0];\n"
            "  for (int64_t i = 1; i < n; i++) if (arr[i] > m) m = arr[i];\n"
            "  return m;\n"
            "}\n"
        ),
        "tests": [
            {"args": [{"id": "a", "ptr": "i32", "data": []}, 0], "ret": 0},
            {"args": [{"id": "a", "ptr": "i32", "data": [3, 1, 4, 1, 5]}, 5], "ret": 5},
            {"args": [{"id": "a", "ptr": "i32", "data": [-9, -3, -7]}, 3], "ret": -3},
            {"args": [{"id": "a", "ptr": "i32", "data": [2, 2, 2]}, 3], "ret": 2},
        ],
    },
    {
        "id": "dot_i32",
        "spec": "Return the dot product of two length-n arrays of 32-bit signed integers, as a 64-bit integer. If n <= 0, return 0.",
        "signature": "int64_t dot_i32(const int32_t *a, const int32_t *b, int64_t n)",
        "reference_c": (
            "int64_t dot_i32(const int32_t *a, const int32_t *b, int64_t n) {\n"
            "  int64_t s = 0;\n"
            "  for (int64_t i = 0; i < n; i++) s += (int64_t)a[i] * (int64_t)b[i];\n"
            "  return s;\n"
            "}\n"
        ),
        "tests": [
            {
                "args": [
                    {"id": "a", "ptr": "i32", "data": []},
                    {"id": "b", "ptr": "i32", "data": []},
                    0,
                ],
                "ret": 0,
            },
            {
                "args": [
                    {"id": "a", "ptr": "i32", "data": [1, 2, 3]},
                    {"id": "b", "ptr": "i32", "data": [4, 5, 6]},
                    3,
                ],
                "ret": 32,
            },
            {
                "args": [
                    {"id": "a", "ptr": "i32", "data": [-1, 2]},
                    {"id": "b", "ptr": "i32", "data": [3, -4]},
                    2,
                ],
                "ret": -11,
            },
        ],
    },
    {
        "id": "reverse_i32",
        "spec": "Reverse n 32-bit signed integers in arr in place. If n <= 1, do nothing.",
        "signature": "void reverse_i32(int32_t *arr, int64_t n)",
        "reference_c": (
            "void reverse_i32(int32_t *arr, int64_t n) {\n"
            "  for (int64_t i = 0; i < n / 2; i++) {\n"
            "    int32_t t = arr[i];\n"
            "    arr[i] = arr[n - 1 - i];\n"
            "    arr[n - 1 - i] = t;\n"
            "  }\n"
            "}\n"
        ),
        "tests": [
            {
                "args": [{"id": "a", "ptr": "i32", "data": [1, 2, 3, 4], "out": True}, 4],
                "bufs": {"a": [4, 3, 2, 1]},
            },
            {
                "args": [{"id": "a", "ptr": "i32", "data": [7], "out": True}, 1],
                "bufs": {"a": [7]},
            },
            {
                "args": [{"id": "a", "ptr": "i32", "data": [1, 2, 3], "out": True}, 3],
                "bufs": {"a": [3, 2, 1]},
            },
            {
                "args": [{"id": "a", "ptr": "i32", "data": [-1, 0, 1, 2], "out": True}, 4],
                "bufs": {"a": [2, 1, 0, -1]},
            },
        ],
    },
    {
        "id": "count_eq_i32",
        "spec": "Return how many of the n 32-bit signed integers in arr are equal to v. If n <= 0, return 0.",
        "signature": "int64_t count_eq_i32(const int32_t *arr, int64_t n, int32_t v)",
        "reference_c": (
            "int64_t count_eq_i32(const int32_t *arr, int64_t n, int32_t v) {\n"
            "  int64_t c = 0;\n"
            "  for (int64_t i = 0; i < n; i++) if (arr[i] == v) c++;\n"
            "  return c;\n"
            "}\n"
        ),
        "tests": [
            {"args": [{"id": "a", "ptr": "i32", "data": [1, 2, 1, 3, 1]}, 5, 1], "ret": 3},
            {"args": [{"id": "a", "ptr": "i32", "data": [0, 0, 0]}, 3, 1], "ret": 0},
            {"args": [{"id": "a", "ptr": "i32", "data": []}, 0, 4], "ret": 0},
            {"args": [{"id": "a", "ptr": "i32", "data": [-2, -2, 2]}, 3, -2], "ret": 2},
        ],
    },
    {
        "id": "memcpy_u8",
        "spec": "Copy n bytes from src to dest. dest and src do not overlap. If n <= 0, do nothing.",
        "signature": "void memcpy_u8(uint8_t *dest, const uint8_t *src, int64_t n)",
        "reference_c": (
            "void memcpy_u8(uint8_t *dest, const uint8_t *src, int64_t n) {\n"
            "  for (int64_t i = 0; i < n; i++) dest[i] = src[i];\n"
            "}\n"
        ),
        "tests": [
            {
                "args": [
                    {"id": "dst", "ptr": "u8", "data": [0, 0, 0, 0, 0], "out": True},
                    {"id": "src", "ptr": "u8", "data": [1, 2, 3, 4, 5]},
                    5,
                ],
                "bufs": {"dst": [1, 2, 3, 4, 5]},
            },
            {
                "args": [
                    {"id": "dst", "ptr": "u8", "data": [9, 9, 9], "out": True},
                    {"id": "src", "ptr": "u8", "data": [1, 2, 3]},
                    0,
                ],
                "bufs": {"dst": [9, 9, 9]},
            },
            {
                "args": [
                    {"id": "dst", "ptr": "u8", "data": [0, 0], "out": True},
                    {"id": "src", "ptr": "u8", "data": [7, 8]},
                    2,
                ],
                "bufs": {"dst": [7, 8]},
            },
        ],
    },
    {
        "id": "strlen_u8",
        "spec": "Return the length of the NUL-terminated byte string s, not counting the terminator.",
        "signature": "int64_t strlen_u8(const char *s)",
        "reference_c": (
            "int64_t strlen_u8(const char *s) {\n"
            "  int64_t n = 0;\n"
            "  while (s[n] != 0) n++;\n"
            "  return n;\n"
            "}\n"
        ),
        "tests": [
            {"args": [{"id": "s", "ptr": "cstr", "data": ""}], "ret": 0},
            {"args": [{"id": "s", "ptr": "cstr", "data": "hello"}], "ret": 5},
            {"args": [{"id": "s", "ptr": "cstr", "data": "a"}], "ret": 1},
            {"args": [{"id": "s", "ptr": "cstr", "data": "English to machine"}], "ret": 18},
        ],
    },
    {
        "id": "strcmp_u8",
        "spec": "Compare two NUL-terminated byte strings. Return a negative int if a<b, zero if equal, positive if a>b (memcmp order on unsigned bytes).",
        "signature": "int32_t strcmp_u8(const uint8_t *a, const uint8_t *b)",
        "reference_c": (
            "int32_t strcmp_u8(const uint8_t *a, const uint8_t *b) {\n"
            "  for (;;) {\n"
            "    uint8_t ca = *a++;\n"
            "    uint8_t cb = *b++;\n"
            "    if (ca != cb) return (int32_t)ca - (int32_t)cb;\n"
            "    if (ca == 0) return 0;\n"
            "  }\n"
            "}\n"
        ),
        "tests": [
            {
                "args": [
                    {"id": "a", "ptr": "cstr", "data": "abc"},
                    {"id": "b", "ptr": "cstr", "data": "abc"},
                ],
                "ret": 0,
                "ret_cmp": "sign",
            },
            {
                "args": [
                    {"id": "a", "ptr": "cstr", "data": "abc"},
                    {"id": "b", "ptr": "cstr", "data": "abd"},
                ],
                "ret": -1,
                "ret_cmp": "sign",
            },
            {
                "args": [
                    {"id": "a", "ptr": "cstr", "data": "abd"},
                    {"id": "b", "ptr": "cstr", "data": "abc"},
                ],
                "ret": 1,
                "ret_cmp": "sign",
            },
            {
                "args": [
                    {"id": "a", "ptr": "cstr", "data": "ab"},
                    {"id": "b", "ptr": "cstr", "data": "abc"},
                ],
                "ret": -1,
                "ret_cmp": "sign",
            },
            {
                "args": [
                    {"id": "a", "ptr": "cstr", "data": ""},
                    {"id": "b", "ptr": "cstr", "data": ""},
                ],
                "ret": 0,
                "ret_cmp": "sign",
            },
        ],
    },
    {
        "id": "count_char",
        "spec": "Return how many times unsigned byte c occurs in the NUL-terminated string s (the terminator is not counted).",
        "signature": "int64_t count_char(const char *s, uint8_t c)",
        "reference_c": (
            "int64_t count_char(const char *s, uint8_t c) {\n"
            "  int64_t n = 0;\n"
            "  for (; *s; s++) if ((uint8_t)*s == c) n++;\n"
            "  return n;\n"
            "}\n"
        ),
        "tests": [
            {"args": [{"id": "s", "ptr": "cstr", "data": "hello"}, 108], "ret": 2},
            {"args": [{"id": "s", "ptr": "cstr", "data": "hello"}, 122], "ret": 0},
            {"args": [{"id": "s", "ptr": "cstr", "data": ""}, 97], "ret": 0},
            {"args": [{"id": "s", "ptr": "cstr", "data": "aaa"}, 97], "ret": 3},
        ],
    },
    {
        "id": "add_f64",
        "spec": "Return the sum of two IEEE-754 doubles a and b.",
        "signature": "double add_f64(double a, double b)",
        "reference_c": "double add_f64(double a, double b) { return a + b; }\n",
        "tests": [
            {"args": [1.5, 2.25], "ret": 3.75},
            {"args": [0.0, 0.0], "ret": 0.0},
            {"args": [-2.0, 2.0], "ret": 0.0},
            {"args": [4.0, 5.0], "ret": 9.0},
        ],
    },
    {
        "id": "dot_f64",
        "spec": "Return the dot product of two length-n arrays of doubles. If n <= 0, return 0.0. Inputs are chosen so the result is exact in IEEE-754.",
        "signature": "double dot_f64(const double *a, const double *b, int64_t n)",
        "reference_c": (
            "double dot_f64(const double *a, const double *b, int64_t n) {\n"
            "  double s = 0.0;\n"
            "  for (int64_t i = 0; i < n; i++) s += a[i] * b[i];\n"
            "  return s;\n"
            "}\n"
        ),
        "tests": [
            {
                "args": [
                    {"id": "a", "ptr": "f64", "data": []},
                    {"id": "b", "ptr": "f64", "data": []},
                    0,
                ],
                "ret": 0.0,
            },
            {
                "args": [
                    {"id": "a", "ptr": "f64", "data": [1.0, 2.0, 3.0]},
                    {"id": "b", "ptr": "f64", "data": [4.0, 5.0, 6.0]},
                    3,
                ],
                "ret": 32.0,
            },
            {
                "args": [
                    {"id": "a", "ptr": "f64", "data": [0.5, 0.5]},
                    {"id": "b", "ptr": "f64", "data": [2.0, 4.0]},
                    2,
                ],
                "ret": 3.0,
            },
        ],
    },
]


def _build() -> list[Task]:
    tasks: list[Task] = []
    for obj in _RAW:
        item = dict(obj)
        item["reference_rust"] = RUST_REFS[item["id"]]
        tasks.append(task_from_dict(item))
    missing = [t.id for t in tasks if not t.reference_rust.strip()]
    if missing:
        raise RuntimeError(f"starter tasks missing Rust reference: {missing}")
    return tasks


STARTER_TASKS: list[Task] = _build()


def write_starter_jsonl(path=None) -> None:
    dest = path or starter_jsonl_path()
    dump_jsonl(STARTER_TASKS, dest)
