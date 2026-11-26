from e2m.harness.abi import parse_signature
from e2m.scale.heldout import is_heldout_name
from e2m.scale.io_convert import heuristic_inputs, io_to_args
from e2m.scale.parse_c import cheap_reject, clean_signature, is_source_trivial
from e2m.scale.pipeline import _candidate_from_row, _compile_ok, _verify_one


def test_clean_static_inline():
    src = "static inline int add(int a, int b) { return a + b; }\n"
    sig = clean_signature(src)
    parsed = parse_signature(sig)
    assert parsed.name == "add"
    assert len(parsed.params) == 2


def test_trivial_return_arg():
    src = "int id(int x) { return x; }\n"
    sig = parse_signature(clean_signature(src))
    assert is_source_trivial(src, sig)
    assert cheap_reject(src, "id") == "trivial_source"


def test_keep_add():
    src = "int64_t add_i64(int64_t a, int64_t b) { return a + b; }\n"
    assert cheap_reject(src, "add_i64") is None


def test_reject_struct_and_main():
    assert cheap_reject("int main(void) { return 0; }\n", "main") == "main_or_empty_name"
    src = "int f(struct s *p) {\n  int x = p->a;\n  return x + 1;\n}\n"
    assert cheap_reject(src, "f") == "struct_or_union"


def test_heldout_distinctive():
    assert is_heldout_name("has_close_elements") is True
    assert is_heldout_name("add") is False


def test_heuristic_and_io_convert():
    sig = parse_signature("int64_t array_sum_i32(const int32_t *arr, int64_t n)")
    args = io_to_args(sig, {"arr": [1, 2, 3, 4], "n": 99})
    assert args is not None
    assert args[1] == 4  # clamped
    hs = heuristic_inputs(sig)
    assert len(hs) >= 3


def test_compile_and_verify_add():
    row = _candidate_from_row(
        {
            "fname": "add_i64",
            "func_def": "int64_t add_i64(int64_t a, int64_t b) { return a + b; }\n",
            "path": "unit/add.c",
            "real_io_pairs": [
                {"input": {"a": 1, "b": 2}},
                {"input": {"a": 0, "b": 0}},
                {"input": {"a": -5, "b": 5}},
            ],
        },
        "exebench",
    )
    assert row and row["status"] == "keep"
    compiled = _compile_ok(row)
    assert compiled["status"] == "keep", compiled
    verified = _verify_one(compiled)
    assert verified["status"] == "pass", verified
    rets = {t["ret"] for t in verified["tests"]}
    assert 3 in rets
    assert 0 in rets
