from e2m.harness.abi import parse_signature


def test_parse_simple():
    s = parse_signature("int64_t add_i64(int64_t a, int64_t b)")
    assert s.name == "add_i64"
    assert s.ret.kind == "int" and s.ret.size == 8
    assert len(s.params) == 2
    assert s.params[0].name == "a"


def test_parse_pointers_and_cstr():
    s = parse_signature("int64_t strlen_u8(const char *s)")
    assert s.params[0].ctype.kind == "ptr"
    assert s.params[0].ctype.cstr


def test_parse_mixed_float():
    s = parse_signature("double dot_f64(const double *a, const double *b, int64_t n)")
    assert s.ret.kind == "float" and s.ret.size == 8
    assert s.params[0].ctype.kind == "ptr"
    assert s.params[0].ctype.pointee.kind == "float"
    assert s.params[2].ctype.kind == "int"


def test_parse_void():
    s = parse_signature("void reverse_i32(int32_t *arr, int64_t n)")
    assert s.ret.kind == "void"
    assert s.params[0].ctype.kind == "ptr"
    assert s.params[0].ctype.pointee.size == 4
