"""Rust references for the starter tasks.

Each body is one ``#[no_mangle] pub extern "C"`` function with the same SysV
signature as the C reference. Integer arithmetic is wrapping so the function
stays a leaf under ``-C overflow-checks`` (a Rust ``+`` that might overflow
lowers to a call into the panic runtime). Division uses inline ``idiv`` via
:data:`RUST_PREAMBLE` for the same reason: stable Rust inserts an external
panic call on the ``%`` overflow and divide-by-zero paths.
"""

from __future__ import annotations

RUST_PREAMBLE = """\
#![no_std]
#![allow(dead_code)]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Signed remainder. `b` must be non-zero and the division must not be
/// `i64::MIN / -1`. Inline so the object has no out-of-function call.
#[inline(always)]
unsafe fn irem(a: i64, b: i64) -> i64 {
    let mut quot = a;
    let mut rem: i64;
    core::arch::asm!(
        "cqo",
        "idiv {divisor}",
        divisor = in(reg) b,
        inout("rax") quot,
        out("rdx") rem,
        options(nomem, nostack, pure),
    );
    let _ = quot;
    rem
}

"""

RUST_REFS: dict[str, str] = {
    "add_i64": """\
#[no_mangle]
pub extern "C" fn add_i64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}
""",
    "max3_i64": """\
#[no_mangle]
pub extern "C" fn max3_i64(a: i64, b: i64, c: i64) -> i64 {
    let mut m = a;
    if b > m {
        m = b;
    }
    if c > m {
        m = c;
    }
    m
}
""",
    "abs_i64": """\
#[no_mangle]
pub extern "C" fn abs_i64(x: i64) -> i64 {
    if x < 0 { x.wrapping_neg() } else { x }
}
""",
    "clamp_i64": """\
#[no_mangle]
pub extern "C" fn clamp_i64(x: i64, lo: i64, hi: i64) -> i64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}
""",
    "sum_to": """\
#[no_mangle]
pub extern "C" fn sum_to(n: i64) -> i64 {
    if n < 1 {
        return 0;
    }
    let mut s: i64 = 0;
    let mut i: i64 = 1;
    while i <= n {
        s = s.wrapping_add(i);
        i = i.wrapping_add(1);
    }
    s
}
""",
    "is_prime": """\
#[no_mangle]
pub extern "C" fn is_prime(n: i64) -> i64 {
    if n < 2 {
        return 0;
    }
    if unsafe { irem(n, 2) } == 0 {
        return if n == 2 { 1 } else { 0 };
    }
    let mut i: i64 = 3;
    while i > 0 && i.wrapping_mul(i) <= n {
        if unsafe { irem(n, i) } == 0 {
            return 0;
        }
        i = i.wrapping_add(2);
    }
    1
}
""",
    "gcd": """\
#[no_mangle]
pub extern "C" fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = unsafe { irem(a, b) };
        a = b;
        b = t;
    }
    if a < 0 { a.wrapping_neg() } else { a }
}
""",
    "factorial": """\
#[no_mangle]
pub extern "C" fn factorial(n: i64) -> i64 {
    if n < 0 {
        return 1;
    }
    let mut r: i64 = 1;
    let mut i: i64 = 2;
    while i <= n {
        r = r.wrapping_mul(i);
        i = i.wrapping_add(1);
    }
    r
}
""",
    "fib": """\
#[no_mangle]
pub extern "C" fn fib(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut a: i64 = 0;
    let mut b: i64 = 1;
    let mut i: i64 = 2;
    while i <= n {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
        i = i.wrapping_add(1);
    }
    b
}
""",
    "array_sum_i32": """\
#[no_mangle]
pub unsafe extern "C" fn array_sum_i32(arr: *const i32, n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s.wrapping_add(i64::from(*arr.offset(i as isize)));
        i = i.wrapping_add(1);
    }
    s
}
""",
    "array_max_i32": """\
#[no_mangle]
pub unsafe extern "C" fn array_max_i32(arr: *const i32, n: i64) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut m = *arr;
    let mut i: i64 = 1;
    while i < n {
        let v = *arr.offset(i as isize);
        if v > m {
            m = v;
        }
        i = i.wrapping_add(1);
    }
    m
}
""",
    "dot_i32": """\
#[no_mangle]
pub unsafe extern "C" fn dot_i32(a: *const i32, b: *const i32, n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let av = i64::from(*a.offset(i as isize));
        let bv = i64::from(*b.offset(i as isize));
        s = s.wrapping_add(av.wrapping_mul(bv));
        i = i.wrapping_add(1);
    }
    s
}
""",
    "reverse_i32": """\
#[no_mangle]
pub unsafe extern "C" fn reverse_i32(arr: *mut i32, n: i64) {
    let mut i: i64 = 0;
    while i < n / 2 {
        let j = n - 1 - i;
        let t = *arr.offset(i as isize);
        *arr.offset(i as isize) = *arr.offset(j as isize);
        *arr.offset(j as isize) = t;
        i = i.wrapping_add(1);
    }
}
""",
    "count_eq_i32": """\
#[no_mangle]
pub unsafe extern "C" fn count_eq_i32(arr: *const i32, n: i64, v: i32) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if *arr.offset(i as isize) == v {
            c = c.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    c
}
""",
    "memcpy_u8": """\
#[no_mangle]
pub unsafe extern "C" fn memcpy_u8(dest: *mut u8, src: *const u8, n: i64) {
    // Volatile so LLVM does not turn the loop into a memcpy call.
    let mut i: i64 = 0;
    while i < n {
        let b = core::ptr::read_volatile(src.offset(i as isize));
        core::ptr::write_volatile(dest.offset(i as isize), b);
        i = i.wrapping_add(1);
    }
}
""",
    "strlen_u8": """\
#[no_mangle]
pub unsafe extern "C" fn strlen_u8(s: *const u8) -> i64 {
    // Volatile so LLVM does not tail-call libc strlen.
    let mut n: i64 = 0;
    while core::ptr::read_volatile(s.offset(n as isize)) != 0 {
        n = n.wrapping_add(1);
    }
    n
}
""",
    "strcmp_u8": """\
#[no_mangle]
pub unsafe extern "C" fn strcmp_u8(a: *const u8, b: *const u8) -> i32 {
    let mut i: isize = 0;
    loop {
        let ca = core::ptr::read_volatile(a.offset(i));
        let cb = core::ptr::read_volatile(b.offset(i));
        if ca != cb {
            return i32::from(ca) - i32::from(cb);
        }
        if ca == 0 {
            return 0;
        }
        i += 1;
    }
}
""",
    "count_char": """\
#[no_mangle]
pub unsafe extern "C" fn count_char(s: *const u8, c: u8) -> i64 {
    let mut n: i64 = 0;
    let mut i: isize = 0;
    loop {
        let ch = core::ptr::read_volatile(s.offset(i));
        if ch == 0 {
            return n;
        }
        if ch == c {
            n = n.wrapping_add(1);
        }
        i += 1;
    }
}
""",
    "add_f64": """\
#[no_mangle]
pub extern "C" fn add_f64(a: f64, b: f64) -> f64 {
    a + b
}
""",
    "dot_f64": """\
#[no_mangle]
pub unsafe extern "C" fn dot_f64(a: *const f64, b: *const f64, n: i64) -> f64 {
    let mut s = 0.0f64;
    let mut i: i64 = 0;
    while i < n {
        s += *a.offset(i as isize) * *b.offset(i as isize);
        i = i.wrapping_add(1);
    }
    s
}
""",
}


def rust_crate(task_id: str) -> str:
    """Full no_std crate source for one starter task."""
    body = RUST_REFS[task_id]
    return RUST_PREAMBLE + body
