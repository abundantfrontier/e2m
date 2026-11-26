"""Turn ExeBench IO / heuristics into harness test args. Expected outputs come later from the harness."""

from __future__ import annotations

import re
from typing import Any

from e2m.harness.abi import Signature

SIZE_NAME = re.compile(r"(len|length|size|count|nmemb|nitems|num$|numsSize|^n$|^n[A-Z])", re.I)


def _ptr_tag(param) -> str:
    pointee = param.ctype.pointee
    if param.ctype.cstr or (pointee and pointee.kind == "int" and pointee.size == 1 and pointee.signed):
        return "cstr"
    if pointee is None:
        return "u8"
    if pointee.kind == "float":
        return "f64" if pointee.size == 8 else "f32"
    if pointee.kind == "int":
        signed = "i" if pointee.signed else "u"
        return f"{signed}{pointee.size * 8}"
    return "u8"


def _looks_string(val: Any) -> bool:
    return isinstance(val, str)


def clamp_io(sig: Signature, raw: dict[str, Any]) -> dict[str, Any] | None:
    """Drop or clamp unsafe ExeBench lengths so the harness cannot OOB."""
    out = dict(raw)
    buf_lens: list[int] = []
    for p in sig.params:
        v = out.get(p.name)
        if p.ctype.kind == "ptr":
            if isinstance(v, list):
                buf_lens.append(len(v))
            elif isinstance(v, str):
                buf_lens.append(len(v))
            elif isinstance(v, (bytes, bytearray)):
                buf_lens.append(len(v))
    min_buf = min(buf_lens) if buf_lens else None
    for p in sig.params:
        if p.ctype.kind != "int":
            continue
        v = out.get(p.name)
        if not isinstance(v, (int, float)):
            continue
        v = int(v)
        if abs(v) > 10_000:
            return None
        if min_buf is not None and SIZE_NAME.search(p.name) and v > min_buf:
            out[p.name] = min_buf
    return out


def io_to_args(sig: Signature, raw: dict[str, Any]) -> list[Any] | None:
    clamped = clamp_io(sig, raw)
    if clamped is None:
        return None
    args: list[Any] = []
    for p in sig.params:
        if p.name not in clamped:
            return None
        val = clamped[p.name]
        if p.ctype.kind == "ptr":
            tag = _ptr_tag(p)
            data: Any
            if val is None:
                return None
            if isinstance(val, dict):
                return None  # struct-shaped
            if tag == "cstr":
                if isinstance(val, list):
                    data = "".join(chr(int(x) & 0xFF) for x in val if int(x) != 0)
                else:
                    data = str(val)
            elif isinstance(val, str):
                data = list(val.encode("utf-8"))
            elif isinstance(val, list):
                data = val
            else:
                data = [val]
            args.append({"id": p.name, "ptr": tag, "data": data, "out": True})
        elif p.ctype.kind == "float":
            try:
                args.append(float(val))
            except (TypeError, ValueError):
                return None
        else:
            if isinstance(val, (list, dict, str)):
                return None
            try:
                args.append(int(val))
            except (TypeError, ValueError):
                return None
    return args


def exebench_inputs(sig: Signature, io_pairs: list[Any], *, cap: int = 8) -> list[list[Any]]:
    out: list[list[Any]] = []
    seen: set[str] = set()
    for pair in io_pairs or []:
        if not isinstance(pair, dict):
            continue
        raw = pair.get("input") or {}
        if not isinstance(raw, dict):
            continue
        args = io_to_args(sig, raw)
        if args is None:
            continue
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        out.append(args)
        if len(out) >= cap:
            break
    return out


def heuristic_inputs(sig: Signature) -> list[list[Any]]:
    """Generate a handful of small inputs from the signature alone."""
    cases: list[list[Any]] = []

    def pack(values: dict[str, Any]) -> list[Any] | None:
        args: list[Any] = []
        for p in sig.params:
            if p.name not in values:
                return None
            val = values[p.name]
            if p.ctype.kind == "ptr":
                tag = _ptr_tag(p)
                args.append({"id": p.name, "ptr": tag, "data": val, "out": True})
            else:
                args.append(val)
        return args

    int_params = [p for p in sig.params if p.ctype.kind == "int"]
    ptr_params = [p for p in sig.params if p.ctype.kind == "ptr"]
    fp_params = [p for p in sig.params if p.ctype.kind == "float"]

    if not sig.params:
        packed = pack({})
        return [packed] if packed is not None else []

    templates: list[dict[str, Any]] = []

    def fill(int_vals: list[int], ptr_data: Any, fp_vals: list[float]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        ii = pi = fi = 0
        for p in sig.params:
            if p.ctype.kind == "int":
                d[p.name] = int_vals[min(ii, len(int_vals) - 1)]
                if SIZE_NAME.search(p.name) and isinstance(ptr_data, list):
                    d[p.name] = len(ptr_data)
                elif SIZE_NAME.search(p.name) and isinstance(ptr_data, str):
                    d[p.name] = len(ptr_data)
                ii += 1
            elif p.ctype.kind == "ptr":
                tag = _ptr_tag(p)
                if tag == "cstr":
                    d[p.name] = ptr_data if isinstance(ptr_data, str) else "ab"
                elif tag.startswith("f"):
                    d[p.name] = ptr_data if isinstance(ptr_data, list) else [1.0, 2.0, 3.0, 4.0]
                else:
                    d[p.name] = ptr_data if isinstance(ptr_data, list) else [1, 2, 3, 4]
                pi += 1
            else:
                d[p.name] = fp_vals[min(fi, len(fp_vals) - 1)]
                fi += 1
        return d

    if ptr_params:
        templates.append(fill([0], [], [0.0]))
        templates.append(fill([1], [7] if _ptr_tag(ptr_params[0]) != "cstr" else "a", [1.0]))
        templates.append(fill([4], [1, 2, 3, 4], [1.0, 2.0]))
        templates.append(fill([4], [-2, 5, -2, 5], [-1.0, 0.5]))
        templates.append(fill([8], list(range(8)), [0.5] * 4))
        if any(_ptr_tag(p) == "cstr" for p in ptr_params):
            templates.append(fill([0], "", [0.0]))
            templates.append(fill([5], "hello", [0.0]))
    elif int_params and not fp_params:
        singles = [0, 1, -1, 2, 7, 10, 42, -7]
        if len(int_params) == 1:
            for v in singles:
                templates.append({int_params[0].name: v})
        elif len(int_params) == 2:
            a, b = int_params[0].name, int_params[1].name
            for x, y in ((0, 0), (1, 2), (-3, 5), (7, 7), (10, 1), (-1, -1)):
                templates.append({a: x, b: y})
        else:
            templates.append({p.name: 0 for p in int_params})
            templates.append({p.name: 1 for p in int_params})
            templates.append({p.name: i + 1 for i, p in enumerate(int_params)})
            templates.append({p.name: -1 for p in int_params})
            d = {p.name: 2 for p in int_params}
            d[int_params[0].name] = 5
            templates.append(d)
    else:
        templates.append(fill([0], [0], [0.0]))
        templates.append(fill([1], [1], [1.0]))
        templates.append(fill([3], [1.0, 2.0, 3.0], [0.5, 1.5]))

    seen: set[str] = set()
    for tmpl in templates:
        packed = pack(tmpl)
        if packed is None:
            continue
        key = repr(packed)
        if key in seen:
            continue
        seen.add(key)
        cases.append(packed)
        if len(cases) >= 8:
            break
    return cases
