"""CPU data for round 5: MBPP leaf tasks, asm lowering, leakage, mix.

HumanEval is never read. Held-out and val tasks are only loaded as a blocklist.
No GPU imports.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import signal
import subprocess
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from e2m.data.extract import ExtractError, extract_function_asm
from e2m.harness.assemble import AssembleError
from e2m.harness.tasks import (
    C_PREAMBLE,
    GCC_COMMON,
    load_jsonl,
    run_task,
    task_from_dict,
)
from e2m.heldout.near_dup import _sig_name
from e2m.loop.leakage import cosine, tokens, word_ngrams
from e2m.scale.parse_c import normalize_body
from e2m.train.failure_analysis import classify_asm_fail

ROOT = Path(__file__).resolve().parents[2]
MBPP_PATH = ROOT / "datasets" / "heldout" / "mbpp.jsonl"
HELDOUT_PATH = ROOT / "eval" / "heldout.jsonl"
VAL_SPLIT = ROOT / "eval" / "val_split.json"
SCALED_TASKS = ROOT / "tasks" / "scaled.jsonl"
STARTER_TASKS = ROOT / "tasks" / "starter.jsonl"
SFT_LORA2 = ROOT / "data" / "scaled" / "sft_lora2.jsonl"
REWRITTEN = ROOT / "data" / "scaled" / "rewritten.jsonl"

SPEC_RATIO = 0.85
C_RATIO = 0.85
EMBED_THRESHOLD = 0.85
WORD_N = 5
VLA_CAP = 8192
# 5-grams keep negation. ExeBench specs share a side-effect tail ("does not
# modify any external variables"), and the stock tokenizer drops "not", so a
# raw overlap drops about half of sft_lora2 against val. Those grams are not
# a copied task. Distinctive grams still drop the row.
_WORD5_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with",
    "is", "are", "be", "by", "as", "at", "from", "that", "this", "it",
}
_BOILER_PHRASES = (
    "does not modify",
    "do not modify",
    "not modify any",
    "modify any",
    "buffer update",
    "external variable",
    "external buffer",
    "input parameter",
    "place buffer",
    "no buffer",
    "returns function does",
    "function returns input",
    "otherwise returns",
    "input value not",
    "not modified",
    "two input value",
)
WORD5_RULE = (
    "word 5-grams over lowercase tokens, negation kept, no short-text fallback; "
    "a shared gram drops the row unless it is the ExeBench side-effect boilerplate "
    "listed in _BOILER_PHRASES"
)

# Do not load datasets/heldout/HumanEval.jsonl.gz. The whole benchmark stays out.


class TranspileError(Exception):
    pass


class _Alarm(Exception):
    pass


def _alarm(_signum: int, _frame: Any) -> None:
    raise _Alarm()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def word5_grams(text: str) -> set[str]:
    toks = [t for t in re.findall(r"[a-z0-9_]+", normalize_ws(text)) if t not in _WORD5_STOP and len(t) > 1]
    if len(toks) < WORD_N:
        return set()
    return {" ".join(toks[i : i + WORD_N]) for i in range(len(toks) - WORD_N + 1)}


def boilerplate_gram(gram: str) -> bool:
    return any(phrase in gram for phrase in _BOILER_PHRASES)


def normalize_signature(signature: str) -> str:
    return normalize_ws(signature)


def _ident(name: str) -> str:
    cleaned = re.sub(r"\W+", "_", name or "").strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "f_" + cleaned
    if cleaned in {"int", "char", "return", "if", "for", "while", "do", "switch", "default", "size"}:
        cleaned = "v_" + cleaned
    return cleaned


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_int(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    return None


def _is_zero(node: ast.AST) -> bool:
    return _const_int(node) == 0


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(elt) for elt in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TranspileError("bad literal")
        return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _literal(node.operand)
    raise TranspileError("non-literal assert arg")


def _py_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple) and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return "list"
    if isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return "list"
    raise TranspileError(f"unsupported value type {type(value).__name__}")


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if abs(value) > (1 << 62):
            raise TranspileError("int out of int64 range")
        return int(value)
    raise TranspileError("expected int")


def parse_assert(src: str) -> tuple[str, list[Any], Any] | None:
    """Return (func_name, args, expected) or None when the assert is not a call compare."""
    try:
        tree = ast.parse(src.strip())
    except SyntaxError as exc:
        raise TranspileError(f"assert syntax: {exc}") from exc
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        raise TranspileError("not an assert")
    test = tree.body[0].test
    expected: Any
    call: ast.AST
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        call = test.left
        try:
            expected = _literal(test.comparators[0])
        except TranspileError:
            return None
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        call = test.operand
        expected = False
    elif isinstance(test, ast.Call):
        call = test
        expected = True
    else:
        return None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.keywords:
        return None
    try:
        args = [_literal(a) for a in call.args]
    except TranspileError:
        return None
    return call.func.id, args, expected


def _oracle_namespace() -> dict[str, Any]:
    safe_builtins = {
        "range": range,
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        "int": int,
        "str": str,
        "bool": bool,
        "sum": sum,
        "enumerate": enumerate,
        "map": map,
        "zip": zip,
        "ord": ord,
        "pow": pow,
        "list": list,
        "tuple": tuple,
        "reversed": reversed,
        "True": True,
        "False": False,
        "None": None,
    }

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "math":
            return math
        raise ImportError(name)

    safe_builtins["__import__"] = _import
    return {"__builtins__": safe_builtins, "math": math}


def _exec_function(code: str, setup: str, name: str) -> Any:
    ns = _oracle_namespace()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        if setup.strip():
            exec(setup, ns, ns)  # noqa: S102
        exec(code, ns, ns)  # noqa: S102
    except _Alarm as exc:
        raise TranspileError("python oracle timeout") from exc
    except Exception as exc:  # noqa: BLE001
        raise TranspileError(f"python oracle: {type(exc).__name__}") from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)
    fn = ns.get(name)
    if not callable(fn):
        raise TranspileError(f"function {name} missing")
    return fn


class _C:
    def __init__(self, params: dict[str, str], module_ints: dict[str, int]):
        self.env = dict(params)
        self.module_ints = module_ints
        self.lines: list[str] = []
        self.hoist: list[str] = []
        self.ind = 1
        self.tmp_i = 0

    def fresh(self, prefix: str = "t") -> str:
        self.tmp_i += 1
        return f"_{prefix}{self.tmp_i}"

    def ensure(self, name: str, kind: str = "int") -> None:
        """Declare a scalar in function scope so a block cannot hide it."""
        if name in self.env:
            return
        self.env[name] = kind
        self.hoist.append(f"int64_t {name} = 0;")

    def hoist_lines(self) -> list[str]:
        return self.hoist

    def line(self, text: str = "") -> None:
        if text:
            self.lines.append("  " * self.ind + text)
        else:
            self.lines.append("")

    def declare(self, name: str, kind: str) -> None:
        if name in self.env and self.env[name] != kind:
            raise TranspileError(f"retyped {name}")
        if name not in self.env:
            self.env[name] = kind
            if kind == "list":
                raise TranspileError("list declare via vla")
            self.line(f"int64_t {name};")

    def stmt(self, node: ast.AST) -> None:
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        if isinstance(node, ast.Return):
            if node.value is None:
                raise TranspileError("return none")
            code, _kind = self.expr(node.value)
            self.line(f"return ({code});")
            return
        if isinstance(node, ast.Assign):
            self._assign(node)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                return
            fake = ast.Assign(targets=[node.target], value=node.value)
            self._assign(fake)
            return
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                if isinstance(node.target, ast.Subscript):
                    self._store_sub(node.target, self._aug_rhs(node.target, node.op, node.value))
                    return
                raise TranspileError("augassign target")
            rhs = self._aug_rhs(node.target, node.op, node.value)
            code, kind = rhs
            self.declare(node.target.id, "int" if kind == "bool" else kind)
            self.line(f"{node.target.id} = ({code});")
            return
        if isinstance(node, ast.If):
            cond, _k = self.expr(node.test)
            self.line(f"if ({cond}) {{")
            self.ind += 1
            for sub in node.body:
                self.stmt(sub)
            self.ind -= 1
            if node.orelse:
                self.line("} else {")
                self.ind += 1
                for sub in node.orelse:
                    self.stmt(sub)
                self.ind -= 1
            self.line("}")
            return
        if isinstance(node, ast.While):
            cond, _k = self.expr(node.test)
            if node.orelse:
                raise TranspileError("while else")
            self.line(f"while ({cond}) {{")
            self.ind += 1
            for sub in node.body:
                self.stmt(sub)
            self.ind -= 1
            self.line("}")
            return
        if isinstance(node, ast.For):
            self._for(node)
            return
        if isinstance(node, ast.Break):
            self.line("break;")
            return
        if isinstance(node, ast.Continue):
            self.line("continue;")
            return
        raise TranspileError(f"stmt {type(node).__name__}")

    def _aug_rhs(self, target: ast.AST, op: ast.operator, value: ast.AST) -> tuple[str, str]:
        left_code, left_kind = self.expr(target)
        right_code, right_kind = self.expr(value)
        return self._binop(op, left_code, left_kind, right_code, right_kind)

    def _assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise TranspileError("multi assign")
        target = node.targets[0]
        if self._is_zero_array(node.value) and isinstance(target, ast.Name):
            self._zero_array(target.id, node.value)
            return
        if isinstance(target, ast.Name):
            code, kind = self.expr(node.value)
            if kind in {"str", "list"}:
                raise TranspileError(f"assign {kind}")
            c_kind = "int" if kind in {"bool", "char"} else kind
            self.ensure(target.id, c_kind)
            self.line(f"{target.id} = ({code});")
            return
        if isinstance(target, ast.Subscript):
            code, _kind = self.expr(node.value)
            self._store_sub(target, (code, "int"))
            return
        raise TranspileError("assign target")

    def _is_zero_array(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            return False
        list_node = node.left if isinstance(node.left, ast.List) else node.right if isinstance(node.right, ast.List) else None
        size_node = node.right if list_node is node.left else node.left if list_node is not None else None
        if list_node is None or size_node is None:
            return False
        if len(list_node.elts) != 1 or not _is_zero(list_node.elts[0]):
            return False
        return True

    def _zero_array(self, name: str, node: ast.BinOp) -> None:
        list_node = node.left if isinstance(node.left, ast.List) else node.right
        size_node = node.right if list_node is node.left else node.left
        size_code, size_kind = self.expr(size_node)
        if size_kind not in {"int", "bool"}:
            raise TranspileError("vla size")
        if name in self.env:
            raise TranspileError(f"retyped {name}")
        n_name = f"{name}_n"
        if n_name in self.env:
            raise TranspileError(f"clash {n_name}")
        self.env[name] = "list"
        self.ensure(n_name, "int")
        self.line(f"{n_name} = ({size_code});")
        self.line(f"if ({n_name} < 0 || {n_name} > {VLA_CAP}) return 0;")
        self.line(f"int64_t {name}[{n_name} > 0 ? {n_name} : 1];")
        idx = self.fresh("i")
        self.line(f"for (int64_t {idx} = 0; {idx} < {n_name}; {idx}++) {name}[{idx}] = 0;")

    def _store_sub(self, target: ast.Subscript, rhs: tuple[str, str]) -> None:
        if not isinstance(target.value, ast.Name):
            raise TranspileError("subscript store")
        name = target.value.id
        if self.env.get(name) != "list":
            raise TranspileError("store into non-list")
        idx, ik = self.expr(target.slice)
        if ik not in {"int", "bool", "char"}:
            raise TranspileError("bad index")
        code, _k = rhs
        self.line(f"{name}[({idx})] = ({code});")

    def _for(self, node: ast.For) -> None:
        if node.orelse:
            raise TranspileError("for else")
        rng = self._range_bounds(node.iter)
        if rng is not None and isinstance(node.target, ast.Name):
            start, stop, step = rng
            var = node.target.id
            self.ensure(var, "int")
            step_c = _const_int(step) if not isinstance(step, str) else None
            start_c, _sk = self.expr(start) if not isinstance(start, str) else (start, "int")
            stop_c, _tk = self.expr(stop) if not isinstance(stop, str) else (stop, "int")
            if step_c == 1:
                self.line(f"for ({var} = ({start_c}); {var} < ({stop_c}); {var}++) {{")
            elif step_c == -1:
                self.line(f"for ({var} = ({start_c}); {var} > ({stop_c}); {var}--) {{")
            else:
                step_code, _ = self.expr(step) if not isinstance(step, str) else (step, "int")
                self.line(
                    f"for ({var} = ({start_c}); ({step_code}) >= 0 ? {var} < ({stop_c}) : {var} > ({stop_c}); "
                    f"{var} += ({step_code})) {{"
                )
            self.ind += 1
            for sub in node.body:
                self.stmt(sub)
            self.ind -= 1
            self.line("}")
            return
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name):
            src = node.iter.id
            kind = self.env.get(src)
            var = node.target.id
            idx = self.fresh("i")
            if kind == "list":
                self.ensure(var, "int")
                self.line(f"for (int64_t {idx} = 0; {idx} < {src}_n; {idx}++) {{")
                self.ind += 1
                self.line(f"{var} = {src}[{idx}];")
            elif kind == "str":
                self.ensure(var, "char")
                self.line(f"for (int64_t {idx} = 0; {src}[{idx}]; {idx}++) {{")
                self.ind += 1
                self.line(f"{var} = (unsigned char){src}[{idx}];")
            else:
                raise TranspileError("for iter")
            for sub in node.body:
                self.stmt(sub)
            self.ind -= 1
            self.line("}")
            return
        raise TranspileError("for pattern")

    def _range_bounds(self, node: ast.AST) -> tuple[Any, Any, Any] | None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "range":
            return None
        args = node.args
        if len(args) == 1:
            return ast.Constant(value=0), args[0], ast.Constant(value=1)
        if len(args) == 2:
            return args[0], args[1], ast.Constant(value=1)
        if len(args) == 3:
            return args[0], args[1], args[2]
        raise TranspileError("range arity")

    def expr(self, node: ast.AST) -> tuple[str, str]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return ("1" if node.value else "0", "bool")
            if isinstance(node.value, int):
                return (str(int(node.value)), "int")
            if isinstance(node.value, str):
                if len(node.value) == 1:
                    escaped = node.value.replace("\\", "\\\\").replace("'", "\\'")
                    return (f"'{escaped}'", "char")
                raise TranspileError("string literal")
            raise TranspileError("constant")
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return (node.id, self.env[node.id])
            if node.id in self.module_ints:
                return (str(self.module_ints[node.id]), "int")
            if node.id in {"True", "False"}:
                return ("1" if node.id == "True" else "0", "bool")
            raise TranspileError(f"name {node.id}")
        if isinstance(node, ast.UnaryOp):
            code, kind = self.expr(node.operand)
            if isinstance(node.op, ast.Not):
                return (f"(!({code}))", "bool")
            if isinstance(node.op, ast.USub):
                return (f"(-({code}))", "int")
            if isinstance(node.op, ast.UAdd):
                return (code, kind)
            if isinstance(node.op, ast.Invert):
                return (f"(~({code}))", "int")
            raise TranspileError("unary")
        if isinstance(node, ast.BinOp):
            left, lk = self.expr(node.left)
            right, rk = self.expr(node.right)
            return self._binop(node.op, left, lk, right, rk)
        if isinstance(node, ast.BoolOp):
            parts = []
            for value in node.values:
                code, _k = self.expr(value)
                parts.append(f"({code})")
            op = " && " if isinstance(node.op, ast.And) else " || "
            return (op.join(parts), "bool")
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            cond, _ck = self.expr(node.test)
            a, ak = self.expr(node.body)
            b, bk = self.expr(node.orelse)
            kind = "bool" if ak == "bool" and bk == "bool" else "int"
            return (f"(({cond}) ? ({a}) : ({b}))", kind)
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise TranspileError(f"expr {type(node).__name__}")

    def _binop(self, op: ast.operator, left: str, lk: str, right: str, rk: str) -> tuple[str, str]:
        if isinstance(op, ast.Add):
            return (f"(({left}) + ({right}))", "int")
        if isinstance(op, ast.Sub):
            return (f"(({left}) - ({right}))", "int")
        if isinstance(op, ast.Mult):
            return (f"(({left}) * ({right}))", "int")
        if isinstance(op, ast.FloorDiv):
            return (
                f"(({left})/({right}) - (((({left})%({right})) != 0 && (({left}) < 0) != (({right}) < 0)) ? 1 : 0))",
                "int",
            )
        if isinstance(op, ast.Mod):
            return (
                f"((({left})%({right})) + ((({right}) < 0) ? ((({left})%({right})) > 0 ? ({right}) : 0) "
                f": ((({left})%({right})) < 0 ? ({right}) : 0)))",
                "int",
            )
        if isinstance(op, ast.BitAnd):
            return (f"(({left}) & ({right}))", "int")
        if isinstance(op, ast.BitOr):
            return (f"(({left}) | ({right}))", "int")
        if isinstance(op, ast.BitXor):
            return (f"(({left}) ^ ({right}))", "int")
        if isinstance(op, ast.LShift):
            return (f"(({left}) << ({right}))", "int")
        if isinstance(op, ast.RShift):
            return (f"(({left}) >> ({right}))", "int")
        if isinstance(op, ast.Pow):
            stripped = right.strip().strip("()")
            if not re.fullmatch(r"-?\d+", stripped):
                raise TranspileError("pow")
            const = int(stripped)
            if not 0 <= const <= 6:
                raise TranspileError("pow")
            if const == 0:
                return ("1", "int")
            return (" * ".join([f"({left})"] * const), "int")
        _ = (lk, rk)
        raise TranspileError(f"binop {type(op).__name__}")

    def _compare(self, node: ast.Compare) -> tuple[str, str]:
        parts = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            lc, _lk = self.expr(left)
            rc, _rk = self.expr(right)
            sym = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }.get(type(op))
            if sym is None:
                if isinstance(op, ast.In):
                    parts.append(self._contains(right, lc))
                elif isinstance(op, ast.NotIn):
                    parts.append(f"(!({self._contains(right, lc)}))")
                else:
                    raise TranspileError("compare")
            else:
                parts.append(f"(({lc}) {sym} ({rc}))")
            left = right
        return (" && ".join(parts), "bool")

    def _contains(self, container: ast.AST, needle: str) -> str:
        if not isinstance(container, ast.Name):
            raise TranspileError("contains")
        kind = self.env.get(container.id)
        idx = self.fresh("i")
        if kind == "str":
            return (
                f"({{ int64_t {idx} = 0, _hit = 0; while ({container.id}[{idx}]) {{ "
                f"if ((unsigned char){container.id}[{idx}] == ({needle})) _hit = 1; {idx}++; }} _hit; }})"
            )
        if kind == "list":
            return (
                f"({{ int64_t {idx}, _hit = 0; for ({idx} = 0; {idx} < {container.id}_n; {idx}++) "
                f"if ({container.id}[{idx}] == ({needle})) _hit = 1; _hit; }})"
            )
        raise TranspileError("contains")

    def _subscript(self, node: ast.Subscript) -> tuple[str, str]:
        if isinstance(node.slice, ast.Slice):
            raise TranspileError("slice")
        if not isinstance(node.value, ast.Name):
            raise TranspileError("subscript")
        name = node.value.id
        idx, ik = self.expr(node.slice)
        if ik not in {"int", "bool", "char"}:
            raise TranspileError("index type")
        kind = self.env.get(name)
        if kind == "list":
            return (f"({name}[({idx})])", "int")
        if kind == "str":
            return (f"((unsigned char){name}[({idx})])", "char")
        raise TranspileError("subscript base")

    def _call(self, node: ast.Call) -> tuple[str, str]:
        if node.keywords:
            raise TranspileError("keywords")
        func = node.func
        if isinstance(func, ast.Attribute):
            return self._method(func, node.args)
        if not isinstance(func, ast.Name):
            raise TranspileError("call")
        name = func.id
        args = node.args
        if name == "len" and len(args) == 1 and isinstance(args[0], ast.Name):
            src = args[0].id
            kind = self.env.get(src)
            if kind == "list":
                return (f"{src}_n", "int")
            if kind == "str":
                tmp = self.fresh("n")
                return (
                    f"({{ int64_t {tmp} = 0; while ({src}[{tmp}]) {tmp}++; {tmp}; }})",
                    "int",
                )
            raise TranspileError("len")
        if name == "abs" and len(args) == 1:
            code, _k = self.expr(args[0])
            return (f"(({code}) < 0 ? -({code}) : ({code}))", "int")
        if name in {"min", "max"} and len(args) == 2:
            a, _ak = self.expr(args[0])
            b, _bk = self.expr(args[1])
            op = "<" if name == "min" else ">"
            return (f"(({a}) {op} ({b}) ? ({a}) : ({b}))", "int")
        if name in {"min", "max"} and len(args) == 1 and isinstance(args[0], ast.Name):
            src = args[0].id
            if self.env.get(src) != "list":
                raise TranspileError(name)
            idx = self.fresh("i")
            best = self.fresh("b")
            op = "<" if name == "min" else ">"
            return (
                f"({{ int64_t {idx}, {best} = 0; if ({src}_n > 0) {{ {best} = {src}[0]; "
                f"for ({idx} = 1; {idx} < {src}_n; {idx}++) if ({src}[{idx}] {op} {best}) {best} = {src}[{idx}]; }} {best}; }})",
                "int",
            )
        if name == "sum" and len(args) == 1 and isinstance(args[0], ast.Name):
            src = args[0].id
            if self.env.get(src) != "list":
                raise TranspileError("sum")
            idx = self.fresh("i")
            acc = self.fresh("s")
            return (
                f"({{ int64_t {idx}, {acc} = 0; for ({idx} = 0; {idx} < {src}_n; {idx}++) {acc} += {src}[{idx}]; {acc}; }})",
                "int",
            )
        if name == "int" and len(args) == 1:
            if (
                isinstance(args[0], ast.Call)
                and isinstance(args[0].func, ast.Attribute)
                and args[0].func.attr == "sqrt"
                and len(args[0].args) == 1
            ):
                inner, _k = self.expr(args[0].args[0])
                r = self.fresh("r")
                x = self.fresh("x")
                return (
                    f"({{ int64_t {x} = ({inner}); int64_t {r} = 0; "
                    f"if ({x} > 0) {{ while ({r} < 3037000499LL && ({r} + 1) <= {x} / ({r} + 1)) {r}++; }} {r}; }})",
                    "int",
                )
            code, kind = self.expr(args[0])
            if kind in {"int", "bool", "char"}:
                return (code, "int")
            raise TranspileError("int()")
        if name == "ord" and len(args) == 1:
            code, kind = self.expr(args[0])
            if kind not in {"char", "int"}:
                raise TranspileError("ord")
            return (f"((int64_t)({code}))", "int")
        if name == "range":
            raise TranspileError("range value")
        raise TranspileError(f"call {name}")

    def _method(self, func: ast.Attribute, args: list[ast.AST]) -> tuple[str, str]:
        if not isinstance(func.value, ast.Name):
            raise TranspileError("method")
        src = func.value.id
        kind = self.env.get(src)
        attr = func.attr
        if kind not in {"str", "char"}:
            raise TranspileError(f"method {attr}")
        if attr in {"isdigit", "isalpha", "islower", "isupper"} and not args and kind == "char":
            pred = {
                "isdigit": f"({src} >= '0' && {src} <= '9')",
                "isalpha": f"(({src} >= 'A' && {src} <= 'Z') || ({src} >= 'a' && {src} <= 'z'))",
                "islower": f"({src} >= 'a' && {src} <= 'z')",
                "isupper": f"({src} >= 'A' && {src} <= 'Z')",
            }[attr]
            return (f"({pred})", "bool")
        if attr == "lower" and not args and kind == "char":
            return (f"(({src} >= 'A' && {src} <= 'Z') ? ({src} - 'A' + 'a') : {src})", "char")
        if attr == "upper" and not args and kind == "char":
            return (f"(({src} >= 'a' && {src} <= 'z') ? ({src} - 'a' + 'A') : {src})", "char")
        if attr in {"isdigit", "isalpha"} and not args and kind == "str":
            idx = self.fresh("i")
            pred = (
                f"({src}[{idx}] >= '0' && {src}[{idx}] <= '9')"
                if attr == "isdigit"
                else f"(({src}[{idx}] >= 'A' && {src}[{idx}] <= 'Z') || ({src}[{idx}] >= 'a' && {src}[{idx}] <= 'z'))"
            )
            return (
                f"({{ int64_t {idx} = 0, _ok = {src}[0] ? 1 : 0; while (_ok && {src}[{idx}]) {{ "
                f"if (!({pred})) _ok = 0; {idx}++; }} _ok; }})",
                "bool",
            )
        raise TranspileError(f"method {attr}")


def _mutated_names(fn: ast.FunctionDef) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(fn):
        target = None
        if isinstance(node, ast.Assign):
            for item in node.targets:
                if isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name):
                    found.add(item.value.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            target = node.target
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Subscript):
            target = node.target
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            found.add(target.value.id)
    return found


def _module_ints(tree: ast.Module) -> dict[str, int]:
    found: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _const_int(node.value)
            if value is not None:
                found[node.targets[0].id] = value
    return found


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise TranspileError("function count")
    fn = found[0]
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise TranspileError("varargs")
    for child in ast.walk(fn):
        if isinstance(child, ast.FunctionDef) and child is not fn:
            raise TranspileError("nested function")
        if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Yield, ast.YieldFrom)):
            raise TranspileError(type(child).__name__)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == name:
            raise TranspileError("recursion")
    return fn


def _param_kinds(cases: list[tuple[list[Any], Any]]) -> tuple[list[str], str]:
    if not cases:
        raise TranspileError("no tests")
    width = len(cases[0][0])
    kinds: list[str] = []
    for index in range(width):
        kind = _py_kind(cases[0][0][index])
        for args, _exp in cases:
            if len(args) != width:
                raise TranspileError("arity mismatch")
            if _py_kind(args[index]) != kind:
                raise TranspileError("mixed param type")
            if kind == "list":
                seq = args[index]
                if len(seq) > 4096:
                    raise TranspileError("list too long")
            if kind == "int":
                _as_int(args[index])
            if kind == "str" and len(args[index]) > 4096:
                raise TranspileError("string too long")
        kinds.append(kind)
    ret_kind = _py_kind(cases[0][1])
    if ret_kind not in {"int", "bool"}:
        raise TranspileError(f"return {ret_kind}")
    for _args, exp in cases:
        if _py_kind(exp) != ret_kind:
            raise TranspileError("mixed return")
        if ret_kind == "int":
            _as_int(exp)
    return kinds, ret_kind


def convert_mbpp_record(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one MBPP row into a harness task, or raise TranspileError."""
    task_num = int(row["task_id"])
    code = str(row.get("code") or "").replace("\r\n", "\n").replace("\r", "\n")
    setup = str(row.get("test_setup_code") or "")
    spec = str(row.get("text") or "").strip()
    if not spec or not code:
        raise TranspileError("empty")
    if len(code) > 6000:
        raise TranspileError("code too long")
    asserts: list[tuple[str, list[Any], Any]] = []
    for blob in list(row.get("test_list") or []) + list(row.get("challenge_test_list") or []):
        parsed = parse_assert(str(blob))
        if parsed is None:
            continue
        asserts.append(parsed)
    if len(asserts) < 1:
        raise TranspileError("no asserts")
    fname = asserts[0][0]
    if any(name != fname for name, _a, _e in asserts):
        raise TranspileError("mixed assert name")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise TranspileError(f"syntax: {exc.msg}") from exc
    fn = _find_function(tree, fname)
    if len(fn.args.args) != len(asserts[0][1]):
        raise TranspileError("signature arity")
    cases = [(args, exp) for _name, args, exp in asserts]
    kinds, ret_kind = _param_kinds(cases)
    py_fn = _exec_function(code, setup, fname)
    checked: list[tuple[list[Any], Any]] = []
    for args, exp in cases:
        try:
            got = py_fn(*args)
        except Exception as exc:  # noqa: BLE001
            raise TranspileError(f"oracle call: {type(exc).__name__}") from exc
        if isinstance(exp, bool):
            if bool(got) != exp:
                raise TranspileError("assert mismatch")
        elif got != exp:
            raise TranspileError("assert mismatch")
        checked.append((args, exp))
    py_names = [arg.arg for arg in fn.args.args]
    mutated = _mutated_names(fn)
    c_params: list[str] = []
    env: dict[str, str] = {}
    harness_params: list[tuple[str, str]] = []
    for py_name, kind in zip(py_names, kinds):
        c_name = _ident(py_name)
        if kind == "list":
            ptr = c_name
            length = f"{c_name}_n"
            if ptr in env or length in env:
                raise TranspileError("name clash")
            env[ptr] = "list"
            env[length] = "int"
            c_type = "int64_t *" if py_name in mutated else "const int64_t *"
            c_params.append(f"{c_type}{ptr}")
            c_params.append(f"int64_t {length}")
            harness_params.append((ptr, "list"))
            harness_params.append((length, "len"))
        elif kind == "str":
            if c_name in env:
                raise TranspileError("name clash")
            env[c_name] = "str"
            c_params.append(f"const char *{c_name}")
            harness_params.append((c_name, "str"))
        else:
            if c_name in env:
                raise TranspileError("name clash")
            env[c_name] = "int"
            c_params.append(f"int64_t {c_name}")
            harness_params.append((c_name, "int"))
    c_fname = _ident(f"mbpp{task_num}_{fname}")
    emitter = _C(env, _module_ints(tree))
    for node in fn.body:
        emitter.stmt(node)
    if not any(ln.strip().startswith("return") for ln in emitter.lines):
        raise TranspileError("no return")
    signature = f"int64_t {c_fname}({', '.join(c_params) if c_params else 'void'})"
    body = "\n".join([*emitter.hoist_lines(), *emitter.lines])
    reference = f"{signature} {{\n{body}\n  return 0;\n}}\n"
    tests = []
    for args, exp in checked:
        h_args: list[Any] = []
        arg_i = 0
        for pname, pkind in harness_params:
            if pkind == "len":
                seq = args[arg_i - 1]
                h_args.append(len(seq))
                continue
            value = args[arg_i]
            arg_i += 1
            if pkind == "list":
                h_args.append({"id": pname, "ptr": "i64", "data": [int(x) for x in value]})
            elif pkind == "str":
                h_args.append({"id": pname, "ptr": "cstr", "data": str(value)})
            else:
                h_args.append(_as_int(value) if not isinstance(value, bool) else int(value))
        ret = int(exp) if not isinstance(exp, bool) else int(exp)
        _ = ret_kind
        tests.append({"args": h_args, "ret": ret})
    if len(tests) > 12:
        tests = tests[:12]
    return {
        "id": c_fname,
        "spec": spec,
        "signature": signature,
        "reference_c": reference,
        "tests": tests,
        "source": f"MBPP/{task_num}",
        "origin": "mbpp",
        "mbpp_task_id": task_num,
    }


def _needs_qemu(run: Any) -> bool:
    if run.assemble_error or run.passed:
        return False
    for case in run.cases:
        if case.emu.status == "invalid_insn":
            return True
    return False


def verify_reference(task: dict[str, Any], *, opts: tuple[str, ...] = ("-O0", "-O2")) -> str | None:
    """Return an error string when gcc -O0/-O2 does not pass the harness."""
    obj = task_from_dict(task)
    for opt in opts:
        try:
            asm = compile_function_asm(task["reference_c"], obj.sig().name, opt=opt)
        except (AssembleError, ExtractError, OSError, subprocess.TimeoutExpired) as exc:
            return f"{opt}: {exc}"[:300]
        run = run_task(obj, asm, syntax="att", timeout_s=1.0, backend="unicorn")
        if _needs_qemu(run):
            run = run_task(obj, asm, syntax="att", timeout_s=2.0, backend="qemu")
        if not run.passed:
            return f"{opt}: {run.fail_reason}"[:300]
    return None


def verify_task_ok(task: dict[str, Any]) -> tuple[str, str | None]:
    try:
        return str(task.get("id")), verify_reference(task)
    except Exception as exc:  # noqa: BLE001
        return str(task.get("id")), f"{type(exc).__name__}: {exc}"[:300]


def compile_function_asm(c_source: str, name: str, *, opt: str = "-O2") -> str:
    if not opt.startswith("-"):
        opt = "-" + opt
    src_body = c_source if "#include" in c_source else C_PREAMBLE + c_source
    with tempfile.TemporaryDirectory(prefix="e2m-r5-") as td:
        src = Path(td) / "f.c"
        asm_path = Path(td) / "f.s"
        src.write_text(src_body, encoding="utf-8")
        cmd = ["gcc", opt, "-S", "-march=x86-64", *GCC_COMMON, "-o", str(asm_path), str(src)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
        if proc.returncode != 0 or not asm_path.is_file():
            raise AssembleError(f"gcc {opt} -S failed", details=(proc.stderr or proc.stdout or "")[-400:])
        raw = asm_path.read_text(encoding="utf-8", errors="replace")
    return extract_function_asm(raw, name)


def score_c_source(task: dict[str, Any], c_source: str) -> bool:
    obj = task_from_dict(task)
    name = obj.sig().name
    # The model must define the task's function name. Reject a renamed helper.
    if name not in c_source:
        return False
    try:
        asm = compile_function_asm(c_source, name, opt="-O2")
    except (AssembleError, ExtractError, OSError, subprocess.TimeoutExpired):
        return False
    run = run_task(obj, asm, syntax="att", timeout_s=1.0, backend="unicorn")
    if _needs_qemu(run):
        run = run_task(obj, asm, syntax="att", timeout_s=2.0, backend="qemu")
    return bool(run.passed)


def lower_asm(task: dict[str, Any], c_source: str, *, opt: str) -> dict[str, Any] | None:
    obj = task_from_dict(task)
    name = obj.sig().name
    try:
        asm = compile_function_asm(c_source, name, opt=opt)
    except (AssembleError, ExtractError, OSError, subprocess.TimeoutExpired):
        return None
    run = run_task(obj, asm, syntax="att", timeout_s=1.0, backend="unicorn")
    backend = "unicorn"
    if _needs_qemu(run):
        run = run_task(obj, asm, syntax="att", timeout_s=2.0, backend="qemu")
        backend = "qemu"
    if not run.passed:
        return None
    goal = "size" if opt in {"-Os", "Os"} else "default"
    flag = "-Os" if goal == "size" else "-O2"
    return {
        "asm": asm if asm.endswith("\n") else asm + "\n",
        "goal": goal,
        "opt": flag,
        "backend": backend,
        "size": int(run.code_size or 0),
        "insn_count": int(run.insn_count or 0),
        "sha256": hashlib.sha256(asm.encode()).hexdigest(),
    }


def distill_pairs(task: dict[str, Any], c_texts: list[str], *, origin: str) -> list[dict[str, Any]]:
    """Keep up to two distinct verified asms (prefer -O2 then -Os of the first good C)."""
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c_source in c_texts:
        text = (c_source or "").strip()
        if not text:
            continue
        produced = False
        for opt in ("-O2", "-Os"):
            lowered = lower_asm(task, text, opt=opt)
            if lowered is None or lowered["sha256"] in seen:
                continue
            seen.add(lowered["sha256"])
            produced = True
            kept.append(
                {
                    "task_id": task["id"],
                    "spec": task["spec"],
                    "signature": task["signature"],
                    "goal": lowered["goal"],
                    "asm": lowered["asm"],
                    "sha256": lowered["sha256"],
                    "size": lowered["size"],
                    "insn_count": lowered["insn_count"],
                    "compiler": "gcc",
                    "flags": [lowered["opt"], "-march=x86-64"],
                    "origin": origin,
                    "reference_c": task.get("reference_c") or "",
                    "model_c": text,
                    "backend": lowered["backend"],
                }
            )
            if len(kept) >= 2:
                return kept
        if produced and len(kept) >= 2:
            break
    return kept


def sft_row(pair: dict[str, Any]) -> dict[str, Any]:
    """Training row. No `c` field, so the asm target stays plain AT&T."""
    return {
        "task_id": pair["task_id"],
        "spec": pair["spec"],
        "signature": pair["signature"],
        "goal": pair["goal"],
        "asm": pair["asm"],
        "sha256": pair.get("sha256"),
        "size": pair.get("size"),
        "insn_count": pair.get("insn_count"),
        "compiler": pair.get("compiler") or "gcc",
        "flags": pair.get("flags") or [],
        "origin": pair.get("origin") or "",
    }


def load_mbpp_rows(path: Path | None = None) -> list[dict[str, Any]]:
    # MBPP only. HumanEval is not a training source.
    src = path or MBPP_PATH
    rows = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def heldout_mbpp_block(path: Path | None = None) -> set[str]:
    """MBPP source ids that already sit in eval/heldout.jsonl, plus every held-out id."""
    blocked: set[str] = set()
    src = path or HELDOUT_PATH
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            blocked.add(str(row.get("id") or ""))
            source = str(row.get("source") or "")
            if source.startswith("MBPP/"):
                blocked.add(source)
    blocked.discard("")
    return blocked


def build_mbpp_pool(
    out_path: Path,
    *,
    mbpp_path: Path | None = None,
    heldout_path: Path | None = None,
    limit_kept: int = 0,
    jobs: int = 8,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    blocked = heldout_mbpp_block(heldout_path)
    existing: list[dict[str, Any]] = []
    seen_src: set[str] = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                existing.append(row)
                seen_src.add(str(row.get("source") or ""))
    rejects: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    scanned = 0
    for raw in load_mbpp_rows(mbpp_path):
        scanned += 1
        source = f"MBPP/{int(raw['task_id'])}"
        if source in blocked or source in seen_src:
            rejects["blocked_or_done"] += 1
            continue
        try:
            task = convert_mbpp_record(raw)
        except TranspileError as exc:
            rejects[str(exc).split(":", 1)[0][:80]] += 1
            continue
        except Exception as exc:  # noqa: BLE001
            rejects[type(exc).__name__] += 1
            continue
        if task["source"] in blocked or task["id"] in blocked:
            rejects["blocked_or_done"] += 1
            continue
        candidates.append(task)
        if limit_kept and len(candidates) >= max(limit_kept * 20, limit_kept):
            break
    verified: list[dict[str, Any]] = []
    if limit_kept:
        for task in candidates:
            _tid, err = verify_task_ok(task)
            if err:
                rejects["verify:" + err.split(":", 1)[0][:60]] += 1
                continue
            verified.append(task)
            if len(verified) >= limit_kept:
                break
    elif candidates:
        workers = max(1, jobs)
        if workers == 1:
            for task in candidates:
                _tid, err = verify_task_ok(task)
                if err:
                    rejects["verify"] += 1
                else:
                    verified.append(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(verify_task_ok, task) for task in candidates]
                ok_ids = set()
                for fut in as_completed(futs):
                    tid, err = fut.result()
                    if err:
                        rejects["verify"] += 1
                    else:
                        ok_ids.add(tid)
            verified = [task for task in candidates if task["id"] in ok_ids]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for task in verified:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    kept = existing + verified
    stats = {
        "scanned": scanned,
        "candidates": len(candidates),
        "verified_new": len(verified),
        "kept": len(kept),
        "rejects": dict(rejects.most_common(40)),
        "out": str(out_path),
    }
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"mbpp pool kept {len(kept)} new {len(verified)} rejects {rejects.most_common(8)}", flush=True)
    return stats


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _task_index(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = str(row.get("id") or row.get("task_id") or "")
                if tid and tid not in found:
                    found[tid] = row
    return found


def load_protected_tasks(
    *,
    heldout_path: Path | None = None,
    val_split_path: Path | None = None,
    scaled_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Held-out tasks plus the 90 val tasks. Read-only."""
    protected: list[dict[str, Any]] = []
    heldout = heldout_path or HELDOUT_PATH
    with heldout.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                row["_split"] = "heldout"
                protected.append(row)
    val_path = val_split_path or VAL_SPLIT
    scaled = scaled_path or SCALED_TASKS
    val_ids = set(json.loads(val_path.read_text(encoding="utf-8"))["task_ids"])
    if scaled.is_file():
        with scaled.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("id") or "") in val_ids:
                    row["_split"] = "val"
                    protected.append(row)
    return protected


class _Ref:
    def __init__(self, row: dict[str, Any]):
        self.id = str(row.get("id") or "")
        self.split = str(row.get("_split") or "")
        self.name = _sig_name(str(row.get("signature") or "")).lower()
        self.signature = normalize_signature(str(row.get("signature") or ""))
        self.spec = str(row.get("spec") or "")
        self.spec_norm = normalize_ws(self.spec)
        c_text = str(row.get("reference_c") or "")
        self.body = normalize_body(c_text) if c_text else ""
        self.c_norm = normalize_ws(c_text)
        self.word5 = word5_grams(self.spec)
        self.spec_len = len(self.spec_norm)
        self.c_len = len(self.c_norm)


def _close_len(a: int, b: int, ratio: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return min(a, b) / max(a, b) >= ratio


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def leakage_reason(row: dict[str, Any], refs: list[_Ref], *, indexes: dict[str, Any]) -> str | None:
    """Return `check:other_id` when the row must be dropped."""
    tid = str(row.get("task_id") or row.get("id") or "")
    if tid and tid in indexes["ids"]:
        return f"id:{tid}"
    source = str(row.get("source") or "")
    if source and source in indexes["sources"]:
        return f"id:{source}"
    name = _sig_name(str(row.get("signature") or "")).lower()
    if name and name in indexes["names"]:
        return f"name:{name}"
    bare = name
    for prefix in ("mbpp_", "he_"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
    if bare and len(bare) >= 6 and bare in indexes["bare_names"]:
        return f"name:{bare}"
    sig = normalize_signature(str(row.get("signature") or ""))
    if sig and sig in indexes["sigs"]:
        return f"signature:{sig[:80]}"
    body = normalize_body(str(row.get("reference_c") or "")) if row.get("reference_c") else ""
    if body and body in indexes["bodies"]:
        return f"body:{indexes['bodies'][body]}"
    spec = str(row.get("spec") or "")
    spec_n = normalize_ws(spec)
    c_n = normalize_ws(str(row.get("reference_c") or ""))
    grams = word5_grams(spec)
    if grams:
        for gram in grams:
            hit = indexes["word5"].get(gram)
            if hit:
                return f"word5:{hit}"
    for ref in refs:
        if spec_n and _close_len(len(spec_n), ref.spec_len, SPEC_RATIO):
            score = _ratio(spec_n, ref.spec_norm)
            if score >= SPEC_RATIO:
                return f"spec_ratio:{ref.id}:{score:.3f}"
        if c_n and ref.c_norm and _close_len(len(c_n), ref.c_len, C_RATIO):
            score = _ratio(c_n, ref.c_norm)
            if score >= C_RATIO:
                return f"c_ratio:{ref.id}:{score:.3f}"
    return None


def _indexes(refs: list[_Ref]) -> dict[str, Any]:
    names = {ref.name for ref in refs if ref.name}
    bare = set()
    for name in names:
        bare_name = name
        for prefix in ("mbpp_", "he_"):
            if bare_name.startswith(prefix):
                bare_name = bare_name[len(prefix) :]
        if len(bare_name) >= 6:
            bare.add(bare_name)
    word5: dict[str, str] = {}
    for ref in refs:
        for gram in ref.word5:
            if boilerplate_gram(gram):
                continue
            word5.setdefault(gram, ref.id)
    bodies = {}
    for ref in refs:
        if ref.body:
            bodies.setdefault(ref.body, ref.id)
    return {
        "ids": {ref.id for ref in refs},
        "sources": set(),
        "names": names,
        "bare_names": bare,
        "sigs": {ref.signature for ref in refs if ref.signature},
        "bodies": bodies,
        "word5": word5,
    }


def screen_text(
    rows: list[dict[str, Any]],
    protected: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    protected_rows = protected if protected is not None else load_protected_tasks()
    refs = [_Ref(row) for row in protected_rows]
    indexes = _indexes(refs)
    for row in protected_rows:
        source = str(row.get("source") or "")
        # "MBPP/3" and "HumanEval/10" identify a task. "exebench" is the whole
        # train corpus, and every val row carries it, so it is not an id.
        if "/" in source:
            indexes["sources"].add(source)
    kept = []
    dropped = []
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        origin = str(row.get("origin") or row.get("source") or "unknown")
        by_source[origin]["checked"] += 1
        reason = leakage_reason(row, refs, indexes=indexes)
        if reason:
            check = reason.split(":", 1)[0]
            counts[check] += 1
            by_source[origin]["dropped"] += 1
            by_source[origin][check] += 1
            dropped.append(
                {
                    "task_id": row.get("task_id") or row.get("id"),
                    "origin": origin,
                    "reason": reason,
                }
            )
        else:
            kept.append(row)
            by_source[origin]["kept"] += 1
    summary = {
        "checked": len(rows),
        "kept_text": len(kept),
        "dropped_text": len(dropped),
        "by_check": dict(counts),
        "by_source": {key: dict(val) for key, val in by_source.items()},
    }
    return kept, dropped, summary


def _embed_one(text: str, *, host: str, model: str, cpu: bool) -> list[float]:
    payload: dict[str, Any] = {"model": model, "prompt": text[:4000], "keep_alive": "20m"}
    if cpu:
        payload["options"] = {"num_gpu": 0}
    data = json.dumps(payload).encode()
    req = __import__("urllib.request", fromlist=["Request"]).Request(
        host.rstrip("/") + "/api/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with __import__("urllib.request", fromlist=["urlopen"]).urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    vec = body.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("empty embedding")
    return [float(x) for x in vec]


def _load_embed_cache(path: Path) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            sha = str(row.get("sha") or "")
            vec = row.get("vec")
            if sha and isinstance(vec, list):
                found[sha] = [float(x) for x in vec]
    return found


def _sha_text(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).encode()).hexdigest()


def screen_embed(
    rows: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    *,
    cache_path: Path,
    host: str = "http://127.0.0.1:11434",
    model: str = "bge-m3:latest",
    cpu: bool = False,
    threshold: float = EMBED_THRESHOLD,
    stop_check: Callable[[], str | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = _load_embed_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    refs = [_Ref(row) for row in protected]

    def vec_for(text: str) -> list[float]:
        sha = _sha_text(text)
        if sha in cache:
            return cache[sha]
        if stop_check is not None:
            why = stop_check()
            if why:
                raise StopIteration(why)
        vec = _embed_one(text, host=host, model=model, cpu=cpu)
        cache[sha] = vec
        with cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"sha": sha, "vec": vec}) + "\n")
        return vec

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen = 0
    try:
        ref_vecs = [(ref.id, vec_for(ref.spec)) for ref in refs if ref.spec]
        for i, row in enumerate(rows, 1):
            spec = str(row.get("spec") or "")
            if not spec.strip():
                dropped.append({"task_id": row.get("task_id") or row.get("id"), "origin": row.get("origin"), "reason": "embed:empty"})
                seen = i
                continue
            vec = vec_for(spec)
            best_id = ""
            best = 0.0
            for rid, rvec in ref_vecs:
                score = cosine(vec, rvec)
                if score > best:
                    best = score
                    best_id = rid
            if best >= threshold:
                dropped.append(
                    {
                        "task_id": row.get("task_id") or row.get("id"),
                        "origin": row.get("origin"),
                        "reason": f"embed:{best_id}:{best:.3f}",
                    }
                )
            else:
                kept.append(row)
            seen = i
            if i % 200 == 0:
                print(f"embed {i}/{len(rows)} kept {len(kept)} dropped {len(dropped)}", flush=True)
    except StopIteration as exc:
        pending = len(rows) - seen
        return kept, dropped, {
            "stopped": str(exc),
            "embedded": False,
            "pending": pending,
            "threshold": threshold,
            "model": model,
            "cpu": cpu,
        }
    return kept, dropped, {
        "stopped": None,
        "embedded": True,
        "pending": 0,
        "threshold": threshold,
        "model": model,
        "cpu": cpu,
    }


def attach_reference_c(rows: list[dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        if not item.get("reference_c"):
            src = tasks.get(str(item.get("task_id") or ""))
            if src:
                item["reference_c"] = src.get("reference_c") or ""
        out.append(item)
    return out


def load_sft_rows(path: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path or SFT_LORA2):
        item = dict(row)
        item["origin"] = item.get("origin") or "sft_lora2"
        item["task_id"] = str(item.get("task_id") or "")
        rows.append(item)
    return rows


def build_train_manifest(
    out_path: Path,
    *,
    scaled_path: Path | None = None,
    val_split_path: Path | None = None,
    rewritten_path: Path | None = None,
    heldout_path: Path | None = None,
    cap: int = 400,
) -> dict[str, Any]:
    """Capped train-pool tasks. Rewritten specs come first. Val and held-out ids are removed."""
    val_ids = set(json.loads((val_split_path or VAL_SPLIT).read_text(encoding="utf-8"))["task_ids"])
    held_ids = set()
    held_sources = set()
    with (heldout_path or HELDOUT_PATH).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            held_ids.add(str(row.get("id") or ""))
            if row.get("source"):
                held_sources.add(str(row["source"]))
    rewritten: dict[str, str] = {}
    rew_path = rewritten_path or REWRITTEN
    if rew_path.is_file():
        with rew_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("dropped"):
                    continue
                spec = str(row.get("spec") or "").strip()
                tid = str(row.get("task_id") or "")
                if tid and spec and tid not in val_ids and tid not in held_ids:
                    rewritten[tid] = spec
    ranked: list[tuple[str, dict[str, Any]]] = []
    with (scaled_path or SCALED_TASKS).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("id") or "")
            if not tid or tid in val_ids or tid in held_ids:
                continue
            if str(row.get("source") or "") in held_sources:
                continue
            spec = rewritten.get(tid) or str(row.get("spec") or "")
            item = {
                "id": tid,
                "spec": spec,
                "signature": row.get("signature") or "",
                "reference_c": row.get("reference_c") or "",
                "tests": row.get("tests") or [],
                "source": row.get("source") or "exebench",
                "origin": "train",
                "spec_kind": "rewritten" if tid in rewritten else "original",
            }
            key = hashlib.sha256(f"r5:{tid}".encode()).hexdigest()
            # Rewritten prompts are the HumanEval-style half of this pool.
            rank = ("0" if tid in rewritten else "1") + key
            ranked.append((rank, item))
    ranked.sort(key=lambda pair: pair[0])
    chosen = [item for _rank, item in ranked[:cap]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in chosen:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats = {
        "cap": cap,
        "chosen": len(chosen),
        "rewritten": sum(1 for item in chosen if item["spec_kind"] == "rewritten"),
        "original": sum(1 for item in chosen if item["spec_kind"] == "original"),
        "out": str(out_path),
    }
    print(f"train manifest {stats}", flush=True)
    return stats


def write_mix(
    out_path: Path,
    *,
    kept_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the SFT mix. Training rows do not carry a `c` field."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    goals: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            origin = str(row.get("origin") or "unknown")
            goal = str(row.get("goal") or "default")
            counts[origin] += 1
            goals[goal] += 1
            fh.write(json.dumps(sft_row(row) if row.get("model_c") or row.get("origin") not in {None, "", "sft_lora2"} else _sft_base_row(row), ensure_ascii=False) + "\n")
    total = sum(counts.values()) or 1
    stats = {
        "rows": sum(counts.values()),
        "by_origin": dict(counts),
        "by_goal": dict(goals),
        "proportions": {key: round(val / total, 4) for key, val in counts.items()},
        "out": str(out_path),
        "auxiliary_c_to_asm": False,
        "auxiliary_reason": (
            "Eval uses the English-to-asm prompt (Goal: default). "
            "A C-to-asm auxiliary would train a different prompt than inference, "
            "and a row that contains both c and asm is treated as C-then-asm text by the trainer. "
            "Distill targets stay plain gcc AT&T of the teacher's passing C."
        ),
    }
    return stats


def _sft_base_row(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve an existing sft_lora2 row and tag its origin. Do not add `c`."""
    kept = {
        "task_id": row.get("task_id"),
        "spec": row.get("spec"),
        "signature": row.get("signature"),
        "goal": row.get("goal") or "default",
        "asm": row.get("asm"),
        "origin": row.get("origin") or "sft_lora2",
    }
    for key in ("sha256", "size", "insn_count", "compiler", "flags", "seed"):
        if key in row:
            kept[key] = row[key]
    return kept


def check_names(row: dict[str, Any]) -> str | None:
    """Guard used by tests: training rows must not carry both c and asm."""
    if row.get("c") and row.get("asm"):
        return "both c and asm"
    return None


def write_failure(
    eval_path: Path,
    out_dir: Path,
    *,
    hybrid_path: Path | None = None,
    model_name: str = "asm-distill-7b",
    split: str = "heldout",
) -> dict[str, Any]:
    rows = []
    if eval_path.is_file():
        for row in _read_jsonl(eval_path):
            if row.get("model") == model_name and row.get("mode") == "asm" and row.get("split") == split:
                rows.append(row)
    by_id = {str(r.get("task_id")): r for r in rows}
    rows = list(by_id.values())
    tally: Counter[str] = Counter()
    passes = []
    failures = []
    for row in rows:
        greedy = row.get("greedy") or {}
        if greedy.get("ok"):
            passes.append(str(row.get("task_id")))
            continue
        bucket = classify_asm_fail(str(greedy.get("category") or ""))
        tally[bucket] += 1
        failures.append(
            {
                "task_id": row.get("task_id"),
                "bucket": bucket,
                "category": greedy.get("category"),
                "detail": str(greedy.get("detail") or "")[:240],
            }
        )
    hybrid_ids: list[str] = []
    hybrid_file = hybrid_path or (ROOT / "runs" / "round4" / "hybrid.jsonl")
    if hybrid_file.is_file():
        seen = set()
        for row in _read_jsonl(hybrid_file):
            if (
                row.get("model") == "c-then-asm"
                and row.get("mode") == "hybrid-O2"
                and row.get("split") == "heldout"
                and (row.get("greedy") or {}).get("ok")
            ):
                tid = str(row.get("task_id") or "")
                if tid and tid not in seen:
                    seen.add(tid)
                    hybrid_ids.append(tid)
    pass_set = set(passes)
    hybrid_set = set(hybrid_ids)
    overlap_pass = sorted(pass_set & hybrid_set)
    hybrid_fail = [row for row in failures if str(row.get("task_id")) in hybrid_set]
    hybrid_tally: Counter[str] = Counter(str(row["bucket"]) for row in hybrid_fail)
    payload = {
        "model": model_name,
        "split": split,
        "n": len(rows),
        "pass_n": len(passes),
        "tally": dict(tally),
        "hybrid_pass_n": len(hybrid_ids),
        "hybrid_pass_copied_from": str(hybrid_file),
        "overlap_pass_n": len(overlap_pass),
        "overlap_pass_ids": overlap_pass,
        "hybrid_fail_tally": dict(hybrid_tally),
        "failures": failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_tally.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Round 5 direct-asm held-out failures",
        "",
        f"Model `{model_name}`, split `{split}`, greedy. "
        f"Scored tasks: {len(rows)}. Passes: {len(passes)}.",
        "",
        "Buckets use `e2m.train.failure_analysis.classify_asm_fail` "
        "(`wrong_answer` is `logic_mismatch`).",
        "",
        "| bucket | n |",
        "|---|---:|",
    ]
    for bucket in ("doesnt_assemble", "abi", "logic_mismatch", "crash", "timeout", "other"):
        lines.append(f"| {bucket} | {tally.get(bucket, 0)} |")
    lines += [
        "",
        f"Round 4 hybrid pass set (c-then-asm, model C compiled gcc -O2, held-out), copied from `{hybrid_file}`: "
        f"{len(hybrid_ids)} tasks. Not re-scored.",
        "",
        f"Direct-asm passes that are also in that hybrid set: {len(overlap_pass)}.",
        "",
        "Failures among those hybrid-pass tasks:",
        "",
        "| bucket | n |",
        "|---|---:|",
    ]
    for bucket in ("doesnt_assemble", "abi", "logic_mismatch", "crash", "timeout", "other"):
        lines.append(f"| {bucket} | {hybrid_tally.get(bucket, 0)} |")
    lines += ["", "## Examples", ""]
    shown: Counter[str] = Counter()
    for row in failures:
        bucket = str(row["bucket"])
        if shown[bucket] >= 1:
            continue
        shown[bucket] += 1
        lines += [
            f"### `{row['task_id']}` — {bucket}",
            "",
            f"Detail: `{row['detail']}`",
            "",
        ]
    if not failures:
        lines.append("No greedy failures in the eval file.")
        lines.append("")
    (out_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"failure analysis n={len(rows)} pass={len(passes)} tally={dict(tally)}", flush=True)
    return payload


def round4_comparison() -> dict[str, Any]:
    """Copied from runs/round4/SUMMARY.md. Do not re-score."""
    return {
        "source": "runs/round4/SUMMARY.md",
        "heldout_direct_asm_pass_at_1": "7/78",
        "heldout_direct_asm_pass_at_5": "9/78",
        "heldout_c_then_asm_loop": "19/78",
        "heldout_hybrid_o2": "48/78",
        "heldout_c_only_pass_at_1": "46/78",
        "heldout_c_only_pass_at_5": "56/78",
        "note": "Copied. Round 5 does not re-score round 4.",
    }


# Silence unused-import warnings if a caller only needs a subset.
_ = (load_jsonl, traceback, Iterable)
