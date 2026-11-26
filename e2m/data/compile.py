"""Compile one C or Rust function to assembly. Never executes the output."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from e2m.harness.rust_refs import RUST_PREAMBLE
from e2m.harness.tasks import C_PREAMBLE, GCC_COMMON

ROOT = Path(__file__).resolve().parents[2]


class CompileError(Exception):
    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


def _search_roots() -> list[Path]:
    roots = [
        Path.home() / ".cargo" / "bin",
        ROOT / "tools",
        Path.home() / "e2m" / "tools",
    ]
    return roots


def find_tool(name: str) -> str | None:
    """Resolve ``name`` from PATH, Cargo, or ``tools``."""
    env = os.environ.get("E2M_TOOL_" + name.upper().replace("-", "_"))
    if env and Path(env).is_file():
        return env
    found = shutil.which(name)
    if found:
        return found
    for root in _search_roots():
        direct = root / name
        if direct.is_file():
            return str(direct)
        if root.is_dir():
            for path in root.glob(f"**/bin/{name}"):
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path)
    return None


def tool_version(path: str) -> str:
    try:
        proc = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(version check failed: {exc})"
    line = (proc.stdout or proc.stderr or "").strip().splitlines()
    return line[0] if line else path


def _run(cmd: list[str], *, timeout: float = 40.0, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
        )
    except subprocess.TimeoutExpired as exc:
        raise CompileError("compile timed out", details=" ".join(cmd[:6])) from exc
    except FileNotFoundError as exc:
        raise CompileError(f"compiler not found: {cmd[0]}") from exc


def c_source(body: str) -> str:
    if "#include" in body:
        return body
    return C_PREAMBLE + body


def compile_c_asm(source: str, compiler: str, flags: list[str]) -> str:
    """``gcc``/``clang -S`` of one translation unit. Returns the ``.s`` text."""
    src = c_source(source)
    cmd = [compiler, *flags, "-S", *GCC_COMMON, "-o", "-", "-x", "c", "-"]
    proc = _run(cmd, input_text=src)
    if proc.returncode != 0 or not proc.stdout.strip():
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        raise CompileError(f"{Path(compiler).name} -S failed", details=err[-800:])
    return proc.stdout


def compile_c_ir(source: str, clang: str) -> str:
    """Unoptimized LLVM IR for a later ``opt`` + ``llc`` sweep.

    ``-disable-O0-optnone`` keeps the IR unoptimized without the ``optnone``
    attribute, which would otherwise make every later ``opt`` pipeline a no-op.
    """
    src = c_source(source)
    cmd = [
        clang,
        "-S",
        "-emit-llvm",
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-fno-discard-value-names",
        *GCC_COMMON,
        "-o",
        "-",
        "-x",
        "c",
        "-",
    ]
    proc = _run(cmd, input_text=src)
    if proc.returncode != 0 or "define" not in proc.stdout:
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        raise CompileError("clang -emit-llvm failed", details=err[-800:])
    return proc.stdout


_ATTR_GROUP = re.compile(r"^(attributes #\d+ = \{)(.*)\}", re.M)


def apply_function_attrs(ir: str, attrs: str) -> str:
    """Append function attributes (``optsize``, ``minsize``) to every group.

    LLVM 23 has no ``default<Os>`` / ``default<Oz>`` pipeline. Size-oriented
    runs are ``default<O2>`` plus these attributes on the functions.
    """
    wanted = [part for part in attrs.split() if part]
    if not wanted:
        return ir

    def repl(match: re.Match[str]) -> str:
        tokens = match.group(2).split()
        for attr in wanted:
            if attr not in tokens:
                tokens.append(attr)
        return match.group(1) + " " + " ".join(tokens) + " }"

    return _ATTR_GROUP.sub(repl, ir)


def opt_llc_asm(
    ir: str,
    *,
    opt_bin: str,
    llc_bin: str,
    passes: str,
    llc_flags: list[str],
) -> str:
    with tempfile.TemporaryDirectory(prefix="e2m-opt-") as td:
        td_path = Path(td)
        ir_path = td_path / "in.ll"
        opt_path = td_path / "opt.ll"
        asm_path = td_path / "out.s"
        ir_path.write_text(ir, encoding="utf-8")
        opt_cmd = [opt_bin, "-S", f"-passes={passes}", "-o", str(opt_path), str(ir_path)]
        proc = _run(opt_cmd, timeout=40)
        if proc.returncode != 0 or not opt_path.is_file():
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise CompileError("opt failed", details=err[-800:])
        llc_cmd = [
            llc_bin,
            str(opt_path),
            "-o",
            str(asm_path),
            "-relocation-model=static",
            "-x86-asm-syntax=att",
            "-function-sections",
            *llc_flags,
        ]
        proc = _run(llc_cmd, timeout=40)
        if proc.returncode != 0 or not asm_path.is_file():
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise CompileError("llc failed", details=err[-800:])
        return asm_path.read_text(encoding="utf-8", errors="replace")


def compile_rust_asm(function_src: str, rustc: str, flags: list[str]) -> str:
    """Compile a no_std crate (``--emit=asm``) and return the assembly text.

    ``flags`` are rustc ``-C`` settings already expanded (``opt-level=2``, ...).
    """
    src = function_src if function_src.lstrip().startswith("#![") else RUST_PREAMBLE + function_src
    with tempfile.TemporaryDirectory(prefix="e2m-rustc-") as td:
        td_path = Path(td)
        src_path = td_path / "task.rs"
        asm_path = td_path / "task.s"
        src_path.write_text(src, encoding="utf-8")
        cmd = [
            rustc,
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--emit=asm",
            "-C",
            "panic=abort",
            "-C",
            "debuginfo=0",
            "-C",
            "debug-assertions=off",
            "-C",
            "relocation-model=static",
            "-C",
            "code-model=small",
            *flags,
            "-o",
            str(asm_path),
            str(src_path),
        ]
        proc = _run(cmd, timeout=60)
        if proc.returncode != 0 or not asm_path.is_file():
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise CompileError("rustc failed", details=err[-800:])
        return asm_path.read_text(encoding="utf-8", errors="replace")


def rustc_flag_list(
    *,
    opt: str,
    cpu: str,
    features: str,
    overflow: str,
    llvm_args: str,
) -> list[str]:
    flags = [
        "-C",
        f"opt-level={opt}",
        "-C",
        f"target-cpu={cpu}",
        "-C",
        f"overflow-checks={overflow}",
    ]
    if features:
        flags.extend(["-C", f"target-feature={features}"])
    if llvm_args:
        flags.extend(["-C", f"llvm-args={llvm_args}"])
    return flags
