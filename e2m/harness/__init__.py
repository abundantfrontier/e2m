"""Safe assemble / emulate / score harness for model-generated x86-64."""

from e2m.harness.abi import Signature, parse_signature
from e2m.harness.assemble import AssembleError, Assembled, assemble
from e2m.harness.emu import CrashReport, EmuResult, emulate

__all__ = [
    "AssembleError",
    "Assembled",
    "CrashReport",
    "EmuResult",
    "Signature",
    "assemble",
    "emulate",
    "parse_signature",
]
