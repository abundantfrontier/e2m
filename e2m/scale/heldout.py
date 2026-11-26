"""Keep HumanEval / MBPP style problems out of the training task set."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from e2m.scale.parse_c import GENERIC_NAMES

ROOT = Path(__file__).resolve().parents[2]
HELDOUT_DIR = ROOT / "datasets" / "heldout"


def _load_humaneval_names() -> set[str]:
    names: set[str] = set()
    gz = HELDOUT_DIR / "HumanEval.jsonl.gz"
    if not gz.is_file():
        return names
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            ep = str(obj.get("entry_point") or "").strip()
            if ep:
                names.add(ep)
                names.add(_to_snake(ep))
    return names


def _load_mbpp_names() -> set[str]:
    names: set[str] = set()
    path = HELDOUT_DIR / "mbpp.jsonl"
    if not path.is_file():
        return names
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = str(obj.get("code") or obj.get("test") or "")
            for m in re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(", code):
                names.add(m.group(1))
    return names


def _to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


_NAMES: set[str] | None = None


def heldout_names() -> set[str]:
    global _NAMES
    if _NAMES is None:
        _NAMES = {_n.lower() for _n in (_load_humaneval_names() | _load_mbpp_names())}
    return _NAMES


def is_heldout_name(fname: str) -> bool:
    key = fname.lower().lstrip("_")
    if key in GENERIC_NAMES:
        return False
    names = heldout_names()
    if key in names:
        return True
    snake = _to_snake(fname).lower()
    return snake in names and snake not in GENERIC_NAMES
