"""Reject held-out tasks that repeat a scaled training task."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from e2m.scale.parse_c import normalize_body

_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def _sig_name(signature: str) -> str:
    match = _NAME_RE.search(signature)
    return match.group(1) if match else ""


def load_scaled_index(path: Path) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    bodies: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            names.add(str(row.get("id", "")))
            sig = str(row.get("signature", ""))
            name = _sig_name(sig)
            if name:
                names.add(name)
            bodies.append(normalize_body(str(row.get("reference_c", ""))))
    return names, bodies


def body_near_dup(candidate: str, bodies: list[str], *, ratio: float = 0.92) -> str | None:
    """Return the matching scaled body when candidate is an exact or near duplicate."""
    if not candidate:
        return None
    clen = len(candidate)
    for other in bodies:
        if candidate == other:
            return other
        olen = len(other)
        if clen == 0 or olen == 0:
            continue
        if min(clen, olen) / max(clen, olen) < ratio:
            continue
        if difflib.SequenceMatcher(None, candidate, other, autojunk=False).ratio() >= ratio:
            return other
    return None


def collision(
    task: dict,
    names: set[str],
    bodies: list[str],
) -> str | None:
    name = _sig_name(str(task.get("signature", "")))
    if name and name in names:
        return f"name:{name}"
    if str(task.get("id", "")) in names:
        return f"id:{task.get('id')}"
    hit = body_near_dup(normalize_body(str(task.get("reference_c", ""))), bodies)
    if hit is not None:
        return "body"
    return None
