"""Drop rewritten training specs that collide with the held-out set.

Never trains on held-out content. This module only *reads* held-out names and
text so rewritten train prompts can be filtered.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from e2m.heldout.near_dup import _sig_name

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "be",
    "by",
    "as",
    "at",
    "from",
    "that",
    "this",
    "it",
    "if",
    "not",
    "no",
    "return",
    "returns",
    "given",
    "function",
    "value",
    "values",
    "integer",
    "integers",
    "array",
    "list",
    "string",
    "otherwise",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(normalize(text)) if t not in _STOP and len(t) > 1]


def char_ngrams(text: str, n: int = 5) -> set[str]:
    s = re.sub(r"[^a-z0-9]+", " ", normalize(text))
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def word_ngrams(text: str, n: int = 3) -> set[str]:
    toks = tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for x, y in zip(u, v):
        dot += x * y
        nu += x * x
        nv += y * y
    if nu <= 0 or nv <= 0:
        return 0.0
    return dot / math.sqrt(nu * nv)


class HeldoutIndex:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.ids: set[str] = set()
        self.names: set[str] = set()
        self.specs: list[tuple[str, str]] = []
        self.char_grams: list[tuple[str, set[str]]] = []
        self.word_grams: list[tuple[str, set[str]]] = []
        for row in rows:
            tid = str(row.get("id") or "")
            spec = str(row.get("spec") or "")
            name = _sig_name(str(row.get("signature") or ""))
            if tid:
                self.ids.add(tid.lower())
            if name:
                self.names.add(name.lower())
                self.names.add(name.lower().replace("mbpp_", "").replace("he_", ""))
            self.specs.append((tid, spec))
            self.char_grams.append((tid, char_ngrams(spec)))
            self.word_grams.append((tid, word_ngrams(spec)))

    @classmethod
    def from_jsonl(cls, path: Path) -> HeldoutIndex:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return cls(rows)


def collision(
    *,
    task_id: str,
    signature: str,
    text: str,
    index: HeldoutIndex,
    char_threshold: float = 0.42,
    word_threshold: float = 0.35,
) -> str | None:
    """Return a short reason when the rewritten text is too close to held-out."""
    tid = (task_id or "").lower()
    if tid in index.ids:
        return f"id:{task_id}"
    name = _sig_name(signature).lower()
    if name and name in index.names:
        return f"name:{name}"
    bare = name.replace("mbpp_", "").replace("he_", "") if name else ""
    if bare and len(bare) >= 6 and bare in index.names:
        return f"name:{bare}"
    # Distinctive held-out identifiers mentioned in the rewritten prompt.
    blob = normalize(text)
    for held_name in index.names:
        if len(held_name) >= 8 and held_name in blob:
            return f"mentions:{held_name}"
    cg = char_ngrams(text)
    for hid, grams in index.char_grams:
        score = jaccard(cg, grams)
        if score >= char_threshold:
            return f"char5:{hid}:{score:.3f}"
    wg = word_ngrams(text)
    for hid, grams in index.word_grams:
        score = jaccard(wg, grams)
        if score >= word_threshold:
            return f"word3:{hid}:{score:.3f}"
    return None


def embedding_hits(
    texts: list[tuple[str, list[float]]],
    held: list[tuple[str, list[float]]],
    *,
    threshold: float = 0.86,
) -> dict[str, str]:
    hits: dict[str, str] = {}
    for tid, vec in texts:
        best_id = None
        best = 0.0
        for hid, hvec in held:
            score = cosine(vec, hvec)
            if score > best:
                best = score
                best_id = hid
        if best >= threshold and best_id is not None:
            hits[tid] = f"embed:{best_id}:{best:.3f}"
    return hits
