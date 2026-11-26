"""Ollama client for specs / user prompts. Think off, JSON only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"
FALLBACK_MODEL = "qwen3:8b"

SYSTEM = (
    "You write short English specs for C functions. Reply with JSON only. "
    "No markdown, no thinking, no code in spec."
)

SPEC_PROMPT = """Write JSON with keys:
spec: 1-3 sentences, plain English, what the function returns and any in-place buffer updates. No code, no C types, no identifier names.
user_prompt: one casual sentence a programmer would type to ask for this.
adequacy: integer 1-5 how completely spec captures the code.

Signature: {signature}

C:
{code}
"""


class LLMError(Exception):
    pass


def chat_json(
    prompt: str,
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    num_predict: int = 400,
    temperature: float = 0.2,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(str(exc)) from exc
    elapsed = time.time() - t0
    content = ((body.get("message") or {}).get("content") or "").strip()
    if not content:
        raise LLMError("empty model content")
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"non-json: {content[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise LLMError(f"json not object: {type(obj)}")
    return obj, elapsed


def write_spec(
    signature: str,
    code: str,
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
) -> tuple[dict[str, Any], float]:
    snippet = code.strip()
    if len(snippet) > 2500:
        snippet = snippet[:2500] + "\n/* truncated */"
    prompt = SPEC_PROMPT.format(signature=signature, code=snippet)
    try:
        obj, elapsed = chat_json(prompt, host=host, model=model)
    except LLMError:
        if model != FALLBACK_MODEL:
            obj, elapsed = chat_json(prompt, host=host, model=FALLBACK_MODEL)
        else:
            raise
    spec = str(obj.get("spec") or "").strip()
    user = str(obj.get("user_prompt") or obj.get("prompt") or "").strip()
    try:
        adequacy = int(obj.get("adequacy") or 0)
    except (TypeError, ValueError):
        adequacy = 0
    if not spec or spec.count(";") > 3 or "{" in spec:
        raise LLMError("spec looks like code")
    if len(spec) < 12:
        raise LLMError("spec too short")
    return {"spec": spec, "user_prompt": user or spec.split(".")[0], "adequacy": adequacy}, elapsed
