"""Rewrite training specs into a short HumanEval/MBPP-style prompt via Ollama.

Reads tasks/scaled.jsonl and tasks/starter.jsonl. Never writes held-out tasks
into the training mix. Leakage is checked against eval/heldout.jsonl.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from e2m.harness.tasks import load_jsonl
from e2m.loop.leakage import HeldoutIndex, collision, embedding_hits
from e2m.scale.llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = (
    "You rewrite C-function specifications in the style of HumanEval and MBPP. "
    "Reply with JSON only. No markdown, no thinking, no assembly."
)

REWRITE_PROMPT = """Rewrite the specification so a coding model sees a HumanEval/MBPP-style prompt.

Return JSON with keys:
docstring: 1-3 sentences, plain English, what the function returns and any in-place updates. Mention important edge cases. No C types unless needed.
examples: array of 1 or 2 objects, each with "call" (string like "foo(1, 2)") and "result" (string). Use the provided tests, do not invent values.
prompt: the full rewritten specification: the docstring, then a blank line, then "Examples:" and the 1-2 examples, one per line as "  call -> result".

Signature: {signature}

Original spec:
{spec}

Tests you may quote as examples:
{tests}
"""


def chat_json(
    prompt: str,
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
    num_predict: int = 500,
    temperature: float = 0.2,
    system: str = SYSTEM,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict},
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


def embed_one(text: str, *, host: str, model: str, timeout: float = 60.0) -> list[float]:
    payload = {"model": model, "prompt": text[:4000]}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    vec = body.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise LLMError("empty embedding")
    return [float(x) for x in vec]


def unload_model(host: str, model: str) -> None:
    payload = {"model": model, "keep_alive": 0, "prompt": ""}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        pass


def _tests_blob(task: Any, n: int = 2) -> str:
    lines = []
    for case in task.tests[:n]:
        lines.append(json.dumps({"args": case.args, "ret": case.expected_return}, default=str))
    return "\n".join(lines) if lines else "(none)"


def _build_prompt(docstring: str, examples: list[Any], signature: str) -> str:
    doc = docstring.strip()
    lines = [doc, ""]
    formatted = []
    for ex in examples[:2]:
        if isinstance(ex, dict):
            call = str(ex.get("call") or ex.get("input") or "").strip()
            result = str(ex.get("result") or ex.get("ret") or ex.get("output") or "").strip()
            if call and result:
                formatted.append(f"  {call} -> {result}")
            elif call:
                formatted.append(f"  {call}")
        elif isinstance(ex, str) and ex.strip():
            formatted.append("  " + ex.strip())
    if formatted:
        lines.append("Examples:")
        lines.extend(formatted)
    prompt = "\n".join(lines).strip()
    if signature.split("(")[0].split()[-1] not in prompt and len(prompt) < 40:
        prompt = prompt + "\nSignature: " + signature
    return prompt


def parse_rewrite(obj: dict[str, Any], *, spec: str, signature: str) -> dict[str, Any]:
    docstring = str(obj.get("docstring") or obj.get("spec") or "").strip()
    examples = obj.get("examples") or []
    if not isinstance(examples, list):
        examples = [examples]
    prompt = str(obj.get("prompt") or "").strip()
    if not prompt:
        prompt = _build_prompt(docstring, examples, signature)
    if len(docstring) < 12:
        raise LLMError("docstring too short")
    if len(prompt) < 20:
        raise LLMError("prompt too short")
    if prompt.count(";") > 8 or "{" in docstring:
        raise LLMError("rewrite looks like code")
    return {
        "docstring": docstring,
        "examples": examples[:2],
        "spec": prompt,
        "spec_original": spec,
        "signature": signature,
    }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id"):
                out[str(row["task_id"])] = row
    return out


def _append(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def rewrite_tasks(
    *,
    scaled_path: Path,
    starter_path: Path,
    heldout_path: Path,
    cache_path: Path,
    out_path: Path,
    leakage_path: Path,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    embed_model: str | None = "bge-m3:latest",
    limit: int = 0,
    unload: bool = True,
) -> dict[str, Any]:
    index = HeldoutIndex.from_jsonl(heldout_path)
    tasks = list(load_jsonl(scaled_path))
    if starter_path.is_file():
        tasks.extend(load_jsonl(starter_path))
    if limit > 0:
        tasks = tasks[:limit]
    cache = _load_cache(cache_path)
    n_llm = 0
    n_fail = 0
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        if task.id in cache:
            continue
        prompt = REWRITE_PROMPT.format(
            signature=task.signature,
            spec=task.spec.strip(),
            tests=_tests_blob(task),
        )
        try:
            obj, elapsed = chat_json(prompt, host=host, model=model)
            parsed = parse_rewrite(obj, spec=task.spec, signature=task.signature)
            row = {
                "task_id": task.id,
                "signature": task.signature,
                "ok": True,
                "elapsed_s": round(elapsed, 3),
                **parsed,
            }
            n_llm += 1
        except LLMError as exc:
            row = {
                "task_id": task.id,
                "signature": task.signature,
                "ok": False,
                "error": str(exc)[:300],
                "spec_original": task.spec,
                "spec": task.spec,
            }
            n_fail += 1
        cache[task.id] = row
        _append(cache_path, row)
        if i % 25 == 0 or i == len(tasks):
            print(f"rewrite {i}/{len(tasks)} llm={n_llm} fail={n_fail} last={task.id}", flush=True)

    ngram_drop = 0
    embed_drop = 0
    fail_drop = 0
    kept_rows: list[dict[str, Any]] = []
    for task in tasks:
        row = cache.get(task.id)
        if row is None or not row.get("ok"):
            fail_drop += 1
            continue
        text = str(row.get("spec") or "")
        reason = collision(task_id=task.id, signature=task.signature, text=text, index=index)
        if reason:
            ngram_drop += 1
            row = {**row, "dropped": True, "leak": reason}
            kept_rows.append(row)
            continue
        kept_rows.append({**row, "dropped": False, "leak": None})

    embed_reason: dict[str, str] = {}
    if embed_model:
        # Free the 14B chat weights before loading the embedding model.
        unload_model(host, model)
        try:
            held_vecs = []
            for hid, spec in index.specs:
                held_vecs.append((hid, embed_one(spec, host=host, model=embed_model)))
            text_vecs = []
            for row in kept_rows:
                if row.get("dropped"):
                    continue
                text_vecs.append((row["task_id"], embed_one(str(row["spec"]), host=host, model=embed_model)))
            embed_reason = embedding_hits(text_vecs, held_vecs)
        except Exception as exc:  # noqa: BLE001
            embed_reason = {"__error__": str(exc)[:300]}
            print(f"embed skip: {exc}", flush=True)
        if "__error__" not in embed_reason:
            for row in kept_rows:
                if row.get("dropped"):
                    continue
                hit = embed_reason.get(row["task_id"])
                if hit:
                    row["dropped"] = True
                    row["leak"] = hit
                    embed_drop += 1

    if unload:
        unload_model(host, model)
        if embed_model:
            unload_model(host, embed_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            if row.get("dropped"):
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept = sum(1 for r in kept_rows if not r.get("dropped"))
    stats = {
        "n_tasks": len(tasks),
        "n_llm_ok": sum(1 for t in tasks if cache.get(t.id, {}).get("ok")),
        "n_llm_fail": n_fail + sum(1 for t in tasks if t.id not in cache),
        "n_kept": kept,
        "n_drop_ngram": ngram_drop,
        "n_drop_embed": embed_drop,
        "n_drop_fail": fail_drop,
        "elapsed_s": round(time.time() - t0, 2),
        "model": model,
        "embed_model": embed_model,
        "embed_error": embed_reason.get("__error__"),
        "out": str(out_path),
    }
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text(json.dumps({"stats": stats, "dropped": [
        {"task_id": r["task_id"], "leak": r.get("leak")}
        for r in kept_rows
        if r.get("dropped")
    ]}, indent=2) + "\n", encoding="utf-8")
    print("rewrite done", json.dumps(stats), flush=True)
    return stats
