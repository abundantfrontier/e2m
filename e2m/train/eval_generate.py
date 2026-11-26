"""Greedy and sampled assembly (and C) evaluation through the harness.

pass@1 is greedy. pass@5 is the unbiased estimator on 5 samples at T=0.8.
A candidate that does not assemble is a failure. Generated code is never
executed on the host; gcc -S output and model bytes both go through the harness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from e2m.harness.tasks import Task, assemble_reference, load_jsonl, run_gcc_baseline, run_task
from e2m.loop.guard import check_gpu_lock, hold_requested, past_deadline
from e2m.train.format import (
    asm_messages,
    c_messages,
    c_then_asm_messages,
    extract_asm,
    extract_asm_from_c_then_asm,
    extract_c,
    extract_c_from_c_then_asm,
    pass_at_k,
    score_task_run,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_model(model_path: Path, adapter: Path | None, *, load_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_path)
    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    if adapter is not None and (adapter / "adapter_config.json").is_file():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if not load_4bit:
        model.to("cuda")
    model.eval()
    model.config.use_cache = True
    # Qwen's pad is <|endoftext|>, distinct from <|im_end|>. Equating them makes
    # greedy generate stop on the first token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "left"
    return model, tokenizer


def _generate(model, tokenizer, messages: list[dict[str, str]], *, n: int, temperature: float, max_new: int) -> list[str]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    kwargs = {
        "max_new_tokens": max_new,
        "num_return_sequences": n,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95
    else:
        kwargs["do_sample"] = False
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    texts = []
    for row in out:
        texts.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
    return texts


def _gcc_reference(task: Task) -> dict[str, Any]:
    run = run_gcc_baseline(task, opt="-O2")
    if not run.passed:
        # One retry with a longer timeout. Reference code should pass.
        try:
            image = assemble_reference(task, opt="-O2")
            run = run_task(task, image, timeout_s=2.0)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "size": 0, "insn": 0, "detail": str(exc)[:200]}
    return {"ok": bool(run.passed), "size": int(run.code_size or 0), "insn": int(run.insn_count or 0)}


def _score_asm(task: Task, text: str) -> dict[str, Any]:
    asm = extract_asm(text)
    run = run_task(task, asm, syntax="auto", timeout_s=1.0)
    scored = score_task_run(run)
    scored["text"] = text[:1500]
    return scored


def _score_c_then_asm(task: Task, text: str) -> dict[str, Any]:
    """Score the assembly half; attach the C-half result for agreement stats."""
    asm = extract_asm_from_c_then_asm(text)
    run = run_task(task, asm, syntax="auto", timeout_s=1.0)
    scored = score_task_run(run)
    scored["text"] = text[:2000]
    c_scored = _score_c(task, extract_c_from_c_then_asm(text))
    scored["c"] = {k: v for k, v in c_scored.items() if k != "text"}
    scored["c_ok"] = bool(c_scored.get("ok"))
    scored["c_category"] = c_scored.get("category")
    scored["agree"] = bool(scored.get("ok")) == bool(c_scored.get("ok"))
    return scored


def _score_c(task: Task, text: str, *, opt: str = "-O2") -> dict[str, Any]:
    code = extract_c(text)
    candidate = Task(
        id=task.id,
        spec=task.spec,
        signature=task.signature,
        reference_c=code,
        tests=task.tests,
    )
    try:
        image = assemble_reference(candidate, opt=opt)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "category": "doesnt_assemble",
            "size": 0,
            "insn": 0,
            "detail": str(exc)[:400],
            "text": text[:1500],
            "opt": opt,
        }
    run = run_task(task, image, timeout_s=1.0)
    scored = score_task_run(run)
    scored["text"] = text[:1500]
    scored["opt"] = opt
    return scored


def _done_ids(path: Path, model_name: str, mode: str, split: str) -> set[str]:
    found = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model") == model_name and row.get("mode") == mode and row.get("split") == split:
                found.add(str(row.get("task_id")))
    return found


def evaluate(
    *,
    model_path: Path,
    adapter: Path | None,
    model_name: str,
    mode: str,
    split: str,
    tasks_path: Path,
    val_path: Path,
    out_path: Path,
    limit: int,
    max_new: int,
    load_4bit: bool = False,
    max_hours: float = 0.0,
) -> dict[str, Any]:
    tasks = load_jsonl(tasks_path)
    if split == "val":
        wanted = set(json.loads(val_path.read_text(encoding="utf-8"))["task_ids"])
        tasks = [t for t in tasks if t.id in wanted]
    if limit > 0:
        tasks = tasks[:limit]
    done = _done_ids(out_path, model_name, mode, split)
    pending = [t for t in tasks if t.id not in done]
    print(
        f"eval {model_name} {mode} {split}: {len(tasks)} tasks, {len(done)} cached, {len(pending)} pending",
        flush=True,
    )
    stopped: str | None = None
    if past_deadline():
        print("deadline reached, skipping eval", flush=True)
        stopped = "deadline"
        pending = []
    else:
        try:
            check_gpu_lock()
        except SystemExit:
            print("gpu.lock lost, skipping eval", flush=True)
            stopped = "gpu-lock"
            pending = []
    if pending and hold_requested():
        print("held", flush=True)
        stopped = "held"
        pending = []
    model = tokenizer = None
    if pending:
        model, tokenizer = _load_model(model_path, adapter, load_4bit=load_4bit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "c-then-asm":
        score = _score_c_then_asm

        def messages_for(spec: str, signature: str, goal: str = "default"):  # noqa: ARG001
            return c_then_asm_messages(spec, signature, goal)

    elif mode == "c":
        score = _score_c
        messages_for = c_messages
    else:
        score = _score_asm
        messages_for = asm_messages
    started = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, task in enumerate(pending, 1):
            if past_deadline():
                print("deadline reached, stopping eval", flush=True)
                stopped = "deadline"
                break
            if max_hours > 0 and (time.time() - started) / 3600.0 >= max_hours:
                print(f"max-hours {max_hours} reached, stopping eval", flush=True)
                stopped = "max-hours"
                break
            if hold_requested():
                print("held", flush=True)
                stopped = "held"
                break
            try:
                check_gpu_lock()
            except SystemExit:
                print("gpu.lock lost, stopping eval", flush=True)
                stopped = "gpu-lock"
                break
            if mode == "c":
                prompt = messages_for(task.spec, task.signature)
            else:
                prompt = messages_for(task.spec, task.signature, "default")
            try:
                greedy_text = _generate(model, tokenizer, prompt, n=1, temperature=0.0, max_new=max_new)[0]
                greedy = score(task, greedy_text)
                samples = []
                sampled = _generate(model, tokenizer, prompt, n=5, temperature=0.8, max_new=max_new)
                for text in sampled:
                    samples.append(score(task, text))
                gcc = _gcc_reference(task)
            except Exception as exc:  # noqa: BLE001 — one bad task must not drop the rest
                detail = f"{type(exc).__name__}: {exc}"[:400]
                greedy = {"ok": False, "category": "crash", "size": 0, "insn": 0, "detail": detail, "text": ""}
                samples = [
                    {"ok": False, "category": "crash", "size": 0, "insn": 0, "detail": detail, "text": ""}
                    for _ in range(5)
                ]
                gcc = {"ok": False, "size": 0, "insn": 0, "detail": detail}
                print(f"  task {task.id} failed: {detail}", flush=True)
            row = {
                "model": model_name,
                "mode": mode,
                "split": split,
                "task_id": task.id,
                "source": task.source,
                "greedy": greedy,
                "samples": samples,
                "gcc": gcc,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {i}/{len(pending)} last={task.id} greedy={greedy['category']}", flush=True)
    summary = summarize(out_path, model_name, mode, split)
    if stopped:
        summary["stopped"] = stopped
    return summary


def summarize(path: Path, model_name: str, mode: str, split: str) -> dict[str, Any]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") == model_name and row.get("mode") == mode and row.get("split") == split:
                rows.append(row)
    # Last write wins if a task was repeated.
    by_id = {r["task_id"]: r for r in rows}
    rows = list(by_id.values())
    n = len(rows)
    categories: dict[str, int] = {}
    sample_categories: dict[str, int] = {}
    pass1 = 0
    pass5_sum = 0.0
    size_ratios = []
    insn_ratios = []
    for row in rows:
        cat = row["greedy"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
        if row["greedy"]["ok"]:
            pass1 += 1
            gcc = row.get("gcc") or {}
            if gcc.get("ok") and gcc.get("size"):
                size_ratios.append(row["greedy"]["size"] / gcc["size"])
            if gcc.get("ok") and gcc.get("insn"):
                insn_ratios.append(row["greedy"]["insn"] / gcc["insn"])
        samples = row.get("samples") or []
        correct = sum(1 for s in samples if s.get("ok"))
        for sample in samples:
            scat = sample.get("category") or "crash"
            sample_categories[scat] = sample_categories.get(scat, 0) + 1
        k = min(5, len(samples))
        pass5_sum += pass_at_k(len(samples), correct, k) if k else 0.0
    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    summary = {
        "model": model_name,
        "mode": mode,
        "split": split,
        "n": n,
        "pass_at_1": (pass1 / n) if n else None,
        "pass_at_5": (pass5_sum / n) if n else None,
        "size_ratio_vs_gcc_o2": _mean(size_ratios),
        "insn_ratio_vs_gcc_o2": _mean(insn_ratios),
        "size_ratio_n": len(size_ratios),
        "insn_ratio_n": len(insn_ratios),
        "greedy_categories": categories,
        "sample_categories": sample_categories,
    }
    if mode == "c-then-asm" and n:
        c_pass = 0
        agree = 0
        both = 0
        c_only = 0
        asm_only = 0
        for row in rows:
            greedy = row.get("greedy") or {}
            c_ok = bool(greedy.get("c_ok"))
            asm_ok = bool(greedy.get("ok"))
            if c_ok:
                c_pass += 1
            if greedy.get("agree") is True or (c_ok == asm_ok):
                agree += 1
            if c_ok and asm_ok:
                both += 1
            elif c_ok:
                c_only += 1
            elif asm_ok:
                asm_only += 1
        summary["c_pass_at_1"] = c_pass / n
        summary["c_pass_n"] = c_pass
        summary["c_asm_agree"] = agree / n
        summary["c_asm_agree_n"] = agree
        summary["c_and_asm_pass"] = both
        summary["c_only_pass"] = c_only
        summary["asm_only_pass"] = asm_only
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate assembly or C generations")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--name", default="lora")
    parser.add_argument("--mode", choices=("asm", "c", "c-then-asm"), default="asm")
    parser.add_argument("--split", choices=("val", "heldout"), default="heldout")
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--val-split", type=Path, default=ROOT / "eval" / "val_split.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "lora1" / "eval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=1536)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--4bit", action="store_true", dest="load_4bit")
    parser.add_argument("--max-hours", type=float, default=0.0)
    args = parser.parse_args(argv)
    tasks = args.tasks
    if tasks is None:
        tasks = ROOT / ("eval/heldout.jsonl" if args.split == "heldout" else "tasks/scaled.jsonl")
    if args.summary_only:
        print(json.dumps(summarize(args.out, args.name, args.mode, args.split), indent=2))
        return 0
    summary = evaluate(
        model_path=args.model,
        adapter=args.adapter,
        model_name=args.name,
        mode=args.mode,
        split=args.split,
        tasks_path=tasks,
        val_path=args.val_split,
        out_path=args.out,
        limit=args.limit,
        max_new=args.max_new,
        load_4bit=args.load_4bit,
        max_hours=args.max_hours,
    )
    print(json.dumps(summary, indent=2), flush=True)
    stopped = summary.get("stopped")
    if stopped == "gpu-lock":
        return 2
    if stopped in {"deadline", "max-hours"}:
        return 3
    if stopped == "held":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
