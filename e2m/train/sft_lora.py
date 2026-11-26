"""LoRA SFT on goal=default AT&T assembly.

Input is the English spec, the SysV signature, and the goal. The target is
GNU as AT&T text for the function (directives and rodata included). A stable
3% of tasks is held out by task id and written to eval/val_split.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from e2m.train.format import (
    asm_messages,
    c_messages,
    c_then_asm_messages,
    extract_c_from_c_then_asm,
    format_c_then_asm_response,
    val_task_ids,
)

ROOT = Path(__file__).resolve().parents[2]


class _JsonDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class _LossJsonl(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        row = {"step": int(state.global_step), "epoch": state.epoch}
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
            if key in logs and logs[key] is not None:
                value = logs[key]
                row[key] = float(value) if not isinstance(value, str) else value
        if "loss" in row or "eval_loss" in row:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


def _load_default_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if '"default"' not in line:
                continue
            obj = json.loads(line)
            if obj.get("goal") != "default":
                continue
            rows.append(obj)
    return rows


def _target_text(row: dict, kind: str = "asm") -> str:
    if kind == "c":
        c_src = str(row.get("c") or "").strip()
        if c_src:
            return c_src
        return extract_c_from_c_then_asm(str(row.get("response") or ""))
    if row.get("response"):
        return str(row["response"])
    if row.get("c") and row.get("asm"):
        return format_c_then_asm_response(str(row["c"]), str(row["asm"]))
    return str(row.get("asm") or "")


def _encode(
    tokenizer,
    spec: str,
    signature: str,
    target: str,
    *,
    goal: str = "default",
    kind: str = "asm",
) -> dict[str, list[int]] | None:
    if kind == "c-then-asm":
        messages = c_then_asm_messages(spec, signature, goal, target)
    elif kind == "c":
        messages = c_messages(spec, signature, target)
    else:
        messages = asm_messages(spec, signature, goal, target)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        answer = tokenizer(target.strip() + "\n" + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(labels) != len(full_ids) or not any(x != -100 for x in labels):
        return None
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[idx]


class _PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        ids, labels, mask = [], [], []
        for feat in features:
            pad = width - len(feat["input_ids"])
            ids.append(feat["input_ids"] + [self.pad_id] * pad)
            labels.append(feat["labels"] + [-100] * pad)
            mask.append(feat["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def _write_svg(loss_path: Path, svg_path: Path) -> None:
    points = []
    if loss_path.is_file():
        for line in loss_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "loss" in row:
                points.append((row["step"], row["loss"]))
    if len(points) < 2:
        return
    width, height = 640, 240
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    coords = []
    for x, y in points:
        px = 40 + (x - xmin) / (xmax - xmin) * (width - 60)
        py = 20 + (ymax - y) / (ymax - ymin) * (height - 40)
        coords.append(f"{px:.1f},{py:.1f}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polyline fill="none" stroke="#1f4b99" stroke-width="2" points="{" ".join(coords)}"/>'
        f'<text x="40" y="16" font-size="12">train loss {ys[0]:.3f} -> {ys[-1]:.3f}</text>'
        f"</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def train(
    *,
    model_path: Path,
    sft_path: Path,
    out_dir: Path,
    val_path: Path,
    epochs: float,
    lr: float,
    rank: int,
    alpha: int,
    dropout: float,
    batch_size: int,
    grad_accum: int,
    load_4bit: bool = False,
    max_steps: int = 0,
    limit: int = 0,
    target: str = "asm",
    max_len: int = 0,
) -> dict:
    from peft import LoraConfig, TaskType, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_default_rows(sft_path)
    if limit > 0:
        rows = rows[:limit]
    ids = [r["task_id"] for r in rows]
    if val_path.is_file():
        val_ids = json.loads(val_path.read_text(encoding="utf-8"))["task_ids"]
    else:
        val_ids = val_task_ids(ids, frac=0.03, seed=0)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.write_text(
            json.dumps({"frac": 0.03, "seed": 0, "task_ids": val_ids}, indent=2) + "\n",
            encoding="utf-8",
        )
    val_set = set(val_ids)
    train_rows = [r for r in rows if r["task_id"] not in val_set]
    val_rows = [r for r in rows if r["task_id"] in val_set]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Measure length on the training rows, then drop the top 1%.
    encoded_all = []
    lengths = []
    for row in train_rows:
        target_text = _target_text(row, kind=target)
        if not target_text.strip():
            continue
        item = _encode(
            tokenizer,
            row["spec"],
            row["signature"],
            target_text,
            goal=str(row.get("goal") or "default"),
            kind=target,
        )
        if item is None:
            continue
        encoded_all.append(item)
        lengths.append(len(item["input_ids"]))
    p99 = _percentile(lengths, 0.99)
    auto_len = max(256, ((p99 + 63) // 64) * 64)
    if max_len <= 0:
        max_len = auto_len
    else:
        max_len = max(256, max_len)
    train_items = [item for item in encoded_all if len(item["input_ids"]) <= max_len]
    dropped = len(encoded_all) - len(train_items)
    val_items = []
    for row in val_rows:
        target_text = _target_text(row, kind=target)
        if not target_text.strip():
            continue
        item = _encode(
            tokenizer,
            row["spec"],
            row["signature"],
            target_text,
            goal=str(row.get("goal") or "default"),
            kind=target,
        )
        if item is not None and len(item["input_ids"]) <= max_len:
            val_items.append(item)

    stats = {
        "syntax": "att",
        "goal": "default",
        "target": target,
        "model": str(model_path),
        "train_tasks": len(train_rows),
        "val_tasks": len(val_rows),
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "dropped_over_p99": dropped,
        "token_p50": _percentile(lengths, 0.50),
        "token_p90": _percentile(lengths, 0.90),
        "token_p99": p99,
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "lora_r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "load_4bit": load_4bit,
        "max_steps": max_steps,
        "limit": limit,
    }
    (out_dir / "train_config.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("train config", json.dumps(stats), flush=True)

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
    model.config.use_cache = False
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if load_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    if not load_4bit:
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    steps_per_epoch = max(1, (len(train_items) + batch_size * grad_accum - 1) // (batch_size * grad_accum))
    planned = max(1, int(steps_per_epoch * epochs))
    total_steps = planned if max_steps <= 0 else min(planned, max_steps)
    warmup = max(1, int(0.03 * total_steps))
    loss_path = out_dir / "loss.jsonl"
    if loss_path.exists():
        loss_path.unlink()
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch" if val_items else "no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=0,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_JsonDataset(train_items),
        eval_dataset=_JsonDataset(val_items) if val_items else None,
        data_collator=_PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[_LossJsonl(loss_path)],
    )
    started = time.time()
    result = trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    _write_svg(loss_path, out_dir / "loss.svg")
    summary = {
        **stats,
        "train_loss": float(result.training_loss),
        "elapsed_s": round(time.time() - started, 2),
        "warmup_steps": warmup,
        "global_steps": int(result.global_step) if hasattr(result, "global_step") else None,
    }
    log_history = []
    for row in trainer.state.log_history:
        if "eval_loss" in row:
            log_history.append({"epoch": row.get("epoch"), "eval_loss": float(row["eval_loss"])})
    summary["eval_loss"] = log_history
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("train done", json.dumps(summary), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an AT&T assembly LoRA")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--sft", type=Path, default=ROOT / "data" / "scaled" / "sft.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "lora1")
    parser.add_argument("--val-split", type=Path, default=ROOT / "eval" / "val_split.json")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--4bit", action="store_true", dest="load_4bit")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target", choices=("asm", "c", "c-then-asm"), default="asm")
    parser.add_argument("--max-len", type=int, default=0, help="0 = p99 rounded up; raise for C-then-asm")
    args = parser.parse_args(argv)
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"model not found: {args.model}")
    train(
        model_path=args.model,
        sft_path=args.sft,
        out_dir=args.out,
        val_path=args.val_split,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        load_4bit=args.load_4bit,
        max_steps=args.max_steps,
        limit=args.limit,
        target=args.target,
        max_len=args.max_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
