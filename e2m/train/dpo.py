"""DPO on compiler preference pairs, starting from the lora2 1.5B adapter.

Reference log-probs use the merged lora2 policy with adapters disabled.
The trainable policy is a fresh LoRA on top of that merged SFT model.

Round 3 (dpo2): each pair's real goal goes in the prompt (the format already
supports Goal: size|speed|default). An RPO/NLL term on the chosen response
anchors likelihood. A collapse guard greedy-generates a fixed val subset
every N steps and stops if assemble-rate falls below 80% of lora2's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2m.harness.assemble import AssembleError, assemble
from e2m.loop.guard import check_gpu_lock
from e2m.train.format import asm_messages, extract_asm
from e2m.train.sft_lora import _percentile, _write_svg

ROOT = Path(__file__).resolve().parents[2]


def pair_key(row: dict[str, Any]) -> str:
    chosen = row.get("chosen") or {}
    rejected = row.get("rejected") or {}
    left = str(chosen.get("sha256") or chosen.get("asm") or "")
    right = str(rejected.get("sha256") or rejected.get("asm") or "")
    return f"{row.get('task_id')}:{left}:{right}"


def blocked_task_ids(val_path: Path, heldout_path: Path) -> set[str]:
    blocked: set[str] = set()
    if val_path.is_file():
        blocked.update(json.loads(val_path.read_text(encoding="utf-8")).get("task_ids") or [])
    if heldout_path.is_file():
        with heldout_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    blocked.add(json.loads(line)["id"])
    return blocked


def load_pair_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            chosen = (row.get("chosen") or {}).get("asm") or ""
            rejected = (row.get("rejected") or {}).get("asm") or ""
            if not chosen.strip() or not rejected.strip():
                continue
            if chosen.strip() == rejected.strip():
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def split_pairs(
    rows: list[dict[str, Any]],
    *,
    blocked_ids: set[str],
    eval_n: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [r for r in rows if str(r.get("task_id") or "") not in blocked_ids]
    ranked = sorted(
        usable,
        key=lambda r: hashlib.sha256(f"{seed}:{pair_key(r)}".encode()).hexdigest(),
    )
    n_eval = max(0, min(int(eval_n), len(ranked)))
    eval_rows = ranked[:n_eval]
    eval_keys = {pair_key(r) for r in eval_rows}
    train_rows = [r for r in usable if pair_key(r) not in eval_keys]
    return train_rows, eval_rows


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
    rpo_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    pi = policy_chosen - policy_rejected
    ref = ref_chosen - ref_rejected
    logits = pi - ref
    pref = -F.logsigmoid(beta * logits)
    nll = -policy_chosen
    loss = pref + float(rpo_weight) * nll
    chosen_reward = (beta * (policy_chosen - ref_chosen)).detach()
    rejected_reward = (beta * (policy_rejected - ref_rejected)).detach()
    acc = (chosen_reward > rejected_reward).float()
    metrics = {
        "loss": float(loss.mean().item()),
        "pref_loss": float(pref.mean().item()),
        "nll_chosen": float(nll.mean().item()),
        "reward_margin": float((chosen_reward - rejected_reward).mean().item()),
        "accuracy": float(acc.mean().item()),
    }
    return loss.mean(), metrics


def pair_goal(row: dict[str, Any], *, use_pair_goal: bool = True) -> str:
    if not use_pair_goal:
        return "default"
    goal = str(row.get("goal") or "default").strip() or "default"
    if goal not in {"default", "size", "speed"}:
        return "default"
    return goal


def collapse_task_ids(val_ids: list[str], n: int = 30, seed: int = 0) -> list[str]:
    """Stable 30-task subset of the val split for the assemble-rate guard."""
    unique = sorted(set(val_ids))
    ranked = sorted(unique, key=lambda i: hashlib.sha256(f"collapse:{seed}:{i}".encode()).hexdigest())
    return ranked[: max(0, min(int(n), len(ranked)))]


def lora2_assemble_counts(eval_path: Path, task_ids: list[str]) -> tuple[int, int]:
    wanted = set(task_ids)
    ok = 0
    n = 0
    if not eval_path.is_file():
        return 0, 0
    with eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") not in wanted:
                continue
            if row.get("model") != "lora2" or row.get("mode") != "asm" or row.get("split") != "val":
                continue
            n += 1
            cat = ((row.get("greedy") or {}).get("category") or "")
            if cat != "doesnt_assemble":
                ok += 1
    return ok, n


def text_assembles(text: str) -> bool:
    try:
        assemble(extract_asm(text), syntax="auto")
        return True
    except AssembleError:
        return False
    except Exception:  # noqa: BLE001
        return False


def encode_response(
    tokenizer,
    spec: str,
    signature: str,
    asm: str,
    *,
    max_len: int,
    goal: str = "default",
) -> dict[str, list[int]] | None:
    messages = asm_messages(spec, signature, goal, asm)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        answer = tokenizer(asm.strip() + "\n" + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer
    if len(prompt_ids) >= max_len - 8:
        return None
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(labels) != len(full_ids) or not any(x != -100 for x in labels):
        return None
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:]
    mask = target != -100
    safe = target.clamp(min=0)
    token = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    token = token * mask
    denom = mask.sum(dim=-1).clamp(min=1)
    return token.sum(dim=-1) / denom


class _PairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_len: int,
        *,
        use_pair_goal: bool = True,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_pair_goal = use_pair_goal

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        row = self.rows[idx]
        spec = str(row.get("spec") or "")
        signature = str(row.get("signature") or "")
        goal = pair_goal(row, use_pair_goal=self.use_pair_goal)
        chosen = encode_response(
            self.tokenizer,
            spec,
            signature,
            str(row["chosen"]["asm"]),
            max_len=self.max_len,
            goal=goal,
        )
        rejected = encode_response(
            self.tokenizer,
            spec,
            signature,
            str(row["rejected"]["asm"]),
            max_len=self.max_len,
            goal=goal,
        )
        if chosen is None or rejected is None:
            return None
        return {"chosen": chosen, "rejected": rejected}


class _PadPairCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def _pad(self, items: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in items)
        ids, labels, mask = [], [], []
        for item in items:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            mask.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, torch.Tensor] | None:
        kept = [f for f in features if f is not None]
        if not kept:
            return None
        chosen = self._pad([f["chosen"] for f in kept])
        rejected = self._pad([f["rejected"] for f in kept])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


def _forward_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return sequence_logps(out.logits, labels)


def _pair_logps(model, batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = _forward_logps(
        model,
        batch["chosen_input_ids"].to(device),
        batch["chosen_attention_mask"].to(device),
        batch["chosen_labels"].to(device),
    )
    rejected = _forward_logps(
        model,
        batch["rejected_input_ids"].to(device),
        batch["rejected_attention_mask"].to(device),
        batch["rejected_labels"].to(device),
    )
    return chosen, rejected


def _sample_max_len(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    cap: int = 1536,
    use_pair_goal: bool = True,
) -> int:
    sample = rows[: min(len(rows), 512)]
    lengths: list[int] = []
    for row in sample:
        item = encode_response(
            tokenizer,
            str(row.get("spec") or ""),
            str(row.get("signature") or ""),
            str(row["chosen"]["asm"]),
            max_len=8192,
            goal=pair_goal(row, use_pair_goal=use_pair_goal),
        )
        if item is not None:
            lengths.append(len(item["input_ids"]))
    p99 = _percentile(lengths, 0.99) if lengths else cap
    return max(256, min(cap, ((p99 + 63) // 64) * 64))


@torch.no_grad()
def _eval_loss(
    model,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    rpo_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    losses, margins, accs, n = [], [], [], 0
    for batch in loader:
        if not batch:
            continue
        policy_c, policy_r = _pair_logps(model, batch, device)
        with model.disable_adapter():
            ref_c, ref_r = _pair_logps(model, batch, device)
        loss, metrics = dpo_loss(
            policy_c, policy_r, ref_c, ref_r, beta=beta, rpo_weight=rpo_weight
        )
        losses.append(metrics["loss"])
        margins.append(metrics["reward_margin"])
        accs.append(metrics["accuracy"])
        n += 1
    model.train()
    if not n:
        return {"eval_loss": None, "eval_reward_margin": None, "eval_accuracy": None, "eval_batches": 0}
    return {
        "eval_loss": sum(losses) / n,
        "eval_reward_margin": sum(margins) / n,
        "eval_accuracy": sum(accs) / n,
        "eval_batches": n,
    }


def load_collapse_prompts(tasks_path: Path, task_ids: list[str]) -> list[dict[str, str]]:
    wanted = {str(i) for i in task_ids}
    found: dict[str, dict[str, str]] = {}
    if tasks_path.is_file():
        with tasks_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = str(row.get("id") or "")
                if tid in wanted:
                    found[tid] = {
                        "id": tid,
                        "spec": str(row.get("spec") or ""),
                        "signature": str(row.get("signature") or ""),
                    }
    return [found[i] for i in task_ids if i in found]


def prepare_dpo2(
    *,
    pairs_path: Path,
    val_path: Path,
    heldout_path: Path,
    out_dir: Path,
    lora2_eval: Path,
    tasks_path: Path,
    collapse_n: int = 30,
) -> dict[str, Any]:
    """CPU stage: record the goal-in-prompt choice and the collapse subset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_pair_rows(pairs_path)
    blocked = blocked_task_ids(val_path, heldout_path)
    goals: dict[str, int] = {}
    for row in rows:
        g = str(row.get("goal") or "default")
        goals[g] = goals.get(g, 0) + 1
    val_ids = []
    if val_path.is_file():
        val_ids = list(json.loads(val_path.read_text(encoding="utf-8")).get("task_ids") or [])
    subset = collapse_task_ids(val_ids, n=collapse_n, seed=0)
    lora2_ok, lora2_n = lora2_assemble_counts(lora2_eval, subset)
    lora2_rate = (lora2_ok / lora2_n) if lora2_n else None
    threshold = (0.8 * lora2_rate) if lora2_rate is not None else None
    meta = {
        "choice": "pair_goal_in_prompt",
        "reason": "user_asm already takes Goal: default|size|speed; each pair keeps its real goal",
        "n_pairs_loaded": len(rows),
        "n_blocked_ids": len(blocked),
        "goal_counts": goals,
        "prompt_goal": "pair",
        "collapse_n": collapse_n,
        "collapse_task_ids": subset,
        "lora2_assemble_ok": lora2_ok,
        "lora2_assemble_n": lora2_n,
        "lora2_assemble_rate": lora2_rate,
        "collapse_threshold": threshold,
        "tasks_path": str(tasks_path),
        "pairs_path": str(pairs_path),
    }
    (out_dir / "pairs_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "collapse_subset.json").write_text(
        json.dumps({"task_ids": subset, "n": len(subset), "seed": 0}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("dpo2 prepare", json.dumps({k: meta[k] for k in meta if k != "collapse_task_ids"}), flush=True)
    return meta


def _adapter_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def _load_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device, dtype=param.dtype))


@torch.no_grad()
def _greedy_text(model, tokenizer, messages: list[dict[str, str]], *, max_new: int, device: torch.device) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    was_cache = model.config.use_cache
    model.config.use_cache = True
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.config.use_cache = was_cache
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def collapse_assemble_rate(
    model,
    tokenizer,
    prompts: list[dict[str, str]],
    *,
    device: torch.device,
    max_new: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    model.eval()
    details: list[dict[str, Any]] = []
    ok = 0
    for row in prompts:
        messages = asm_messages(row["spec"], row["signature"], "default")
        text = _greedy_text(model, tokenizer, messages, max_new=max_new, device=device)
        assembled = text_assembles(text)
        if assembled:
            ok += 1
        details.append(
            {
                "task_id": row.get("id"),
                "assembles": assembled,
                "text_head": (text or "")[:160],
            }
        )
    model.train()
    model.config.use_cache = False
    return ok, len(prompts), details


def train(
    *,
    model_path: Path,
    init_adapter: Path,
    pairs_path: Path,
    out_dir: Path,
    val_path: Path,
    heldout_path: Path,
    beta: float,
    epochs: float,
    lr: float,
    rank: int,
    alpha: int,
    dropout: float,
    batch_size: int,
    grad_accum: int,
    eval_n: int,
    max_steps: int,
    limit: int,
    max_len: int,
    rpo_weight: float = 0.0,
    use_pair_goal: bool = False,
    collapse_every: int = 0,
    collapse_n: int = 30,
    collapse_max_new: int = 1024,
    lora2_eval: Path | None = None,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_pair_rows(pairs_path, limit=limit)
    blocked = blocked_task_ids(val_path, heldout_path)
    if eval_n < 0:
        eval_n = 0
    if limit > 0:
        eval_n = max(1, min(eval_n, max(1, limit // 4)))
    train_rows, eval_rows = split_pairs(rows, blocked_ids=blocked, eval_n=eval_n, seed=0)
    tokenizer = AutoTokenizer.from_pretrained(init_adapter if (init_adapter / "tokenizer.json").is_file() else model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if max_len <= 0:
        max_len = _sample_max_len(tokenizer, train_rows, use_pair_goal=use_pair_goal)

    prompt_goal = "pair" if use_pair_goal else "default"
    stats: dict[str, Any] = {
        "model": str(model_path),
        "init_adapter": str(init_adapter),
        "pairs": str(pairs_path),
        "n_pairs_loaded": len(rows),
        "n_blocked_ids": len(blocked),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "beta": beta,
        "epochs": epochs,
        "lr": lr,
        "lora_r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_len": max_len,
        "max_steps": max_steps,
        "prompt_goal": prompt_goal,
        "rpo_weight": rpo_weight,
        "collapse_every": collapse_every,
        "collapse_n": collapse_n,
    }
    (out_dir / "train_config.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("dpo config", json.dumps(stats), flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if (init_adapter / "adapter_config.json").is_file():
        base = PeftModel.from_pretrained(base, init_adapter)
        base = base.merge_and_unload()
    base.config.use_cache = False
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    device = torch.device("cuda")
    model.to(device)
    model.train()

    collator = _PadPairCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        _PairDataset(train_rows, tokenizer, max_len, use_pair_goal=use_pair_goal),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )
    eval_loader = DataLoader(
        _PairDataset(eval_rows, tokenizer, max_len, use_pair_goal=use_pair_goal),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    ) if eval_rows else None

    steps_per_epoch = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    opt_steps_per_epoch = max(1, (steps_per_epoch + grad_accum - 1) // grad_accum)
    planned = int(opt_steps_per_epoch * epochs)
    total_opt_steps = planned if max_steps <= 0 else min(planned, max_steps)
    warmup = max(1, int(0.03 * total_opt_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * float(step + 1) / float(warmup)
        remain = max(1, total_opt_steps - warmup)
        progress = min(1.0, float(step - warmup) / float(remain))
        return lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())

    loss_path = out_dir / "loss.jsonl"
    if loss_path.exists():
        loss_path.unlink()

    started = time.time()
    opt_step = 0
    micro = 0
    running: list[float] = []
    last_eval: dict[str, Any] = {}
    last_loss: float | None = None
    optimizer.zero_grad(set_to_none=True)
    stop = False
    stop_reason = "completed"
    collapsed = False
    last_good_step = 0
    last_good_state = _adapter_state(model)
    collapse_curve: list[dict[str, Any]] = []
    collapse_path = out_dir / "collapse.jsonl"
    if collapse_path.exists():
        collapse_path.unlink()

    collapse_prompts: list[dict[str, str]] = []
    lora2_rate: float | None = None
    collapse_threshold: float | None = None
    if collapse_every > 0:
        val_ids: list[str] = []
        if val_path.is_file():
            val_ids = list(json.loads(val_path.read_text(encoding="utf-8")).get("task_ids") or [])
        subset = collapse_task_ids(val_ids, n=collapse_n, seed=0)
        tpath = tasks_path or (ROOT / "tasks" / "scaled.jsonl")
        collapse_prompts = load_collapse_prompts(tpath, subset)
        eval_path = lora2_eval or (ROOT / "runs" / "lora2" / "eval.jsonl")
        lora2_ok, lora2_n = lora2_assemble_counts(eval_path, subset)
        lora2_rate = (lora2_ok / lora2_n) if lora2_n else 1.0
        collapse_threshold = 0.8 * lora2_rate
        stats["lora2_assemble_rate"] = lora2_rate
        stats["collapse_threshold"] = collapse_threshold
        stats["collapse_subset_n"] = len(collapse_prompts)
        print(
            f"collapse guard every {collapse_every} steps, n={len(collapse_prompts)} "
            f"lora2_rate={lora2_rate:.3f} threshold={collapse_threshold:.3f}",
            flush=True,
        )

        def _run_collapse(step: int) -> bool:
            nonlocal last_good_step, last_good_state, collapsed, stop, stop_reason
            if not collapse_prompts:
                return True
            check_gpu_lock()
            ok, n, _details = collapse_assemble_rate(
                model, tokenizer, collapse_prompts, device=device, max_new=collapse_max_new
            )
            rate = (ok / n) if n else 0.0
            keep = collapse_threshold is None or rate >= collapse_threshold
            row = {
                "step": step,
                "assembles": ok,
                "n": n,
                "rate": rate,
                "lora2_rate": lora2_rate,
                "threshold": collapse_threshold,
                "keep": keep,
            }
            collapse_curve.append(row)
            with collapse_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print("dpo collapse", json.dumps(row), flush=True)
            if keep:
                last_good_step = step
                last_good_state = _adapter_state(model)
                ckpt = out_dir / f"checkpoint-{step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                return True
            collapsed = True
            stop = True
            stop_reason = (
                f"collapse at step {step}: assemble {ok}/{n}={rate:.3f} "
                f"< {collapse_threshold:.3f} (80% of lora2 {lora2_rate:.3f})"
            )
            print(stop_reason, flush=True)
            _load_adapter_state(model, last_good_state)
            print(f"restored last good checkpoint step={last_good_step}", flush=True)
            return False

        _run_collapse(0)

    n_epochs = 1 if epochs < 1 else max(1, int(round(epochs)))
    for epoch in range(n_epochs):
        if stop:
            break
        for batch in train_loader:
            if batch is None:
                continue
            if opt_step > 0 and opt_step % 20 == 0:
                try:
                    check_gpu_lock()
                except SystemExit:
                    stop = True
                    stop_reason = "gpu lock lost"
                    break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(opt_step)
            policy_c, policy_r = _pair_logps(model, batch, device)
            with torch.no_grad():
                was_train = model.training
                model.eval()
                with model.disable_adapter():
                    ref_c, ref_r = _pair_logps(model, batch, device)
                if was_train:
                    model.train()
            loss, metrics = dpo_loss(
                policy_c, policy_r, ref_c, ref_r, beta=beta, rpo_weight=rpo_weight
            )
            (loss / grad_accum).backward()
            running.append(metrics["loss"])
            micro += 1
            if micro % grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            last_loss = sum(running) / len(running)
            running = []
            row = {
                "step": opt_step,
                "epoch": epoch + opt_step / max(1, opt_steps_per_epoch),
                "loss": last_loss,
                "pref_loss": metrics.get("pref_loss"),
                "nll_chosen": metrics.get("nll_chosen"),
                "reward_margin": metrics["reward_margin"],
                "accuracy": metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with loss_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if opt_step == 1 or opt_step % 20 == 0:
                print(
                    f"dpo step {opt_step}/{total_opt_steps} loss={last_loss:.4f} "
                    f"margin={metrics['reward_margin']:.3f} acc={metrics['accuracy']:.3f}",
                    flush=True,
                )
            if eval_loader is not None and (opt_step % 500 == 0 or opt_step == total_opt_steps):
                last_eval = _eval_loss(
                    model, eval_loader, device, beta, rpo_weight=rpo_weight
                )
                print("dpo eval", json.dumps(last_eval), flush=True)
                with loss_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": opt_step, **last_eval}) + "\n")
            if collapse_every > 0 and opt_step % collapse_every == 0:
                if not _run_collapse(opt_step):
                    break
            if opt_step >= total_opt_steps:
                stop = True
                if collapse_every > 0 and opt_step % collapse_every != 0:
                    _run_collapse(opt_step)
                break

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if eval_loader is not None and not last_eval:
        last_eval = _eval_loss(model, eval_loader, device, beta, rpo_weight=rpo_weight)
        print("dpo eval", json.dumps(last_eval), flush=True)

    merged_dir = out_dir / "merged"
    print("merging dpo adapter into sft weights", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    _write_svg(loss_path, out_dir / "loss.svg")

    summary = {
        **stats,
        "train_loss": last_loss,
        "eval": last_eval,
        "elapsed_s": round(time.time() - started, 2),
        "warmup_steps": warmup,
        "global_steps": opt_step,
        "last_good_step": last_good_step,
        "collapsed": collapsed,
        "stop_reason": stop_reason,
        "collapse_curve": collapse_curve,
        "adapter": str(adapter_dir),
        "merged": str(merged_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("dpo done", json.dumps({k: summary[k] for k in summary if k != "collapse_curve"}), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DPO on compiler preference pairs from lora2")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--init-adapter", type=Path, default=ROOT / "runs" / "lora2" / "adapter")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data" / "scaled" / "pairs.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "dpo")
    parser.add_argument("--val-split", type=Path, default=ROOT / "eval" / "val_split.json")
    parser.add_argument("--heldout", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-n", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--rpo-weight", type=float, default=0.0, help="NLL weight on the chosen response (RPO)")
    parser.add_argument("--pair-goal", action="store_true", help="Put each pair's real goal in the prompt")
    parser.add_argument("--collapse-every", type=int, default=0, help="Greedy assemble-rate check every N steps (0=off)")
    parser.add_argument("--collapse-n", type=int, default=30)
    parser.add_argument("--collapse-max-new", type=int, default=1024)
    parser.add_argument("--lora2-eval", type=Path, default=ROOT / "runs" / "lora2" / "eval.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks" / "scaled.jsonl")
    parser.add_argument("--prepare-only", action="store_true", help="Write pair-goal meta and collapse subset, no GPU")
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare_dpo2(
            pairs_path=args.pairs,
            val_path=args.val_split,
            heldout_path=args.heldout,
            out_dir=args.out,
            lora2_eval=args.lora2_eval,
            tasks_path=args.tasks,
            collapse_n=args.collapse_n,
        )
        return 0
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not (args.init_adapter / "adapter_config.json").is_file():
        raise SystemExit(f"init adapter not found: {args.init_adapter}")
    train(
        model_path=args.model,
        init_adapter=args.init_adapter,
        pairs_path=args.pairs,
        out_dir=args.out,
        val_path=args.val_split,
        heldout_path=args.heldout,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_n=args.eval_n,
        max_steps=args.max_steps,
        limit=args.limit,
        max_len=args.max_len,
        rpo_weight=args.rpo_weight,
        use_pair_goal=args.pair_goal,
        collapse_every=args.collapse_every,
        collapse_n=args.collapse_n,
        collapse_max_new=args.collapse_max_new,
        lora2_eval=args.lora2_eval,
        tasks_path=args.tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
