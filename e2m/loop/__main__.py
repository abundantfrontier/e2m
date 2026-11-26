"""CLI: python -m e2m.loop <split-heldout|rewrite|run|collect|mix|summarize|round2|round3|round4>"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cmd_split(args: argparse.Namespace) -> int:
    from e2m.loop.split import write_heldout_split

    payload = write_heldout_split(args.tasks, args.out)
    print(json.dumps({"n_tasks": payload["n_tasks"], "out": str(args.out)}))
    return 0


def _cmd_rewrite(args: argparse.Namespace) -> int:
    from e2m.loop.rewrite import rewrite_tasks

    rewrite_tasks(
        scaled_path=args.scaled,
        starter_path=args.starter,
        heldout_path=args.heldout,
        cache_path=args.cache,
        out_path=args.out,
        leakage_path=args.leakage,
        model=args.model,
        embed_model=None if args.no_embed else args.embed_model,
        limit=args.limit,
        unload=not args.keep_ollama,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from e2m.loop.run import run_loop

    summary = run_loop(
        model_path=args.model,
        adapter=args.adapter,
        model_name=args.name,
        mode=args.mode,
        split=args.split,
        tasks_path=args.tasks,
        val_path=args.val_split,
        out_dir=args.out,
        k=args.k,
        rounds=args.rounds,
        temperature=args.temperature,
        max_new=args.max_new,
        limit=args.limit,
        spec_map_path=args.specs,
        spec_kind=args.spec_kind,
        load_4bit=args.load_4bit,
        timeout_s=args.timeout,
        batch_initial=args.batch_initial,
        max_hours=args.max_hours,
    )
    stopped = (summary or {}).get("stopped")
    if stopped == "gpu-lock":
        return 2
    if stopped in {"deadline", "max-hours"}:
        return 3
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from e2m.loop.run import collect_reject

    stats = collect_reject(args.loop, args.out, spec_kind=args.spec_kind)
    print(json.dumps(stats))
    return 0


def _cmd_mix(args: argparse.Namespace) -> int:
    from e2m.loop.mix import mix_sft

    val_ids = set()
    if args.val_split.is_file():
        val_ids.update(json.loads(args.val_split.read_text(encoding="utf-8"))["task_ids"])
    heldout_ids = set()
    if args.heldout.is_file():
        with args.heldout.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    heldout_ids.add(json.loads(line)["id"])
    stats = mix_sft(
        gcc_sft=args.sft,
        rewritten=args.rewritten,
        reject=args.reject,
        out_path=args.out,
        val_ids=val_ids,
        heldout_ids=heldout_ids,
        starter_sft=args.starter_sft,
    )
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from e2m.loop.report import collect

    print(json.dumps(collect(), indent=2, default=str))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from e2m.loop.summarize import summarize_loop

    print(json.dumps(summarize_loop(args.loop, model=args.name, split=args.split), indent=2))
    return 0


def _cmd_round2(args: argparse.Namespace) -> int:
    from e2m.loop.round2 import main as round2_main

    extra: list[str] = []
    if args.smoke:
        extra.append("--smoke")
    if args.resume:
        extra.append("--resume")
    if args.round2b:
        extra.append("--round2b")
    if args.dry_run:
        extra.append("--dry-run")
    if args.carry_hours:
        extra.append("--carry-hours")
    if args.only:
        extra += ["--only", *args.only]
    if args.prior:
        for path in args.prior:
            extra += ["--prior", str(path)]
    if args.out:
        extra += ["--out", str(args.out)]
    if args.lock_note:
        extra += ["--lock-note", args.lock_note]
    return round2_main(extra)


def _cmd_round3(args: argparse.Namespace) -> int:
    from e2m.loop.round3 import main as round3_main

    extra: list[str] = []
    if args.smoke:
        extra.append("--smoke")
    if args.resume:
        extra.append("--resume")
    if args.dry_run:
        extra.append("--dry-run")
    if args.only:
        extra += ["--only", *args.only]
    if args.out:
        extra += ["--out", str(args.out)]
    if args.lock_note:
        extra += ["--lock-note", args.lock_note]
    return round3_main(extra)


def _cmd_round4(args: argparse.Namespace) -> int:
    from e2m.loop.round4 import main as round4_main

    extra: list[str] = []
    if args.smoke:
        extra.append("--smoke")
    if args.resume:
        extra.append("--resume")
    if args.dry_run:
        extra.append("--dry-run")
    if args.only:
        extra += ["--only", *args.only]
    if args.out:
        extra += ["--out", str(args.out)]
    if args.lock_note:
        extra += ["--lock-note", args.lock_note]
    return round4_main(extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2M repair loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split-heldout", help="Write eval/heldout_visible.json")
    p.add_argument("--tasks", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "heldout_visible.json")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("rewrite", help="Rewrite train specs via Ollama")
    p.add_argument("--scaled", type=Path, default=ROOT / "tasks" / "scaled.jsonl")
    p.add_argument("--starter", type=Path, default=ROOT / "tasks" / "starter.jsonl")
    p.add_argument("--heldout", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "scale" / "rewrite-cache.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "scaled" / "rewritten.jsonl")
    p.add_argument("--leakage", type=Path, default=ROOT / "data" / "scaled" / "rewrite-leakage.json")
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--embed-model", default="bge-m3:latest")
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--keep-ollama", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=_cmd_rewrite)

    p = sub.add_parser("run", help="Repair loop (GPU)")
    p.add_argument("--model", type=Path, default=ROOT / "runs" / "models" / "Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--name", default="lora1")
    p.add_argument("--mode", choices=("asm", "c", "c-then-asm"), default="asm")
    p.add_argument("--split", choices=("val", "heldout", "train"), default="heldout")
    p.add_argument("--tasks", type=Path, default=None)
    p.add_argument("--val-split", type=Path, default=ROOT / "eval" / "val_split.json")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "loop-lora1-heldout")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--specs", type=Path, default=None)
    p.add_argument("--spec-kind", default="original")
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--batch-initial", action="store_true")
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--4bit", action="store_true", dest="load_4bit")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("collect", help="Keep passing programs from a loop JSONL")
    p.add_argument("--loop", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "scaled" / "reject.jsonl")
    p.add_argument("--spec-kind", default="original")
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("mix", help="Build sft_lora2.jsonl")
    p.add_argument("--sft", type=Path, default=ROOT / "data" / "scaled" / "sft.jsonl")
    p.add_argument("--starter-sft", type=Path, default=ROOT / "data" / "sft.jsonl")
    p.add_argument("--rewritten", type=Path, default=ROOT / "data" / "scaled" / "rewritten.jsonl")
    p.add_argument("--reject", type=Path, default=ROOT / "data" / "scaled" / "reject.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "scaled" / "sft_lora2.jsonl")
    p.add_argument("--val-split", type=Path, default=ROOT / "eval" / "val_split.json")
    p.add_argument("--heldout", type=Path, default=ROOT / "eval" / "heldout.jsonl")
    p.set_defaults(func=_cmd_mix)

    p = sub.add_parser("report", help="Collect STATUS numbers from runs/")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("summarize", help="Print a loop summary")
    p.add_argument("--loop", type=Path, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--split", default=None)
    p.set_defaults(func=_cmd_summarize)

    p = sub.add_parser("round2", help="Round-2 GPU pipeline (lora2-val, DPO, 7B)")
    p.add_argument("--smoke", action="store_true", help="Run each stage on 1-2 tasks, then exit")
    p.add_argument("--resume", action="store_true", help="Skip stages already recorded ok")
    p.add_argument("--only", nargs="+", metavar="STAGE", help="Run only these stages")
    p.add_argument("--round2b", action="store_true", help="Finish 7B stages into runs/round2b")
    p.add_argument("--dry-run", action="store_true", help="Print the stage plan without GPU work")
    p.add_argument("--carry-hours", action="store_true", help="Count round2 GPU hours against the 60h cap")
    p.add_argument("--prior", type=Path, action="append", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lock-note", default=None)
    p.set_defaults(func=_cmd_round2)

    p = sub.add_parser("round3", help="Round-3 GPU pipeline (DPO repair, C-then-asm)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", nargs="+", metavar="STAGE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lock-note", default=None)
    p.set_defaults(func=_cmd_round3)

    p = sub.add_parser("round4", help="Round-4 GPU pipeline (7B C-only, hybrid, loop resume)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", nargs="+", metavar="STAGE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lock-note", default=None)
    p.set_defaults(func=_cmd_round4)

    args = parser.parse_args(argv)
    if args.cmd == "run" and args.tasks is None:
        if args.split == "heldout":
            args.tasks = ROOT / "eval" / "heldout.jsonl"
        else:
            args.tasks = ROOT / "tasks" / "scaled.jsonl"
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
