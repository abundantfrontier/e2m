"""GPU lock, deadline, and pid/exit files. No torch."""

from __future__ import annotations

import atexit
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
# Round 2: Boyd approved work past the original 2026-11-14 cutoff.
DEADLINE = datetime(2026, 11, 18, 18, 0, tzinfo=PT)
ROUND2_GPU_HOURS = 60.0
ROUND3_GPU_HOURS = 30.0
ROUND4_GPU_HOURS = 30.0
LOCK_PATH = Path.home() / "gpu.lock"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_gpu_lock(pid: int, note: str, *, path: Path | None = None) -> str:
    """Take ~/gpu.lock if free or stale. Refuse if another live process holds it."""
    lock = path or LOCK_PATH
    if lock.is_file():
        text = lock.read_text(encoding="utf-8").strip()
        parts = text.split()
        if parts and parts[0] != "e2m":
            raise SystemExit(f"gpu lock held by someone else: {text[:200]}")
        if len(parts) > 1 and parts[1].isdigit():
            holder = int(parts[1])
            if holder != int(pid) and pid_alive(holder):
                raise SystemExit(f"gpu lock held by live pid {holder}: {text[:200]}")
    return write_gpu_lock(pid, note, path=path)


def check_gpu_lock(*, path: Path | None = None) -> str:
    lock = path or LOCK_PATH
    if not lock.is_file():
        raise SystemExit("gpu.lock missing; refusing to use the GPU")
    text = lock.read_text(encoding="utf-8").strip()
    first = text.split()[0] if text else ""
    if first != "e2m":
        raise SystemExit(f"gpu lock held by someone else: {text[:200]}")
    return text


def write_gpu_lock(pid: int, note: str, *, path: Path | None = None) -> str:
    lock = path or LOCK_PATH
    now = datetime.now(tz=PT).isoformat()
    text = f"e2m {pid} {now} {note}\n"
    lock.write_text(text, encoding="utf-8")
    return text.strip()


def release_gpu_lock(*, owner_pid: int | None = None, path: Path | None = None) -> bool:
    """Delete the lock if we still own it. Returns True when a file was removed."""
    lock = path or LOCK_PATH
    if not lock.is_file():
        return False
    text = lock.read_text(encoding="utf-8").strip()
    parts = text.split()
    if not parts or parts[0] != "e2m":
        return False
    if owner_pid is not None and len(parts) > 1 and parts[1].isdigit():
        if int(parts[1]) != int(owner_pid):
            return False
    lock.unlink(missing_ok=True)
    return True


def past_deadline(now: datetime | None = None) -> bool:
    if os.environ.get("E2M_NO_DEADLINE", "").strip().lower() in {"1", "true", "yes"}:
        return False
    stamp = now if now is not None else datetime.now(tz=PT)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=PT)
    return stamp >= DEADLINE


def install_run_files(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = out_dir / "pid"
    exit_path = out_dir / "exit"
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    state = {"code": 1}

    def _write_exit() -> None:
        if not exit_path.exists():
            exit_path.write_text(str(state["code"]) + "\n", encoding="utf-8")

    atexit.register(_write_exit)
    return state
