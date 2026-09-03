"""Ідентифікатор зрізу коду для логів і Telegram (git tag/SHA)."""

from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def git_revision(cwd: Path | None = None) -> str:
    """`git describe --tags --always --dirty`, або unknown якщо немає git."""
    root = cwd if cwd is not None else _ROOT
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def startup_banner(*, mode: str, extra: str = "") -> str:
    from scalper_hft.config import get_settings

    settings = get_settings()
    dry = "DRY_RUN=true" if settings.dry_run else "DRY_RUN=false"
    line = f"scalper-hft {mode} | {git_revision()} | {dry}"
    if extra:
        line = f"{line} | {extra}"
    return line
