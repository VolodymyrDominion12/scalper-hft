"""Fingerprint коду для invalidation довгоживучих worker-процесів."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]


def package_code_hash() -> str:
    """SHA-256[:12] ключових модулів research/validation/live."""
    parts: list[str] = []
    for rel in (
        "validation/cell_audit.py",
        "validation/sweep.py",
        "research/job_handlers.py",
        "research/sweep_store.py",
        "live/pairs_engine.py",
        "backtest/pairs.py",
    ):
        path = _PKG_ROOT / rel
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    blob = "\n".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def worker_code_hash() -> str:
    """Аліас для логів worker."""
    return package_code_hash()
