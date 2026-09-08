"""Сховище вердиктів overfitting-аудиту (PASS/FAIL) для гейту paper/live.

Кожен прогін `audit_cell` (CLI `overfit` або job-worker) додає рядок у
JSONL: {ts, strategy, symbol, interval, label, reasons}. Live/paper-раннери
читають останній вердикт комірки через `latest_verdict` — це перетворює
advisory-вердикт на перевірюваний артефакт (overfitting-audit skill:
PASS обов'язковий перед paper/live).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VERDICTS_PATH = Path("results/audit_verdicts.jsonl")


def record_verdict(
    strategy: str,
    symbol: str,
    interval: str,
    label: str,
    reasons: str = "",
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Додати вердикт комірки у JSONL-журнал (append-only)."""
    p = path or DEFAULT_VERDICTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "ts": (now or datetime.now(UTC)).isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "label": label,
        "reasons": reasons,
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Вердикт %s %s %s: %s", strategy, symbol, interval, label)


def load_verdicts(path: Path | None = None) -> list[dict[str, Any]]:
    """Прочитати весь журнал вердиктів (биті рядки пропускаються)."""
    p = path or DEFAULT_VERDICTS_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Битий рядок вердикту у %s — пропускаю", p)
    return rows


def latest_verdict(
    strategy: str,
    symbol: str,
    interval: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Останній (за часом запису) вердикт комірки або None, якщо аудиту не було."""
    best: dict[str, Any] | None = None
    for row in load_verdicts(path):
        if row.get("strategy") == strategy and row.get("symbol") == symbol and row.get("interval") == interval:
            best = row  # журнал append-only → останній збіг = найсвіжіший
    return best
