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


def pair_cell(leg1: str, leg2: str) -> str:
    """Ключ комірки пари: `LEG1/LEG2` (порядок ніг як у конфігу, не сортований)."""
    return f"{leg1}/{leg2}"


def record_verdict(
    strategy: str,
    symbol: str,
    interval: str,
    label: str,
    reasons: str = "",
    *,
    path: Path | None = None,
    now: datetime | None = None,
    pair: str = "",
    dedupe: bool = True,
) -> None:
    """Додати вердикт комірки у JSONL-журнал (append-only).

    `dedupe=True` (дефолт) пропускає запис, якщо останній рядок тієї самої
    комірки має ідентичні label+reasons: повторні прогони одного аудиту не
    мають засмічувати журнал. У старому журналі 70 із 77 рядків були дублями
    однієї комірки — реальних вердиктів було неможливо порахувати.
    """
    p = path or DEFAULT_VERDICTS_PATH
    row: dict[str, Any] = {
        "ts": (now or datetime.now(UTC)).isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "label": label,
        "reasons": reasons,
        "pair": pair,
    }
    if dedupe and _is_duplicate(p, row):
        logger.info("Вердикт %s %s %s: без змін (%s) — не дублюємо", strategy, symbol, interval, label)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Вердикт %s %s %s pair=%s: %s", strategy, symbol, interval, pair or "—", label)


def _is_duplicate(path: Path, row: dict[str, Any]) -> bool:
    """Чи дорівнює останній рядок журналу для цієї комірки новому вердикту."""
    if not path.exists():
        return False
    last: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    rec.get("strategy") == row["strategy"]
                    and rec.get("symbol") == row["symbol"]
                    and rec.get("interval") == row["interval"]
                    and rec.get("pair", "") == row["pair"]
                ):
                    last = rec
    except OSError:
        return False
    if last is None:
        return False
    return last.get("label") == row["label"] and last.get("reasons", "") == row["reasons"]


def record_pair_verdict(
    strategy: str,
    leg1: str,
    leg2: str,
    interval: str,
    label: str,
    reasons: str = "",
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Вердикт комірки пари (не двох directional-символів)."""
    cell = pair_cell(leg1, leg2)
    record_verdict(strategy, cell, interval, label, reasons, path=path, now=now, pair=cell)


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


def latest_verdicts(path: Path | None = None) -> list[dict[str, Any]]:
    """Останній вердикт на комірку (strategy, symbol, interval). Порядок — як у журналі."""
    best: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any, Any]] = []
    for row in load_verdicts(path):
        key = (row.get("strategy"), row.get("symbol"), row.get("interval"))
        if key not in best:
            order.append(key)
        best[key] = row
    return [best[k] for k in order]


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


def latest_pair_verdict(
    strategy: str,
    pair: str,
    interval: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Останній вердикт комірки пари (`pair` = `LEG1/LEG2`) або None."""
    best: dict[str, Any] | None = None
    for row in load_verdicts(path):
        if row.get("strategy") == strategy and row.get("pair") == pair and row.get("interval") == interval:
            best = row
    return best
