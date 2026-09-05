"""Спільні константи сторінок дашборду.

SYMBOLS — канонічний універсум інструментів (див. `scalper_hft.symbols`),
той самий, що є fallback-списком `DEFAULT_SYMBOLS` у config. Якщо у .env
задано свій `DEFAULT_SYMBOLS`, CLI/sweep/MCP підуть за ним, а дашборд
показує курований список нижче. PAIR_CHOICES — окремий курований список
(лише перевірені пари для pairs_arb).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from scalper_hft.symbols import CANONICAL_SYMBOLS

SYMBOLS: list[str] = list(CANONICAL_SYMBOLS)

# Пари для pairs_arb: лише ті, що пройшли коінтеграційний скринінг/валідацію.
PAIR_CHOICES = ["XRPUSDT/BTCUSDT", "BTCUSDT/ETHUSDT", "LINKUSDT/BTCUSDT", "LINKUSDT/ETHUSDT"]

# Ті самі ТФ, що validation.sweep.DEFAULT_INTERVALS — без імпорту validation
# (дашборд піднімає _common до site-packages).
BT_INTERVALS: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h"]
RESEARCH_BT_PREFILL = "research_bt_prefill"
RESEARCH_AUDIT_PREFILL = "research_audit_prefill"


def combo_prefill(row: Mapping[str, Any], *, default_days: int = 60) -> dict[str, Any]:
    """Нормалізувати рядок sweep/топа до prefill для бектесту й аудиту."""
    days_raw = row.get("days", default_days)
    try:
        days_f = float(days_raw)  # type: ignore[arg-type]
        days = default_days if days_f != days_f else int(days_f)
    except (TypeError, ValueError):
        days = default_days
    return {
        "strategy": str(row["strategy"]),
        "symbol": str(row["symbol"]),
        "interval": str(row["interval"]),
        "days": days,
    }


def apply_research_bt_prefill(state: MutableMapping[str, Any]) -> None:
    """Перенести prefill зі сторінки досліджень у ключі віджетів бектесту.

    Викликати до створення sidebar-віджетів.
    """
    raw = state.pop(RESEARCH_BT_PREFILL, None)
    if not isinstance(raw, dict):
        return
    if raw.get("strategy"):
        state["bt_strategy"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["bt_symbol"] = str(raw["symbol"])
    if raw.get("interval"):
        state["bt_interval_single"] = str(raw["interval"])
    if raw.get("days") is not None:
        try:
            state["bt_days"] = int(raw["days"])
        except (TypeError, ValueError):
            pass


def apply_research_audit_prefill(state: MutableMapping[str, Any]) -> None:
    """Перенести prefill топа у віджети вкладки аудиту. Викликати до віджетів."""
    raw = state.pop(RESEARCH_AUDIT_PREFILL, None)
    if not isinstance(raw, dict):
        return
    if raw.get("strategy"):
        state["au_strat"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["au_sym"] = str(raw["symbol"])
    if raw.get("interval"):
        state["au_iv"] = str(raw["interval"])
    if raw.get("days") is not None:
        try:
            state["au_days"] = int(raw["days"])
        except (TypeError, ValueError):
            pass
    if raw.get("strategy"):
        state["rg_strat"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["rg_sym"] = str(raw["symbol"])
    if raw.get("interval"):
        state["rg_iv"] = str(raw["interval"])
    if raw.get("days") is not None:
        try:
            state["rg_days"] = int(raw["days"])
        except (TypeError, ValueError):
            pass


def overfit_job_payload(
    strategy: str,
    symbol: str,
    interval: str,
    days: int,
    *,
    train_bars: int,
    test_bars: int,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "days": int(days),
        "train_bars": int(train_bars),
        "test_bars": int(test_bars),
    }


def single_backtest_payload(strategy: str, symbol: str, interval: str, days: int) -> dict[str, Any]:
    """Той самий payload, що сторінка Бектест для одиночної стратегії."""
    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "days": int(days),
        "params": {},
        "maker": False,
        "trace": False,
        "base_interval": "1m",
        "breakeven_gate": False,
    }
