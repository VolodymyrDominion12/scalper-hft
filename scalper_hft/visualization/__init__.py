"""Візуалізація бектестів (Plotly, без UI-залежностей).

Модуль будує інтерактивні багатопанельні графіки бектесту:
    - ціна: свічки + індикатори + точки входу/виходу + рівні SL/TP;
    - об'єм;
    - позиція (лонг/шорт);
    - equity + просадка (drawdown).

Використовується з Streamlit-дашборду, CLI (`plot`) та ноутбуків.

Основні функції:
    make_backtest_figure — повний графік угод;
    trade_detail_figure — деталі однієї угоди (клік по маркеру);
    find_trade_by_ts — пошук угоди за часом входу/виходу;
    equity_figure — equity + drawdown;
    trades_table — людсько-читабельна таблиця угод;
    drawdown_series — серія просадки.
"""

from __future__ import annotations

from scalper_hft.visualization.charts import (
    LONG_COLOR,
    SHORT_COLOR,
    add_indicator_overlays,
    add_sl_tp_levels,
    add_trade_markers,
    auto_indicator_columns,
    drawdown_series,
    equity_figure,
    find_trade_by_ts,
    make_backtest_figure,
    make_pairs_figure,
    trade_detail_figure,
    trades_table,
)

__all__ = [
    "LONG_COLOR",
    "SHORT_COLOR",
    "add_indicator_overlays",
    "add_sl_tp_levels",
    "add_trade_markers",
    "auto_indicator_columns",
    "drawdown_series",
    "equity_figure",
    "find_trade_by_ts",
    "make_backtest_figure",
    "make_pairs_figure",
    "trade_detail_figure",
    "trades_table",
]
