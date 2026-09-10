"""Побудова інтерактивних графіків бектесту (Plotly, без UI-залежності).

Багатопанельна фігура `make_backtest_figure`:
    1. Ціна: свічки + індикатори (BB/EMA/SMA/VWAP) + точки входу/виходу
       (▲/▼ входи, × виходи) + пунктирні сегменти рівнів SL/TP;
    2. Об'єм (колір за напрямком бару);
    3. Позиція (заливка: зелений лонг / червоний шорт);
    4. Equity + просадка (drawdown, %).

Модуль не імпортує Streamlit — працює в CLI, ноутбуках та тестах.
Для великих даних є вікно (start/end) і даунсемплінг (max_bars), причому
бари входу/виходу угод завжди зберігаються на графіку.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.event_engine import EventBacktestResult
from scalper_hft.backtest.pairs import PairsResult

LONG_COLOR = "#00b894"
SHORT_COLOR = "#ef5350"
SL_COLOR = "#e74c3c"
TP_COLOR = "#16a085"
EQUITY_COLOR = "#2c3e50"

# Стилі цінових індикаторів-оверлеїв (за префіксом колонки)
_INDICATOR_STYLES: dict[str, dict] = {
    "bb_up": {"color": "#9b59b6", "width": 1, "dash": "dash"},
    "bb_low": {"color": "#9b59b6", "width": 1, "dash": "dash"},
    "bb_mid": {"color": "#9b59b6", "width": 1, "dash": "dot"},
    "ema_9": {"color": "#e67e22", "width": 1, "dash": None},
    "ema_21": {"color": "#2980b9", "width": 1, "dash": None},
    "ema_50": {"color": "#7f8c8d", "width": 1, "dash": None},
}
_DEFAULT_STYLE = {"color": "#95a5a6", "width": 1, "dash": "dot"}


# ── Серії та таблиці ─────────────────────────────────────────────────────────
def drawdown_series(equity: pd.Series) -> pd.Series:
    """Просадка відносно історичного максимуму: equity/cummax − 1 (≤ 0)."""
    return equity / equity.cummax() - 1.0


def _ukr_plural(n: int, one: str, few: str, many: str) -> str:
    """Українська плюралізація: 1 бар, 2 бари, 5 барів."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not (12 <= n100 <= 14):
        return few
    return many


def trades_table(result: BacktestResult | PairsResult, initial_capital: float = 10_000.0) -> pd.DataFrame:
    """Людсько-читабельна таблиця угод для UI (ціни, SL/TP, PnL)."""
    t = result.trades
    out = pd.DataFrame(
        {
            "Вхід": t["entry_ts"],
            "Вихід": t["exit_ts"],
            "Сторона": np.where(t["side"] == 1, "Лонг", "Шорт"),
            "Ціна входу": t["entry_price"].astype(float) if "entry_price" in t.columns else np.nan,
            "Ціна виходу": t["exit_price"].astype(float) if "exit_price" in t.columns else np.nan,
            "SL": t["sl_price"].astype(float) if "sl_price" in t.columns else np.nan,
            "TP": t["tp_price"].astype(float) if "tp_price" in t.columns else np.nan,
            "PnL, %": t["ret"] * 100.0,
            "PnL, $": t["ret"] * initial_capital,
        }
    )
    return out


# ── Індикатори ───────────────────────────────────────────────────────────────
def auto_indicator_columns(df: pd.DataFrame) -> list[str]:
    """Цінові індикаторні колонки у df (bb_*/ema_*/sma_*/vwap*/wma*)."""
    out: list[str] = []
    for col in df.columns:
        base = col.split("_")[0]
        if base in {"bb", "ema", "sma", "vwap", "wma"}:
            out.append(col)
    return out


def add_indicator_overlays(fig: go.Figure, df: pd.DataFrame, columns: Sequence[str], row: int = 1) -> None:
    """Додати лінії індикаторів на панель ціни (row) з канонічними стилями.

    Всі оверлеї належать одній legend-групі "indicators" — клік по легенді
    вмикає/вимикає їх разом.
    """
    for col in columns:
        if col not in df.columns:
            continue
        style = _INDICATOR_STYLES.get(col, _DEFAULT_STYLE)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                legendgroup="indicators",
                line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                hovertemplate=f"{col}: %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=1,
        )


# ── Угоди та SL/TP ───────────────────────────────────────────────────────────
def add_trade_markers(fig: go.Figure, trades: pd.DataFrame, df: pd.DataFrame | None = None, row: int = 1) -> None:
    """Точки входу (▲/▼) та виходу (×) на панелі ціни.

    Колір за стороною дії: купівля = зелений, продаж = червоний.
    Якщо у trades немає entry_price/exit_price — береться close з df.
    """
    if trades is None or trades.empty:
        return
    t = trades
    if "entry_price" in t.columns:
        px_entry = t["entry_price"].to_numpy(dtype=float)
    elif df is not None:
        px_entry = df["close"].reindex(t["entry_ts"]).to_numpy(dtype=float)
    else:
        px_entry = np.full(len(t), np.nan)
    if "exit_price" in t.columns:
        px_exit = t["exit_price"].to_numpy(dtype=float)
    elif df is not None:
        px_exit = df["close"].reindex(t["exit_ts"]).to_numpy(dtype=float)
    else:
        px_exit = np.full(len(t), np.nan)

    ts_entry = t["entry_ts"].to_numpy()
    ts_exit = t["exit_ts"].to_numpy()
    side = t["side"].to_numpy()
    ret = t["ret"].to_numpy()

    for s, label, color in ((1, "Лонг-вхід", LONG_COLOR), (-1, "Шорт-вхід", SHORT_COLOR)):
        sel = side == s
        if not sel.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=ts_entry[sel],
                y=px_entry[sel],
                mode="markers",
                name=label,
                legendgroup="trades",
                marker=dict(
                    symbol="triangle-up" if s == 1 else "triangle-down",
                    size=12,
                    color=color,
                    line=dict(width=1, color="#333333"),
                ),
                hovertemplate=(f"{label}<br>%{{x|%Y-%m-%d %H:%M}}<br>вхід: %{{y:.4f}}<extra></extra>"),
            ),
            row=row,
            col=1,
        )
    # Виходи: закриття лонга (продаж) = червоний ×, закриття шорта (купівля) = зелений ×
    for s, label, color in ((1, "Лонг-вихід", SHORT_COLOR), (-1, "Шорт-вихід", LONG_COLOR)):
        sel = side == s
        if not sel.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=ts_exit[sel],
                y=px_exit[sel],
                customdata=ret[sel],
                mode="markers",
                name=label,
                legendgroup="trades",
                marker=dict(symbol="x", size=11, color=color, line=dict(width=1, color="#333333")),
                hovertemplate=(
                    f"{label}<br>%{{x|%Y-%m-%d %H:%M}}<br>вихід: %{{y:.4f}}<br>PnL: %{{customdata:.3%}}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def add_sl_tp_levels(fig: go.Figure, trades: pd.DataFrame, row: int = 1, max_trades: int = 500) -> None:
    """Горизонтальні пунктирні сегменти SL/TP від входу до виходу угоди.

    Потребує колонок sl_price/tp_price у trades (додаються рушієм зі
    стратегії через Strategy.exit_levels). При великій кількості угод
    малюються лише останні max_trades сегментів (щоб не вантажити браузер).
    """
    if trades is None or trades.empty or "sl_price" not in trades.columns:
        return
    t = trades.dropna(subset=["sl_price", "tp_price"])
    if t.empty:
        return
    if len(t) > max_trades:
        t = t.tail(max_trades)

    for key, color, label, group in (
        ("sl_price", SL_COLOR, "Stop-loss", "SL"),
        ("tp_price", TP_COLOR, "Take-profit", "TP"),
    ):
        first = True
        for _, r in t.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["entry_ts"], r["exit_ts"]],
                    y=[r[key], r[key]],
                    mode="lines",
                    name=label,
                    legendgroup=group,
                    showlegend=first,
                    line=dict(color=color, width=1, dash="dash"),
                    hovertemplate=f"{label}: %{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            first = False


DEFAULT_VIEW_BARS = 250
PAD_BARS_BEFORE = 20


def default_price_window(
    index: pd.DatetimeIndex,
    trades: pd.DataFrame | None = None,
    *,
    bars: int = DEFAULT_VIEW_BARS,
    pad_before: int = PAD_BARS_BEFORE,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Робоче вікно свічок: ~``bars`` від першого входу (інакше від старту ряду)."""
    if index is None or len(index) == 0:
        raise ValueError("index порожній")
    width = max(int(bars), 2)
    pad = max(int(pad_before), 0)
    start_pos = 0
    if trades is not None and not trades.empty and "entry_ts" in trades.columns:
        first = pd.to_datetime(trades["entry_ts"], errors="coerce").dropna()
        if not first.empty:
            loc = int(index.searchsorted(first.iloc[0], side="left"))
            start_pos = max(0, loc - pad)
    end_pos = min(len(index) - 1, start_pos + width - 1)
    start_pos = max(0, end_pos - width + 1)
    return pd.Timestamp(index[start_pos]), pd.Timestamp(index[end_pos])


def shift_window(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    index: pd.DatetimeIndex,
    direction: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Зсунути [start, end] на половину ширини. ``direction``: −1 ліворуч, +1 праворуч."""
    if index is None or len(index) == 0:
        raise ValueError("index порожній")
    i0 = int(index.searchsorted(pd.Timestamp(start), side="left"))
    i1 = int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    i0 = max(0, min(i0, len(index) - 1))
    i1 = max(i0, min(i1, len(index) - 1))
    width = max(i1 - i0 + 1, 2)
    step = max(width // 2, 1) * (1 if int(direction) >= 0 else -1)
    new0 = max(0, min(i0 + step, len(index) - width))
    new1 = min(len(index) - 1, new0 + width - 1)
    return pd.Timestamp(index[new0]), pd.Timestamp(index[new1])


def window_around_trade(
    entry_ts: pd.Timestamp | str,
    index: pd.DatetimeIndex,
    *,
    bars: int = DEFAULT_VIEW_BARS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Вікно ``bars`` барів, центроване на вході угоди (обрізане індексом)."""
    if index is None or len(index) == 0:
        raise ValueError("index порожній")
    width = max(int(bars), 2)
    loc = int(index.searchsorted(pd.Timestamp(entry_ts), side="left"))
    loc = max(0, min(loc, len(index) - 1))
    half = width // 2
    start_pos = max(0, loc - half)
    end_pos = min(len(index) - 1, start_pos + width - 1)
    start_pos = max(0, end_pos - width + 1)
    return pd.Timestamp(index[start_pos]), pd.Timestamp(index[end_pos])


def neighboring_entry_ts(
    trades: pd.DataFrame,
    current: pd.Timestamp | str | None,
    *,
    step: int,
) -> pd.Timestamp | None:
    """Попередній/наступний ``entry_ts``. ``step`` −1 або +1. None, якщо краю досягнуто."""
    if trades is None or trades.empty or "entry_ts" not in trades.columns:
        return None
    entries = pd.to_datetime(trades["entry_ts"], errors="coerce").dropna().sort_values().unique()
    if len(entries) == 0:
        return None
    if current is None:
        pick = entries[0] if int(step) >= 0 else entries[-1]
        return pd.Timestamp(pick)
    pos = int(pd.Index(entries).searchsorted(pd.Timestamp(current), side="left"))
    if pos < len(entries) and pd.Timestamp(entries[pos]) == pd.Timestamp(current):
        nxt = pos + int(step)
    else:
        nxt = pos if int(step) >= 0 else pos - 1
    if nxt < 0 or nxt >= len(entries):
        return None
    return pd.Timestamp(entries[nxt])


# ── Вікно / даунсемплінг ─────────────────────────────────────────────────────
def _window(
    df: pd.DataFrame,
    result: BacktestResult | EventBacktestResult | PairsResult,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    max_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Зріз даних: вікно (start/end) і/або даунсемплінг (max_bars).

    Бари входу/виходу угод завжди включаються у вибірку, щоб маркери не
    зникали при даунсемплінгу. Повертає (df, trades, positions, equity),
    вирівняні за спільним індексом.
    """
    trades = result.trades if result.trades is not None else pd.DataFrame()
    sub = df

    if start is not None or end is not None:
        lo = pd.Timestamp(start) if start is not None else df.index[0]
        hi = pd.Timestamp(end) if end is not None else df.index[-1]
        mask = (df.index >= lo) & (df.index <= hi)
        sub = df.loc[mask]
        if not trades.empty:
            trades = trades[(trades["entry_ts"] <= hi) & (trades["exit_ts"] >= lo)]

    if len(sub) > max_bars and not sub.empty:
        stride = max(1, math.ceil(len(sub) / max_bars))
        pos = np.arange(0, len(sub), stride)
        if not trades.empty:
            extra = np.concatenate(
                [
                    sub.index.get_indexer(trades["entry_ts"]),
                    sub.index.get_indexer(trades["exit_ts"]),
                ]
            )
            extra = extra[(extra >= 0) & (extra < len(sub))]
            pos = np.unique(np.concatenate([pos, extra]))
        sub = sub.iloc[pos]

    positions = result.positions.reindex(sub.index).fillna(0.0)
    equity = result.equity.reindex(sub.index)
    return sub, trades, positions, equity


# ── Основні фігури ───────────────────────────────────────────────────────────
def make_backtest_figure(
    df: pd.DataFrame,
    result: BacktestResult | EventBacktestResult | PairsResult,
    *,
    indicators: Sequence[str] | None = None,
    symbol: str | None = None,
    show_volume: bool = True,
    show_position: bool = True,
    show_equity: bool = True,
    with_trades: bool = True,
    with_sl_tp: bool = True,
    max_sl_tp_trades: int = 500,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    max_bars: int = 20_000,
    template: str = "plotly_white",
    title: str | None = None,
) -> go.Figure:
    """Багатопанельний інтерактивний графік бектесту.

    Панелі (спільна вісь часу): ціна (свічки + індикатори + угоди + SL/TP),
    об'єм, позиція, equity + просадка. Внизу — range slider для зуму.

    Args:
        df: OHLCV DataFrame (опційно з колонками індикаторів).
        result: результат run_backtest.
        indicators: колонки-оверлеї; None = автодетект (bb_*/ema_*/sma_*...).
        symbol: назва символу для заголовка.
        with_trades/with_sl_tp: вмикати маркери угод / сегменти SL/TP.
        start/end: вікно графіка (ISO-строка або Timestamp).
        max_bars: даунсемплінг для великих даних (бари угод зберігаються).
    """
    if df is None or df.empty:
        raise ValueError("df порожній — немає що малювати")

    sub_df, sub_trades, sub_pos, sub_eq = _window(df, result, start, end, max_bars)

    n_rows = 1 + int(show_volume) + int(show_position) + int(show_equity)
    rows: dict[str, int] = {"price": 1}
    cur = 1
    if show_volume:
        cur += 1
        rows["volume"] = cur
    if show_position:
        cur += 1
        rows["position"] = cur
    if show_equity:
        cur += 1
        rows["equity"] = cur

    row_heights = [0.52]
    if show_volume:
        row_heights.append(0.10)
    if show_position:
        row_heights.append(0.13)
    if show_equity:
        row_heights.append(0.25)
    row_specs: list[list[dict[str, Any]]] = [
        [{"secondary_y": True}] if (show_equity and r == rows["equity"]) else [{}] for r in range(1, n_rows + 1)
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=row_specs,
        subplot_titles=["Ціна", "Об'єм", "Позиція", "Equity"][:n_rows],
    )

    # 1. Ціна
    fig.add_trace(
        go.Candlestick(
            x=sub_df.index,
            open=sub_df["open"],
            high=sub_df["high"],
            low=sub_df["low"],
            close=sub_df["close"],
            name="OHLC",
            increasing_line_color=LONG_COLOR,
            decreasing_line_color=SHORT_COLOR,
            increasing_fillcolor=LONG_COLOR,
            decreasing_fillcolor=SHORT_COLOR,
        ),
        row=rows["price"],
        col=1,
    )
    if indicators is None:
        indicators = auto_indicator_columns(df)
    add_indicator_overlays(fig, sub_df, indicators, row=rows["price"])
    if with_trades:
        add_trade_markers(fig, sub_trades, df=sub_df, row=rows["price"])
    if with_sl_tp:
        add_sl_tp_levels(fig, sub_trades, row=rows["price"], max_trades=max_sl_tp_trades)

    # 2. Об'єм
    if show_volume:
        vol_colors = np.where(sub_df["close"] >= sub_df["open"], LONG_COLOR, SHORT_COLOR)
        fig.add_trace(
            go.Bar(
                x=sub_df.index,
                y=sub_df["volume"],
                name="Об'єм",
                marker_color=vol_colors,
                opacity=0.6,
                hovertemplate="Об'єм: %{y:.3g}<extra></extra>",
            ),
            row=rows["volume"],
            col=1,
        )

    # 3. Позиція (заливка: лонг/шорт)
    if show_position:
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            row=rows["position"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub_pos.index,
                y=sub_pos.clip(lower=0.0),
                mode="lines",
                line_shape="hv",
                line=dict(color=LONG_COLOR, width=1),
                fill="tozeroy",
                fillcolor="rgba(0,184,148,0.25)",
                name="Лонг",
                legendgroup="position",
                hovertemplate="Позиція: %{y:.4f}<extra></extra>",
            ),
            row=rows["position"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub_pos.index,
                y=sub_pos.clip(upper=0.0),
                mode="lines",
                line_shape="hv",
                line=dict(color=SHORT_COLOR, width=1),
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.25)",
                name="Шорт",
                legendgroup="position",
                hovertemplate="Позиція: %{y:.4f}<extra></extra>",
            ),
            row=rows["position"],
            col=1,
        )

    # 4. Equity + просадка (%)
    if show_equity:
        fig.add_trace(
            go.Scatter(
                x=sub_eq.index,
                y=sub_eq.values,
                mode="lines",
                name="Equity",
                legendgroup="equity",
                line=dict(color=EQUITY_COLOR, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(44,62,80,0.08)",
                hovertemplate="Equity: %{y:,.2f}<extra></extra>",
            ),
            row=rows["equity"],
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sub_eq.index,
                y=drawdown_series(sub_eq) * 100.0,
                mode="lines",
                name="Просадка, %",
                legendgroup="equity",
                line=dict(color=SL_COLOR, width=1),
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.15)",
                hovertemplate="Просадка: %{y:.2f}%<extra></extra>",
            ),
            row=rows["equity"],
            col=1,
            secondary_y=True,
        )

    head = f"{symbol} · " if symbol else ""
    n_bars = _ukr_plural(len(sub_df), "бар", "бари", "барів")
    n_tr = _ukr_plural(len(result.trades), "угода", "угоди", "угод")
    fig.update_layout(
        title=title or f"{head}Бектест · {len(sub_df):,} {n_bars} · {len(result.trades)} {n_tr}",
        template=template,
        height=340 + 200 * n_rows,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True, row=n_rows, col=1)
    fig.update_yaxes(title_text="Ціна", row=rows["price"], col=1)
    if show_volume:
        fig.update_yaxes(title_text="Об'єм", row=rows["volume"], col=1)
    if show_position:
        fig.update_yaxes(title_text="Позиція", row=rows["position"], col=1)
    if show_equity:
        fig.update_yaxes(title_text="Equity", row=rows["equity"], col=1, secondary_y=False)
        fig.update_yaxes(title_text="DD, %", row=rows["equity"], col=1, secondary_y=True, showgrid=False)
    return fig


def make_pairs_figure(
    result: PairsResult,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    max_bars: int = 20_000,
    with_trades: bool = True,
    template: str = "plotly_white",
    title: str | None = None,
    symbol: str | None = None,
) -> go.Figure:
    """Графік парного бектесту: спред + маркери угод + позиція + equity.

    Без OHLC-свічок — `PairsResult` зберігає log-ratio спред, не ціну ноги.
    """
    if result.spread is None or result.spread.empty:
        raise ValueError("spread порожній — немає що малювати")
    df = result.spread.rename("close").to_frame()
    sub_df, sub_trades, sub_pos, sub_eq = _window(df, result, start, end, max_bars)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.18, 0.32],
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=["Спред", "Позиція", "Equity"],
    )
    fig.add_trace(
        go.Scatter(
            x=sub_df.index,
            y=sub_df["close"],
            mode="lines",
            name="Спред",
            line=dict(color=EQUITY_COLOR, width=1.2),
            hovertemplate="Спред: %{y:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if with_trades:
        add_trade_markers(fig, sub_trades, df=sub_df, row=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=sub_pos.index,
            y=sub_pos.clip(lower=0.0),
            mode="lines",
            line_shape="hv",
            line=dict(color=LONG_COLOR, width=1),
            fill="tozeroy",
            fillcolor="rgba(0,184,148,0.25)",
            name="Лонг",
            legendgroup="position",
            hovertemplate="Позиція: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_pos.index,
            y=sub_pos.clip(upper=0.0),
            mode="lines",
            line_shape="hv",
            line=dict(color=SHORT_COLOR, width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.25)",
            name="Шорт",
            legendgroup="position",
            hovertemplate="Позиція: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_eq.index,
            y=sub_eq.values,
            mode="lines",
            name="Equity",
            legendgroup="equity",
            line=dict(color=EQUITY_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44,62,80,0.08)",
            hovertemplate="Equity: %{y:,.2f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_eq.index,
            y=drawdown_series(sub_eq) * 100.0,
            mode="lines",
            name="Просадка, %",
            legendgroup="equity",
            line=dict(color=SL_COLOR, width=1),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.15)",
            hovertemplate="Просадка: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    head = f"{symbol} · " if symbol else ""
    n_bars = _ukr_plural(len(sub_df), "бар", "бари", "барів")
    n_tr = _ukr_plural(len(result.trades), "угода", "угоди", "угод")
    fig.update_layout(
        title=title or f"{head}Pairs · {len(sub_df):,} {n_bars} · {len(result.trades)} {n_tr}",
        template=template,
        height=780,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
    fig.update_yaxes(title_text="Спред", row=1, col=1)
    fig.update_yaxes(title_text="Позиція", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="DD, %", row=3, col=1, secondary_y=True, showgrid=False)
    return fig


def equity_figure(
    result: BacktestResult,
    *,
    template: str = "plotly_white",
    title: str | None = None,
) -> go.Figure:
    """Equity-крива + просадка (для звітів та дашборду)."""
    eq = result.equity
    dd = drawdown_series(eq) * 100.0
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq.values,
            mode="lines",
            name="Equity",
            line=dict(color=EQUITY_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44,62,80,0.08)",
            hovertemplate="Equity: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Просадка, %",
            line=dict(color=SL_COLOR, width=1),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.15)",
            hovertemplate="Просадка: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title or f"Equity · {len(eq):,} барів",
        template=template,
        height=420,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="DD, %", row=2, col=1)
    return fig


# ── Деталі однієї угоди (клік по маркеру) ───────────────────────────────────
def find_trade_by_ts(trades: pd.DataFrame, ts) -> pd.Series | None:
    """Знайти угоду за часом входу/виходу (для кліку по маркеру на графіку).

    ts: pd.Timestamp (або ISO-строка) з маркера входу/виходу. None, якщо
    угоди з таким часом немає або ts непарсований.
    """
    if trades is None or trades.empty or ts is None:
        return None
    try:
        ts = pd.Timestamp(ts)
    except (ValueError, TypeError):
        return None
    hit = trades[(trades["entry_ts"] == ts) | (trades["exit_ts"] == ts)]
    return hit.iloc[0] if not hit.empty else None


def trade_detail_figure(
    df: pd.DataFrame,
    result: BacktestResult,
    entry_ts: pd.Timestamp,
    *,
    before: int = 40,
    after: int = 20,
    indicators: Sequence[str] | None = None,
    template: str = "plotly_white",
    title: str | None = None,
) -> go.Figure:
    """Детальний графік однієї угоди — вікно навколо входу.

    Панелі: ціна (свічки + індикатори + маркер входу/виходу + сегмент SL/TP),
    позиція, equity. Викликається при кліку по маркеру угоди у дашборді.
    """
    trade = find_trade_by_ts(result.trades, entry_ts)
    if trade is None:
        raise ValueError(f"Немає угоди з входом {entry_ts}")

    i_entry = df.index.get_indexer([trade["entry_ts"]], method="nearest")[0]
    i_exit = df.index.get_indexer([trade["exit_ts"]], method="nearest")[0]
    lo = max(0, i_entry - before)
    hi = min(len(df), i_exit + after)
    sub = df.iloc[lo:hi]
    sub_pos = result.positions.reindex(sub.index).fillna(0.0)
    sub_eq = result.equity.reindex(sub.index)
    one = pd.DataFrame([trade])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.18, 0.22],
        subplot_titles=["Ціна", "Позиція", "Equity"],
    )
    fig.add_trace(
        go.Candlestick(
            x=sub.index,
            open=sub["open"],
            high=sub["high"],
            low=sub["low"],
            close=sub["close"],
            name="OHLC",
            increasing_line_color=LONG_COLOR,
            decreasing_line_color=SHORT_COLOR,
            increasing_fillcolor=LONG_COLOR,
            decreasing_fillcolor=SHORT_COLOR,
        ),
        row=1,
        col=1,
    )
    if indicators is None:
        indicators = auto_indicator_columns(df)
    add_indicator_overlays(fig, sub, indicators, row=1)
    add_trade_markers(fig, one, df=sub, row=1)
    add_sl_tp_levels(fig, one, row=1, max_trades=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)
    for sign, name, color, fill in (
        (1.0, "Лонг", LONG_COLOR, "rgba(0,184,148,0.25)"),
        (-1.0, "Шорт", SHORT_COLOR, "rgba(239,83,80,0.25)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=sub_pos.index,
                y=sub_pos.where(sub_pos * sign > 0, 0.0),
                mode="lines",
                line_shape="hv",
                line=dict(color=color, width=1),
                fill="tozeroy",
                fillcolor=fill,
                name=name,
                legendgroup="position",
                hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=sub_eq.index,
            y=sub_eq.values,
            mode="lines",
            name="Equity",
            line=dict(color=EQUITY_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44,62,80,0.08)",
            hovertemplate="Equity: %{y:,.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    side = "Лонг" if trade["side"] == 1 else "Шорт"
    t0 = pd.Timestamp(trade["entry_ts"])
    if title is None:
        title = (
            f"{side} · вхід {t0:%d.%m %H:%M} · "
            f"{trade['entry_price']:.2f} → {trade['exit_price']:.2f} · PnL {trade['ret']:.2%}"
        )
    fig.update_layout(
        title=title,
        template=template,
        height=560,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
    fig.update_yaxes(title_text="Ціна", row=1, col=1)
    fig.update_yaxes(title_text="Позиція", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=3, col=1)
    return fig
