"""Session & Regime Analytics — розбивка метрик по часу доби, дню тижня та режиму.

Мета: виявити, в якій «сесії» стратегія найкраще/найгірше торгує,
і як ведуть себе угоди в різних ринкових режимах.

Використання:
    from scalper_hft.research.session_analysis import (
        session_breakdown, weekday_breakdown, regime_breakdown,
        structure_breakdown, named_regime_breakdown, mae_mfe_analysis
    )
    trades_df = res.trades
    session = session_breakdown(trades_df)
    weekday = weekday_breakdown(trades_df)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from scalper_hft.features.regimes import COMPOSITE_ORDER, STRUCTURE_ORDER, VOL_ORDER


def session_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Метрики по годинах UTC (0–23).

    Для кожної години: кількість угод, win_rate, avg_pnl, total_pnl.
    Допомагає виявити найкращий/найгірший час входу.
    """
    if trades_df is None or trades_df.empty or "entry_ts" not in trades_df.columns:
        return pd.DataFrame()

    df = trades_df.copy()
    df["hour"] = pd.to_datetime(df["entry_ts"]).dt.hour

    rows = []
    for hour in range(24):
        sub = df[df["hour"] == hour]
        if sub.empty:
            continue
        rets = sub["ret"] if "ret" in sub.columns else pd.Series(dtype=float)
        rows.append(
            {
                "hour_utc": hour,
                "n_trades": len(sub),
                "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
                "avg_pnl": float(rets.mean()) if len(rets) else 0.0,
                "total_pnl": float(rets.sum()) if len(rets) else 0.0,
                "median_pnl": float(rets.median()) if len(rets) else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("hour_utc")


def weekday_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Метрики по днях тижня (0=Пн, 6=Нд).

    Для кожного дня: кількість угод, win_rate, avg_pnl.
    """
    if trades_df is None or trades_df.empty or "entry_ts" not in trades_df.columns:
        return pd.DataFrame()

    df = trades_df.copy()
    ts = pd.to_datetime(df["entry_ts"])
    df["weekday"] = ts.dt.dayofweek
    df["weekday_name"] = ts.dt.day_name()

    rows = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for wd in range(7):
        sub = df[df["weekday"] == wd]
        rets = sub["ret"] if "ret" in sub.columns and not sub.empty else pd.Series(dtype=float)
        rows.append(
            {
                "weekday": wd,
                "day": day_names[wd],
                "n_trades": len(sub),
                "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
                "avg_pnl": float(rets.mean()) if len(rets) else 0.0,
                "total_pnl": float(rets.sum()) if len(rets) else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("weekday")


def hourly_fill_rate(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Fill-rate по годинах UTC з журналу ордерів (status filled/unfilled)."""
    if orders_df is None or orders_df.empty or "ts" not in orders_df.columns:
        return pd.DataFrame()
    if "status" not in orders_df.columns:
        return pd.DataFrame()
    df = orders_df.copy()
    df["hour_utc"] = pd.to_datetime(df["ts"]).dt.hour
    rows: list[dict] = []
    for hour in range(24):
        sub = df[df["hour_utc"] == hour]
        if sub.empty:
            continue
        filled = int((sub["status"] == "filled").sum())
        unfilled = int((sub["status"] == "unfilled").sum())
        total = filled + unfilled
        rows.append(
            {
                "hour_utc": hour,
                "n_filled": filled,
                "n_unfilled": unfilled,
                "fill_rate": (filled / total) if total else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("hour_utc")


def _metrics_for_returns(rets: pd.Series, n_trades: int) -> dict[str, float]:
    """Trade-level metrics. Sharpe = mean(ret)/std(ret), not annualized."""
    if rets.empty:
        return {
            "n_trades": float(n_trades),
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "median_pnl": 0.0,
            "profit_factor": float("nan"),
            "sharpe": float("nan"),
        }
    losses = rets[rets < 0]
    gains = rets[rets > 0]
    loss_sum = float(-losses.sum()) if not losses.empty else 0.0
    if loss_sum > 0:
        profit_factor = float(gains.sum() / loss_sum)
    else:
        profit_factor = float("nan")
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    mean = float(rets.mean())
    sharpe = mean / std if std > 0 else float("nan")
    return {
        "n_trades": float(n_trades),
        "win_rate": float((rets > 0).mean()),
        "avg_pnl": mean,
        "total_pnl": float(rets.sum()),
        "median_pnl": float(rets.median()),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
    }


def _regime_at_entry(entry_ts: pd.Series, regime_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(entry_ts)
    return ts.map(lambda t: regime_series.asof(t))


def _breakdown_by_labels(
    trades_df: pd.DataFrame,
    labels: pd.Series,
    order: Sequence[str],
    index_name: str,
) -> pd.DataFrame:
    rets_col = "ret" if "ret" in trades_df.columns else None
    rows: list[dict[str, float | str]] = []
    for name in order:
        mask = labels == name
        n = int(mask.sum())
        rets = trades_df.loc[mask, "ret"] if rets_col is not None else pd.Series(dtype=float)
        row: dict[str, float | str] = {index_name: name}
        row.update(_metrics_for_returns(rets, n))
        rows.append(row)
    return pd.DataFrame(rows).set_index(index_name)


def regime_breakdown(trades_df: pd.DataFrame, regime_series: pd.Series | None = None) -> pd.DataFrame:
    """Метрики угод по режимах волатильності (low/normal/high).

    regime_series: Series з індексом datetime і значеннями "low"/"normal"/"high"
    (результат `volatility_regime()`). Якщо None — повертає порожній DataFrame.

    Sharpe тут — mean/std прибутків угод, не річний. Гіпотеза preferred_regimes
    підтверджується лише якщо OOS Sharpe у «своєму» режимі стійко кращий.
    """
    if trades_df is None or trades_df.empty or regime_series is None:
        return pd.DataFrame()

    df = trades_df.copy()
    labels = _regime_at_entry(df["entry_ts"], regime_series)
    return _breakdown_by_labels(df, labels, VOL_ORDER, "regime")


def structure_breakdown(
    trades_df: pd.DataFrame,
    structure_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Метрики угод по структурі ринку (range / trend_up / trend_down)."""
    if trades_df is None or trades_df.empty or structure_series is None:
        return pd.DataFrame()
    df = trades_df.copy()
    labels = _regime_at_entry(df["entry_ts"], structure_series)
    return _breakdown_by_labels(df, labels, STRUCTURE_ORDER, "structure")


def named_regime_breakdown(
    trades_df: pd.DataFrame,
    state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Метрики угод по складеному стану `structure|vol` (9 комірок).

    `state` — результат `named_market_state(close)`. Клас стратегії не
    оновлюється з цієї таблиці автоматично: потрібен окремий OOS-вердикт.
    """
    if trades_df is None or trades_df.empty or state is None or state.empty:
        return pd.DataFrame()
    if "label" in state.columns:
        label_series = state["label"]
    elif {"structure", "vol"} <= set(state.columns):
        label_series = state["structure"].astype(str) + "|" + state["vol"].astype(str)
    else:
        raise ValueError("state must have 'label' or both 'structure' and 'vol'")
    df = trades_df.copy()
    labels = _regime_at_entry(df["entry_ts"], label_series)
    return _breakdown_by_labels(df, labels, COMPOSITE_ORDER, "regime")


def mae_mfe_analysis(trades_df: pd.DataFrame, df_bars: pd.DataFrame) -> pd.DataFrame:
    """Maximum Adverse / Favorable Excursion аналіз угод.

    MAE: наскільки максимально ціна йшла ПРОТИ позиції до виходу.
    MFE: наскільки максимально ціна йшла ЗА позицією до виходу.

    Якщо MAE >> фактичний PnL → стоп-лос можна підтягнути.
    Якщо MFE >> фактичний PnL → тейк-профіт занизький (виходимо рано).

    Повертає DataFrame з колонками: entry_ts, ret, mae, mfe, efficiency.
    efficiency = ret / mfe (наскільки ефективно захопили рух, 1.0 = ідеально).
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    if "high" not in df_bars.columns or "low" not in df_bars.columns:
        return pd.DataFrame()

    rows = []
    for _, trade in trades_df.iterrows():
        entry_ts = pd.Timestamp(trade["entry_ts"])
        exit_ts = pd.Timestamp(trade["exit_ts"])
        side = int(trade.get("side", 0))
        entry_px = float(trade.get("entry_price", float("nan")))
        ret = float(trade.get("ret", float("nan")))
        if side == 0 or not np.isfinite(entry_px) or entry_px <= 0:
            continue

        # Бари під час утримання
        mask = (df_bars.index >= entry_ts) & (df_bars.index <= exit_ts)
        window = df_bars[mask]
        if window.empty:
            continue

        if side == 1:  # лонг
            mae = float((window["low"].min() - entry_px) / entry_px)  # від'ємне
            mfe = float((window["high"].max() - entry_px) / entry_px)  # додатнє
        else:  # шорт
            mae = float((entry_px - window["high"].max()) / entry_px)  # від'ємне
            mfe = float((entry_px - window["low"].min()) / entry_px)  # додатнє

        efficiency = ret / mfe if mfe > 0 else float("nan")
        rows.append(
            {
                "entry_ts": entry_ts,
                "side": side,
                "ret": ret,
                "mae": mae,
                "mfe": mfe,
                "efficiency": efficiency,
            }
        )
    return pd.DataFrame(rows)


def hourly_heatmap_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Матриця win_rate: рядки=дні тижня, колонки=години UTC.

    Зручна для побудови heatmap у Plotly/Streamlit.
    Повертає DataFrame (7×24): значення win_rate або NaN якщо немає угод.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    ts = pd.to_datetime(df["entry_ts"])
    df["weekday"] = ts.dt.dayofweek
    df["hour"] = ts.dt.hour

    result = pd.DataFrame(index=range(7), columns=range(24), dtype=float)
    for wd in range(7):
        for h in range(24):
            sub = df[(df["weekday"] == wd) & (df["hour"] == h)]
            if len(sub) >= 3:  # мінімум 3 угоди для статистики
                rets = sub["ret"] if "ret" in sub.columns else pd.Series(dtype=float)
                result.loc[wd, h] = float((rets > 0).mean())

    result.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    result.columns = [f"{h:02d}:00" for h in range(24)]
    return result
