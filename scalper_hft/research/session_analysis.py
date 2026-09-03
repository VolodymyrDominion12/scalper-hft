"""Session & Regime Analytics — розбивка метрик по часу доби, дню тижня та режиму.

Мета: виявити, в якій «сесії» стратегія найкраще/найгірше торгує,
і як ведуть себе угоди в різних ринкових режимах.

Використання:
    from scalper_hft.research.session_analysis import (
        session_breakdown, weekday_breakdown, regime_breakdown, mae_mfe_analysis
    )
    trades_df = res.trades
    session = session_breakdown(trades_df)
    weekday = weekday_breakdown(trades_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def regime_breakdown(trades_df: pd.DataFrame, regime_series: pd.Series | None = None) -> pd.DataFrame:
    """Метрики угод по режимах волатильності (low/normal/high).

    regime_series: Series з індексом datetime і значеннями "low"/"normal"/"high"
    (результат `volatility_regime()`). Якщо None — повертає порожній DataFrame.
    """
    if trades_df is None or trades_df.empty or regime_series is None:
        return pd.DataFrame()

    df = trades_df.copy()
    # Беремо режим на барі входу
    entry_ts = pd.to_datetime(df["entry_ts"])
    df["regime"] = entry_ts.map(lambda ts: regime_series.asof(ts) if ts in regime_series.index or True else "unknown")

    rows = []
    for regime in ["low", "normal", "high"]:
        sub = df[df["regime"] == regime]
        rets = sub["ret"] if "ret" in sub.columns and not sub.empty else pd.Series(dtype=float)
        rows.append(
            {
                "regime": regime,
                "n_trades": len(sub),
                "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
                "avg_pnl": float(rets.mean()) if len(rets) else 0.0,
                "total_pnl": float(rets.sum()) if len(rets) else 0.0,
                "profit_factor": (
                    float(rets[rets > 0].sum() / (-rets[rets < 0].sum()))
                    if len(rets[rets < 0]) > 0 and -rets[rets < 0].sum() > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


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
