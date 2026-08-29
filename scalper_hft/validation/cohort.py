"""Cohort analysis: деградація edge за когортами угод.

Джерело: Predictive Marketing (Artun & Levin) — Ch.4, 8, 13.
Аналогія: «клієнтські когорти» → когорти трейдів за періодом входу.

Ідея: групувати угоди за місяцем входу (або когортою режиму) і дивитися
per-cohort Sharpe / PnL на угоду / win rate. Стабільна кількість угод при
падінні PnL на угоду = «silent attrition» — ранній сигнал смерті edge.

Приклад використання:
    from scalper_hft.backtest.engine import run_backtest
    res = run_backtest(df, strategy, ...)
    cohort = cohort_metrics(res.trades)
    print(cohort.to_string())
    print(cohort_decay(cohort))
"""

from __future__ import annotations

import pandas as pd


def cohort_metrics(trades: pd.DataFrame, freq: str = "ME", pnl_col: str = "ret") -> pd.DataFrame:
    """Per-cohort метрики угод.

    Args:
        trades: DataFrame з колонками ['entry_ts', 'exit_ts', 'side', 'ret']
            (формат _extract_trades з backtest/engine.py).
        freq: частота когорт (pandas offset; 'ME' = місяць, 'W' = тиждень).
        pnl_col: колонка PnL на угоду (за замовч. 'ret').

    Returns:
        DataFrame з індексом — початок когорти, колонки:
            n_trades, pnl_per_trade, win_rate, cum_pnl, sharpe.
    """
    if trades is None or trades.empty or pnl_col not in trades.columns:
        return pd.DataFrame(columns=["n_trades", "pnl_per_trade", "win_rate", "cum_pnl", "sharpe"])

    df = trades.copy()
    ts = pd.to_datetime(df["entry_ts"])
    try:
        df["cohort"] = ts.dt.to_period(freq)
    except ValueError:
        # старі pandas: 'ME' → 'M', 'QE' → 'Q', 'YE' → 'Y'
        df["cohort"] = ts.dt.to_period(freq.replace("E", ""))
    rows: dict[str, list] = {"cohort": [], "n_trades": [], "pnl_per_trade": [],
                             "win_rate": [], "cum_pnl": [], "sharpe": []}
    for cohort, g in df.groupby("cohort"):
        pnl = g[pnl_col]
        rows["cohort"].append(str(cohort))
        rows["n_trades"].append(len(pnl))
        rows["pnl_per_trade"].append(float(pnl.mean()) if len(pnl) else 0.0)
        rows["win_rate"].append(float((pnl > 0).mean()) if len(pnl) else 0.0)
        rows["cum_pnl"].append(float(pnl.sum()))
        rows["sharpe"].append(
            float(pnl.mean() / pnl.std(ddof=0)) if len(pnl) > 1 and pnl.std(ddof=0) > 0 else 0.0
        )
    out = pd.DataFrame(rows).set_index("cohort")
    return out


def cohort_decay(
    cohort: pd.DataFrame,
    metric: str = "pnl_per_trade",
    min_cohorts: int = 4,
) -> dict:
    """Тест тренду метрики по когортах (деградація edge?).

    Використовує лінійну регресію метрики на порядковий номер когорти
    (scipy.stats.linregress). Негативний значущий нахил = деградація.

    Returns:
        dict: slope, rvalue, pvalue, n_cohorts, decaying (bool).
    """
    if cohort is None or cohort.empty or metric not in cohort.columns:
        return {"slope": 0.0, "rvalue": 0.0, "pvalue": 1.0, "n_cohorts": 0, "decaying": False}
    if len(cohort) < min_cohorts:
        return {"slope": 0.0, "rvalue": 0.0, "pvalue": 1.0,
                "n_cohorts": len(cohort), "decaying": False, "note": "замало когорт"}
    try:
        from scipy.stats import linregress
    except ImportError:  # pragma: no cover
        return {"slope": 0.0, "rvalue": 0.0, "pvalue": 1.0,
                "n_cohorts": len(cohort), "decaying": False, "note": "потрібен scipy"}
    x = pd.Series(range(len(cohort)), index=cohort.index)
    y = cohort[metric].astype(float)
    res = linregress(x, y)
    return {
        "slope": float(res.slope),
        "rvalue": float(res.rvalue),
        "pvalue": float(res.pvalue),
        "n_cohorts": len(cohort),
        "decaying": bool(res.slope < 0 and res.pvalue < 0.05),
    }


def cohort_report(trades: pd.DataFrame, freq: str = "ME") -> str:
    """Людиночитабельний звіт cohort decay."""
    c = cohort_metrics(trades, freq=freq)
    d = cohort_decay(c)
    lines = [
        "Cohort analysis (по когортах входу):",
        "",
        c.to_string(),
        "",
        f"Decay-тест ({'pnl_per_trade'}): slope={d['slope']:+.5f}, p={d['pvalue']:.3f}, "
        f"когорт={d['n_cohorts']}",
        "→ edge деградує ⚠" if d["decaying"] else "→ деградації не виявлено",
    ]
    return "\n".join(lines)


# ── Експорт ───────────────────────────────────────────────────────────────────
__all__ = ["cohort_metrics", "cohort_decay", "cohort_report"]
