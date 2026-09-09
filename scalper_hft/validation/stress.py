"""Стрес-тестування стратегій (Narang гл. 4, 10).

Замість одного історичного шляху — переграємо прибутковості під
сценаріями, які історія могла не показати:

    crash       — краш: найгірше історичне вікно посилюється у K разів
                   (ампліфікація хвоста);
    liquidity   — криза ліквідності: витрати ×10 (спред розширюється,
                   глибина тане) — штраф на кожен бар;
    vol_spike   — стрибок волатильності: прибутковості масштабуються
                   волатильністю (хвости жирнішають);
    funding_shock — раптовий перекіс фандінгу: додаткова вартість
                   утримання позиції.

Підхід чесний: ми НЕ підганяємо сценарії під дані — беремо найгірші
реалізовані епізоди і посилюємо їх (worst-case amplification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS: dict[str, dict] = {
    "crash": {"crash_mult": 2.0, "crash_bars": 24, "cost_mult": 1.0, "vol_mult": 1.0, "funding_rate": 0.0},
    "liquidity": {"crash_mult": 1.0, "crash_bars": 24, "cost_mult": 10.0, "vol_mult": 1.0, "funding_rate": 0.0},
    "vol_spike": {"crash_mult": 1.0, "crash_bars": 24, "cost_mult": 3.0, "vol_mult": 2.0, "funding_rate": 0.0},
    "funding_shock": {"crash_mult": 1.0, "crash_bars": 24, "cost_mult": 1.0, "vol_mult": 1.0, "funding_rate": 0.001},
}


def apply_stress(
    returns: pd.Series,
    crash_mult: float = 2.0,
    crash_bars: int = 24,
    cost_mult: float = 1.0,
    vol_mult: float = 1.0,
    funding_rate: float = 0.0,
    cost_frac: float = 0.0005,
) -> pd.Series:
    """Перетворює прибутковості під стрес-сценарій.

    Args:
        returns: барні прибутковості стратегії (вже з комісіями).
        crash_mult: у скільки разів посилити найгірше вікно `crash_bars`.
        crash_bars: довжина посилюваного вікна.
        cost_mult: множник додаткових витрат на бар (ліквідність-криза).
        vol_mult: множник волатильності (масштаб прибутковостей хвостів).
        funding_rate: додаткова per-bar вартість утримання (фандінг-шок).
        cost_frac: базова частка витрат на бар для cost_mult/funding_rate.

    Returns:
        Series стресованих прибутковостей (той самий індекс).
    """
    out = returns.copy().astype(float)

    # vol_spike: жирніші хвости — масштабуємо відхилення від середнього
    if vol_mult != 1.0:
        center = out.mean()
        out = center + (out - center) * vol_mult

    # crash: найгірше ковзне вікно посилюється
    # rolling-сума у позиції t покриває [t−crash_bars+1, t] → посилюємо цей інтервал
    if crash_mult > 1.0 and len(out) > crash_bars:
        roll = out.rolling(crash_bars).sum()
        worst_end = int(np.nanargmin(roll.to_numpy(dtype=float)))
        if np.isfinite(roll.iloc[worst_end]):
            idx = np.arange(max(0, worst_end - crash_bars + 1), worst_end + 1)
            out.iloc[idx] = out.iloc[idx] * crash_mult

    # liquidity / funding: додаткові витрати лише на АКТИВНИХ барах
    # (бар з нульовою позицією має strat_ret = 0 — заряджати його нечесно)
    if cost_mult != 1.0 or funding_rate != 0.0:
        extra = (cost_mult - 1.0) * cost_frac + funding_rate
        active = out != 0
        out = out.where(~active, out - extra)

    return out


def stress_test(returns: pd.Series, scenarios: list[str] | None = None) -> dict[str, pd.Series]:
    """Прогін усіх (або обраних) сценаріїв.

    Returns:
        dict: scenario_name → стресовані прибутковості (базовий ряд включено
        під ключем 'baseline').
    """
    out: dict[str, pd.Series] = {"baseline": returns.copy()}
    names = scenarios or list(SCENARIOS.keys())
    for name in names:
        if name not in SCENARIOS:
            raise KeyError(f"Невідомий сценарій '{name}'. Доступні: {list(SCENARIOS)}")
        out[name] = apply_stress(returns, **SCENARIOS[name])
    return out


def stress_report(
    returns: pd.Series,
    scenarios: list[str] | None = None,
    per_period: int = 1,
) -> pd.DataFrame:
    """Зведений звіт: базові метрики vs кожен сценарій.

    Returns:
        DataFrame з колонками total_return, sharpe, max_drawdown,
        worst_day для baseline + сценаріїв.
    """
    from scalper_hft.backtest.metrics import compute_metrics

    stressed = stress_test(returns, scenarios)
    rows: dict[str, list] = {"scenario": [], "total_return": [], "sharpe": [], "max_drawdown": [], "worst_period": []}
    for name, r in stressed.items():
        if not isinstance(r.index, pd.DatetimeIndex):
            r = r.copy()
            r.index = pd.date_range("2025-01-01", periods=len(r), freq="1min")
        equity = (1.0 + r).cumprod()
        m = compute_metrics(equity)
        rows["scenario"].append(name)
        rows["total_return"].append(m.total_return)
        rows["sharpe"].append(m.sharpe)
        rows["max_drawdown"].append(m.max_drawdown)
        rows["worst_period"].append(float(r.min()) if len(r) else 0.0)
    return pd.DataFrame(rows).set_index("scenario")


def drop_top_trade_returns(trades: pd.DataFrame, n: int = 5) -> pd.Series:
    """Прибутковості угод без N найкращих (концентрація edge)."""
    if trades is None or trades.empty or "ret" not in trades.columns:
        return pd.Series(dtype=float)
    r = trades["ret"].astype(float)
    n_drop = min(int(n), len(r))
    if n_drop <= 0:
        return r
    return r.drop(r.nlargest(n_drop).index)


def cost_concentration_stress(
    df: pd.DataFrame,
    strategy,
    cost,
    *,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
    n_drop: int = 5,
    baseline=None,
) -> pd.DataFrame:
    """Стрес роадмапу 5.3: комісії×2, slippage×2, вилучення топ-N угод.

    Повертає таблицю scenario × (sharpe, total_return, max_drawdown, n_trades).
    """
    from dataclasses import replace

    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.metrics import compute_metrics

    base = baseline or run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=position_pct)
    cost2 = replace(
        cost,
        maker_fee=float(cost.maker_fee) * 2,
        taker_fee=float(cost.taker_fee) * 2,
        slippage_frac=float(cost.slippage_frac) * 2,
    )
    fees2 = run_backtest(df, strategy, cost=cost2, trades=trades, funding=funding, position_pct=position_pct)

    dropped = drop_top_trade_returns(base.trades, n=n_drop)
    if dropped.empty:
        drop_metrics = base.metrics
    else:
        eq = (1.0 + dropped).cumprod()
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq.index = pd.date_range("2025-01-01", periods=len(eq), freq="1h")
        drop_metrics = compute_metrics(eq, trade_return=dropped)

    rows = [
        {
            "scenario": "baseline",
            "sharpe": float(base.metrics.sharpe),
            "total_return": float(base.metrics.total_return),
            "max_drawdown": float(base.metrics.max_drawdown),
            "n_trades": int(base.metrics.n_trades),
        },
        {
            "scenario": "fees_slippage_x2",
            "sharpe": float(fees2.metrics.sharpe),
            "total_return": float(fees2.metrics.total_return),
            "max_drawdown": float(fees2.metrics.max_drawdown),
            "n_trades": int(fees2.metrics.n_trades),
        },
        {
            "scenario": f"drop_top_{n_drop}_trades",
            "sharpe": float(drop_metrics.sharpe),
            "total_return": float(drop_metrics.total_return),
            "max_drawdown": float(drop_metrics.max_drawdown),
            "n_trades": int(getattr(drop_metrics, "n_trades", max(0, len(dropped)))),
        },
    ]
    return pd.DataFrame(rows).set_index("scenario")


__all__ = [
    "SCENARIOS",
    "apply_stress",
    "stress_test",
    "stress_report",
    "drop_top_trade_returns",
    "cost_concentration_stress",
]
