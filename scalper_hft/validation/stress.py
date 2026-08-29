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
        worst_end = int(np.nanargmin(roll.values))
        if np.isfinite(roll.iloc[worst_end]):
            idx = np.arange(max(0, worst_end - crash_bars + 1), worst_end + 1)
            out.iloc[idx] = out.iloc[idx] * crash_mult

    # liquidity / funding: додаткові витрати на бар (еквівалент turnover ≈ 1)
    if cost_mult != 1.0 or funding_rate != 0.0:
        extra = (cost_mult - 1.0) * cost_frac + funding_rate
        out = out - extra

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
    rows: dict[str, list] = {"scenario": [], "total_return": [], "sharpe": [],
                             "max_drawdown": [], "worst_period": []}
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


__all__ = ["SCENARIOS", "apply_stress", "stress_test", "stress_report"]
