"""Аналіз роботи стратегій по ринкових режимах.

Функції:
    compare_strategies_by_regime  — порівняльна таблиця Sharpe/PF по режимах.
    regime_performance_attribution — attribution: скільки PnL дав кожен режим.
    supervisor_vs_baseline         — зведена таблиця: supervisor vs одиночні стратегії.

Всі функції працюють з результатами бектесту (DataFrame trades/returns).
Без lookahead: режим визначається за закритими барами і не залежить від trades.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scalper_hft.features.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Допоміжні функції
# ────────────────────────────────────────────────────────────────────────────


def _sharpe(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Annualized Sharpe Ratio (без rf)."""
    r = returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def _profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def _win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((r > 0).sum() / len(r)) if len(r) > 0 else 0.0


def _max_drawdown(cum_returns: pd.Series) -> float:
    roll_max = cum_returns.cummax()
    dd = (cum_returns - roll_max) / (roll_max + 1e-12)
    return float(dd.min())


# ────────────────────────────────────────────────────────────────────────────
# Основні функції аналізу
# ────────────────────────────────────────────────────────────────────────────


def compare_strategies_by_regime(
    close: pd.Series,
    strategy_returns: dict[str, pd.Series],
    *,
    n_hmm_states: int = 3,
    hmm_fit_bars: int = 2000,
    periods_per_year: int = 8760,
) -> pd.DataFrame:
    """Порівняльна таблиця метрик по режимах для кожної стратегії.

    Args:
        close:             Series цін закриття.
        strategy_returns:  dict {назва_стратегії: pd.Series барних прибутковостей}.
        n_hmm_states:      кількість HMM станів.
        hmm_fit_bars:      бари для навчання HMM.
        periods_per_year:  для annualization (8760 для годинних барів).

    Returns:
        DataFrame з MultiIndex (strategy, regime) × (sharpe, profit_factor,
        win_rate, n_bars, pct_time).
    """
    detector = RegimeDetector(n_hmm_states=n_hmm_states, hmm_fit_bars=hmm_fit_bars)
    regime_df = detector.detect(close)

    rows: list[dict] = []
    for strat_name, ret in strategy_returns.items():
        ret_aligned = ret.reindex(regime_df.index).fillna(0.0)

        for regime_label in sorted(regime_df["label"].unique()):
            mask = regime_df["label"] == regime_label
            r_regime = ret_aligned[mask]

            rows.append(
                {
                    "strategy": strat_name,
                    "regime": regime_label,
                    "sharpe": _sharpe(r_regime, periods_per_year),
                    "profit_factor": _profit_factor(r_regime),
                    "win_rate": _win_rate(r_regime),
                    "n_bars": int(mask.sum()),
                    "pct_time": float(mask.sum() / len(mask) * 100),
                    "total_return_pct": float(r_regime.sum() * 100),
                }
            )

    df = pd.DataFrame(rows).set_index(["strategy", "regime"])
    return df


def regime_performance_attribution(
    close: pd.Series,
    strategy_returns: dict[str, pd.Series],
    *,
    n_hmm_states: int = 3,
    hmm_fit_bars: int = 2000,
) -> pd.DataFrame:
    """Attribution: скільки PnL (%) принесло кожен режим для кожної стратегії.

    Returns:
        DataFrame (strategy × regime) з колонками contribution_pct та pct_time.
    """
    detector = RegimeDetector(n_hmm_states=n_hmm_states, hmm_fit_bars=hmm_fit_bars)
    regime_df = detector.detect(close)

    rows: list[dict] = []
    for strat_name, ret in strategy_returns.items():
        ret_aligned = ret.reindex(regime_df.index).fillna(0.0)
        total = ret_aligned.sum()

        for regime_label in sorted(regime_df["label"].unique()):
            mask = regime_df["label"] == regime_label
            regime_total = ret_aligned[mask].sum()
            rows.append(
                {
                    "strategy": strat_name,
                    "regime": regime_label,
                    "contribution_pct": float(regime_total / total * 100) if total != 0 else 0.0,
                    "pct_time": float(mask.sum() / len(mask) * 100),
                    "cumulative_return": float(regime_total * 100),
                }
            )

    return pd.DataFrame(rows).set_index(["strategy", "regime"])


def supervisor_vs_baseline(
    close: pd.Series,
    baseline_returns: dict[str, pd.Series],
    supervisor_returns: pd.Series,
    *,
    periods_per_year: int = 8760,
) -> pd.DataFrame:
    """Зведена таблиця: RegimeSupervisor vs кожна базова стратегія.

    Args:
        close:               Series цін закриття.
        baseline_returns:    dict {назва: returns} для базових стратегій.
        supervisor_returns:  Series прибутковостей supervisor-а.
        periods_per_year:    для annualization.

    Returns:
        DataFrame (strategy,) × (sharpe, profit_factor, win_rate, max_dd, total_return_pct).
    """
    all_returns = dict(baseline_returns)
    all_returns["regime_supervisor"] = supervisor_returns

    rows: list[dict] = []
    for name, ret in all_returns.items():
        r = ret.dropna()
        cum = (1 + r).cumprod()
        rows.append(
            {
                "strategy": name,
                "sharpe": _sharpe(r, periods_per_year),
                "profit_factor": _profit_factor(r),
                "win_rate": _win_rate(r),
                "max_drawdown_pct": _max_drawdown(cum) * 100,
                "total_return_pct": float((cum.iloc[-1] - 1) * 100) if len(cum) > 0 else 0.0,
                "n_bars": len(r),
                "is_supervisor": name == "regime_supervisor",
            }
        )

    df = pd.DataFrame(rows).set_index("strategy")
    # Сортуємо за Sharpe (supervisor першим якщо він кращий)
    return df.sort_values("sharpe", ascending=False)


def regime_transition_matrix(
    close: pd.Series,
    *,
    n_hmm_states: int = 3,
    hmm_fit_bars: int = 2000,
) -> pd.DataFrame:
    """Матриця переходів між режимами (нормована по рядках).

    Показує: якщо зараз range|low — з якою ймовірністю наступний бар буде
    trend_up|normal тощо. Корисно для розуміння персистентності режимів.
    """
    detector = RegimeDetector(n_hmm_states=n_hmm_states, hmm_fit_bars=hmm_fit_bars)
    regime_df = detector.detect(close)
    labels = regime_df["label"]
    transitions = pd.crosstab(
        labels.iloc[:-1].values,
        labels.iloc[1:].values,
        normalize="index",
    )
    transitions.index.name = "from_regime"
    transitions.columns.name = "to_regime"
    return transitions


__all__ = [
    "compare_strategies_by_regime",
    "regime_performance_attribution",
    "supervisor_vs_baseline",
    "regime_transition_matrix",
]
