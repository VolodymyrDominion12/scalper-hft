"""Спільні розширені тести аудиту (quintile, time-decay, stress, benchmark).

Використовуються і в `cmd_report`, і в `audit_cell` для паритету протоколу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy
from scalper_hft.validation.benchmark import buy_and_hold_sharpe

QUINTILE_SPEARMAN_MIN = 0.7


@dataclass(frozen=True, slots=True)
class QuintileAudit:
    spearman: float
    monotonic: bool
    pass_: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TimeDecayAudit:
    sharpes: tuple[float, ...]
    pass_: bool
    warn: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StressAudit:
    pass_: bool
    crash_dd: float
    liquidity_dd: float
    baseline_dd: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExtendedAudit:
    quintile: QuintileAudit | None = None
    time_decay: TimeDecayAudit | None = None
    stress: StressAudit | None = None
    benchmark_sharpe: float | None = None
    holdout_sharpe: float | None = None


def run_quintile_audit(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> QuintileAudit:
    from scalper_hft.validation.quintile import quintile_spread_study

    try:
        signals = strategy.generate_signals(df, trades=trades, funding=funding)
        fwd_ret = df["close"].pct_change().shift(-1).fillna(0.0)
        if signals.abs().sum() <= 5:
            return QuintileAudit(0.0, False, False, error="недостатньо сигналів")
        q_res = quintile_spread_study(signals.astype(float), fwd_ret)
        ok = q_res.monotonic and abs(q_res.spearman) >= QUINTILE_SPEARMAN_MIN
        return QuintileAudit(float(q_res.spearman), bool(q_res.monotonic), ok)
    except Exception as exc:  # noqa: BLE001
        return QuintileAudit(0.0, False, False, error=str(exc)[:120])


def run_time_decay_audit(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    cost: CostModel,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
    max_lag: int = 3,
) -> TimeDecayAudit:
    from scalper_hft.validation.time_decay import time_decay_test

    try:
        td = time_decay_test(
            df,
            strategy,
            max_lag=max_lag,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=position_pct,
        )
        sharpes = tuple(float(x) for x in td.sharpes)
        if len(sharpes) < 2:
            return TimeDecayAudit(sharpes, False, False, error="мало лагів")
        lag0, lag1 = sharpes[0], sharpes[1]
        if lag0 > 0 and lag1 < 0:
            return TimeDecayAudit(sharpes, False, True)
        if lag0 > 0 and lag1 < lag0 * 0.5:
            return TimeDecayAudit(sharpes, False, True)
        return TimeDecayAudit(sharpes, True, False)
    except Exception as exc:  # noqa: BLE001
        return TimeDecayAudit((), False, False, error=str(exc)[:120])


def run_stress_audit(
    bar_returns: pd.Series,
    baseline_max_dd: float,
) -> StressAudit:
    from scalper_hft.validation.stress import stress_report

    try:
        if bar_returns is None or len(bar_returns) < 10:
            return StressAudit(False, 0.0, 0.0, abs(baseline_max_dd), error="мало барів")
        tbl = stress_report(bar_returns)
        baseline_dd = abs(float(baseline_max_dd))
        crash_dd = abs(float(tbl.loc["crash", "max_drawdown"])) if "crash" in tbl.index else 0.0
        liq_dd = abs(float(tbl.loc["liquidity", "max_drawdown"])) if "liquidity" in tbl.index else 0.0
        ok = True
        if baseline_dd > 0 and crash_dd > baseline_dd * 2.5:
            ok = False
        if baseline_dd > 0 and liq_dd > baseline_dd * 4.0:
            ok = False
        return StressAudit(ok, crash_dd, liq_dd, baseline_dd)
    except Exception as exc:  # noqa: BLE001
        return StressAudit(False, 0.0, 0.0, abs(baseline_max_dd), error=str(exc)[:120])


def run_extended_audit(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    cost: CostModel,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
    bar_returns: pd.Series | None = None,
    baseline_max_dd: float = 0.0,
    holdout_df: pd.DataFrame | None = None,
    strategy_params: dict[str, Any] | None = None,
) -> ExtendedAudit:
    """Повний набір розширених тестів для однієї комірки."""
    quintile = run_quintile_audit(df, strategy, trades=trades, funding=funding)
    time_decay = run_time_decay_audit(
        df, strategy, cost=cost, trades=trades, funding=funding, position_pct=position_pct
    )
    benchmark = buy_and_hold_sharpe(df)
    stress = run_stress_audit(bar_returns, baseline_max_dd) if bar_returns is not None else None
    holdout_sharpe: float | None = None
    if holdout_df is not None and not holdout_df.empty:
        from scalper_hft.validation.optimize import evaluate_holdout

        ho = evaluate_holdout(
            holdout_df,
            type(strategy),
            strategy_params or {},
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=position_pct,
        )
        holdout_sharpe = float(ho) if ho == ho else None
    return ExtendedAudit(
        quintile=quintile,
        time_decay=time_decay,
        stress=stress,
        benchmark_sharpe=benchmark,
        holdout_sharpe=holdout_sharpe,
    )
