"""Спільні розширені тести аудиту (quintile, time-decay, stress, benchmark).

Використовуються і в `cmd_report`, і в `audit_cell` для паритету протоколу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy
from scalper_hft.validation.benchmark import buy_and_hold_sharpe

QUINTILE_SPEARMAN_MIN = 0.7


def has_pair_legs(df: pd.DataFrame) -> bool:
    return {"leg1", "leg2"}.issubset(df.columns)


def pairs_spread(df: pd.DataFrame) -> pd.Series:
    """log(leg1/leg2) — спред, на якому pairs_arb рахує PnL (−pos·Δspread)."""
    ratio = np.log(df["leg1"].to_numpy(dtype=float) / df["leg2"].to_numpy(dtype=float))
    return pd.Series(ratio, index=df.index, dtype=float)


def pairs_z_and_forward(
    df: pd.DataFrame,
    lookback: int = 240,
) -> tuple[pd.Series, pd.Series]:
    """z спреду на закритті t і forward = −Δspread бару t+1 (без lookahead)."""
    ratio = pairs_spread(df)
    window = max(int(lookback), 2)
    min_p = max(window // 2, 2)
    mean = ratio.rolling(window, min_periods=min_p).mean()
    std = ratio.rolling(window, min_periods=min_p).std(ddof=0).replace(0, np.nan)
    z = (ratio - mean) / std
    fwd = -ratio.diff().shift(-1)
    return z, fwd


def _pairs_lookback(strategy: Strategy) -> int:
    raw = strategy.get("lookback", 240)
    try:
        return max(int(raw), 2)
    except (TypeError, ValueError):
        return 240


@dataclass(frozen=True, slots=True)
class QuintileAudit:
    spearman: float
    monotonic: bool
    # None = тест не застосовний (дискретний сигнал / мало даних): це НЕ FAIL,
    # інакше будь-яка directional-стратегія автоматично отримувала "quintile_fail".
    pass_: bool | None
    error: str | None = None
    summary: str = ""
    on_spread: bool = False


@dataclass(frozen=True, slots=True)
class TimeDecayAudit:
    sharpes: tuple[float, ...]
    pass_: bool
    warn: bool
    error: str | None = None
    summary: str = ""
    on_spread: bool = False


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
    from scalper_hft.validation.quintile import discrete_signal_study, quintile_spread_study

    try:
        if has_pair_legs(df):
            z, fwd_ret = pairs_z_and_forward(df, lookback=_pairs_lookback(strategy))
            on_spread = True
            q_res = quintile_spread_study(z.astype(float), fwd_ret)
            ok = q_res.monotonic and abs(q_res.spearman) >= QUINTILE_SPEARMAN_MIN
            return QuintileAudit(
                float(q_res.spearman),
                bool(q_res.monotonic),
                bool(ok),
                summary=q_res.summary(),
                on_spread=on_spread,
            )
        if getattr(strategy, "name", "") == "pairs_arb":
            return QuintileAudit(0.0, False, False, error="потрібні колонки leg1/leg2 (−Δspread)")
        # Directional-стратегії: сигнал дискретний ({-1,0,+1}), тому квінтилі
        # вироджуються (2–3 біни → ρ=±1 незалежно від наявності інформації).
        # Замість цього — монотонність середнього forward-return за знаком
        # сигналу + Welch t на спреді країв. N/A (pass_=None) ставимо ЛИШЕ коли
        # тест структурно незастосовний (замало кошиків); помилки даних і далі
        # валять комірку (pass_=False).
        sig = strategy.generate_signals(df, trades=trades, funding=funding).astype(float)
        if sig.abs().sum() <= 5:
            return QuintileAudit(0.0, False, False, error="недостатньо сигналів")
        fwd_ret = df["close"].pct_change().shift(-1).fillna(0.0)
        try:
            d_res = discrete_signal_study(sig, fwd_ret)
        except ValueError as exc:
            return QuintileAudit(0.0, False, None, error=f"тест не застосовний: {exc}")
        return QuintileAudit(
            float(d_res.spearman),
            bool(d_res.monotonic),
            bool(d_res.pass_),
            summary=d_res.summary(),
            on_spread=False,
        )
    except Exception as exc:  # noqa: BLE001
        return QuintileAudit(0.0, False, False, error=str(exc)[:120])


def _time_decay_verdict(sharpes: tuple[float, ...], *, summary: str, on_spread: bool) -> TimeDecayAudit:
    if len(sharpes) < 2:
        return TimeDecayAudit(sharpes, False, False, error="мало лагів", summary=summary, on_spread=on_spread)
    lag0, lag1 = sharpes[0], sharpes[1]
    if lag0 > 0 and lag1 < 0:
        return TimeDecayAudit(sharpes, False, True, summary=summary, on_spread=on_spread)
    if lag0 > 0 and lag1 < lag0 * 0.5:
        return TimeDecayAudit(sharpes, False, True, summary=summary, on_spread=on_spread)
    return TimeDecayAudit(sharpes, True, False, summary=summary, on_spread=on_spread)


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
    from scalper_hft.validation.time_decay import pairs_time_decay, time_decay_test

    try:
        if has_pair_legs(df):
            signals = strategy.generate_signals(df, trades=trades, funding=funding)
            td = pairs_time_decay(signals, pairs_spread(df), max_lag=max_lag)
            sharpes = tuple(float(x) for x in td.sharpes)
            return _time_decay_verdict(sharpes, summary=td.summary(), on_spread=True)
        if getattr(strategy, "name", "") == "pairs_arb":
            return TimeDecayAudit((), False, False, error="потрібні колонки leg1/leg2 (−Δspread)")
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
        return _time_decay_verdict(sharpes, summary=td.summary(), on_spread=False)
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


def pair_frame_from_klines(leg1: pd.DataFrame, leg2: pd.DataFrame) -> pd.DataFrame:
    """Спільний індекс close → колонки leg1/leg2 для quintile/decay пар."""
    return (
        leg1[["close"]]
        .rename(columns={"close": "leg1"})
        .join(leg2[["close"]].rename(columns={"close": "leg2"}), how="inner")
        .dropna()
    )


def format_pairs_signal_quality(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    max_lag: int = 3,
) -> str:
    """Текстовий блок CLI: quintile + pairs_time_decay на −Δspread."""
    q = run_quintile_audit(df, strategy)
    td = run_time_decay_audit(df, strategy, cost=CostModel(), max_lag=max_lag)
    lines = ["", "Quintile / time-decay на −Δspread (не close ноги):"]
    if q.error:
        lines.append(f"  Quintile: помилка: {q.error}")
    else:
        flag = "PASS" if q.pass_ else "FAIL"
        lines.append(f"  Quintile: {flag} | ρ={q.spearman:+.3f} | monotonic={q.monotonic}")
        if q.summary:
            lines.append(q.summary)
    if td.error:
        lines.append(f"  Time-decay: помилка: {td.error}")
    else:
        flag = "PASS" if td.pass_ else "WARN/FAIL"
        lines.append(f"  Time-decay: {flag} | {td.summary}")
    return "\n".join(lines)
