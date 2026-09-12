"""Factor Tearsheet & Alpha Decay Engine.

Аналітичний інструмент дослідження альфа-факторів за методологією Alphalens та Microsoft Qlib.
Дозволяє оцінити прогнозуючу здатність сирого сигналу на різних горизонтах прогнозування (horizons)
до запуску важкого бектесту:
- Information Coefficient (Pearson IC & Spearman Rank IC)
- Статистична значущість (t-stat, p-value)
- Квантильний аналіз (Quantile returns, top minus bottom spread)
- Монотонність квантилів (Monotonicity score)
- Експоненційний розпад альфи (Alpha decay half-life)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class FactorHorizonReport:
    """Метрики альфа-фактора для фіксованого горизонту прогнозу h."""

    horizon: int
    count: int
    ic_pearson: float
    ic_spearman: float
    t_stat: float
    p_value: float
    quantile_returns: dict[int, float] = field(default_factory=dict)
    spread_top_minus_bottom: float = 0.0
    monotonicity: float = 0.0
    is_monotonic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "count": self.count,
            "ic_pearson": round(self.ic_pearson, 4),
            "ic_spearman": round(self.ic_spearman, 4),
            "t_stat": round(self.t_stat, 2),
            "p_value": round(self.p_value, 4),
            "quantile_returns": {k: round(v, 6) for k, v in self.quantile_returns.items()},
            "spread_top_minus_bottom": round(self.spread_top_minus_bottom, 6),
            "monotonicity": round(self.monotonicity, 3),
            "is_monotonic": self.is_monotonic,
        }


@dataclass(frozen=True, slots=True)
class FactorTearsheetResult:
    """Підсумковий результат аналізу альфа-фактора на множині горизонтів."""

    factor_name: str
    horizons: tuple[int, ...]
    reports: dict[int, FactorHorizonReport]
    half_life_bars: float | None
    mean_rank_ic: float
    best_horizon: int
    is_significant: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "horizons": list(self.horizons),
            "reports": {h: r.to_dict() for h, r in self.reports.items()},
            "half_life_bars": round(self.half_life_bars, 2) if self.half_life_bars is not None else None,
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "best_horizon": self.best_horizon,
            "is_significant": self.is_significant,
            "summary": self.summary,
        }


def compute_forward_returns(
    prices: pd.Series,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 24),
) -> dict[int, pd.Series]:
    """Обчислення майбутніх дохідностей без lookahead (t+1 до t+h).

    Форвардний ретьорн на закритті бару t вимірюється як рух від close[t] до close[t+h].
    """
    clean_p = prices.dropna().astype(float)
    result: dict[int, pd.Series] = {}
    for h in horizons:
        # P[t+h] / P[t] - 1.0 (зсув уперед на -h для вирівнювання з фічею в момент t)
        fwd = (clean_p.shift(-h) / clean_p) - 1.0
        result[h] = fwd
    return result


def compute_factor_horizon_report(
    factor: pd.Series,
    forward_return: pd.Series,
    horizon: int,
    quantiles: int = 5,
) -> FactorHorizonReport:
    """Розрахунок IC, t-stat, p-value та квантильних ретьорнів для одного горизонту."""
    # Злиття та видалення NaN / Inf
    df = pd.DataFrame({"factor": factor, "fwd": forward_return}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 10:
        return FactorHorizonReport(
            horizon=horizon,
            count=n,
            ic_pearson=0.0,
            ic_spearman=0.0,
            t_stat=0.0,
            p_value=1.0,
        )

    f_vals = df["factor"].to_numpy(dtype=float)
    r_vals = df["fwd"].to_numpy(dtype=float)

    # Pearson IC
    if np.all(f_vals == f_vals[0]) or np.all(r_vals == r_vals[0]):
        pearson_ic = 0.0
    else:
        p_corr, _ = stats.pearsonr(f_vals, r_vals)
        pearson_ic = float(p_corr) if not math.isnan(p_corr) else 0.0

    # Spearman Rank IC
    s_corr, s_pval = stats.spearmanr(f_vals, r_vals)
    spearman_ic = float(s_corr) if not math.isnan(s_corr) else 0.0
    p_value = float(s_pval) if not math.isnan(s_pval) else 1.0

    # T-статистика
    if abs(spearman_ic) >= 1.0:
        t_stat = math.copysign(99.0, spearman_ic)
    else:
        denom = math.sqrt(max(1e-9, 1.0 - spearman_ic**2))
        t_stat = spearman_ic * math.sqrt(max(1, n - 2)) / denom

    # Квантильний аналіз
    q_returns: dict[int, float] = {}
    try:
        # qcut з захистом від дублікатів значень
        df["quantile"] = pd.qcut(df["factor"], q=quantiles, labels=False, duplicates="drop")
        grouped = df.groupby("quantile", observed=True)["fwd"].mean()
        for q_idx, mean_ret in grouped.items():
            q_returns[int(q_idx) + 1] = float(mean_ret)
    except Exception:
        pass

    # Spread top minus bottom
    if len(q_returns) >= 2:
        top_q = max(q_returns.keys())
        bot_q = min(q_returns.keys())
        spread = q_returns[top_q] - q_returns[bot_q]
        # Монотонність (рангова кореляція індексу квантиля з його середнім ретьорном)
        q_idx_arr = np.array(list(q_returns.keys()), dtype=float)
        q_ret_arr = np.array(list(q_returns.values()), dtype=float)
        mono_corr, _ = stats.spearmanr(q_idx_arr, q_ret_arr)
        monotonicity = float(mono_corr) if not math.isnan(mono_corr) else 0.0
        is_mono = bool(monotonicity >= 0.8 or monotonicity <= -0.8)
    else:
        spread = 0.0
        monotonicity = 0.0
        is_mono = False

    return FactorHorizonReport(
        horizon=horizon,
        count=n,
        ic_pearson=pearson_ic,
        ic_spearman=spearman_ic,
        t_stat=float(t_stat),
        p_value=p_value,
        quantile_returns=q_returns,
        spread_top_minus_bottom=spread,
        monotonicity=monotonicity,
        is_monotonic=is_mono,
    )


def estimate_alpha_half_life(reports: dict[int, FactorHorizonReport]) -> float | None:
    """Оцінка періоду напіврозпаду альфи (Half-Life) через експоненційне затухання Rank IC.

    Модель: |IC(h)| = IC_0 * exp(-lambda * h) => ln(|IC(h)|) = a - lambda * h
    Half-life: h_{1/2} = ln(2) / lambda (у барах).
    """
    valid_points = [
        (h, abs(r.ic_spearman)) for h, r in sorted(reports.items()) if abs(r.ic_spearman) > 0.005 and r.p_value <= 0.15
    ]
    if len(valid_points) < 3:
        return None

    hs = np.array([p[0] for p in valid_points], dtype=float)
    ics = np.array([p[1] for p in valid_points], dtype=float)

    try:
        log_ics = np.log(ics)
        slope, _, _, _, _ = stats.linregress(hs, log_ics)
        decay_rate = -float(slope)
        if decay_rate > 1e-4:
            half_life = math.log(2.0) / decay_rate
            return float(half_life)
        return None
    except Exception:
        return None


def run_factor_tearsheet(
    factor: pd.Series,
    prices: pd.Series | None = None,
    forward_returns: dict[int, pd.Series] | None = None,
    factor_name: str = "signal",
    horizons: tuple[int, ...] = (1, 2, 4, 8, 24),
    quantiles: int = 5,
) -> FactorTearsheetResult:
    """Головна точка входу для побудови повного tearsheet-звіту альфа-фактора."""
    clean_factor = factor.dropna()
    if clean_factor.empty:
        raise ValueError("Factor series is empty or contains only NaNs")

    if forward_returns is None:
        if prices is None:
            raise ValueError("Either prices or forward_returns must be provided")
        forward_returns = compute_forward_returns(prices, horizons=horizons)

    reports: dict[int, FactorHorizonReport] = {}
    rank_ics: list[float] = []

    for h in horizons:
        fwd = forward_returns.get(h)
        if fwd is None:
            continue
        rep = compute_factor_horizon_report(clean_factor, fwd, horizon=h, quantiles=quantiles)
        reports[h] = rep
        rank_ics.append(rep.ic_spearman)

    if not reports:
        raise ValueError("No valid forward returns computed for the requested horizons")

    mean_rank_ic = float(np.mean(rank_ics)) if rank_ics else 0.0
    half_life = estimate_alpha_half_life(reports)

    # Найкращий горизонт (максимальний за модулем |Rank IC| із p-value < 0.05)
    sig_reports = [r for r in reports.values() if r.p_value <= 0.05]
    if sig_reports:
        best_rep = max(sig_reports, key=lambda r: abs(r.ic_spearman))
        best_horizon = best_rep.horizon
        is_significant = True
    else:
        best_rep = max(reports.values(), key=lambda r: abs(r.ic_spearman))
        best_horizon = best_rep.horizon
        is_significant = False

    # Формування підсумкового рядка
    hl_str = f"{half_life:.1f} bars" if half_life is not None else "N/A"
    summary = (
        f"Factor '{factor_name}': mean Rank IC={mean_rank_ic:+.3f}, "
        f"best horizon={best_horizon}h (IC={best_rep.ic_spearman:+.3f}, t={best_rep.t_stat:.1f}, p={best_rep.p_value:.3f}), "
        f"half-life={hl_str}, significant={'YES' if is_significant else 'NO'}"
    )

    return FactorTearsheetResult(
        factor_name=factor_name,
        horizons=horizons,
        reports=reports,
        half_life_bars=half_life,
        mean_rank_ic=mean_rank_ic,
        best_horizon=best_horizon,
        is_significant=is_significant,
        summary=summary,
    )


def format_tearsheet_table(result: FactorTearsheetResult) -> str:
    """Форматування результатів у вигляді текстової таблиці для терміналу або звітів."""
    lines = [
        f"=== Factor Tearsheet: {result.factor_name} ===",
        f"{'Horizon':<8} {'Count':<8} {'Pearson IC':<12} {'Rank IC':<10} {'t-stat':<8} {'p-value':<9} {'Monotonic':<10} {'Top-Bottom Spread'}",
        "-" * 82,
    ]
    for h in result.horizons:
        r = result.reports.get(h)
        if r is None:
            continue
        mono_str = "YES" if r.is_monotonic else f"NO ({r.monotonicity:+.2f})"
        lines.append(
            f"{r.horizon:<2}h      {r.count:<8} {r.ic_pearson:+10.4f}  {r.ic_spearman:+8.4f}  {r.t_stat:+6.2f}  {r.p_value:<9.4f} {mono_str:<10} {r.spread_top_minus_bottom:+.4%}"
        )
    lines.append("-" * 82)
    lines.append(result.summary)
    return "\n".join(lines)
