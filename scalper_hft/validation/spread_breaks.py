"""CSW CUSUM на спреді — дешевий детектор структурного зламу (AFML Ch.17)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CusumBreakResult:
    n_breaks: int
    break_times: list[pd.Timestamp]
    statistic_max: float

    def summary(self) -> str:
        return f"CUSUM спред: breaks={self.n_breaks} maxS={self.statistic_max:.2f}"


def csw_cusum(
    spread: pd.Series,
    *,
    threshold: float = 4.6,
    warmup: int = 60,
) -> CusumBreakResult:
    """Homogeneous CUSUM на стандартизованих Δspread. b0.05≈4.6 (AFML)."""
    x = spread.diff().dropna()
    if len(x) < warmup + 10:
        return CusumBreakResult(0, [], 0.0)
    mu = float(x.iloc[:warmup].mean())
    sd = float(x.iloc[:warmup].std(ddof=0)) or 1e-12
    z = (x - mu) / sd
    s_pos = 0.0
    s_neg = 0.0
    breaks: list[pd.Timestamp] = []
    s_max = 0.0
    for t, v in z.items():
        s_pos = max(0.0, s_pos + float(v))
        s_neg = min(0.0, s_neg + float(v))
        s_max = max(s_max, s_pos, -s_neg)
        if s_pos > threshold or s_neg < -threshold:
            breaks.append(pd.Timestamp(str(t)))
            s_pos = 0.0
            s_neg = 0.0
    return CusumBreakResult(len(breaks), breaks, s_max)


def rank_corr(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Kendall τ / Spearman — робастні до хвостів (FSPML Ch.8.2)."""
    aligned = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(aligned) < 10:
        return {"spearman": 0.0, "kendall": 0.0}
    return {
        "spearman": float(aligned["a"].corr(aligned["b"], method="spearman")),
        "kendall": float(aligned["a"].corr(aligned["b"], method="kendall")),
    }
