"""Систематичний скан коінтеграції пар перпів (замість ручного списку)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairScanRow:
    leg1: str
    leg2: str
    adf_pvalue: float
    spearman: float
    kendall: float
    half_life: float
    tradable: bool

    def summary(self) -> str:
        flag = "кандидат" if self.tradable else "ні"
        return (
            f"{self.leg1}/{self.leg2}: ADF p={self.adf_pvalue:.3f} "
            f"ρ={self.spearman:+.2f} τ={self.kendall:+.2f} hl={self.half_life:.1f} → {flag}"
        )


def _half_life(spread: pd.Series) -> float:
    x = spread.shift(1).dropna()
    y = spread.diff().dropna()
    common = x.index.intersection(y.index)
    if len(common) < 20:
        return float("inf")
    xx, yy = x.loc[common].to_numpy(dtype=float), y.loc[common].to_numpy(dtype=float)
    if np.std(xx) < 1e-12:
        return float("inf")
    b = float(np.polyfit(xx, yy, 1)[0])
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def _adf_pvalue(spread: pd.Series) -> float:
    try:
        import warnings

        from statsmodels.tsa.stattools import adfuller

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res = adfuller(spread.dropna(), maxlag=1, autolag=None)
            p_val = getattr(res, "pvalue", None)
            return float(p_val if p_val is not None else res[1])
    except Exception:  # noqa: BLE001
        return 1.0


def scan_pairs(
    closes: dict[str, pd.Series],
    *,
    adf_max_p: float = 0.05,
    max_half_life: float = 200.0,
) -> list[PairScanRow]:
    """Усі унікальні пари; tradable якщо ADF значущий і half-life помірний."""
    symbols = sorted(closes)
    rows: list[PairScanRow] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            common = pd.concat({"a": closes[a], "b": closes[b]}, axis=1).dropna()
            if len(common) < 80:
                continue
            spread = np.log(common["a"]) - np.log(common["b"])
            from scalper_hft.validation.spread_breaks import rank_corr

            rc = rank_corr(common["a"].pct_change(), common["b"].pct_change())
            p = _adf_pvalue(spread)
            hl = _half_life(spread)
            rows.append(
                PairScanRow(
                    a,
                    b,
                    p,
                    rc["spearman"],
                    rc["kendall"],
                    hl,
                    tradable=p <= adf_max_p and 1.0 < hl < max_half_life,
                )
            )
    rows.sort(key=lambda r: (not r.tradable, r.adf_pvalue, r.half_life))
    return rows
