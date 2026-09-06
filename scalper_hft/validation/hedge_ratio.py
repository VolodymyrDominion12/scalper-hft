"""Hedge-ratio research: 1:1 log-ratio vs rolling OLS / Johansen (OOS only).

Не змінює live, доки OOS Sharpe OLS/Johansen не перевищить log-ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HedgeCompareResult:
    logratio_oos_sharpe: float
    ols_oos_sharpe: float
    johansen_oos_sharpe: float | None
    prefer: str
    ols_beta: float

    def summary(self) -> str:
        j = f"{self.johansen_oos_sharpe:+.3f}" if self.johansen_oos_sharpe is not None else "n/a"
        return (
            f"Hedge OOS: log-ratio {self.logratio_oos_sharpe:+.3f} | "
            f"OLS β={self.ols_beta:.3f} {self.ols_oos_sharpe:+.3f} | "
            f"Johansen {j} | prefer={self.prefer}"
        )


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """β у y = α + β x на ковзному вікні (без lookahead: β_t з даних до t)."""
    yv, xv = y.astype(float), x.astype(float)
    out = pd.Series(np.nan, index=y.index)
    for i in range(window, len(y) + 1):
        yy = yv.iloc[i - window : i].to_numpy(dtype=float)
        xx = xv.iloc[i - window : i].to_numpy(dtype=float)
        if np.std(xx) < 1e-12:
            continue
        x1 = np.column_stack([np.ones(len(xx)), xx])
        coef, *_ = np.linalg.lstsq(x1, yy, rcond=None)
        out.iloc[i - 1] = float(coef[1])
    return out.ffill()


def johansen_beta(y: pd.Series, x: pd.Series) -> float | None:
    """Перший коінтеграційний вектор (нормалізований на y): hedge ≈ |v_x/v_y|."""
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        return None
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < 40:
        return None
    try:
        res = coint_johansen(data.values, det_order=0, k_ar_diff=1)
        vec = res.evec[:, 0]
        if abs(vec[0]) < 1e-12:
            return None
        return float(abs(vec[1] / vec[0]))
    except Exception:  # noqa: BLE001
        return None


def _spread_z(spread: pd.Series, lookback: int) -> pd.Series:
    mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
    sd = spread.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
    return ((spread - mu) / sd).fillna(0.0)


def _signal_from_z(z: pd.Series, entry_z: float = 2.0, exit_z: float = 0.3) -> pd.Series:
    sig = pd.Series(np.nan, index=z.index)
    sig[z > entry_z] = 1.0
    sig[z < -entry_z] = -1.0
    prev = sig.ffill().shift(1).fillna(0.0)
    sig[(prev != 0.0) & (z.abs() < exit_z)] = 0.0
    return sig.ffill().fillna(0.0)


def _oos_sharpe(signals: pd.Series, spread: pd.Series, oos: slice) -> float:
    pos = signals.astype(float).shift(1).fillna(0.0)
    ret = (-pos * spread.diff().fillna(0.0)).iloc[oos]
    if ret.std(ddof=0) == 0 or len(ret) < 5:
        return 0.0
    return float(ret.mean() / ret.std(ddof=0) * np.sqrt(len(ret)))


def compare_hedge_oos(
    leg1: pd.Series,
    leg2: pd.Series,
    *,
    lookback: int = 240,
    train_frac: float = 0.7,
    ols_window: int | None = None,
) -> HedgeCompareResult:
    """IS/OOS порівняння 1:1 log-ratio vs OLS (і Johansen, якщо доступний)."""
    common = pd.concat({"l1": leg1, "l2": leg2}, axis=1).dropna()
    if len(common) < lookback + 50:
        raise ValueError("Замало барів для hedge-порівняння")
    split = int(len(common) * train_frac)
    oos = slice(split, len(common))
    win = ols_window or lookback
    l1_log = pd.Series(np.log(common["l1"]), index=common.index)
    l2_log = pd.Series(np.log(common["l2"]), index=common.index)
    log_s = pd.Series(np.log(common["l1"] / common["l2"]), index=common.index)
    beta = rolling_ols_beta(l1_log, l2_log, win)
    ols_s = l1_log - beta * l2_log

    sr_log = _oos_sharpe(_signal_from_z(_spread_z(log_s, lookback)), log_s, oos)
    sr_ols = _oos_sharpe(_signal_from_z(_spread_z(ols_s, lookback)), ols_s, oos)

    j_beta = johansen_beta(l1_log.iloc[:split], l2_log.iloc[:split])
    sr_j = None
    if j_beta is not None:
        j_s = l1_log - j_beta * l2_log
        sr_j = _oos_sharpe(_signal_from_z(_spread_z(j_s, lookback)), j_s, oos)

    prefer = "logratio"
    if sr_ols > sr_log + 1e-9 and (sr_j is None or sr_ols >= sr_j):
        prefer = "ols"
    elif sr_j is not None and sr_j > sr_log + 1e-9 and sr_j > sr_ols:
        prefer = "johansen"

    return HedgeCompareResult(
        logratio_oos_sharpe=sr_log,
        ols_oos_sharpe=sr_ols,
        johansen_oos_sharpe=sr_j,
        prefer=prefer,
        ols_beta=float(beta.dropna().iloc[-1]) if beta.notna().any() else 1.0,
    )
