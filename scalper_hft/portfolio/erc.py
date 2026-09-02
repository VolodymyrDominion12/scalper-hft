"""Портфельна алокація: ERC / risk-parity з turnover tax (Narang гл. 6).

Замість рівних ваг — алокація за внеском у ризик:
    ERC (Equal Risk Contribution): ваги w такі, що всі w_i·(Σw)_i рівні.

Додатково: turnover tax — штраф за зміну ваг при ребалансуванні
(для низькочастотних пар істотно: кожна зміна ваг = комісії на спреді).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Внесок кожного активу в портфельний ризик: rc_i = w_i·(Σw)_i."""
    w = np.asarray(weights, dtype=float)
    total = float(np.sqrt(np.dot(w, np.dot(cov, w)))) if np.dot(w, np.dot(cov, w)) > 0 else 0.0
    if total <= 0:
        return np.zeros_like(w)
    return w * np.dot(cov, w) / total


def erc_weights(
    returns: np.ndarray | pd.DataFrame,
    target_rc: np.ndarray | None = None,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Equal Risk Contribution ваги (Narang гл. 6; Maillard et al. 2010).

    Мінімізує дисперсію часток ризику Σ_i (rc_i/Σrc − target_i)² при Σw = 1,
    0 ≤ w_i ≤ max_weight. target_rc: цільові частки (за замовч. рівні 1/n).
    max_weight: обмеження концентрації (1.0 = без обмеження).
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns має бути 2D (T × N), N ≥ 2")
    stds = r.std(axis=0)
    tol = max(float(stds.max()) * 1e-10, 1e-12)
    dead = stds <= tol
    if dead.all():
        return np.full(r.shape[1], 1.0 / r.shape[1])
    if dead.any():
        # «мертвий» актив (≈нульова дисперсія) не має ризику → нульова вага;
        # решта — ERC на живих активах. Раніше нульова дисперсія одного
        # активу спотворювала коваріацію і могла захопити ERC-портфель.
        live = ~dead
        if live.sum() == 1:
            w = np.zeros(r.shape[1])
            w[live] = 1.0
            return w
        w_live = erc_weights(r[:, live], max_weight=max_weight)
        w = np.zeros(r.shape[1])
        w[live] = w_live
        return w
    cov = np.cov(r, rowvar=False)
    # робастність: floor на дисперсії
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    n = r.shape[1]
    if target_rc is None:
        target_rc = np.full(n, 1.0 / n)

    def obj(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        tot = rc.sum()
        shares = rc / tot if tot > 0 else np.zeros_like(rc)
        return float(np.sum((shares - target_rc) ** 2))

    from scipy.optimize import minimize

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n
    best_w = np.full(n, 1.0 / n)
    best_obj = obj(best_w)
    # кілька стартів (рівні + випадкові) — SLSQP чутливий до початкової точки
    rng = np.random.default_rng(42)
    starts = [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(3)]
    for x0 in starts:
        res = minimize(
            obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12}
        )
        if res.success and res.fun < best_obj:
            best_obj = float(res.fun)
            best_w = np.clip(res.x, 0.0, max_weight)
            best_w = best_w / best_w.sum()
    return best_w


def risk_parity_weights(returns: np.ndarray | pd.DataFrame, max_weight: float = 0.5) -> np.ndarray:
    """Синонім ERC для зручності (risk-parity = ERC на коваріації прибутків)."""
    return erc_weights(returns, max_weight=max_weight)


def allocate_portfolio(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    turnover_rate: float = 0.0,
    rebalance: str | None = "ME",
) -> pd.Series:
    """Портфельна прибутковість з (опційним) turnover tax.

    returns: (T × N) прибутковості стратегій на спільному індексі.
    weights: фіксовані ваги; None → рівні.
    turnover_rate: частка капіталу, що списується за зміну ваг при ребалансі
        (наприклад, 2·taker_fee для двосторонньої зміни).
    rebalance: частота ребалансу (None = жодного, ваги сталі).

    Returns:
        Series портфельної прибутковості (без комісій самих стратегій —
        вони вже у їхніх прибутковостях).
    """
    n = returns.shape[1]
    if weights is None:
        weights = np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if len(w) != n:
        raise ValueError("weights і returns мають мати однакову кількість колонок")

    if turnover_rate <= 0 or rebalance is None:
        return (returns * w).sum(axis=1)

    # ребаланс: ваги сталі між датами ребалансу, turnover = Σ|Δw| у моменти зміни
    idx = returns.index
    try:
        periods = idx.to_period(rebalance)
    except ValueError:
        # старі pandas: 'ME' → 'M', 'QE' → 'Q', 'YE' → 'Y'
        periods = idx.to_period(rebalance.replace("E", ""))
    port = pd.Series(0.0, index=idx)
    prev_w = np.zeros(n)
    for _, group in returns.groupby(periods):
        port.loc[group.index] = (group * w).sum(axis=1)
        turnover = float(np.sum(np.abs(w - prev_w)))
        if turnover > 0:
            port.loc[group.index[0]] -= turnover * turnover_rate
        prev_w = w
    return port


__all__ = ["risk_contributions", "erc_weights", "risk_parity_weights", "allocate_portfolio"]
