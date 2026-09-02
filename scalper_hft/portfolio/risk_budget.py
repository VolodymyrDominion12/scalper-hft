"""Портфельний risk budget: vol-targeting, VaR, розподіл ліміту збитків.

Narang гл. 4/10: ризик на рівні ПОРТФЕЛЯ, а не лише окремих стратегій.

    vol_target_scale   — масштаб позицій, щоб портфельна волатильність
                         вийшла на цільовий рівень (vol targeting);
    portfolio_var      — Value-at-Risk портфеля (історичний);
    loss_budget_split  — розподіл денного ліміту збитків між стратегіями
                         пропорційно їхньому внеску в ризик.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.05,
    horizon: int = 1,
) -> float:
    """Історичний VaR портфеля (частка капіталу, додатне число).

    returns: (T × N) прибутковості стратегій; weights: ваги.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 2 or len(w) != r.shape[1]:
        raise ValueError("returns (T×N) і weights (N,) мають збігатися")
    port = r @ w
    if horizon > 1:
        port = pd.Series(port).rolling(horizon).sum().dropna().values
    return float(-np.quantile(port, alpha)) if len(port) else 0.0


def vol_target_scale(
    returns: pd.DataFrame,
    weights: np.ndarray,
    target_vol: float,
    ann_factor: float = 1.0,
) -> float:
    """Масштаб s, щоб s·Σ w_i·r_i мав річну волатильність ≈ target_vol.

    s = target_vol / (σ_port · √ann_factor). Якщо σ_port ≈ 0 — повертає 1.0.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) != r.shape[1]:
        raise ValueError("weights і returns мають збігатися")
    port = r @ w
    vol = float(np.std(port, ddof=0))
    if vol <= 0 or not np.isfinite(vol):
        return 1.0
    return float(target_vol / (vol * np.sqrt(max(ann_factor, 1e-12))))


def loss_budget_split(
    returns: pd.DataFrame,
    weights: np.ndarray,
    daily_loss_limit: float,
) -> dict[str, float]:
    """Розподіл денного ліміту збитків між стратегіями.

    Кожна стратегія отримує ліміт пропорційно своєму внеску в портфельну
    волатильність (risk contribution) — так жодна нога не «з'їдає» весь ліміт.

    Returns:
        dict: назва стратегії → ліміт (частка капіталу).
    """
    from scalper_hft.portfolio.erc import risk_contributions

    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 2 or len(w) != r.shape[1]:
        raise ValueError("returns (T×N) і weights (N,) мають збігатися")
    cov = np.cov(r, rowvar=False) + np.eye(r.shape[1]) * 1e-12
    rc = risk_contributions(w, cov)
    tot = rc.sum()
    shares = rc / tot if tot > 0 else np.full(len(w), 1.0 / len(w))
    cols = list(getattr(returns, "columns", range(len(w))))
    return {str(cols[i]): float(daily_loss_limit * shares[i]) for i in range(len(w))}


def estimate_tail_dependence(
    returns: pd.DataFrame | np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Оцінка коефіцієнта нижньої хвостової залежності λ_L (FSPML Ch. 8.4).

    λ_L = lim_{q->0} P(U_1 <= q | U_2 <= q) ≈ (1 / (q * T)) * Σ I(U_{1,t} <= q, U_{2,t} <= q)
    Вимірює схильність активів до синхронного обвалу під час крипто-крахів.

    Args:
        returns: DataFrame (T x N) прибутковостей активів/стратегій.
        alpha: квантиль нижнього хвоста (за замовчуванням 5%).

    Returns:
        DataFrame (N x N) попарної хвостової залежності у [0.0, 1.0].
    """
    if isinstance(returns, pd.DataFrame):
        cols = list(returns.columns)
        r = returns.dropna().values
    else:
        r = np.asarray(returns, dtype=float)
        cols = [f"asset_{i}" for i in range(r.shape[1])]

    t, n = r.shape
    if t < 20 or n < 2:
        return pd.DataFrame(np.eye(n), index=cols, columns=cols)

    # Перетворення у емпіричні рівномірні маргінали U_i у (0, 1)
    from scipy.stats import rankdata

    u = np.zeros((t, n), dtype=float)
    for j in range(n):
        u[:, j] = rankdata(r[:, j]) / (t + 1.0)

    lambda_l = np.eye(n, dtype=float)
    q = min(max(alpha, 0.01), 0.5)

    for i in range(n):
        for j in range(i + 1, n):
            joint_co_crash = np.sum((u[:, i] <= q) & (u[:, j] <= q))
            coef = float(joint_co_crash / (q * t))
            coef = min(max(coef, 0.0), 1.0)
            lambda_l[i, j] = coef
            lambda_l[j, i] = coef

    return pd.DataFrame(lambda_l, index=cols, columns=cols)


__all__ = ["portfolio_var", "vol_target_scale", "loss_budget_split", "estimate_tail_dependence"]
