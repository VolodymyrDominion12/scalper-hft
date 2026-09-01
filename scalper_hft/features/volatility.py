"""Прогноз волатильності: GARCH(1,1) та EWMA (FSPML Ch.7–8).

Застосування в скальпінгу:
    - σ̂_{t+1} як ML-фіча;
    - інверсне sizing: менша позиція у високій волатильності;
    - ARCH-LM на залишках як діагностика моделі.

Реалізація без бібліотеки `arch` (недоступна офлайн):
    σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1},  ε_t = r_t
    variance targeting: ω = γ·V, V — довгострокова дисперсія, γ = 1−α−β;
    (α, β) — MLE через scipy.optimize з fallback на класичні дефолти.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def garch11_vol(
    returns: pd.Series | np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    long_var: float | None = None,
) -> pd.Series:
    """Умовна волатильність σ_t за GARCH(1,1) (рекурсія, без lookahead).

    σ²_0 = довгострокова дисперсія (або var(returns)); повертає σ_t.
    """
    r = np.asarray(returns, dtype=float)
    if long_var is None or not np.isfinite(long_var) or long_var <= 0:
        long_var = float(np.var(r)) if len(r) > 1 else 1e-8
    var = np.zeros(len(r))
    var[0] = long_var
    for t in range(1, len(r)):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    out = pd.Series(np.sqrt(var), index=getattr(returns, "index", None))
    return out


def garch11_fit(
    returns: np.ndarray | pd.Series,
    alpha0: float = 0.06,
    beta0: float = 0.92,
) -> tuple[float, float, float]:
    """Оцінка (α, β) GARCH(1,1) через MLE з variance targeting.

    Returns:
        (omega, alpha, beta). При невдачі оптимізації — дефолти α=0.06, β=0.92.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return 0.0, alpha0, beta0
    long_var = float(np.var(r))
    if long_var <= 0:
        return 0.0, alpha0, beta0

    def neg_ll(params: np.ndarray) -> float:
        alpha, beta = params
        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10
        omega = (1.0 - alpha - beta) * long_var
        var = np.empty(len(r))
        var[0] = long_var
        for t in range(1, len(r)):
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        var = np.maximum(var, 1e-12)
        return float(0.5 * np.sum(np.log(var) + r**2 / var))

    try:
        from scipy.optimize import minimize

        res = minimize(
            neg_ll,
            x0=np.array([alpha0, beta0]),
            bounds=[(1e-4, 0.4), (0.5, 0.999)],
            method="L-BFGS-B",
        )
        alpha, beta = float(res.x[0]), float(res.x[1])
    except Exception:  # noqa: BLE001
        alpha, beta = alpha0, beta0
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha + beta >= 1:
        alpha, beta = alpha0, beta0
    omega = max((1.0 - alpha - beta) * long_var, 1e-12)
    return omega, alpha, beta


def garch_forecast(
    returns: pd.Series,
    window: int = 500,
    refit_every: int = 100,
    warmup: int = 50,
) -> pd.Series:
    """Прогноз σ̂_{t+1} на кожному барі (rolling GARCH(1,1) з періодичним рефітом).

    На кожному барі t прогноз використовує лише дані до t (без lookahead):
    параметри рефітяться кожні `refit_every` барів; рекурсія волатильності
    продовжується інкрементально між рефітами.

    Виправлення: прибрано вкладений loop що перегравав рекурсію при кожному рефіті
    (O(window×T/refit_every) → O(T)).
    """
    r = returns.astype(float)
    n = len(r)
    out = np.full(n, np.nan)
    omega, alpha, beta = 0.0, 0.0, 0.0
    v = 1e-8  # поточна дисперсія σ²_{t-1}
    t = 0
    while t < n:
        if t < warmup:
            t += 1
            continue
        if t % refit_every == 0:
            # рефіт параметрів на останньому вікні
            hist = r.iloc[max(0, t - window): t]
            omega, alpha, beta = garch11_fit(hist.values)
            # ініціалізуємо v з довгострокової дисперсії при першому рефіті
            if t == warmup or v <= 0:
                v = float(np.var(hist.values)) if len(hist) > 1 else 1e-8
            # v продовжується інкрементально (не перегравається)
        # σ²_t → прогноз σ²_{t+1}
        var_t = omega + alpha * r.iloc[t - 1] ** 2 + beta * v
        var_t1 = omega + alpha * r.iloc[t] ** 2 + beta * var_t
        out[t] = float(np.sqrt(max(var_t1, 1e-12)))
        v = var_t
        t += 1
    return pd.Series(out, index=r.index)



def ewma_vol(returns: pd.Series, span: int = 20) -> pd.Series:
    """EWMA волатильність (дешева альтернатива GARCH для фіч)."""
    return returns.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)


def arch_lm_test(returns: pd.Series, lags: int = 5) -> tuple[float, float]:
    """ARCH-LM тест на залишках: чи є кластеризація волатильності.

    Returns:
        (lm_stat, p_value): H0 — немає ARCH-ефектів. p < 0.05 → є.
    """
    r2 = np.asarray(returns, dtype=float) ** 2
    r2 = r2[np.isfinite(r2)]
    n = len(r2)
    if n <= lags + 2:
        return 0.0, 1.0
    y = r2[lags:]
    X = np.column_stack([np.ones(n - lags)] + [r2[lags - j - 1 : n - j - 1] for j in range(lags)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = float(np.dot(resid, resid))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2_stat = 1.0 - rss / tss if tss > 0 else 0.0
        lm = float((n - lags) * r2_stat)
        from scipy.stats import chi2

        p = float(chi2.sf(lm, lags))
        return lm, p
    except Exception:  # noqa: BLE001
        return 0.0, 1.0


__all__ = ["garch11_vol", "garch11_fit", "garch_forecast", "ewma_vol", "arch_lm_test"]
