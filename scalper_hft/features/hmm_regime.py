"""Режими ринку через приховану марковську модель (HMM).

Джерело: Financial Signal Processing and ML (Akansu et al.), Ch.4.5
(MS-TCM / Markov-switching); hmmlearn недоступний офлайн — реалізація
класичного Gaussian HMM на numpy (Baum–Welch EM + Viterbi).

Модель: латентний стан s_t ∈ {1..K} — Markov chain, емісії x_t ~ N(μ_s, Σ_s)
(діагональні коваріації). EM:
    E-крок: forward–backward у log-просторі → γ_t(i), ξ_t(i,j);
    M-крок: оновлення π, A, μ_i, Σ_i.

Застосування: фічі P(режим) для ML-класифікатора та gate для стратегій
(mean-reversion ↔ momentum ↔ висока волатильність), замість/поверх
rule-based `features/regimes.py`.

⚠ Label switching: імена станів не стабільні між прогонами — сортуємо стани
за середнім значенням першої фічі (для [ret, |ret|, vol] це монотонний порядок).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Лог-густина багатовимірного нормального (діагональна коваріація).

    x: (T, D); mu: (D,); var: (D,) — дисперсії (> 0).
    Returns: (T,)
    """
    d = x.shape[1]
    diff = x - mu
    quad = np.sum(diff * diff / var, axis=1)
    logdet = np.sum(np.log(var))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def _scaled_forward(B: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Масштабований forward (Rabiner 1989). Returns (alpha, c), log_lik = Σ ln c."""
    t, k = B.shape
    alpha = np.zeros((t, k))
    c = np.zeros(t)
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum()
    if c[0] > 0:
        alpha[0] /= c[0]
    for t_ in range(1, t):
        alpha[t_] = (alpha[t_ - 1] @ A) * B[t_]
        c[t_] = alpha[t_].sum()
        if c[t_] > 0:
            alpha[t_] /= c[t_]
    return alpha, c


def _scaled_backward(B: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Масштабований backward."""
    t, k = B.shape
    beta = np.zeros((t, k))
    beta[-1] = 1.0
    for t_ in range(t - 2, -1, -1):
        beta[t_] = (A @ (B[t_ + 1] * beta[t_ + 1])) / c[t_ + 1] if c[t_ + 1] > 0 else (A @ (B[t_ + 1] * beta[t_ + 1]))
    return beta


def _viterbi(log_B: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Найімовірніша послідовність станів."""
    t, k = log_B.shape
    delta = np.zeros((t, k))
    psi = np.zeros((t, k), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t_ in range(1, t):
        for j in range(k):
            tmp = delta[t_ - 1] + log_A[:, j]
            psi[t_, j] = int(np.argmax(tmp))
            delta[t_, j] = tmp[psi[t_, j]] + log_B[t_, j]
    states = np.zeros(t, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t_ in range(t - 2, -1, -1):
        states[t_] = psi[t_ + 1, states[t_ + 1]]
    return states


class GaussianHMM:
    """Gaussian HMM з діагональними коваріаціями (Baum–Welch)."""

    def __init__(self, n_states: int = 3, n_iter: int = 60, tol: float = 1e-4, seed: int | None = 42) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.states_: np.ndarray | None = None
        self.posteriors_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> GaussianHMM:
        """Навчання EM. X: (T, D)."""
        X = np.asarray(X, dtype=float)
        t, d = X.shape
        k = self.n_states
        if t < k * 10 or d < 1:
            raise ValueError(f"Замало даних для HMM: {t} точок, {k} станів")

        rng = np.random.default_rng(self.seed)
        # ініціалізація: центри — квантилі кожної фічі вздовж станів + шум
        q = np.linspace(0.05, 0.95, k)
        means = np.array([np.quantile(X[:, j], q) + rng.normal(0, 0.01 * X[:, j].std(), k) for j in range(d)]).T
        var_all = np.var(X, axis=0) + 1e-12
        covars = np.tile(var_all, (k, 1))
        # діагонально-домінантна transition-матриця: інакше EM зливає стани
        transmat = np.full((k, k), 0.1 / (k - 1)) if k > 1 else np.ones((1, 1))
        np.fill_diagonal(transmat, 0.9)
        startprob = np.full(k, 1.0 / k)
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            B = np.column_stack([np.exp(_log_gaussian(X, means[i], covars[i])) for i in range(k)])
            B = np.maximum(B, 1e-300)  # захист від underflow

            # ── масштабований forward–backward (Rabiner 1989) ──
            alpha = np.zeros((t, k))
            c = np.zeros(t)
            alpha[0] = startprob * B[0]
            c[0] = alpha[0].sum()
            if c[0] <= 0:
                break
            alpha[0] /= c[0]
            for t_ in range(1, t):
                alpha[t_] = (alpha[t_ - 1] @ transmat) * B[t_]
                c[t_] = alpha[t_].sum()
                if c[t_] <= 0:
                    break
                alpha[t_] /= c[t_]
            log_lik = float(np.sum(np.log(np.maximum(c, 1e-300))))
            if not np.isfinite(log_lik):
                break
            if abs(log_lik - prev_ll) < self.tol:
                prev_ll = log_lik
                break
            prev_ll = log_lik

            beta = np.zeros((t, k))
            beta[-1] = 1.0
            for t_ in range(t - 2, -1, -1):
                beta[t_] = (transmat @ (B[t_ + 1] * beta[t_ + 1])) / c[t_ + 1]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((t - 1, k, k))
            for t_ in range(t - 1):
                xi[t_] = alpha[t_][:, None] * transmat * B[t_ + 1][None, :] * beta[t_ + 1][None, :]
                s = xi[t_].sum()
                if s > 0:
                    xi[t_] /= s
            xi_sum = xi.sum(axis=0)

            # M-крок
            startprob = gamma[0] / gamma[0].sum()
            transmat = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            denom = gamma.sum(axis=0)
            means = (gamma.T @ X) / denom[:, None]
            for i in range(k):
                diff = X - means[i]
                covars[i] = (gamma[:, i][:, None] * diff**2).sum(axis=0) / denom[i]
            covars = np.maximum(covars, 1e-10)

        # фінальний прогін (масштабований forward–backward)
        B = np.column_stack([np.exp(_log_gaussian(X, means[i], covars[i])) for i in range(k)])
        B = np.maximum(B, 1e-300)
        alpha, c = _scaled_forward(B, startprob, transmat)
        beta = _scaled_backward(B, transmat, c)
        posteriors = alpha * beta
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        states = _viterbi(np.log(B + 1e-300), np.log(transmat + 1e-12), np.log(startprob + 1e-12))

        # стабільність імен станів: сортуємо за середнім першої фічі
        order = np.argsort(means[:, 0])
        inv = np.argsort(order)
        self.means_ = means[order]
        self.covars_ = covars[order]
        self.transmat_ = transmat[order][:, order]
        self.startprob_ = startprob[order]
        self.posteriors_ = posteriors[:, inv]
        self.states_ = inv[states]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("Модель не навчена — викличте fit()")
        X = np.asarray(X, dtype=float)
        log_B = np.column_stack([_log_gaussian(X, self.means_[i], self.covars_[i]) for i in range(self.n_states)])
        return _viterbi(log_B, np.log(self.transmat_ + 1e-12), np.log(self.startprob_ + 1e-12))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("Модель не навчена — викличте fit()")
        X = np.asarray(X, dtype=float)
        B = np.column_stack([np.exp(_log_gaussian(X, self.means_[i], self.covars_[i])) for i in range(self.n_states)])
        B = np.maximum(B, 1e-300)
        alpha, c = _scaled_forward(B, self.startprob_, self.transmat_)
        beta = _scaled_backward(B, self.transmat_, c)
        post = alpha * beta
        post /= post.sum(axis=1, keepdims=True)
        return post

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """Фільтрована ймовірність стану P(s_t | x_1..x_t) — БЕЗ lookahead.

        Використовує лише forward-прохід (масштабований alpha), тож значення
        у момент t не залежить від майбутніх даних — на відміну від
        predict_proba (згладжена, forward+backward).
        """
        if self.means_ is None:
            raise RuntimeError("Модель не навчена — викличте fit()")
        X = np.asarray(X, dtype=float)
        B = np.column_stack([np.exp(_log_gaussian(X, self.means_[i], self.covars_[i])) for i in range(self.n_states)])
        B = np.maximum(B, 1e-300)
        alpha, _ = _scaled_forward(B, self.startprob_, self.transmat_)
        alpha /= alpha.sum(axis=1, keepdims=True)
        return alpha


def hmm_regime_features(
    close: pd.Series,
    n_states: int = 3,
    extra: pd.DataFrame | None = None,
    window: int | None = None,
    n_iter: int = 60,
    seed: int | None = 42,
    causal: bool = False,
    fit_window: int | None = None,
) -> pd.DataFrame:
    """Режимні HMM-фічі для ряду close.

    Спостереження: [ret, |ret|, realized_vol] + (опційно) колонки extra.
    Returns DataFrame з колонками:
        hmm_state     — стан (0..K-1, відсортований за середнім ret);
        hmm_p0..hmm_pK-1 — P(режим = k) на кожному барі.

    causal=True: модель навчається лише на ПЕРШИХ `fit_window` барах, далі
    використовуються фільтровані ймовірності (forward-only) — БЕЗ lookahead
    (для ML-фіч у walk-forward). За замовчуванням — згладжені (smoothed)
    ймовірності (для аналізу/гейтів поза ML).
    """
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
    obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol})
    if extra is not None:
        obs = obs.join(extra.reindex(obs.index).fillna(0.0))
    obs = obs.iloc[20:]  # прогрів
    X = obs.values

    if causal:
        # навчання лише на першому відрізку (фіксована модель, без рефітів)
        fw = fit_window or min(len(X), 2000)
        model = GaussianHMM(n_states=n_states, n_iter=n_iter, seed=seed).fit(X[:fw])
        post = model.filtered_proba(X)  # фільтрована (каузальна) ймовірність
        states = np.argmax(post, axis=1)
    else:
        if window is not None and len(X) > window:
            X = X[-window:]
        model = GaussianHMM(n_states=n_states, n_iter=n_iter, seed=seed).fit(X)
        states = model.states_
        post = model.posteriors_

    out = pd.DataFrame(index=close.index, dtype=float)
    out.loc[obs.index[-len(states) :], "hmm_state"] = states.astype(float)
    for k in range(n_states):
        out.loc[obs.index[-len(states) :], f"hmm_p{k}"] = post[:, k]
    out["hmm_state"] = out["hmm_state"].ffill().fillna(0.0)
    out = out.fillna(0.0)
    return out


__all__ = ["GaussianHMM", "hmm_regime_features"]
