"""Hedge / Exponentially Weighted Aggregation — онлайн-блендінг експертів.

Джерело: Eyal Gofer, "Machine Learning Algorithms with Applications in
Finance" (PhD, Tel Aviv University, 2014), Ch.2 §2.1 — базовий no-regret
алгоритм best-expert:

    w_{i,t+1} = w_{i,t} · e^{−η·l_{i,t}}
    p_{i,t}   = w_{i,t} / Σ_j w_{j,t}

Regret O(√(T·ln N)) проти найкращого експерта; з адаптивним
η = √((8/q′)·ln N), де q′ — відносна квадратична варіація loss-векторів.

Застосування в scalper-hft: заміна статичного mean/vote в EnsembleStrategy
динамічним зважуванням суб-стратегій за їхньою історією прибутковості —
адаптація до концепт-дрейфу без перетренування.

Векторизована версія для бектесту: w̃_t = e^{−η·cumloss}, p_t = w̃_t / Σ w̃_t
(нормалізація на кожному кроці скасовується в кумулятивній формі).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_weights(
    returns: np.ndarray | pd.DataFrame,
    eta: float | None = None,
    loss_clip: float = 0.99,
) -> pd.DataFrame:
    """Ваги експертів у часі за алгоритмом Hedge (без lookahead).

    Args:
        returns: (T × N) матриця прибутковостей експертів за бар (з комісіями
            або без — на розсуд користувача). Рядки — час, колонки — експерти.
        eta: параметр швидкості навчання. None → адаптивний
            η = √((8/q′)·ln N), q′ = відносна квадратична варіація loss.
        loss_clip: обрізання loss = −ln(1+r) (захист від r ≤ −1).

    Returns:
        DataFrame (T × N) ваг p_{i,t} (рядки сумуються до 1).
        Вага у момент t використовує лише прибутковості до t (включно з t−1) —
        тобто p_t коректна для рішення на барі t.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns має бути 2D (T × N)")
    t, n = r.shape
    if n < 2:
        raise ValueError("Потрібно щонайменше 2 експерти")

    # loss = −ln(1 + r) з обрізанням
    loss = -np.log(np.clip(1.0 + r, 1.0 - loss_clip, None))

    if eta is None:
        # адаптивне η з відносної квадратичної варіації loss
        q = float(np.sum((loss - loss.mean(axis=1, keepdims=True)) ** 2, axis=1).sum())
        q = max(q, 1e-12)
        eta = float(np.sqrt((8.0 / q) * np.log(n)))

    # кумулятивний loss → не-нормалізовані ваги → нормовані p_t
    cumloss = np.cumsum(loss, axis=0)
    # стабільність: віднімаємо row-max перед exp (нормалізація row-wise)
    w_tilde = np.exp(-eta * (cumloss - cumloss.max(axis=1, keepdims=True)))
    p = w_tilde / w_tilde.sum(axis=1, keepdims=True)
    # перший рядок: рівні ваги (loss ще не спостережено)
    p = np.vstack([np.full((1, n), 1.0 / n), p[:-1]])

    index = returns.index if isinstance(returns, pd.DataFrame) else None
    columns = returns.columns if isinstance(returns, pd.DataFrame) else None
    return pd.DataFrame(p, index=index, columns=columns)


class HedgeBlend:
    """Інкрементальний Hedge для live/paper: крок за кроком."""

    def __init__(self, n_experts: int, eta: float | None = None) -> None:
        self.n = n_experts
        self.eta = eta
        self._w = np.ones(n_experts)
        self._q = 0.0  # накопичена квадратична варіація loss

    def weights(self) -> np.ndarray:
        return self._w / self._w.sum()

    def blend(self, signals: np.ndarray) -> float:
        """Зважена сума сигналів експертів поточними вагами."""
        return float(np.dot(self.weights(), np.asarray(signals, dtype=float)))

    def step(self, returns: np.ndarray) -> None:
        """Оновити ваги після спостереження прибутковостей експертів."""
        r = np.asarray(returns, dtype=float)
        loss = -np.log(np.clip(1.0 + r, 1e-3, 10.0))
        if self.eta is None:
            self._q += float(np.sum((loss - loss.mean()) ** 2))
            self.eta = float(np.sqrt((8.0 / max(self._q, 1e-12)) * np.log(self.n)))
        self._w = self._w * np.exp(-self.eta * loss)
        if not np.all(np.isfinite(self._w)) or self._w.sum() <= 0:
            self._w = np.ones(self.n)  # аварійний скид до рівних ваг
        self._w = self._w / self._w.sum()


def hedge_blend_signals(
    signal_matrix: pd.DataFrame,
    close: pd.Series,
    eta: float | None = None,
) -> pd.Series:
    """Комбінований сигнал з Hedge-ваг для бектесту.

    Для кожної суб-стратегії рахується барна прибутковість
    r_{i,t} = sig_{i,t−1}·ret_t (без комісій — їх стягує рушій на сумарному
    turnover), потім Hedge-ваги і зважена сума сигналів.

    Args:
        signal_matrix: DataFrame (T × N) сигналів суб-стратегій у [-1, 1].
        close: Series цін закриття, індексована як signal_matrix.
        eta: параметр Hedge (None = адаптивний).

    Returns:
        Series комбінованого сигналу у [-1, 1] (кліпнуто).
    """
    ret = close.pct_change().fillna(0.0)
    r = signal_matrix.shift(1).fillna(0.0).mul(ret, axis=0)
    w = hedge_weights(r, eta=eta)
    combined = (signal_matrix * w).sum(axis=1)
    return combined.clip(-1.0, 1.0)
