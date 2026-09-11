"""Exp3 Multi-Armed Bandit для адаптивного онлайн-вибору стратегій/режимів.

Джерело: Eyal Gofer, "Machine Learning Algorithms with Applications in Finance", Ch. 4 §4.7.
Алгоритм Exponential-weight for Exploration and Exploitation (Exp3):
    1. Ймовірності вибору: p_{i,t} = (1 - γ) * (w_{i,t} / Σ w_{j,t}) + γ / K
    2. Вибір дії: I_t ~ p_t
    3. Оцінка винагороди: r̂_{i,t} = r_{i,t} / p_{i,t} для i = I_t (інакше 0)
    4. Оновлення ваг: w_{i,t+1} = w_{i,t} * exp(γ * r̂_{i,t} / K)

Гарантує обмеження регрету O(√(K T ln K)) без припущення про стаціонарність ринку.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Exp3Bandit:
    """Multi-Armed Bandit на базі Exp3."""

    # Дефолтний seed для бектест-шляхів. Аудит 2026-09-11 показав, що `seed=None`
    # у `RegimeSupervisor(blend_mode="exp3")` робив комірку НЕВІДТВОРЮВАНОЮ: два
    # прогони того самого скрипта дали `meta:sup_exp3` Sharpe −0.583 (5714 угод) і
    # −0.499 (4490 угод). Артефакт, який змінюється між прогонами, не є доказом.
    DEFAULT_SEED = 42

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.05,
        arm_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("Потрібно щонайменше 2 руки (arms)")
        self.n_arms = n_arms
        self.gamma = gamma
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]
        self._weights = np.ones(n_arms, dtype=float)
        self.history: list[dict[str, float]] = []
        # Seeded RNG для відтворюваності (1D): однаковий seed → однаковий вибір рук.
        # `seed=None` лишається явним opt-in у стохастичність (напр. для ансамблів
        # різних шляхів); дефолт для продакшн-шляху задає `DEFAULT_SEED` через
        # `RegimeSupervisor._blend_exp3`.
        self._rng = np.random.default_rng(seed)

    def probabilities(self) -> np.ndarray:
        """Поточний розподіл ймовірностей вибору рук."""
        total_w = np.sum(self._weights)
        if total_w <= 0 or not np.all(np.isfinite(self._weights)):
            self._weights = np.ones(self.n_arms, dtype=float)
            total_w = float(self.n_arms)

        p = (1.0 - self.gamma) * (self._weights / total_w) + (self.gamma / self.n_arms)
        return p / np.sum(p)

    def select_arm(self, rng: np.random.Generator | None = None) -> int:
        """Вибір руки згідно з ймовірнісним розподілом Exp3."""
        p = self.probabilities()
        if rng is not None:
            return int(rng.choice(self.n_arms, p=p))
        return int(self._rng.choice(self.n_arms, p=p))

    def update(self, arm: int, reward: float, turnover_cost: float = 0.0) -> None:
        """Оновлення ваг після отримання винагороди.

        reward: скаляр у діапазоні [-1.0, 1.0] (напр. барний PnL після комісій).
        turnover_cost (2C): вартість зміни позиції (|Δsignal| × cost_per_unit).
            Віднімається від reward — net-PnL-свідоме навчання. 0 = як раніше.
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Невалідний індекс руки: {arm}")

        p = self.probabilities()
        p_arm = max(p[arm], 1e-6)

        # Net reward: брутто PnL мінус turnover-вартість (2C).
        net_reward = float(reward) - float(turnover_cost)
        # Зсув винагороди у [0, 1] для гарантії додатності
        shifted_reward = (np.clip(net_reward, -1.0, 1.0) + 1.0) / 2.0
        est_reward = shifted_reward / p_arm

        # Оновлення ваги
        growth = np.exp((self.gamma * est_reward) / self.n_arms)
        self._weights[arm] *= growth

        # Чисельна нормалізація для запобігання overflow
        max_w = np.max(self._weights)
        if max_w > 1e10:
            self._weights /= max_w

        self.history.append(
            {
                "arm": float(arm),
                "reward": float(reward),
                "net_reward": float(net_reward),
                "turnover_cost": float(turnover_cost),
                "p_arm": float(p_arm),
            }
        )

    def weights_series(self) -> pd.Series:
        """Поточні нормовані ваги стратегій."""
        p = self.probabilities()
        return pd.Series(p, index=self.arm_names)


def exp3_select_signals(
    signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    gamma: float = 0.05,
    seed: int | None = None,
    turnover_penalty: float = 0.0,
) -> pd.Series:
    """Векторизований бектест вибору сигналів через Exp3 Bandit (без lookahead).

    На кожному барі t обирає сигнал однієї зі стратегій, отримує прибуток на барі t+1
    та оновлює розподіл ваг.

    Args:
        signals_df: (T x N) матриця сигналів суб-стратегій у [-1, 1].
        returns_df: (T x N) матриця фактичних повернень суб-стратегій.
        gamma: exploration rate.
        seed: seed для відтворюваного вибору рук (1D). None = nondeterministic.
        turnover_penalty (2C): вартість зміни позиції при перемиканні рук.
            Віднімається від reward у update() — net-PnL-свідоме навчання.
            Штраф = turnover_penalty × |Δsignal| при зміні руки. 0 = як раніше.

    Returns:
        Series обраного сигналу в [-1, 1], індексована як signals_df.
    """
    t, n = signals_df.shape
    bandit = Exp3Bandit(n_arms=n, gamma=gamma, arm_names=list(signals_df.columns), seed=seed)

    chosen_signals = []
    sig_vals = signals_df.values
    ret_vals = returns_df.values

    last_arm: int | None = None
    last_signal: float = 0.0
    for i in range(t):
        arm = bandit.select_arm()
        chosen_signals.append(sig_vals[i, arm])

        # Винагорода за ПОПЕРЕДНІЙ обраний крок. select_arm() НЕ пише у history
        # (туди пише лише update()), тому history[-1] не відповідає обраній руці —
        # раніше це давало fixed point: update() завжди цілив у руку 0, і бандит
        # не навчався. Тримаємо обрану руку явно.
        if last_arm is not None:
            prev_ret = ret_vals[i - 1, int(last_arm)]
            # Turnover-штраф (2C): вартість зміни позиції при перемиканні руки.
            turnover_cost = 0.0
            if turnover_penalty > 0:
                turnover_cost = turnover_penalty * abs(float(sig_vals[i, arm]) - last_signal)
            bandit.update(int(last_arm), prev_ret, turnover_cost=turnover_cost)
        last_arm = arm
        last_signal = float(sig_vals[i, arm])

    return pd.Series(chosen_signals, index=signals_df.index, dtype=float).clip(-1.0, 1.0)
