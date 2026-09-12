"""Тести Exp3 seeded RNG + --trials floor (1D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.strategies.bandit import Exp3Bandit, exp3_select_signals


def test_exp3_seeded_rng_reproducible() -> None:
    """Однаковий seed → однаковий вибір рук (відтворюваність)."""
    b1 = Exp3Bandit(n_arms=3, gamma=0.05, seed=42)
    b2 = Exp3Bandit(n_arms=3, gamma=0.05, seed=42)
    arms1 = [b1.select_arm() for _ in range(50)]
    arms2 = [b2.select_arm() for _ in range(50)]
    assert arms1 == arms2, "seeded RNG має бути відтворюваним"


def test_exp3_different_seed_different_path() -> None:
    """Різний seed → (з високою ймовірністю) різний шлях."""
    b1 = Exp3Bandit(n_arms=4, gamma=0.1, seed=1)
    b2 = Exp3Bandit(n_arms=4, gamma=0.1, seed=2)
    arms1 = [b1.select_arm() for _ in range(200)]
    arms2 = [b2.select_arm() for _ in range(200)]
    assert arms1 != arms2


def test_exp3_select_signals_seeded_reproducible() -> None:
    """exp3_select_signals з seed — відтворюваний бектест."""
    idx = pd.date_range("2025-01-01", periods=40, freq="1h")
    sig = pd.DataFrame(
        {
            "a": np.random.default_rng(0).choice([-1, 0, 1], size=40),
            "b": np.random.default_rng(1).choice([-1, 0, 1], size=40),
        },
        index=idx,
    )
    ret = pd.DataFrame(
        {"a": np.random.default_rng(2).normal(0, 0.001, 40), "b": np.random.default_rng(3).normal(0, 0.001, 40)},
        index=idx,
    )
    s1 = exp3_select_signals(sig, ret, gamma=0.05, seed=7)
    s2 = exp3_select_signals(sig, ret, gamma=0.05, seed=7)
    pd.testing.assert_series_equal(s1, s2)


def test_audit_cell_n_trials_floor() -> None:
    """n_trials_floor задає мінімум числа спроб для DSR (не занижує корекцію)."""
    # Викликаємо з n_trials_floor > combos — має взяти max.
    # audit_cell потребує дані; використовуємо мок через monkeypatch у повному тесті.
    # Тут лише перевіряємо, що параметр приймається (сигнатура).
    import inspect

    from scalper_hft.validation.cell_audit import audit_cell

    sig = inspect.signature(audit_cell)
    assert "n_trials_floor" in sig.parameters


def test_regime_supervisor_exp3_is_reproducible_by_default() -> None:
    """Дефолт Exp3 у supervisor має бути детермінованим.

    Аудит 2026-09-11: `meta:sup_exp3` давав Sharpe −0.583 (5714 угод) і −0.499
    (4490 угод) у двох прогонах ОДНОГО й того самого коду — бо `exp3_seed`
    за замовчуванням був None. Артефакт, що змінюється між прогонами, не є доказом.
    """
    import pandas as pd
    from scalper_hft.strategies import get_strategy

    n = 1200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 0.4, n)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 10.0}, index=idx
    )

    def signals(**kw):
        return get_strategy(
            "regime_supervisor",
            strategies="supertrend,stoch_rsi,mean_reversion",
            n_hmm_states=3,
            hmm_fit_bars=400,
            blend_mode="exp3",
            **kw,
        ).generate_signals(df)

    assert (signals() == signals()).all(), "дефолтний Exp3 має бути відтворюваним"
    assert (signals(exp3_seed=7) == signals(exp3_seed=7)).all()


def test_exp3_explicit_none_keeps_stochastic_opt_in() -> None:
    """Явний `seed=None` лишається opt-in у стохастичність (як було раніше)."""
    import pandas as pd
    from scalper_hft.strategies.bandit import Exp3Bandit, exp3_select_signals

    assert Exp3Bandit.DEFAULT_SEED is not None

    rng = np.random.default_rng(0)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    cols = ["a", "b", "c"]
    sig = pd.DataFrame({c: rng.integers(-1, 2, n).astype(float) for c in cols}, index=idx)
    ret = pd.DataFrame(rng.normal(0, 0.001, (n, 3)), index=idx, columns=cols)

    s1 = exp3_select_signals(sig, ret, gamma=0.05, seed=None)
    s2 = exp3_select_signals(sig, ret, gamma=0.05, seed=None)
    assert not (s1 == s2).all(), "seed=None має лишатись стохастичним"

    s3 = exp3_select_signals(sig, ret, gamma=0.05, seed=Exp3Bandit.DEFAULT_SEED)
    s4 = exp3_select_signals(sig, ret, gamma=0.05, seed=Exp3Bandit.DEFAULT_SEED)
    assert (s3 == s4).all()
