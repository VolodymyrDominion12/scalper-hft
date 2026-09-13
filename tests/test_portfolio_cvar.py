"""Тести CVaR / Expected Shortfall (дослідження §6.2) та інтеграції в pairs_runner.

Дослідження §6.2: CVaR — середня втрата за хвостом розподілу, критичне для
«товстих хвостів» крипто. На відміну від VaR, враховує magnitude втрат за
порогом. Перевіряємо:
  - historical_cvar на нормальному розподілі ≈ VaR
  - на важкому хвості CVaR > VaR
  - insufficient data → 0.0
  - _portfolio_cvar_ok: дефолт off, блокує при перевищенні, no-op при мало даних
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scalper_hft.portfolio.risk_budget import historical_cvar

# ── historical_cvar ───────────────────────────────────────────────────────────


class TestHistoricalCVar:
    def test_normal_distribution_cvar_close_to_var(self) -> None:
        """На нормальному розподілі CVaR ≈ VaR (хвіст тонкий)."""
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.0, 0.01, 10_000))
        var = float(-np.quantile(r, 0.05))
        cvar = historical_cvar(r, alpha=0.05)
        # CVaR трохи більший за VaR (бо бере середнє за хвостом)
        assert cvar >= var * 0.95
        assert cvar <= var * 1.3  # але не набагато більший на нормальному

    def test_fat_tail_cvar_exceeds_var(self) -> None:
        """На важкому хвості (t-розподіл) CVaR > VaR суттєво."""
        rng = np.random.default_rng(7)
        # t-розподіл з df=3 — важкі хвости
        r = pd.Series(rng.standard_t(df=3, size=10_000) * 0.01)
        var = float(-np.quantile(r, 0.05))
        cvar = historical_cvar(r, alpha=0.05)
        assert cvar > var * 1.3, "CVaR має суттєво перевищувати VaR на важкому хвості"

    def test_insufficient_data_returns_zero(self) -> None:
        assert historical_cvar(pd.Series([0.01, -0.02]), alpha=0.05) == 0.0
        assert historical_cvar(np.array([]), alpha=0.05) == 0.0

    def test_positive_returns_zero_cvar(self) -> None:
        """Якщо всі прибутковості додатні — хвоста нема, CVaR ≈ 0."""
        r = pd.Series([0.01, 0.02, 0.005, 0.03, 0.01] * 4)
        cvar = historical_cvar(r, alpha=0.05)
        assert cvar >= 0.0

    def test_extreme_loss_dominates_cvar(self) -> None:
        """Один екстремальний збиток піднімає CVaR вище за VaR."""
        r = pd.Series([-0.001] * 50 + [-0.50] + [0.001] * 49)
        var = float(-np.quantile(r, 0.05))
        cvar = historical_cvar(r, alpha=0.05)
        assert cvar > var, "екстрим має підняти CVaR над VaR"
        assert cvar > 0.005  # велика втрата (mean хвоста ≈ 1%)

    def test_alpha_01_more_conservative(self) -> None:
        """alpha=0.01 (99% CVaR) ≥ alpha=0.05 (95% CVaR) на важкому хвості."""
        rng = np.random.default_rng(11)
        r = pd.Series(rng.standard_t(df=3, size=10_000) * 0.01)
        cvar_95 = historical_cvar(r, alpha=0.05)
        cvar_99 = historical_cvar(r, alpha=0.01)
        assert cvar_99 >= cvar_95 * 0.9  # 99% зазвичай більший

    def test_accepts_numpy_array(self) -> None:
        r = np.array([-0.01, -0.02, 0.005, -0.03, 0.01] * 4)
        cvar = historical_cvar(r, alpha=0.05)
        assert cvar > 0.0


# ── Інтеграція в PairsPortfolioRunner._portfolio_cvar_ok ──────────────────────


def _runner_with_window(equity_window: list[float], **settings_kw) -> object:
    """Створити PairsPortfolioRunner-подібний об'єкт з заданим equity_window.

    Уникає повного __init__ (мережа, SQLite, ccxt) — лише атрибути потрібні
    для _portfolio_cvar_ok.
    """
    from scalper_hft.live.pairs_runner import PairsPortfolioRunner

    r = PairsPortfolioRunner.__new__(PairsPortfolioRunner)
    r.portfolio_cvar_limit = float(settings_kw.get("portfolio_cvar_limit", 0.0))
    r.portfolio_cvar_alpha = float(settings_kw.get("portfolio_cvar_alpha", 0.05))
    r._equity_window = deque(equity_window, maxlen=100)
    return r


class TestPortfolioCvarOk:
    def test_default_off_returns_true(self) -> None:
        r = _runner_with_window([100, 99, 98, 97, 96, 95, 94, 93, 92, 91], portfolio_cvar_limit=0.0)
        assert r._portfolio_cvar_ok() is True

    def test_insufficient_data_returns_true(self) -> None:
        r = _runner_with_window([100, 99, 98], portfolio_cvar_limit=0.05)
        assert r._portfolio_cvar_ok() is True

    def test_blocks_when_cvar_exceeds_limit(self) -> None:
        # серія збитків: -1% щокроку → CVaR ~0.01 > limit 0.005
        eq = [100.0]
        for i in range(60):
            eq.append(eq[-1] * 0.99)
        r = _runner_with_window(eq, portfolio_cvar_limit=0.005, portfolio_cvar_alpha=0.05)
        assert r._portfolio_cvar_ok() is False

    def test_passes_when_cvar_within_limit(self) -> None:
        # малі коливання ~0.1% → CVaR ~0.001 < limit 0.05
        rng = np.random.default_rng(3)
        eq = [100.0]
        for _ in range(60):
            eq.append(eq[-1] * (1.0 + rng.normal(0.0, 0.001)))
        r = _runner_with_window(eq, portfolio_cvar_limit=0.05, portfolio_cvar_alpha=0.05)
        assert r._portfolio_cvar_ok() is True

    def test_extreme_loss_triggers_cvar_halt(self) -> None:
        # переважно спокій + один крах -10%. alpha=0.01 (99% CVaR) ловить саме крах.
        eq = [100.0]
        for i in range(60):
            eq.append(eq[-1] * (1.0 + 0.0001))
        eq.append(eq[-1] * 0.90)  # крах -10%
        eq.append(eq[-1] * (1.0 + 0.0001))
        r = _runner_with_window(eq, portfolio_cvar_limit=0.05, portfolio_cvar_alpha=0.01)
        assert r._portfolio_cvar_ok() is False
