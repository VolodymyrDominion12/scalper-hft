"""Тести Fractional Kelly sizing overlay для PairsEngine (дослідження §6.2).

Дослідження §6.2: фракційний Kelly (0.25–0.5) для sizing. `fractional_kelly()`
вже є в `portfolio/risk_budget.py`, але не підключено. Тестуємо:
  - fractional_kelly на синтетичних μ/σ²
  - PairsEngine._update_kelly_size_mult: дефолт off (1.0), з μ>0 зменшує, з μ≤0 → 0
  - _quote: kelly множник застосовується до size_pct
  - backward-compat: kelly_fraction=0 → поведінка як без Kelly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_engine import PairsEngine
from scalper_hft.portfolio.risk_budget import fractional_kelly
from scalper_hft.strategies.pairs_arb import PairsArb

# ── fractional_kelly (unit) ───────────────────────────────────────────────────


class TestFractionalKelly:
    def test_positive_edge_positive_size(self) -> None:
        f = fractional_kelly(0.001, 0.0001, fraction=0.25)
        assert f > 0.0
        assert f == pytest.approx(0.25 * 0.001 / 0.0001)

    def test_negative_edge_zero_size(self) -> None:
        f = fractional_kelly(-0.001, 0.0001, fraction=0.25)
        assert f == 0.0  # clip до 0 — шортимо негативний edge

    def test_zero_variance_zero_size(self) -> None:
        assert fractional_kelly(0.001, 0.0, fraction=0.25) == 0.0

    def test_max_leverage_caps(self) -> None:
        f = fractional_kelly(0.1, 0.0001, fraction=0.5, max_leverage=2.0)
        assert f <= 2.0

    def test_half_kelly_half_of_full(self) -> None:
        full = fractional_kelly(0.001, 0.0001, fraction=1.0, max_leverage=100.0)
        half = fractional_kelly(0.001, 0.0001, fraction=0.5, max_leverage=100.0)
        assert half == pytest.approx(full * 0.5)


# ── PairsEngine._update_kelly_size_mult ──────────────────────────────────────


def _engine(kelly_fraction: float = 0.0, kelly_lookback: int = 168) -> PairsEngine:
    """PairsEngine з заданим kelly_fraction (через settings monkeypatch)."""
    acc = PaperAccount(1_000_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
    )
    eng.kelly_fraction = kelly_fraction
    eng.kelly_lookback = kelly_lookback
    eng._kelly_size_mult = 1.0
    return eng


def _feed_spread(eng: PairsEngine, n: int, spread_values: np.ndarray) -> None:
    """Заповнити _spread_hist тестовими значеннями log(leg1/leg2)."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    eng._spread_hist = pd.Series(spread_values, index=idx)


class TestKellySizeMult:
    def test_default_off_is_noop(self) -> None:
        eng = _engine(kelly_fraction=0.0)
        _feed_spread(eng, 100, np.random.default_rng(1).normal(0, 0.01, 100))
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult == 1.0

    def test_insufficient_data_is_noop(self) -> None:
        eng = _engine(kelly_fraction=0.25, kelly_lookback=168)
        _feed_spread(eng, 15, np.random.default_rng(2).normal(0.001, 0.01, 15))
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult == 1.0  # < 20 точок → no-op

    def test_positive_mean_reduces_size(self) -> None:
        """Mean-reverting spread з μ≈0, великою σ² → Kelly ≈ 0 → size_mult малий.

        Fractional Kelly на μ≈0 дає f≈0 (нема edge), тому множник малий —
        це правильно: Kelly каже «не входь» коли edge невидимий. Велика σ²
        потрібна щоб kelly = fraction·μ/σ² був малим (μ шумить ~σ/√N).
        """
        eng = _engine(kelly_fraction=0.25)
        rng = np.random.default_rng(3)
        # diffs з mean=0, великою σ=0.1 → kelly = 0.25·~0/0.01 ≈ малий
        _feed_spread(eng, 200, np.cumsum(rng.normal(0.0, 0.3, 200)))
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult < 0.2  # малий (велика σ² → kelly малий)

    def test_positive_edge_keeps_size(self) -> None:
        """Спред з позитивним mean-reverting edge → Kelly > 0 → size_mult > 0."""
        eng = _engine(kelly_fraction=0.25)
        # спред з позитивним дрейфом (mean-reverting, edge > 0)
        rng = np.random.default_rng(4)
        diffs = rng.normal(0.002, 0.01, 200)  # mean=0.002 > 0
        spread = np.cumsum(diffs)
        _feed_spread(eng, 200, spread)
        eng._update_kelly_size_mult()
        assert 0.0 < eng._kelly_size_mult <= 1.0

    def test_negative_edge_zero_size(self) -> None:
        """Спред з негативним mean → Kelly clip до 0 → size_mult = 0."""
        eng = _engine(kelly_fraction=0.25)
        rng = np.random.default_rng(5)
        diffs = rng.normal(-0.002, 0.01, 200)  # mean=-0.002 < 0
        spread = np.cumsum(diffs)
        _feed_spread(eng, 200, spread)
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult == 0.0

    def test_zero_variance_is_noop(self) -> None:
        """Константний спред (σ²=0) → no-op (1.0)."""
        eng = _engine(kelly_fraction=0.25)
        _feed_spread(eng, 200, np.full(200, 1.0))  # константа
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult == 1.0

    def test_size_mult_in_unit_range(self) -> None:
        """Множник завжди у [0, 1] — лише зменшує, без плеча."""
        for seed in range(10):
            eng = _engine(kelly_fraction=0.25)
            rng = np.random.default_rng(seed)
            _feed_spread(eng, 200, np.cumsum(rng.normal(0.001, 0.02, 200)))
            eng._update_kelly_size_mult()
            assert 0.0 <= eng._kelly_size_mult <= 1.0


# ── backward-compat: kelly off не змінює поведінку ───────────────────────────


class TestBackwardCompat:
    def test_kelly_off_default(self) -> None:
        eng = _engine()
        assert eng.kelly_fraction == 0.0
        assert eng._kelly_size_mult == 1.0

    def test_kelly_off_does_not_affect_quote(self) -> None:
        """З kelly_fraction=0 _quote поводиться як раніше (size_pct без kelly)."""

        acc = PaperAccount(1_000_000.0, taker_fee=0.0, maker_fee=0.0)
        eng = PairsEngine(
            "AAA",
            "BBB",
            PairsArb(lookback=20, regime_scale=False),
            acc,
            wait_bars=1,
            is_maker=True,
            coint_kill=False,
        )
        # kelly за замовчуванням off
        assert eng.kelly_fraction == 0.0
        # _kelly_size_mult лишається 1.0 після update
        _feed_spread(eng, 100, np.random.default_rng(7).normal(0, 0.01, 100))
        eng._update_kelly_size_mult()
        assert eng._kelly_size_mult == 1.0
