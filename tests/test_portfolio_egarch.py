"""Тести EGARCH/HAR-RV vol-target overlay для PairsEngine (дослідження §6.1).

Дослідження §6.1: EGARCH/GJR-GARCH для асиметрії шоків; HAR-RV для 1-денних
прогнозів. У проєкті вони у `features/volatility.py`, але не у portfolio risk.
Тестуємо:
  - _estimate_ann_vol: realized (дефолт), egarch, har_rv
  - fallback на realized при помилці/нестачі даних
  - backward-compat: vol_method="realized" = стара поведінка
  - vol_size_mult у [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_engine import PairsEngine
from scalper_hft.strategies.pairs_arb import PairsArb


def _engine(vol_method: str = "realized", vol_target_ann: float | None = 0.10) -> PairsEngine:
    acc = PaperAccount(1_000_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
        vol_target_ann=vol_target_ann,
    )
    eng.vol_method = vol_method
    return eng


def _feed_spread(eng: PairsEngine, n: int, seed: int = 7, vol: float = 0.01) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    eng._spread_hist = pd.Series(np.cumsum(rng.normal(0, vol, n)), index=idx)


# ── _estimate_ann_vol ─────────────────────────────────────────────────────────


class TestEstimateAnnVol:
    def test_realized_default(self) -> None:
        eng = _engine(vol_method="realized")
        diffs = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
        vol = eng._estimate_ann_vol(diffs, method="realized")
        expected = float(diffs.std(ddof=0)) * float(np.sqrt(eng.bars_per_year))
        assert vol == pytest.approx(expected, rel=1e-6)

    def test_egarch_returns_finite(self) -> None:
        eng = _engine(vol_method="egarch")
        rng = np.random.default_rng(2)
        # EGARCH потребує достатньо даних + ненульову варіацію
        diffs = pd.Series(rng.normal(0, 0.02, 600), index=pd.date_range("2025-01-01", periods=600, freq="1h"))
        vol = eng._estimate_ann_vol(diffs, method="egarch")
        assert np.isfinite(vol)
        assert vol > 0

    def test_har_rv_returns_finite(self) -> None:
        eng = _engine(vol_method="har_rv")
        rng = np.random.default_rng(3)
        diffs = pd.Series(rng.normal(0, 0.02, 600), index=pd.date_range("2025-01-01", periods=600, freq="1h"))
        vol = eng._estimate_ann_vol(diffs, method="har_rv")
        assert np.isfinite(vol)
        assert vol > 0

    def test_egarch_fallback_on_short_data(self) -> None:
        """Мало даних → fallback на realized."""
        eng = _engine(vol_method="egarch")
        diffs = pd.Series(np.random.default_rng(4).normal(0, 0.01, 30))
        vol_egarch = eng._estimate_ann_vol(diffs, method="egarch")
        vol_realized = eng._estimate_ann_vol(diffs, method="realized")
        # при нестачі даних egarch fallback на realized
        assert vol_egarch == pytest.approx(vol_realized, rel=1e-6)

    def test_unknown_method_falls_back_to_realized(self) -> None:
        eng = _engine(vol_method="garbage")
        diffs = pd.Series(np.random.default_rng(5).normal(0, 0.01, 200))
        vol = eng._estimate_ann_vol(diffs, method="garbage")
        expected = eng._estimate_ann_vol(diffs, method="realized")
        assert vol == pytest.approx(expected, rel=1e-6)


# ── _update_vol_size_mult ────────────────────────────────────────────────────


class TestUpdateVolSizeMult:
    def test_vol_target_none_is_noop(self) -> None:
        eng = _engine(vol_target_ann=None)
        _feed_spread(eng, 200)
        eng._update_vol_size_mult()
        assert eng._vol_size_mult == 1.0

    def test_insufficient_data_is_noop(self) -> None:
        eng = _engine(vol_method="realized")
        _feed_spread(eng, 15)
        eng._update_vol_size_mult()
        assert eng._vol_size_mult == 1.0

    def test_realized_method_backward_compat(self) -> None:
        """vol_method='realized' = стара поведінка (rolling std)."""
        eng = _engine(vol_method="realized", vol_target_ann=0.10)
        _feed_spread(eng, 200, vol=0.01)
        eng._update_vol_size_mult()
        assert 0.0 <= eng._vol_size_mult <= 1.0

    def test_egarch_method_produces_multiplier(self) -> None:
        eng = _engine(vol_method="egarch", vol_target_ann=0.10)
        _feed_spread(eng, 600, vol=0.02)
        eng._update_vol_size_mult()
        assert 0.0 <= eng._vol_size_mult <= 1.0

    def test_har_rv_method_produces_multiplier(self) -> None:
        eng = _engine(vol_method="har_rv", vol_target_ann=0.10)
        _feed_spread(eng, 600, vol=0.02)
        eng._update_vol_size_mult()
        assert 0.0 <= eng._vol_size_mult <= 1.0

    def test_all_methods_in_unit_range(self) -> None:
        """Усі методи дають множник у [0, 1] — лише зменшує."""
        for method in ("realized", "egarch", "har_rv"):
            eng = _engine(vol_method=method, vol_target_ann=0.10)
            _feed_spread(eng, 600, vol=0.02, seed=42)
            eng._update_vol_size_mult()
            assert 0.0 <= eng._vol_size_mult <= 1.0, f"method={method} дав {eng._vol_size_mult}"


# ── backward-compat ─────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_default_vol_method_is_realized(self) -> None:
        eng = _engine()
        assert eng.vol_method == "realized"

    def test_realized_matches_old_formula(self) -> None:
        """vol_method='realized' дає ту саму формулу що й раніше."""
        eng = _engine(vol_method="realized", vol_target_ann=0.15)
        _feed_spread(eng, 200, vol=0.01, seed=7)
        eng._update_vol_size_mult()
        # стара формула: clip(target / (std * sqrt(bars_per_year)), 0, 1)
        diffs = eng._spread_hist.diff().dropna().tail(eng.vol_lookback)
        expected = float(np.clip(0.15 / (diffs.std(ddof=0) * np.sqrt(eng.bars_per_year)), 0.0, 1.0))
        assert eng._vol_size_mult == pytest.approx(expected, rel=1e-6)
