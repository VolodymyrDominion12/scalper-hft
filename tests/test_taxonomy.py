"""Таксономія стратегій: family + preferred_regimes (гіпотези, не live-гейт)."""

from __future__ import annotations

import pytest
from scalper_hft.strategies import (
    FAMILIES,
    PREFERRED_REGIME_LABELS,
    REGISTRY,
    get_strategy,
    regime_capital_weight,
    registry_taxonomy,
)

EXPECTED_FAMILIES: dict[str, str] = {
    "mean_reversion": "mean_reversion",
    "cvd_momentum": "momentum",
    "ob_imbalance": "flow",
    "market_maker": "market_making",
    "funding_carry": "carry",
    "funding_arb": "carry",
    "basis_reversion": "relative_value",
    "pairs_arb": "relative_value",
    "ml_strategy": "ml",
    "ensemble": "meta",
    "hmm_reversion": "mean_reversion",
    "sparse_basket": "relative_value",
    "cross_momentum": "momentum",
    "ts_momentum": "momentum",
    "supertrend": "momentum",
    "stoch_rsi": "mean_reversion",
    "smc_fvg": "flow",
    "regime_supervisor": "meta",
}


def test_registry_has_eighteen_named_strategies() -> None:
    # 17 → 18: iter9 додав `ts_momentum` (TSMOM, винесений з single-symbol
    # гілки cross_momentum, яку блокує capability contract).
    assert len(REGISTRY) == 18
    assert set(REGISTRY) == set(EXPECTED_FAMILIES)


@pytest.mark.parametrize("name, family", sorted(EXPECTED_FAMILIES.items()))
def test_registered_strategy_family(name: str, family: str) -> None:
    cls = REGISTRY[name]
    assert cls.family == family
    assert family in FAMILIES
    extra = set(cls.preferred_regimes) - PREFERRED_REGIME_LABELS
    assert not extra
    inst = get_strategy(name)
    assert inst.family == family
    assert inst.preferred_regimes == cls.preferred_regimes


def test_relative_value_has_empty_preferred_regimes() -> None:
    """pairs/basis/basket — гіпотеза «всі режими», поки OOS не скаже інакше."""
    for name in ("pairs_arb", "basis_reversion", "sparse_basket"):
        assert REGISTRY[name].preferred_regimes == frozenset()


def test_mean_reversion_prefers_range_not_high_vol() -> None:
    pref = REGISTRY["mean_reversion"].preferred_regimes
    assert "range" in pref
    assert "high" not in pref
    assert "trend_up" not in pref


def test_momentum_prefers_trends() -> None:
    for name in ("cvd_momentum", "cross_momentum"):
        pref = REGISTRY[name].preferred_regimes
        assert pref == frozenset({"trend_up", "trend_down"})


def test_iter7_evidence_based_regime_tags() -> None:
    """iter7 (1h × 10 символів × 3y): теги виправлено за OOS-даними.

    supertrend/stoch_rsi/smc_fvg — лише trend_up (у trend_down 0/10 символів
    позитивних); funding_carry — range+trend_down (у trend_up 0/10).
    Референс: docs/reports/strategy_rating_regime.md.
    """
    for name in ("supertrend", "stoch_rsi", "smc_fvg"):
        assert REGISTRY[name].preferred_regimes == frozenset({"trend_up"}), name
    assert REGISTRY["funding_carry"].preferred_regimes == frozenset({"range", "trend_down"})
    # supertrend НЕ має trend_down (найгірший режим: −4.82, 0/10 символів)
    assert "trend_down" not in REGISTRY["supertrend"].preferred_regimes


def test_registry_taxonomy_matches_classes() -> None:
    book = registry_taxonomy()
    assert set(book) == set(REGISTRY)
    for name, meta in book.items():
        assert meta["family"] == REGISTRY[name].family
        assert meta["preferred_regimes"] == frozenset(REGISTRY[name].preferred_regimes)


@pytest.mark.parametrize(
    ("preferred", "structure", "vol", "expected"),
    [
        (frozenset(), "range", "high", 1.0),
        (frozenset({"range"}), "range", "high", 1.0),
        (frozenset({"range"}), "trend_up", "low", 0.25),
        (frozenset({"trend_up", "trend_down"}), "trend_down", "normal", 1.0),
        (frozenset({"range", "low", "normal"}), "range", "high", 0.25),
        (frozenset({"range", "low", "normal"}), "trend_up", "high", 0.25 * 0.25),
        (frozenset({"range", "low", "normal"}), "range", "low", 1.0),
    ],
)
def test_regime_capital_weight(
    preferred: frozenset[str],
    structure: str,
    vol: str,
    expected: float,
) -> None:
    assert regime_capital_weight(preferred, structure, vol) == expected


def test_regime_capital_weight_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown preferred"):
        regime_capital_weight(frozenset({"bull"}), "range", "low")


def test_regime_capital_weight_rejects_bad_unfavorable() -> None:
    with pytest.raises(ValueError, match="unfavorable"):
        regime_capital_weight(frozenset(), "range", "low", unfavorable=1.5)
