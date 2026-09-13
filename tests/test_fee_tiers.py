"""Тести VIP-рівнів комісій (дослідження §7.1) та інтеграції в CostModel.

Дослідження §7.1: Binance VIP 4–9 дає зниження maker аж до 0%; Bybit Supreme VIP
0.000% maker. Для MFT з 50 угод/день це визначає прибутковість. Перевіряємо:
  - таблиці рівнів Binance/Bybit
  - resolve_fees для різних рівнів
  - BNB-дисконт 25%
  - CostModel.from_settings читає FEE_TIER і перекриває maker/taker
  - дефолт vip0 = базові ставки (без змін)
  - невідома біржа/рівень → fallback на явні MAKER_FEE/TAKER_FEE
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.data.fees import (
    BINANCE_USDTM_TIERS,
    BYBIT_DERIVATIVES_TIERS,
    available_tiers,
    is_supported_tier,
    resolve_fees,
)

# ── Таблиці рівнів ────────────────────────────────────────────────────────────


class TestTierTables:
    def test_binance_has_vip0_to_vip9(self) -> None:
        for i in range(10):
            assert f"vip{i}" in BINANCE_USDTM_TIERS

    def test_binance_vip0_is_base(self) -> None:
        assert BINANCE_USDTM_TIERS["vip0"] == (0.00020, 0.00050)

    def test_binance_vip9_zero_maker(self) -> None:
        maker, taker = BINANCE_USDTM_TIERS["vip9"]
        assert maker == 0.0
        assert taker == pytest.approx(0.00017)

    def test_binance_maker_monotone_decreasing(self) -> None:
        makers = [BINANCE_USDTM_TIERS[f"vip{i}"][0] for i in range(10)]
        assert makers == sorted(makers, reverse=True)

    def test_bybit_has_supreme(self) -> None:
        assert "supreme" in BYBIT_DERIVATIVES_TIERS

    def test_bybit_supreme_zero_maker(self) -> None:
        maker, taker = BYBIT_DERIVATIVES_TIERS["supreme"]
        assert maker == 0.0

    def test_bybit_base_taker_higher_than_binance(self) -> None:
        # Дослідження §7.1: Bybit базовий taker 0.055% > Binance 0.050%
        assert BYBIT_DERIVATIVES_TIERS["vip0"][1] > BINANCE_USDTM_TIERS["vip0"][1]


# ── resolve_fees ─────────────────────────────────────────────────────────────


class TestResolveFees:
    def test_binance_vip0_base(self) -> None:
        m, t = resolve_fees("vip0", "binanceusdm")
        assert m == pytest.approx(0.0002)
        assert t == pytest.approx(0.0005)

    def test_binance_vip9_zero_maker(self) -> None:
        m, t = resolve_fees("vip9", "binance")
        assert m == 0.0
        assert t == pytest.approx(0.00017)

    def test_bybit_supreme(self) -> None:
        m, t = resolve_fees("supreme", "bybit")
        assert m == 0.0
        assert t == pytest.approx(0.0003)

    def test_case_insensitive_tier(self) -> None:
        m1, _ = resolve_fees("VIP9", "binance")
        m2, _ = resolve_fees("vip9", "binance")
        assert m1 == m2

    def test_case_insensitive_exchange(self) -> None:
        m1, _ = resolve_fees("vip0", "BinanceUSDM")
        m2, _ = resolve_fees("vip0", "binanceusdm")
        assert m1 == m2

    def test_bnb_discount_lowers_fees(self) -> None:
        m_base, t_base = resolve_fees("vip0", "binance", bnb_discount=False)
        m_bnb, t_bnb = resolve_fees("vip0", "binance", bnb_discount=True)
        assert m_bnb == pytest.approx(m_base * 0.75)
        assert t_bnb == pytest.approx(t_base * 0.75)

    def test_bnb_discount_only_binance(self) -> None:
        # Bybit не має BNB-дисконту — не повинен змінювати комісію.
        m, t = resolve_fees("vip0", "bybit", bnb_discount=True)
        assert m == BYBIT_DERIVATIVES_TIERS["vip0"][0]
        assert t == BYBIT_DERIVATIVES_TIERS["vip0"][1]

    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Невідомий FEE_TIER"):
            resolve_fees("vip99", "binance")

    def test_unknown_exchange_raises(self) -> None:
        with pytest.raises(ValueError, match="Невідома біржа"):
            resolve_fees("vip0", "okx")

    def test_is_supported_tier(self) -> None:
        assert is_supported_tier("vip9", "binance") is True
        assert is_supported_tier("supreme", "bybit") is True
        assert is_supported_tier("supreme", "binance") is False
        assert is_supported_tier("vip0", "okx") is False

    def test_available_tiers_binance(self) -> None:
        tiers = available_tiers("binance")
        assert "vip0" in tiers
        assert "vip9" in tiers
        assert "supreme" not in tiers

    def test_available_tiers_unknown_exchange_empty(self) -> None:
        assert available_tiers("okx") == []


# ── Інтеграція в CostModel.from_settings ──────────────────────────────────────


def _settings(**kw) -> SimpleNamespace:
    """Мінімальний Settings-подібний об'єкт для CostModel.from_settings."""
    defaults = dict(
        maker_fee=0.0002,
        taker_fee=0.0005,
        slippage_bps=2.0,
        vol_aware_slippage_ref=0.0,
        vol_aware_slippage_exp=1.0,
        impact_k=0.0,
        fee_tier="vip0",
        fee_tier_bnb_discount=False,
        data_exchange="binanceusdm",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestCostModelTierIntegration:
    def test_vip0_uses_explicit_fees(self) -> None:
        """vip0 (дефолт) — лишає явні MAKER_FEE/TAKER_FEE."""
        cm = CostModel.from_settings(_settings(fee_tier="vip0"))
        assert cm.maker_fee == pytest.approx(0.0002)
        assert cm.taker_fee == pytest.approx(0.0005)

    def test_vip9_overrides_to_zero_maker(self) -> None:
        cm = CostModel.from_settings(_settings(fee_tier="vip9"))
        assert cm.maker_fee == 0.0
        assert cm.taker_fee == pytest.approx(0.00017)

    def test_vip4_overrides_both(self) -> None:
        cm = CostModel.from_settings(_settings(fee_tier="vip4"))
        assert cm.maker_fee == pytest.approx(0.00010)
        assert cm.taker_fee == pytest.approx(0.00030)

    def test_bnb_discount_applied(self) -> None:
        cm = CostModel.from_settings(_settings(fee_tier="vip0", fee_tier_bnb_discount=True))
        assert cm.maker_fee == pytest.approx(0.0002 * 0.75)
        assert cm.taker_fee == pytest.approx(0.0005 * 0.75)

    def test_bybit_supreme(self) -> None:
        cm = CostModel.from_settings(_settings(fee_tier="supreme", data_exchange="bybit"))
        assert cm.maker_fee == 0.0
        assert cm.taker_fee == pytest.approx(0.0003)

    def test_unknown_tier_falls_back_to_explicit(self) -> None:
        """Невідомий рівень — лишає явні MAKER_FEE/TAKER_FEE (не падає)."""
        cm = CostModel.from_settings(_settings(fee_tier="vip99"))
        assert cm.maker_fee == pytest.approx(0.0002)
        assert cm.taker_fee == pytest.approx(0.0005)

    def test_unknown_exchange_falls_back(self) -> None:
        cm = CostModel.from_settings(_settings(fee_tier="vip9", data_exchange="okx"))
        # okx не підтримується → лишає явні
        assert cm.maker_fee == pytest.approx(0.0002)
        assert cm.taker_fee == pytest.approx(0.0005)

    def test_round_trip_maker_vip9_zero(self) -> None:
        """vip9 + maker → round-trip maker = 0 (тільки slippage/impact)."""
        cm = CostModel.from_settings(_settings(fee_tier="vip9", slippage_bps=0.0, impact_k=0.0))
        assert cm.round_trip_maker() == 0.0

    def test_round_trip_maker_vip4_lower_than_vip0(self) -> None:
        cm0 = CostModel.from_settings(_settings(fee_tier="vip0", slippage_bps=0.0))
        cm4 = CostModel.from_settings(_settings(fee_tier="vip4", slippage_bps=0.0))
        assert cm4.round_trip_maker() < cm0.round_trip_maker()

    def test_explicit_fees_still_work_without_tier(self) -> None:
        """Без fee_tier атрибута — лишає дефолт (backward-compat)."""
        s = SimpleNamespace(
            maker_fee=0.0003,
            taker_fee=0.0007,
            slippage_bps=1.0,
            vol_aware_slippage_ref=0.0,
            vol_aware_slippage_exp=1.0,
            impact_k=0.0,
        )
        cm = CostModel.from_settings(s)
        assert cm.maker_fee == pytest.approx(0.0003)
        assert cm.taker_fee == pytest.approx(0.0007)
