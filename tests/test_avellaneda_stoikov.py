"""Тести для Avellaneda-Stoikov Market Maker та VPIN захисту."""

import numpy as np
import pytest

from scalper_hft.strategies.market_maker import PassiveMarketMaker


def test_reservation_price_inventory_skew():
    """Тест: при накопиченні лонгу ціна бронювання падає (бажання продати); при шорті — росте."""
    mm = PassiveMarketMaker(gamma=0.1)
    mid = 100.0
    vol = 0.02

    r_neutral = mm.reservation_price(mid, inventory=0.0, vol=vol)
    r_long = mm.reservation_price(mid, inventory=1.0, vol=vol)
    r_short = mm.reservation_price(mid, inventory=-1.0, vol=vol)

    assert r_neutral == mid
    assert r_long < mid  # дисконт, щоб швидше закрити лонг
    assert r_short > mid  # премія, щоб швидше викупити шорт


def test_optimal_half_spread_volatility():
    """Тест: напівспред розширюється зі збільшенням волатильності ринку."""
    mm = PassiveMarketMaker(gamma=0.1, kappa=1.5)
    delta_low_vol = mm.optimal_half_spread(vol=0.01)
    delta_high_vol = mm.optimal_half_spread(vol=0.05)

    assert delta_high_vol > delta_low_vol > 0.0


def test_compute_quotes_inventory_cap():
    """Тест: при досягненні inventory_cap котирування вимикаються у напрямку ризику."""
    mm = PassiveMarketMaker(inventory_cap=1.0)
    mid = 100.0
    vol = 0.02

    # Long inventory limit reached (inventory = 1.0 >= cap 1.0)
    bid, ask, active = mm.compute_quotes(mid, inventory=1.0, vol=vol, base_spread=0.1)
    assert bid == 0.0  # заборона нових покупок
    assert ask > mid
    assert active is True

    # Short inventory limit reached (inventory = -1.0 <= -cap)
    bid, ask, active = mm.compute_quotes(mid, inventory=-1.0, vol=vol, base_spread=0.1)
    assert bid < mid
    assert np.isinf(ask)  # заборона нових продажів


def test_vpin_circuit_breaker():
    """Тест: сплеск VPIN деактивує котирування (захист від токсичного потоку)."""
    mm = PassiveMarketMaker(use_vpin_shield=True, vpin_threshold=0.80)
    mid = 100.0
    vol = 0.02

    # Normal flow: VPIN = 0.40 -> active
    bid, ask, active = mm.compute_quotes(mid, inventory=0.0, vol=vol, vpin=0.40)
    assert active is True

    # Toxic flow: VPIN = 0.85 (> 0.80) -> inactive (circuit breaker)
    bid, ask, active = mm.compute_quotes(mid, inventory=0.0, vol=vol, vpin=0.85)
    assert active is False
