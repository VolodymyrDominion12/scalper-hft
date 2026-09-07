"""Тести Decimal-обліку PaperAccount та money helpers."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.money import as_float, money, quantize, to_decimal


def test_money_avoids_float_drift() -> None:
    """0.1 + 0.2 у Decimal ≠ float artifact 0.30000000000000004."""
    a = money("0.1")
    b = money("0.2")
    assert as_float(a + b) == 0.3


def test_many_micro_fees_stable() -> None:
    """1000 дрібних maker-комісій не накопичують float-помилку."""
    acc = PaperAccount(initial_capital=10_000.0, maker_fee=0.0002, taker_fee=0.0005)
    ts = pd.Timestamp("2025-01-01")
    start = acc.cash
    for i in range(1000):
        sym = f"SYM{i}"
        acc.open_position(sym, "long", 0.001, 100.0 + i * 0.01, ts, is_maker=True)
        acc.close_position(sym, 100.0 + i * 0.01, ts, is_maker=True)
    # Кожна round-trip: open fee + close fee = 2 * 0.0002 * notional
    expected_fees = sum(2 * 0.0002 * (0.001 * (100.0 + i * 0.01)) for i in range(1000))
    assert abs(acc.cash - (start - expected_fees)) < 1e-6


def test_snapshot_roundtrip_preserves_cash() -> None:
    acc = PaperAccount(initial_capital=5_000.0)
    ts = pd.Timestamp("2025-06-01")
    acc.open_position("BTCUSDT", "long", 0.01, 50_000.0, ts, is_maker=True)
    snap = acc.to_snapshot()
    restored = PaperAccount.from_snapshot(snap)
    assert restored.cash == acc.cash
    assert restored.positions["BTCUSDT"].size == acc.positions["BTCUSDT"].size


def test_partial_close_decimal_fraction() -> None:
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("ETHUSDT", "long", 1.0, 3000.0, ts)
    tr = acc.close_position("ETHUSDT", 3100.0, ts, size=0.33333333)
    assert tr["pnl"] == pytest.approx(33.333333, rel=1e-6)
    assert acc.positions["ETHUSDT"].size == pytest.approx(0.66666667, rel=1e-6)


def test_quantize_constants() -> None:
    assert quantize(to_decimal("1.234567891")) == Decimal("1.23456789")
