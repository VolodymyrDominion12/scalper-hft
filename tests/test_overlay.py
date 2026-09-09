"""Cell Overlay: резолвер, YAML, apply без lookahead, sweep-вікна 4h."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.overlay.apply import apply_cell_overlay, hours_to_bars
from scalper_hft.overlay.loader import load_overlay_book, overlay_book_from_dict
from scalper_hft.overlay.policy import CellPolicy
from scalper_hft.overlay.resolver import resolve_policy
from scalper_hft.validation.sweep import run_sweep


def _signals(values: list[float], freq: str = "1h") -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_hours_to_bars_scales_with_interval() -> None:
    assert hours_to_bars(0, "1h") == 0
    assert hours_to_bars(4, "1h") == 4
    assert hours_to_bars(4, "1m") == 240
    assert hours_to_bars(4, "15m") == 16


def test_resolver_specificity_and_tie_later_wins() -> None:
    book = overlay_book_from_dict(
        {
            "clusters": {"liquid_alts": ["AAVEUSDT"]},
            "defaults": {"enabled": False},
            "cells": [
                {"match": {"strategy": "*", "interval": ["1h"]}, "enabled": False},
                {
                    "match": {"strategy": "stoch_rsi", "interval": ["1h"], "cluster": "liquid_alts"},
                    "enabled": True,
                    "min_hold_hours": 4,
                },
            ],
        }
    )
    on = resolve_policy(book, "stoch_rsi", "AAVEUSDT", "1h")
    off = resolve_policy(book, "stoch_rsi", "BTCUSDT", "1h")
    assert on.enabled is True
    assert on.min_hold_hours == 4.0
    assert off.enabled is False


def test_loader_rejects_unknown_cluster() -> None:
    with pytest.raises(ValueError, match="невідомі кластери"):
        overlay_book_from_dict(
            {
                "clusters": {"majors": ["BTCUSDT"]},
                "cells": [{"match": {"strategy": "*", "cluster": "nope"}, "enabled": False}],
            }
        )


def test_default_yaml_playbook_cells() -> None:
    book = load_overlay_book()
    aave_1h = resolve_policy(book, "stoch_rsi", "AAVEUSDT", "1h")
    btc_1h = resolve_policy(book, "stoch_rsi", "BTCUSDT", "1h")
    aave_1m = resolve_policy(book, "stoch_rsi", "AAVEUSDT", "1m")
    xrp_fund = resolve_policy(book, "funding_carry", "XRPUSDT", "15m")
    cross = resolve_policy(book, "cross_momentum", "AAVEUSDT", "1h")
    pairs = resolve_policy(book, "pairs_arb", "BTCUSDT", "1h")
    assert aave_1h.enabled is True
    assert aave_1h.allow_short is True
    assert aave_1h.execution == "maker"
    assert aave_1h.min_hold_hours == 4.0
    assert btc_1h.enabled is False
    assert aave_1m.enabled is False
    assert xrp_fund.enabled is True
    assert xrp_fund.enabled_when == "funding_regime"
    assert cross.enabled is False
    assert pairs.enabled is True


def test_apply_disabled_is_flat() -> None:
    sig = _signals([1.0, -1.0, 1.0])
    out = apply_cell_overlay(sig, CellPolicy(enabled=False), interval="1h")
    assert (out == 0).all()


def test_apply_clips_short_when_disallowed() -> None:
    sig = _signals([1.0, -1.0, -1.0])
    out = apply_cell_overlay(sig, CellPolicy(allow_short=False), interval="1h")
    assert list(out) == [1.0, 0.0, 0.0]


def test_apply_min_hold_blocks_flip() -> None:
    sig = _signals([1.0] * 3 + [-1.0] * 5)
    out = apply_cell_overlay(sig, CellPolicy(min_hold_hours=4), interval="1h")
    # flip на барі 3 при held=3 < 4 — лишаємо лонг ще бар, далі можна фліпнути
    assert out.iloc[3] == 1.0
    assert out.iloc[4] == -1.0


def test_apply_max_trades_per_day() -> None:
    # два входи за один день
    sig = _signals([1.0, 0.0, 1.0, 1.0], freq="1h")
    out = apply_cell_overlay(sig, CellPolicy(max_trades_per_day=1), interval="1h")
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert out.iloc[2] == 0.0  # другий вхід заблоковано


def test_apply_funding_regime_fail_closed() -> None:
    sig = _signals([1.0, 1.0, 1.0])
    out = apply_cell_overlay(
        sig,
        CellPolicy(enabled_when="funding_regime", funding_annual_min=0.10),
        interval="1h",
        funding=None,
    )
    assert (out == 0).all()


def test_apply_no_lookahead() -> None:
    sig = _signals([1.0, 0.0, -1.0, 1.0, 0.0, 1.0] * 5)
    policy = CellPolicy(min_hold_hours=2, allow_short=True)
    base = apply_cell_overlay(sig, policy, interval="1h")
    mutated = sig.copy()
    mutated.iloc[-3:] = -1.0
    after = apply_cell_overlay(mutated, policy, interval="1h")
    mid = len(sig) // 2
    pd.testing.assert_series_equal(base.iloc[:mid], after.iloc[:mid])


def _klines(n: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range(end="2024-06-01", periods=n, freq=freq)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )


def test_sweep_walkforward_4h_uses_per_interval_windows() -> None:
    def provider(symbol: str, interval: str, days: int):
        return _klines(400, "4h"), None, None

    auto = run_sweep(
        strategies=["mean_reversion"],
        symbols=["BTCUSDT"],
        intervals=["4h"],
        days=1,
        mode="walkforward",
        data_provider=provider,
    )
    assert auto.iloc[0]["status"] == "ok"

    old = run_sweep(
        strategies=["mean_reversion"],
        symbols=["BTCUSDT"],
        intervals=["4h"],
        days=1,
        mode="walkforward",
        train_bars=2000,
        test_bars=500,
        data_provider=provider,
    )
    assert old.iloc[0]["status"] == "error"
    assert "train+purge+test" in str(old.iloc[0]["error"])


def test_sweep_overlay_disables_majors_and_keeps_alts() -> None:
    book = load_overlay_book()

    def provider(symbol: str, interval: str, days: int):
        return _klines(80, "1h"), None, None

    btc = run_sweep(
        strategies=["stoch_rsi"],
        symbols=["BTCUSDT"],
        intervals=["1h"],
        days=1,
        data_provider=provider,
        overlay_book=book,
    )
    aave = run_sweep(
        strategies=["stoch_rsi"],
        symbols=["AAVEUSDT"],
        intervals=["1h"],
        days=1,
        data_provider=provider,
        overlay_book=book,
    )
    assert btc.iloc[0]["status"] == "disabled"
    assert aave.iloc[0]["status"] == "ok"
