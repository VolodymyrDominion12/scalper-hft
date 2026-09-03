"""CLI helpers: Kalman flag і попередження pairs на коротких ТФ."""

from types import SimpleNamespace

import pandas as pd
from scalper_hft.cli import _apply_use_kalman, _warn_pairs_short_interval
from scalper_hft.research.session_analysis import hourly_fill_rate
from scalper_hft.strategies import get_strategy


def test_apply_use_kalman_sets_param() -> None:
    got = _apply_use_kalman(SimpleNamespace(use_kalman=True), {})
    assert got["use_kalman"] is True
    strat = get_strategy("pairs_arb", **got)
    assert strat.get("use_kalman", False) is True


def test_apply_use_kalman_default_off() -> None:
    got = _apply_use_kalman(SimpleNamespace(use_kalman=False), {"entry_z": 2.0})
    assert "use_kalman" not in got


def test_warn_pairs_short_interval(caplog) -> None:
    with caplog.at_level("WARNING"):
        _warn_pairs_short_interval("pairs_arb", "1m")
    assert "1h maker" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_pairs_short_interval("pairs_arb", "1h")
    assert caplog.text == ""


def test_hourly_fill_rate() -> None:
    idx = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:30", "2026-01-01 11:00"])
    df = pd.DataFrame({"ts": idx, "status": ["filled", "unfilled", "filled"]})
    out = hourly_fill_rate(df)
    assert out.loc[10, "fill_rate"] == 0.5
    assert out.loc[11, "fill_rate"] == 1.0
