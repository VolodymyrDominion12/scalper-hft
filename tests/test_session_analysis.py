"""Умовні метрики угод по vol / структурі / складеному режиму."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.features.regimes import COMPOSITE_ORDER, STRUCTURE_ORDER, VOL_ORDER, named_market_state
from scalper_hft.research.session_analysis import (
    named_regime_breakdown,
    regime_breakdown,
    structure_breakdown,
)


def _trades() -> pd.DataFrame:
    idx = pd.date_range("2025-01-10", periods=6, freq="1D")
    return pd.DataFrame(
        {
            "entry_ts": idx,
            "ret": [0.02, 0.01, -0.01, 0.03, -0.02, 0.04],
        }
    )


def test_regime_breakdown_empty_without_series() -> None:
    assert regime_breakdown(_trades(), None).empty
    assert structure_breakdown(_trades(), None).empty
    assert named_regime_breakdown(_trades(), None).empty


def test_regime_breakdown_by_vol_includes_sharpe() -> None:
    trades = _trades()
    vol = pd.Series(
        ["low", "low", "normal", "high", "high", "high"],
        index=pd.to_datetime(trades["entry_ts"]),
    )
    out = regime_breakdown(trades, vol)
    assert list(out.index) == list(VOL_ORDER)
    assert out.loc["low", "n_trades"] == 2
    assert out.loc["normal", "n_trades"] == 1
    assert out.loc["high", "n_trades"] == 3
    assert "sharpe" in out.columns
    assert out.loc["low", "win_rate"] == 1.0
    assert out.loc["low", "avg_pnl"] == pytest.approx(0.015)


def test_structure_breakdown_counts() -> None:
    trades = _trades()
    structure = pd.Series(
        ["range", "range", "trend_up", "trend_up", "trend_down", "range"],
        index=pd.to_datetime(trades["entry_ts"]),
    )
    out = structure_breakdown(trades, structure)
    assert list(out.index) == list(STRUCTURE_ORDER)
    assert out.loc["range", "n_trades"] == 3
    assert out.loc["trend_up", "n_trades"] == 2
    assert out.loc["trend_down", "n_trades"] == 1


def test_named_regime_breakdown_has_nine_cells() -> None:
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100.0 + np.linspace(0.0, 5.0, n), index=idx)
    state = named_market_state(close)
    entry = idx[50:56]
    trades = pd.DataFrame({"entry_ts": entry, "ret": np.linspace(-0.01, 0.02, len(entry))})
    out = named_regime_breakdown(trades, state)
    assert list(out.index) == list(COMPOSITE_ORDER)
    assert int(out["n_trades"].sum()) == len(entry)
    assert "sharpe" in out.columns


def test_named_regime_breakdown_rejects_bad_state() -> None:
    trades = _trades()
    bad = pd.DataFrame({"foo": [1]}, index=pd.to_datetime(trades["entry_ts"]))
    with pytest.raises(ValueError, match="structure"):
        named_regime_breakdown(trades, bad)


def test_sharpe_is_mean_over_std_of_trade_returns() -> None:
    trades = pd.DataFrame(
        {
            "entry_ts": pd.date_range("2025-01-01", periods=4, freq="1D"),
            "ret": [0.01, 0.03, -0.01, 0.05],
        }
    )
    vol = pd.Series("low", index=pd.to_datetime(trades["entry_ts"]))
    out = regime_breakdown(trades, vol)
    rets = trades["ret"]
    expected = float(rets.mean() / rets.std(ddof=1))
    assert out.loc["low", "sharpe"] == pytest.approx(expected)
    assert out.loc["normal", "n_trades"] == 0
    assert np.isnan(out.loc["normal", "sharpe"])


def test_mae_mfe_falls_back_to_close_when_entry_price_missing() -> None:
    from scalper_hft.research.session_analysis import mae_mfe_analysis

    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    bars = pd.DataFrame(
        {"open": 100.0, "high": [101, 103, 102, 100, 99], "low": [99, 100, 98, 97, 96], "close": 100.0},
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "entry_ts": [idx[1]],
            "exit_ts": [idx[3]],
            "side": [1],
            "ret": [0.01],
        }
    )
    out = mae_mfe_analysis(trades, bars)
    assert len(out) == 1
    assert out.iloc[0]["mfe"] == pytest.approx(0.03)
    assert out.iloc[0]["mae"] == pytest.approx(-0.03)
