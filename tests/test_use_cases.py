"""W0-R6: application.run_backtest takes injected frames and CostModel.from_settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scalper_hft.application import RunBacktest, run_backtest
from scalper_hft.backtest import router as router_mod
from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.event_engine import EventBacktestResult
from scalper_hft.strategies.base import Strategy


def _bars_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )


def _boom(*_a: object, **_k: object) -> None:
    raise AssertionError("use_cases must not load market data")


def test_run_backtest_requires_injected_df() -> None:
    req = RunBacktest("mean_reversion", "BTCUSDT", "1h", 10)
    with pytest.raises(TypeError):
        run_backtest(req)  # type: ignore[call-arg]


def test_run_backtest_uses_injected_df_without_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scalper_hft.data.access.ensure_klines", _boom)
    monkeypatch.setattr("scalper_hft.data.downloader.download_agg_trades", _boom)
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", _boom)

    df = _bars_df()
    captured: dict[str, Any] = {}
    orig = router_mod.run_strategy_backtest

    def _wrap(frame: pd.DataFrame, strategy: Strategy, **kwargs: Any) -> BacktestResult | EventBacktestResult:
        captured["df"] = frame
        captured["cost"] = kwargs.get("cost")
        captured["is_maker"] = kwargs.get("is_maker")
        captured["trace"] = kwargs.get("trace")
        return orig(frame, strategy, **kwargs)

    monkeypatch.setattr(router_mod, "run_strategy_backtest", _wrap)

    req = RunBacktest(
        strategy="mean_reversion",
        symbol="BTCUSDT",
        interval="1h",
        days=20,
        maker=True,
        trace=True,
    )
    res = run_backtest(req, df)

    assert captured["df"] is df
    assert captured["is_maker"] is True
    assert captured["trace"] is True
    cost = captured["cost"]
    assert cost is not None
    assert float(cost.vol_ref) > 0
    assert float(cost.impact_k) == 0.0
    assert len(res.equity) == len(df)


def test_run_backtest_vol_ref_override(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _bars_df()
    captured: dict[str, Any] = {}
    orig = router_mod.run_strategy_backtest

    def _wrap(frame: pd.DataFrame, strategy: Strategy, **kwargs: Any) -> BacktestResult | EventBacktestResult:
        captured["cost"] = kwargs.get("cost")
        return orig(frame, strategy, **kwargs)

    monkeypatch.setattr(router_mod, "run_strategy_backtest", _wrap)

    run_backtest(RunBacktest("mean_reversion", "BTCUSDT", "1h", 20), df, vol_ref=0.042)
    assert captured["cost"].vol_ref == pytest.approx(0.042)
