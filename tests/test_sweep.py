"""Тести матричного прогону sweep: всі стратегії × символи × таймфрейми."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.resample import resample_klines
from scalper_hft.validation.sweep import (
    DEFAULT_INTERVALS,
    MULTI_SYMBOL_STRATEGIES,
    SLOW_STRATEGIES,
    default_strategies,
    run_sweep,
    save_sweep_report,
)


def _make_data(symbol: str, interval: str, days: int):
    """Синтетичний GBM-провайдер даних (без мережі)."""
    rng = np.random.default_rng(42 + len(symbol) + len(interval))  # локальний RNG — thread-safe
    n = 60 * 24 * int(days)
    idx = pd.date_range(end="2024-01-05", periods=n, freq="1min")
    ret = rng.normal(0, 0.0005, n)
    close = 100 * np.exp(np.cumsum(ret))
    df = pd.DataFrame(
        {
            "open": close * (1 - 1e-5),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.abs(rng.normal(10, 2, n)),
        },
        index=idx,
    )
    if interval != "1m":
        df = resample_klines(df, interval, source="1m")
    return df, None, None


def test_default_strategies_exclusions() -> None:
    names = default_strategies()
    assert not (set(names) & MULTI_SYMBOL_STRATEGIES)
    assert not (set(names) & SLOW_STRATEGIES)
    assert "mean_reversion" in names


def test_default_strategies_include_slow() -> None:
    names = default_strategies(include_slow=True)
    assert "ml_strategy" in names and "ensemble" in names


def test_sweep_runs_all_cells() -> None:
    strategies = ["mean_reversion", "cvd_momentum"]
    res = run_sweep(
        strategies=strategies,
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1m", "5m", "1h"],
        days=2,
        data_provider=_make_data,
    )
    assert len(res) == len(strategies) * 2 * 3
    assert (res["status"] == "ok").all()
    cols = {"strategy", "symbol", "interval", "n_bars", "n_trades", "sharpe", "total_return"}
    assert cols.issubset(res.columns)
    # n_bars зменшується зі старшим таймфреймом
    bt = res[res["interval"] == "1m"].iloc[0]
    h1 = res[res["interval"] == "1h"].iloc[0]
    assert bt["n_bars"] > h1["n_bars"]


def test_sweep_parallel_matches_sequential() -> None:
    strategies = ["mean_reversion"]
    seq = run_sweep(
        strategies=strategies,
        symbols=["BTCUSDT"],
        intervals=["1m", "5m"],
        days=2,
        data_provider=_make_data,
        workers=1,
    )
    par = run_sweep(
        strategies=strategies,
        symbols=["BTCUSDT"],
        intervals=["1m", "5m"],
        days=2,
        data_provider=_make_data,
        workers=2,
    )
    sort_cols = [c for c in seq.columns if c not in {"status", "error"}]
    seq_sorted = seq.sort_values(sort_cols).reset_index(drop=True)
    par_sorted = par.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(seq_sorted, par_sorted, check_exact=False)


def test_sweep_walkforward_mode() -> None:
    strategies = ["mean_reversion"]
    res = run_sweep(
        strategies=strategies,
        symbols=["BTCUSDT"],
        intervals=["1m"],
        days=4,
        mode="walkforward",
        train_bars=1000,
        test_bars=250,
        data_provider=_make_data,
    )
    row = res.iloc[0]
    assert row["status"] == "ok"
    assert "avg_oos_sharpe" in res.columns
    assert pd.notna(row["avg_oos_sharpe"])


def test_sweep_isolates_cell_errors() -> None:
    def bad_provider(symbol, interval, days):
        raise RuntimeError("network down")

    res = run_sweep(
        strategies=["mean_reversion"],
        symbols=["BTCUSDT"],
        intervals=["1m"],
        days=2,
        data_provider=bad_provider,
    )
    row = res.iloc[0]
    assert row["status"] == "error"
    assert "network down" in row["error"]


def test_sweep_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError):
        run_sweep(strategies=["nope_strategy"], symbols=["BTCUSDT"], intervals=["1m"], days=1)


def test_sweep_defaults_use_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_SYMBOLS", "AAAUSDT,BBBUSDT")
    # скинути кешований синглтон Settings, щоб підхопити нове оточення
    monkeypatch.setattr("scalper_hft.config._settings", None)
    res = run_sweep(strategies=["mean_reversion"], intervals=["1m"], days=2, data_provider=_make_data)
    assert set(res["symbol"]) == {"AAAUSDT", "BBBUSDT"}
    monkeypatch.setattr("scalper_hft.config._settings", None)  # відновити для інших тестів


def test_save_sweep_report(tmp_path) -> None:
    res = run_sweep(
        strategies=["mean_reversion"],
        symbols=["BTCUSDT"],
        intervals=["1m", "5m"],
        days=2,
        data_provider=_make_data,
    )
    csv_path = tmp_path / "sweep.csv"
    save_sweep_report(res, out_csv=str(csv_path), out_md=str(csv_path.with_suffix(".md")))
    assert csv_path.exists()
    assert csv_path.with_suffix(".md").exists()
    from_csv = pd.read_csv(csv_path)
    assert len(from_csv) == len(res)


def test_default_intervals_include_user_requested() -> None:
    assert {"1m", "5m", "15m", "30m", "1h"} <= set(DEFAULT_INTERVALS)
