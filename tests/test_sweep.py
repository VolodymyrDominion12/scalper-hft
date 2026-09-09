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


def test_sweep_warms_agg_trades_once_then_cells_read_store(monkeypatch) -> None:
    """needs_trades стратегії не мають качати REST у кожній клітинці."""
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl
    from scalper_hft.data import store as store_mod

    klines = _make_data("BTCUSDT", "1m", 2)[0]
    trades_calls: list[str] = []
    load_trades_calls: list[str] = []
    sample_trades = pd.DataFrame(
        {"trade_id": [1], "price": [1.0], "amount": [1.0], "side": ["buy"]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-04")]),
    )

    monkeypatch.setattr(acc, "warm_base_cache", lambda *a, **k: {"BTCUSDT": klines})
    monkeypatch.setattr(acc, "can_derive", lambda iv, base: iv != base)
    monkeypatch.setattr(acc, "ensure_klines", lambda *a, **k: klines)

    def fake_download_agg(symbol: str, days: int, **kwargs: object) -> pd.DataFrame:
        trades_calls.append(symbol)
        return sample_trades

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame:
            load_trades_calls.append(symbol)
            return sample_trades

        def load_funding(self, symbol: str) -> None:
            return None

    monkeypatch.setattr(dl, "download_agg_trades", fake_download_agg)
    monkeypatch.setattr(dl, "download_funding", lambda *a, **k: None)
    monkeypatch.setattr(store_mod, "get_store", lambda: _Store())

    res = run_sweep(
        strategies=["cvd_momentum", "mean_reversion"],
        symbols=["BTCUSDT"],
        intervals=["1m", "5m"],
        days=2,
        workers=1,
    )
    assert (res["status"] == "ok").all()
    assert trades_calls == ["BTCUSDT"]
    assert load_trades_calls.count("BTCUSDT") == 2


def test_sweep_resume_skips_ok_cells(tmp_path) -> None:
    from scalper_hft.research.sweep_store import SweepStore

    calls: list[tuple] = []
    inner = _make_data

    def counting_provider(symbol, interval, days):
        calls.append((symbol, interval))
        return inner(symbol, interval, days)

    db = tmp_path / "sweep.db"
    strategies = ["mean_reversion"]
    symbols = ["BTCUSDT"]
    intervals = ["1m", "5m"]
    with SweepStore(db) as store:
        first = run_sweep(
            strategies=strategies,
            symbols=symbols,
            intervals=intervals,
            days=2,
            data_provider=counting_provider,
            store=store,
            resume=True,
        )
    n_first = len(calls)
    assert n_first == 2
    assert (first["status"] == "ok").all()
    calls.clear()
    with SweepStore(db) as store:
        run_sweep(
            strategies=strategies,
            symbols=symbols,
            intervals=intervals,
            days=2,
            data_provider=counting_provider,
            store=store,
            resume=True,
        )
    assert calls == []


def test_default_intervals_include_user_requested() -> None:
    assert {"1m", "5m", "15m", "30m", "1h"} <= set(DEFAULT_INTERVALS)


def test_sweep_store_load_coerces_numeric_columns(tmp_path) -> None:
    from scalper_hft.research.sweep_store import SweepRow, SweepStore

    db = tmp_path / "sweep_test.db"
    with SweepStore(db) as store:
        # Додаємо запис, де OOS-метрики не заповнені (NULL в SQLite)
        row = SweepRow(
            strategy="mean_reversion",
            symbol="BTCUSDT",
            interval="5m",
            days=10,
            mode="backtest",
            sharpe=1.8,
        )
        # явно вказуємо None для OOS колонок
        d = row.as_dict()
        d["avg_oos_sharpe"] = None
        d["oos_positive_frac"] = None
        store._conn.execute(
            """
            INSERT INTO sweep_results (strategy, symbol, interval, days, mode, sharpe, avg_oos_sharpe, oos_positive_frac)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("mean_reversion", "BTCUSDT", "5m", 10, "backtest", 1.8, None, None),
        )
        store._conn.commit()

        loaded = store.load()
        assert not loaded.empty
        # Перевірка що dtype не object, а числовий float64
        assert loaded["avg_oos_sharpe"].dtype == "float64"
        assert loaded["oos_positive_frac"].dtype == "float64"
        assert loaded["sharpe"].dtype == "float64"


def _sweep_df(rows: list[dict]) -> pd.DataFrame:
    base = {"status": "ok", "symbol": "BTCUSDT", "n_trades": 50}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_sweep_winners_haircut_prefers_oos_metric() -> None:
    """Walkforward-режим: haircut рахується по avg_oos_sharpe, не по IS sharpe."""
    from scalper_hft.validation.sweep import sweep_winners_haircut

    df = _sweep_df(
        [
            {"strategy": "a", "interval": "1h", "sharpe": 9.9, "avg_oos_sharpe": 0.5},
            {"strategy": "b", "interval": "1h", "sharpe": 0.1, "avg_oos_sharpe": 1.5},
            {"strategy": "c", "interval": "1h", "sharpe": 0.2, "avg_oos_sharpe": 0.4},
        ]
    )
    w = sweep_winners_haircut(df)
    assert len(w) == 1
    row = w.iloc[0]
    assert row["metric"] == "avg_oos_sharpe"
    assert row["winner_strategy"] == "b"
    assert row["n_candidates"] == 3
    assert row["selection_sr0"] > 0.0
    assert row["deflated_sharpe"] == pytest.approx(row["winner_sharpe"] - row["selection_sr0"])


def test_sweep_winners_haircut_flags_random_maximum() -> None:
    """Якщо переможець не кращий за очікуваний випадковий максимум → survives=False."""
    from scalper_hft.validation.sweep import sweep_winners_haircut

    # 20 кандидатів з близькими Sharpe — max серед них ≈ випадковий максимум
    rng = np.random.default_rng(0)
    sharpes = rng.normal(0.0, 0.3, 20)
    df = _sweep_df([{"strategy": f"s{i}", "interval": "1h", "avg_oos_sharpe": float(s)} for i, s in enumerate(sharpes)])
    w = sweep_winners_haircut(df)
    assert len(w) == 1
    assert bool(w.iloc[0]["survives_haircut"]) is False


def test_sweep_winners_haircut_backtest_metric_and_empty() -> None:
    """Backtest-режим (немає avg_oos_sharpe) → метрика sharpe; порожній df → порожній результат."""
    from scalper_hft.validation.sweep import sweep_winners_haircut

    df = _sweep_df(
        [
            {"strategy": "a", "interval": "5m", "sharpe": 0.5},
            {"strategy": "b", "interval": "5m", "sharpe": 0.9},
        ]
    )
    w = sweep_winners_haircut(df)
    assert w.iloc[0]["metric"] == "sharpe"
    assert w.iloc[0]["winner_strategy"] == "b"

    assert sweep_winners_haircut(pd.DataFrame()).empty


def test_save_sweep_report_includes_haircut_section(tmp_path) -> None:
    """MD-звіт sweep містить секцію переможців зі selection haircut."""
    from scalper_hft.validation.sweep import save_sweep_report

    df = _sweep_df(
        [
            {"strategy": "a", "interval": "1h", "avg_oos_sharpe": 0.5},
            {"strategy": "b", "interval": "1h", "avg_oos_sharpe": 1.5},
        ]
    )
    out_md = tmp_path / "sweep.md"
    save_sweep_report(df, out_md=str(out_md))
    text = out_md.read_text(encoding="utf-8")
    assert "selection haircut" in text
    assert "deflated" in text
