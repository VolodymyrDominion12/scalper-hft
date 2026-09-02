"""MAE/MFE forensics, paper-audit, risk gate, experiment catalog, OOD, PSR."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_runner import PairsEngine
from scalper_hft.live.risk_gate import (
    CooldownState,
    correlated_size_mult,
    decide_entry,
    open_pair_size_pcts,
    pairs_share_leg,
)
from scalper_hft.live.store import PaperStore
from scalper_hft.live.trader import LiveTrader, TradeDecision
from scalper_hft.ml.ood import apply_ood_veto, dissimilarity_index, fit_ood_stats, ood_mask
from scalper_hft.strategies.pairs_arb import PairsArb
from scalper_hft.validation.deflated_sharpe import probabilistic_sharpe_ratio
from scalper_hft.validation.experiments import (
    Experiment,
    Verdict,
    ZoneMetrics,
    compare_to_baseline,
    is_test_window_burned,
    parse_catalog,
    qualifies_for_oos,
)
from scalper_hft.validation.forensics import analyze_trades, compute_mfe_mae, excursions_from_bars
from scalper_hft.validation.oos_registry import parse_registry
from scalper_hft.validation.paper_audit import (
    audit_paper_store,
    audit_paper_vs_backtest,
    fill_rate,
    max_drawdown,
)


def test_compute_mfe_mae_long_and_short() -> None:
    mfe, mae = compute_mfe_mae(100.0, 110.0, 95.0, is_short=False)
    assert mfe == pytest.approx(0.10)
    assert mae == pytest.approx(0.05)
    mfe_s, mae_s = compute_mfe_mae(100.0, 110.0, 95.0, is_short=True)
    assert mfe_s == pytest.approx(0.05)
    assert mae_s == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("entry", "high", "low"),
    [(0.0, 1.0, 0.5), (-1.0, 1.0, 0.5), (float("nan"), 1.0, 0.5)],
)
def test_compute_mfe_mae_invalid_entry(entry: float, high: float, low: float) -> None:
    assert compute_mfe_mae(entry, high, low, is_short=False) == (None, None)


def test_analyze_trades_empty() -> None:
    report = analyze_trades(pd.DataFrame())
    assert report.n_closed == 0
    assert report.winrate is None


def test_analyze_trades_entry_fail_vs_roundtrip() -> None:
    idx = pd.date_range("2025-01-06", periods=6, freq="1h")  # Monday
    trades = pd.DataFrame(
        {
            "entry_ts": [idx[0], idx[1], idx[2]],
            "exit_ts": [idx[3], idx[4], idx[5]],
            "side": [1, 1, 1],
            "ret": [0.02, -0.01, -0.01],
            "mfe": [0.04, 0.001, 0.03],
            "mae": [0.005, 0.02, 0.02],
            "pair": ["A/B", "A/B", "C/D"],
            "fill_status": ["filled", "filled", "filled"],
            "exit_reason": ["z_exit", "z_exit", "stop"],
        }
    )
    report = analyze_trades(trades)
    assert report.n_winners == 1
    assert report.n_losers == 2
    assert report.mfe_mae.entry_failures == 1
    assert report.mfe_mae.roundtrip_exits == 1
    assert "A/B" in report.by_pair
    assert report.by_weekday["Пн"].count == 3


def test_excursions_from_bars_long() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    bars = pd.DataFrame(
        {"open": 100.0, "high": [100, 105, 103, 101, 100], "low": [99, 99, 98, 97, 99], "close": 100.0},
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "entry_ts": [idx[0]],
            "exit_ts": [idx[3]],
            "side": [1],
            "ret": [-0.02],
            "entry_price": [100.0],
        }
    )
    out = excursions_from_bars(trades, bars)
    assert float(out.iloc[0]["mfe"]) == pytest.approx(0.05)
    assert float(out.iloc[0]["mae"]) == pytest.approx(0.03)


def test_paper_audit_dd_gate_and_tracking_error(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="1h")
    bt = pd.Series(np.linspace(10_000, 10_100, 10), index=idx)
    paper = bt.copy()
    paper.iloc[-1] = 9_000.0
    audit = audit_paper_vs_backtest(
        paper,
        paper_fill_rate=0.5,
        bt_equity=bt,
        bt_fill_rate=0.8,
        dd_mult=1.5,
    )
    assert audit.dd_gate_ok is False
    assert audit.fill_gap == pytest.approx(-0.3)
    assert audit.tracking_error is not None and audit.tracking_error > 0


def test_paper_audit_from_store(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    ts = pd.Timestamp("2025-01-01")
    store.log_equity(ts, "A/B", 10_000.0, 10_000.0, 0.0)
    store.log_equity(ts + pd.Timedelta(hours=1), "A/B", 10_100.0, 10_100.0, 100.0)
    store.log_order(ts, "A/B", "A", "buy", 1.0, 10.0, "filled", "filled")
    store.log_order(ts, "A/B", "B", "sell", 1.0, 10.0, "unfilled", "no_touch")
    store.log_trade(
        ts,
        "A/B",
        {
            "type": "trade",
            "symbol": "A",
            "side": "long",
            "size": 1.0,
            "entry_price": 10.0,
            "exit_price": 11.0,
            "pnl": 1.0,
        },
    )
    audit = audit_paper_store(store)
    assert audit.n_bars == 2
    assert audit.paper_fill_rate == pytest.approx(0.5)
    assert audit.forensics.n_closed >= 1
    store.close()


def test_fill_rate_empty() -> None:
    assert fill_rate({}) == 0.0
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


@pytest.mark.parametrize(
    ("losses", "max_loss", "expected"),
    [(0, 3, "allow"), (2, 3, "cooldown"), (3, 3, "reject")],
)
def test_decide_entry_cooldown_then_halt(losses: int, max_loss: int, expected: str) -> None:
    now = pd.Timestamp("2025-01-01 12:00")
    d = decide_entry(
        consecutive_losses=losses,
        now=now,
        cooldown=CooldownState(),
        cooldown_losses=2,
        max_consecutive_losses=max_loss,
        cooldown_hours=12.0,
        cooldown_size_mult=0.5,
    )
    assert d.status == expected
    if expected == "cooldown":
        assert d.size_mult == 0.5
        assert d.cooldown.active(now + pd.Timedelta(hours=1))
        assert not d.cooldown.active(now + pd.Timedelta(hours=13))


def test_decide_entry_flatten_never_blocked() -> None:
    d = decide_entry(
        consecutive_losses=10,
        now=pd.Timestamp("2025-01-01"),
        cooldown=CooldownState(),
        max_consecutive_losses=3,
        flattening=True,
    )
    assert d.status == "allow" and d.size_mult == 1.0


def test_correlated_size_mult_shared_leg() -> None:
    assert pairs_share_leg("XRPUSDT/BTCUSDT", "LINKUSDT/BTCUSDT")
    assert not pairs_share_leg("AAA/BBB", "CCC/DDD")
    open_pcts = {"XRPUSDT/BTCUSDT": 0.20}
    assert correlated_size_mult("LINKUSDT/BTCUSDT", 0.20, open_pcts, 0.40) == pytest.approx(1.0)
    assert correlated_size_mult("LINKUSDT/BTCUSDT", 0.20, open_pcts, 0.20) == pytest.approx(0.0)
    assert correlated_size_mult("LINKUSDT/BTCUSDT", 0.20, open_pcts, 0.30) == pytest.approx(0.5)


def test_pairs_engine_halt_after_max_losses() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1)
    eng.max_consecutive_losses = 3
    eng.cooldown_losses = 2
    eng.consecutive_pair_losses = 3
    msg = eng.on_bar(pd.Timestamp("2025-03-01"), 101, 99, 100, 51, 49, 50, signal=1)
    assert "blocked" in msg
    assert acc.is_flat


def test_pairs_engine_corr_cap_blocks_second_pair() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("XXXUSDT/BTCUSDT:XXXUSDT", "long", 1.0, 1.0, ts)
    acc.open_position("XXXUSDT/BTCUSDT:BTCUSDT", "short", 0.01, 100.0, ts)
    eng = PairsEngine("YYYUSDT", "BTCUSDT", PairsArb(lookback=20), acc, wait_bars=1, n_pairs=3)
    eng.corr_notional_cap = 0.20
    eng.size_pct = 0.20
    msg = eng.on_bar(pd.Timestamp("2025-01-02"), 101, 99, 100, 51, 49, 50, signal=1)
    assert "корельований" in msg


def test_live_trader_cooldown_allows_smaller_size() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    acc.consecutive_losses = 2
    trader = LiveTrader(PairsArb(lookback=20), "BTCUSDT", "1h", account=acc)
    trader.settings = SimpleNamespace(
        dry_run=True,
        maker_execution=True,
        position_pct=0.01,
        max_open_positions=2,
        daily_loss_limit=0.99,
        max_consecutive_losses=3,
        cooldown_losses=2,
        cooldown_hours=12.0,
        cooldown_size_mult=0.5,
    )
    decision = TradeDecision(action="open_long", symbol="BTCUSDT", size=2.0)
    allowed, reason = trader.risk_check(decision, mark_price=100.0, now=pd.Timestamp("2025-01-01"))
    assert allowed is True
    assert reason == "cooldown"
    assert trader._entry_size_mult == 0.5


def test_experiment_catalog_gate() -> None:
    text = (
        "| id | strategy | hypothesis | val_start | val_end | test_start | test_end | baseline |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| ml-strategy-v1 | ml_strategy | OOD veto | 2025-01-01 | 2025-06-01 | 2025-07-01 | 2025-12-01 | - |\n"
    )
    rows = parse_catalog(text)
    assert len(rows) == 1
    assert rows[0].strategy == "ml_strategy"
    baseline = ZoneMetrics(trades=30, sharpe=0.4, total_return=0.05)
    better = ZoneMetrics(trades=40, sharpe=0.8, total_return=0.10)
    worse = ZoneMetrics(trades=40, sharpe=0.1, total_return=0.01)
    assert compare_to_baseline(better, baseline) is Verdict.IMPROVED
    assert compare_to_baseline(worse, baseline) is Verdict.REGRESSED
    assert qualifies_for_oos(Verdict.IMPROVED, 40, min_trades=20)
    assert not qualifies_for_oos(Verdict.IMPROVED, 5, min_trades=20)
    assert not qualifies_for_oos(Verdict.REGRESSED, 100, min_trades=20)


def test_experiment_skips_burned_oos() -> None:
    exp = Experiment(
        id="ml-strategy-v1",
        strategy="ml_strategy",
        hypothesis="x",
        val_start=date(2025, 1, 1),
        val_end=date(2025, 6, 1),
        test_start=date(2025, 7, 1),
        test_end=date(2025, 12, 1),
    )
    text = (
        "| strategy | symbol | start | end | purpose |\n"
        "|---|---|---|---|---|\n"
        "| ml_strategy | ml-strategy-v1 | 2025-06-01 | 2025-08-01 | wf |\n"
    )
    assert is_test_window_burned(exp, parse_registry(text))


def test_ood_veto_zeros_far_rows() -> None:
    train = pd.DataFrame({"a": np.zeros(20), "b": np.zeros(20)})
    mean, std = fit_ood_stats(train)
    assert dissimilarity_index(np.array([0.0, 0.0]), mean, std) == pytest.approx(0.0)
    far = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]})
    mask = ood_mask(far, mean, std, threshold=1.0)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    signals = pd.Series([1.0, -1.0], index=far.index)
    out = apply_ood_veto(signals, mask)
    assert float(out.iloc[0]) == 1.0
    assert float(out.iloc[1]) == 0.0


def test_ood_disabled_when_threshold_nonpositive() -> None:
    x = np.ones((4, 2))
    mean, std = fit_ood_stats(x)
    assert ood_mask(x * 100, mean, std, threshold=0.0).all()


def test_probabilistic_sharpe_positive_edge() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.normal(0.0, 0.01, 400)
    edge = rng.normal(0.02, 0.01, 400)
    psr_noisy = probabilistic_sharpe_ratio(noisy)
    psr_edge = probabilistic_sharpe_ratio(edge)
    assert 0.1 < psr_noisy < 0.9
    assert psr_edge > 0.95
    assert probabilistic_sharpe_ratio([0.01]) == 0.0


def test_open_pair_size_pcts_parses_keys() -> None:
    keys = ["XRPUSDT/BTCUSDT:XRPUSDT", "XRPUSDT/BTCUSDT:BTCUSDT"]
    got = open_pair_size_pcts(keys, 0.2)
    assert got == {"XRPUSDT/BTCUSDT": 0.2}


def test_cli_paper_audit_missing_db(tmp_path: Path) -> None:
    from scalper_hft.cli import cmd_paper_audit

    args = SimpleNamespace(db=str(tmp_path / "missing.sqlite"), bt_equity=None, bt_fill_rate=None, dd_mult=1.5)
    with pytest.raises(SystemExit):
        cmd_paper_audit(args)
