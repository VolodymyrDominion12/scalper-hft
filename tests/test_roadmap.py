"""Спринти 5–8: pairs live-path, ML purge/CUSUM, дані, книжкові тести."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.backtest.router import EVENT_STRATEGIES, run_strategy_backtest
from scalper_hft.data.validate import validate_bars
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import decide_fill, fill_probability
from scalper_hft.live.is_log import IsJournal, shortfall_bps
from scalper_hft.live.orders import next_client_order_id
from scalper_hft.live.reconcile import (
    ExchangePosition,
    KillSwitch,
    halt_if_drift,
    reconcile_exchange_state,
    reconcile_positions,
)
from scalper_hft.ml.labeling import add_vertical_barrier, get_t_events
from scalper_hft.ml.trainer import _purge_train_slice, _simulate_sharpe
from scalper_hft.validation.coint_scan import scan_pairs
from scalper_hft.validation.hedge_ratio import compare_hedge_oos, rolling_ols_beta
from scalper_hft.validation.oos_registry import is_burned, parse_registry
from scalper_hft.validation.quintile import quintile_spread_study
from scalper_hft.validation.spread_breaks import csw_cusum, rank_corr
from scalper_hft.validation.time_decay import pairs_time_decay


def _series(n: int = 200, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=idx)


def test_fill_probability_at_mid_is_one() -> None:
    assert fill_probability("buy", 100.0, 100.0) == 1.0
    assert fill_probability("sell", 100.0, 100.0) == 1.0
    assert fill_probability("buy", 99.0, 100.0) < 1.0


def test_decide_fill_touch_compat() -> None:
    hit = decide_fill("buy", 100.0, high=101.0, low=99.0)
    miss = decide_fill("buy", 100.0, high=102.0, low=100.5)
    assert hit.filled
    assert not miss.filled


def test_client_order_id_unique() -> None:
    a, b = next_client_order_id(), next_client_order_id()
    assert a != b and a.startswith("sh-") and len(a) <= 36


def test_reconcile_detects_drift() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("BTCUSDT", "long", 1.0, 100.0, ts)
    ok, _ = reconcile_positions(acc, {"BTCUSDT": ExchangePosition("BTCUSDT", "long", 1.0)})
    assert ok
    bad, reason = reconcile_positions(acc, {"BTCUSDT": ExchangePosition("BTCUSDT", "short", 1.0)})
    assert not bad and "short" in reason


def test_halt_if_drift_paper_noop() -> None:
    acc = PaperAccount(10_000.0)
    halt_if_drift(acc, [{"symbol": "BTCUSDT", "side": "long", "contracts": 1}], dry_run=True)


def test_halt_if_drift_live_raises() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    acc.open_position("BTCUSDT", "long", 1.0, 100.0, pd.Timestamp("2025-01-01"))
    try:
        halt_if_drift(acc, [], dry_run=False)
    except KillSwitch:
        return
    raise AssertionError("очікували KillSwitch")


def test_reconcile_exchange_state_paper_skips_fetch() -> None:
    class _C:
        n = 0

        def fetch_positions(self, symbols: list | None = None) -> list:
            self.n += 1
            return []

    client = _C()
    reconcile_exchange_state(PaperAccount(10_000.0), client, dry_run=True)
    assert client.n == 0


def test_reconcile_exchange_state_live_calls_fetch() -> None:
    class _C:
        n = 0

        def fetch_positions(self, symbols: list | None = None) -> list:
            self.n += 1
            return []

    client = _C()
    reconcile_exchange_state(PaperAccount(10_000.0), client, dry_run=False)
    assert client.n == 1


def test_is_shortfall_sign() -> None:
    assert shortfall_bps("buy", 100.0, 100.1) > 0
    assert shortfall_bps("sell", 100.0, 99.9) > 0
    j = IsJournal()
    j.log(pd.Timestamp("2025-01-01"), "A/B", "A", "buy", 100.0, 100.0)
    assert j.mean_bps() == 0.0


def test_ols_hedge_and_compare() -> None:
    y = _series(180, 1)
    x = y * 0.5 + 10
    beta = rolling_ols_beta(np.log(y), np.log(x), 40)
    assert beta.notna().sum() > 10
    res = compare_hedge_oos(y, x, lookback=40, train_frac=0.6)
    assert res.prefer in {"logratio", "ols", "johansen"}


def test_cusum_events_sparser_than_all_bars() -> None:
    close = _series(300, 2)
    ev = get_t_events(close, threshold=0.01)
    assert 0 < len(ev) < len(close)


def test_purge_drops_overlapping_labels() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="1h")
    t1 = pd.Series(idx + pd.Timedelta(hours=5), index=idx)
    keep = _purge_train_slice(idx, t1, 10, 15)
    assert keep.max() < 10
    assert all(t1.iloc[i] <= idx[10] for i in keep)


def test_fee_aware_sharpe_worse_than_gross() -> None:
    idx = pd.date_range("2025-01-01", periods=50, freq="1h")
    close = pd.Series(np.linspace(100, 110, 50), index=idx)
    preds = pd.Series(1, index=idx)
    from scalper_hft.backtest.execution import CostModel

    gross = _simulate_sharpe(preds, close, cost=CostModel(taker_fee=0.0, maker_fee=0.0, slippage_frac=0.0))
    net = _simulate_sharpe(preds, close, cost=CostModel(taker_fee=0.001, maker_fee=0.0, slippage_frac=0.0))
    assert net < gross


def test_validate_bars_catches_ohlc() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    good = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.2, "volume": 1.0},
        index=idx,
    )
    assert validate_bars(good, interval="1h").ok
    bad = good.copy()
    bad.loc[idx[0], "high"] = 0.1
    assert not validate_bars(bad).ok


def test_attach_imbalance_and_router_names() -> None:
    from scalper_hft.data.research import attach_imbalance

    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    klines = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    book = pd.DataFrame(
        {"bid": [99, 99, 99], "ask": [101, 101, 101], "bid_qty": [2, 2, 2], "ask_qty": [1, 1, 1]},
        index=idx,
    )
    out = attach_imbalance(klines, book)
    assert "imbalance" in out.columns
    assert abs(float(out["imbalance"].iloc[-1]) - (1 / 3)) < 1e-9
    # event-рушій лишається лише для пасивного MM; ob_imbalance — напрямкова
    # стратегія з власним generate_signals → векторний рушій (див. router.py)
    assert "market_maker" in EVENT_STRATEGIES
    assert "ob_imbalance" not in EVENT_STRATEGIES


def test_router_vector_path() -> None:
    from tests.test_system import AlwaysLong, make_klines

    df = make_klines(80)
    res = run_strategy_backtest(df, AlwaysLong(), position_pct=0.1)
    assert len(res.equity) == len(df)


def test_pairs_time_decay_and_quintile() -> None:
    z = _series(200, 3)
    z = (z - z.mean()) / z.std()
    fwd = -z.diff().shift(-1)
    q = quintile_spread_study(z, fwd)
    assert len(q.means) >= 3
    sig = pd.Series(np.sign(z), index=z.index)
    td = pairs_time_decay(sig, z, max_lag=2)
    assert td.lags == [0, 1, 2]


def test_csw_and_rank_corr() -> None:
    s = _series(120, 4)
    br = csw_cusum(s, threshold=8.0, warmup=20)
    assert br.statistic_max >= 0
    rc = rank_corr(s, s)
    assert rc["spearman"] > 0.9


def test_coint_scan_self_pair_skipped() -> None:
    a, b = _series(120, 5), _series(120, 6)
    rows = scan_pairs({"AAA": a, "BBB": b})
    assert len(rows) == 1
    assert rows[0].leg1 == "AAA"


def test_oos_registry_overlap(tmp_path: Path) -> None:
    text = (
        "| strategy | symbol | start | end | purpose |\n"
        "|---|---|---|---|---|\n"
        "| pairs_arb | X/Y | 2025-01-01 | 2025-06-01 | wf |\n"
    )
    windows = parse_registry(text)
    assert is_burned(windows, "pairs_arb", "X/Y", date(2025, 5, 1), date(2025, 7, 1))
    assert not is_burned(windows, "pairs_arb", "X/Y", date(2025, 7, 1), date(2025, 8, 1))


def test_vertical_barrier_bars_still() -> None:
    close = _series(40)
    t1 = add_vertical_barrier(close.index[:10], close, num_days=3)
    hours = (t1.values - t1.index.values).astype("timedelta64[h]").astype(int)
    assert set(hours.tolist()) == {3}
