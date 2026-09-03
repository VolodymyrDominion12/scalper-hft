"""Дослідницький бриф дашборду: свіжість кешу, paper KPI, hurdle, книга стратегій."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.live.store import PaperStore
from scalper_hft.research.dashboard_brief import (
    cache_inventory,
    classify_staleness,
    compute_paper_kpis,
    dsr_verdict,
    fill_rate,
    format_age_hours,
    hurdle_note,
    inventory_kpis,
    klines_interval_from_name,
    paper_pair_table,
    peek_parquet,
    sweep_highlights,
)
from scalper_hft.research.strategy_book import lane_for, select_label


def test_classify_staleness_thresholds() -> None:
    now = pd.Timestamp("2026-09-03 12:00")
    assert classify_staleness(None, now) == "missing"
    assert classify_staleness(now + pd.Timedelta(hours=1), now) == "future"
    assert classify_staleness(now - pd.Timedelta(minutes=30), now) == "fresh"
    assert classify_staleness(now - pd.Timedelta(hours=6), now) == "aging"
    assert classify_staleness(now - pd.Timedelta(hours=30), now) == "stale"


@pytest.mark.parametrize(
    ("hours", "label"),
    [
        (float("nan"), "—"),
        (-1.0, "майбутнє"),
        (0.5, "30 хв"),
        (3.0, "3.0 год"),
        (72.0, "3.0 дн"),
    ],
)
def test_format_age_hours(hours: float, label: str) -> None:
    assert format_age_hours(hours) == label


def test_klines_interval_from_name_skips_spot() -> None:
    assert klines_interval_from_name("BTCUSDT", Path("BTCUSDT_1h_klines.parquet")) == "1h"
    assert klines_interval_from_name("BTCUSDT", Path("BTCUSDT_1h_spot_klines.parquet")) is None
    assert klines_interval_from_name("ETHUSDT", Path("BTCUSDT_1m_klines.parquet")) is None


def test_peek_and_inventory_from_tmp_parquet(tmp_path: Path) -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="1min")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        },
        index=idx,
    )
    path = tmp_path / "BTCUSDT_1m_klines.parquet"
    df.to_parquet(path)
    peek = peek_parquet(path)
    assert peek.n_rows == 10
    assert peek.start == idx[0]
    assert peek.end == idx[-1]

    now = pd.Timestamp("2026-01-01 00:30")
    inv = cache_inventory(tmp_path, ["BTCUSDT", "ETHUSDT"], now=now)
    assert list(inv["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    btc = inv.iloc[0]
    assert btc["klines_1m"] == 10
    assert btc["freshness"] == "fresh"
    assert "1m" in str(btc["intervals"])
    eth = inv.iloc[1]
    assert eth["klines_1m"] == 0
    assert eth["freshness"] == "missing"
    kpis = inventory_kpis(inv)
    assert kpis["symbols"] == 2
    assert kpis["with_1m"] == 1
    assert kpis["missing"] == 1
    assert kpis["klines_1m"] == 10


def test_fill_rate_and_paper_kpis() -> None:
    assert fill_rate(0, 0) != fill_rate(0, 0)  # NaN
    assert fill_rate(8, 2) == pytest.approx(0.8)
    eq = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "pair": ["A/B", "A/B", "C/D", "C/D"],
            "equity": [10_000.0, 10_100.0, 10_000.0, 9_900.0],
        }
    )
    kpis = compute_paper_kpis({"filled": 8, "unfilled": 2, "pending": 1}, eq)
    assert kpis.fill_rate == pytest.approx(0.8)
    assert kpis.last_equity_by_pair["A/B"] == 10_100.0
    assert kpis.last_equity_by_pair["C/D"] == 9_900.0
    assert len(kpis.sparkline) >= 2
    table = paper_pair_table(eq)
    assert set(table["pair"]) == {"A/B", "C/D"}
    a = table.set_index("pair").loc["A/B"]
    assert a["return"] == pytest.approx(0.01)


def test_empty_paper_kpis() -> None:
    kpis = compute_paper_kpis({}, pd.DataFrame())
    assert kpis.filled == 0
    assert kpis.fill_rate != kpis.fill_rate
    assert paper_pair_table(pd.DataFrame()).empty


def test_hurdle_and_dsr() -> None:
    cost = CostModel(maker_fee=0.0002, slippage_frac=0.0)
    note = hurdle_note(0.001, cost.round_trip_maker(), n_legs=2)
    assert "8.0 bps" in note
    assert "додатне" in note
    drag = hurdle_note(-0.01, cost.round_trip_maker(), n_legs=1)
    assert "fee-drag" in drag
    assert dsr_verdict(0.99) == "significant"
    assert dsr_verdict(0.7) == "weak"
    assert dsr_verdict(0.1) == "none"
    assert dsr_verdict(float("nan")) == "none"


def test_sweep_highlights_prefers_oos() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["mean_reversion", "pairs_arb", "cvd_momentum"],
            "symbol": ["BTCUSDT", "XRPUSDT", "ETHUSDT"],
            "interval": ["5m", "1h", "1m"],
            "status": ["ok", "ok", "error"],
            "sharpe": [2.0, 0.8, 9.0],
            "avg_oos_sharpe": [-0.2, 1.1, 5.0],
            "n_trades": [10, 28, 3],
        }
    )
    top = sweep_highlights(df, top_n=1)
    assert len(top) == 1
    assert top.iloc[0]["strategy"] == "pairs_arb"
    assert sweep_highlights(pd.DataFrame()).empty


def test_strategy_book_pairs_arb_validated() -> None:
    assert lane_for("pairs_arb") == "validated"
    assert lane_for("mean_reversion") == "rejected"
    assert lane_for("unknown_alpha") == "research"
    assert "валідована" in select_label("pairs_arb")
    assert "відхилена" in select_label("mean_reversion")


def test_paper_store_all_months(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    store.log_month("XRPUSDT/BTCUSDT", "2026-08", 12.5)
    store.log_month("XRPUSDT/BTCUSDT", "2026-09", -3.0)
    months = store.all_months()
    store.close()
    assert list(months["month"]) == ["2026-08", "2026-09"]
    assert months["pnl"].tolist() == [12.5, -3.0]
