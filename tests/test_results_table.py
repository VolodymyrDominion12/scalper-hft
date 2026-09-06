"""Фільтри та підказки колонок таблиці sweep-результатів."""

from __future__ import annotations

import pandas as pd
from scalper_hft.research.jobs import Job
from scalper_hft.research.results_table import (
    SWEEP_DISPLAY_KEYS,
    default_sort_column,
    display_columns,
    filter_sweep_results,
    is_audit_action,
    job_matches_combo,
    row_actions,
    trade_quality_blockers,
)


def _sample_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["mean_reversion", "cvd_momentum", "supertrend"],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "interval": ["15m", "1h", "15m"],
            "mode": ["backtest", "walkforward", "backtest"],
            "n_trades": [40, 12, 5],
            "sharpe": [1.2, 0.4, -0.1],
            "win_rate": [0.55, 0.48, 0.30],
            "avg_oos_sharpe": [float("nan"), 0.35, float("nan")],
            "status": ["ok", "ok", "error"],
            "error": ["", "", "немає даних"],
        }
    )


def test_filter_sweep_by_strategy_and_ok_only() -> None:
    view = filter_sweep_results(_sample_sweep(), strategies=["mean_reversion", "supertrend"])
    assert list(view["strategy"]) == ["mean_reversion"]


def test_filter_sweep_query_and_mode() -> None:
    view = filter_sweep_results(_sample_sweep(), query="eth", ok_only=False)
    assert list(view["symbol"]) == ["ETHUSDT"]
    wf = filter_sweep_results(_sample_sweep(), mode="walkforward")
    assert len(wf) == 1
    assert wf.iloc[0]["strategy"] == "cvd_momentum"


def test_filter_min_sharpe_and_trades() -> None:
    view = filter_sweep_results(_sample_sweep(), min_trades=20, min_sharpe=1.0)
    assert len(view) == 1
    assert view.iloc[0]["strategy"] == "mean_reversion"


def test_default_sort_prefers_oos() -> None:
    df = _sample_sweep()
    assert default_sort_column(df) == "avg_oos_sharpe"
    df2 = df.drop(columns=["avg_oos_sharpe"])
    assert default_sort_column(df2) == "sharpe"


def test_display_columns_keep_canonical_order() -> None:
    cols = display_columns(_sample_sweep())
    assert cols[0] == "strategy"
    assert "sharpe" in cols
    assert all(c in SWEEP_DISPLAY_KEYS for c in cols)


def test_row_actions_always_details_and_audit() -> None:
    assert row_actions("backtest") == row_actions("walkforward")
    assert any("Деталі" in a for a in row_actions(None))
    assert any("Аудит" in a for a in row_actions("backtest"))
    assert is_audit_action(":material/fact_check: Аудит")
    assert not is_audit_action(":material/candlestick_chart: Деталі")


def test_open_action_cells_are_homogeneous_lists() -> None:
    from scalper_hft.research.results_table import open_action_cells

    cells = open_action_cells(["backtest", "walkforward", "backtest"])
    assert all(isinstance(cell, list) for cell in cells)
    assert len({len(cell) for cell in cells}) == 1
    assert all(all(isinstance(label, str) for label in cell) for cell in cells)


def test_job_matches_combo_optional_days() -> None:
    params = {"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "15m", "days": 60}
    assert job_matches_combo(params, strategy="mean_reversion", symbol="BTCUSDT", interval="15m")
    assert job_matches_combo(params, strategy="mean_reversion", symbol="BTCUSDT", interval="15m", days=60)
    assert not job_matches_combo(params, strategy="mean_reversion", symbol="BTCUSDT", interval="15m", days=90)
    assert not job_matches_combo(params, strategy="cvd_momentum", symbol="BTCUSDT", interval="15m")


def test_trade_quality_blockers_explain_missing_backtest() -> None:
    reasons = trade_quality_blockers(job_status=None, has_artifacts=False, n_trades=0, has_klines=False)
    assert any("Sweep" in r for r in reasons)
    assert any("свічок" in r for r in reasons)


def test_trade_quality_blockers_empty_trades() -> None:
    reasons = trade_quality_blockers(job_status="succeeded", has_artifacts=True, n_trades=0, has_klines=True)
    assert any("немає закритих угод" in r for r in reasons)
    assert len(reasons) == 1


def test_job_matches_from_dataclass() -> None:
    job = Job(
        id=1,
        fingerprint="abc",
        kind="backtest",
        params={"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 30},
        status="succeeded",
    )
    assert job_matches_combo(job.params, strategy="mean_reversion", symbol="BTCUSDT", interval="1h", days=30)
