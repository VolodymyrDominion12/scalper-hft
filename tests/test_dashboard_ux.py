"""UX-хелпери дашборду: prefill, підписи job, FilterTrace roundtrip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.app_pages._common import (
    RESEARCH_BT_PREFILL,
    RESEARCH_SECTION,
    RESEARCH_SECTIONS,
    apply_research_bt_prefill,
    apply_research_section_prefill,
    capacity_job_payload,
    combo_from_job_params,
    job_label,
    job_open_target,
    job_status_caption,
    single_backtest_payload,
)
from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.metrics import BacktestMetrics
from scalper_hft.research.filter_trace import FilterTrace, SignalEvent
from scalper_hft.research.job_artifacts import (
    load_backtest_result,
    load_capacity_curve,
    save_backtest_result,
    save_capacity_curve,
)
from scalper_hft.research.jobs import Job, JobStore


def test_apply_research_bt_prefill_pair() -> None:
    state: dict[str, object] = {
        RESEARCH_BT_PREFILL: {
            "strategy": "pairs_arb",
            "leg1": "XRPUSDT",
            "leg2": "BTCUSDT",
            "interval": "1h",
            "days": 90,
        }
    }
    apply_research_bt_prefill(state)
    assert state["bt_strategy"] == "pairs_arb"
    assert state["bt_pair"] == "XRPUSDT/BTCUSDT"
    assert state["bt_interval_pairs"] == "1h"
    assert state["bt_days"] == 90


def test_apply_research_section_prefill() -> None:
    state: dict[str, object] = {RESEARCH_SECTION: "Повний цикл"}
    apply_research_section_prefill(state)
    assert state["research_section"] == "Повний цикл"

    state2: dict[str, object] = {RESEARCH_SECTION: "Invalid Tab"}
    apply_research_section_prefill(state2)
    assert "research_section" not in state2


def test_research_sections_cover_ui() -> None:
    assert "Масовий пошук" in RESEARCH_SECTIONS


def test_job_label_pairs_and_sweep() -> None:
    assert "XRPUSDT/BTCUSDT" in job_label("pairs", {"leg1": "XRPUSDT", "leg2": "BTCUSDT", "interval": "1h"})
    assert "2×1×3" in job_label(
        "sweep", {"strategies": ["a", "b"], "symbols": ["BTCUSDT"], "intervals": ["5m", "15m", "1h"]}
    )
    assert "mean_reversion" in job_label(
        "backtest", {"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "15m"}
    )


def test_job_open_target_routes() -> None:
    page, upd = job_open_target(
        "overfit", {"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 30}
    )
    assert page.endswith("research.py")
    assert upd[RESEARCH_SECTION] == "Аудит комірки"
    page_bt, upd_bt = job_open_target(
        "pairs", {"strategy": "pairs_arb", "leg1": "XRPUSDT", "leg2": "BTCUSDT", "interval": "1h", "days": 90}
    )
    assert page_bt.endswith("backtest.py")
    assert upd_bt[RESEARCH_BT_PREFILL]["pair"] == "XRPUSDT/BTCUSDT"
    page_sw, upd_sw = job_open_target("sweep", {"strategies": ["a"], "symbols": ["BTCUSDT"], "intervals": ["1h"]})
    assert upd_sw[RESEARCH_SECTION] == "Sweep matrix"
    assert page_sw.endswith("research.py")


def test_combo_from_job_params_pairs() -> None:
    combo = combo_from_job_params("pairs", {"leg1": "LINKUSDT", "leg2": "ETHUSDT", "interval": "1h", "days": 60})
    assert combo["pair"] == "LINKUSDT/ETHUSDT"
    assert combo["strategy"] == "pairs_arb"


def test_single_backtest_payload_trace_flag() -> None:
    plain = single_backtest_payload("mean_reversion", "BTCUSDT", "15m", 30)
    traced = single_backtest_payload("mean_reversion", "BTCUSDT", "15m", 30, trace=True)
    assert plain["trace"] is False
    assert traced["trace"] is True
    cap = capacity_job_payload("mean_reversion", "BTCUSDT", "1h", 60)
    assert cap["scales"] == [1.0, 2.0, 5.0, 10.0]


def test_job_status_caption() -> None:
    level, msg = job_status_caption(None)
    assert level == "empty"
    job = Job(id=3, fingerprint="abc", kind="backtest", params={}, status="failed", error="boom")
    level, msg = job_status_caption(job)
    assert level == "error"
    assert "boom" in msg


def test_filter_trace_from_dataframe_empty() -> None:
    assert len(FilterTrace.from_dataframe(pd.DataFrame())) == 0
    assert len(FilterTrace.from_dataframe(None)) == 0


def test_filter_trace_roundtrip_dataframe() -> None:
    ts = pd.Timestamp("2025-03-01 12:00")
    src = FilterTrace()
    src.add(
        SignalEvent(
            ts=ts,
            raw_signal=1,
            final_signal=0,
            blocked_by=["vol_ok", "trend_ok"],
            context={"rsi": 72.0},
        )
    )
    src.add(SignalEvent(ts=ts + pd.Timedelta(hours=1), raw_signal=-1, final_signal=-1, blocked_by=[]))
    loaded = FilterTrace.from_dataframe(src.to_dataframe())
    assert len(loaded) == 2
    assert loaded.n_blocked() == 1
    blocked = loaded.blocked_events()[0]
    assert blocked.blocked_by == ["vol_ok", "trend_ok"]
    assert blocked.context["rsi"] == pytest.approx(72.0)


def test_save_load_backtest_includes_trace(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    equity = pd.Series([10000.0, 10010.0, 9990.0, 10020.0, 10015.0], index=idx, name="equity")
    metrics = BacktestMetrics(
        total_return=0.0015,
        cagr=0.1,
        ann_vol=0.2,
        sharpe=1.0,
        sortino=1.1,
        calmar=0.5,
        max_drawdown=0.02,
        win_rate=0.5,
        profit_factor=1.2,
        n_trades=1,
        avg_trade_return=0.001,
        exposure=0.5,
        trades_per_day=1.0,
        turnover=0.1,
        risk_of_ruin=0.0,
    )
    trades = pd.DataFrame({"entry_ts": [idx[0]], "exit_ts": [idx[2]], "ret": [0.01], "side": [1]})
    trace = FilterTrace()
    trace.add(SignalEvent(ts=idx[0], raw_signal=1, final_signal=0, blocked_by=["vol_ok"]))
    res = BacktestResult(equity=equity, positions=pd.Series(dtype=float), trades=trades, metrics=metrics, trace=trace)
    save_backtest_result(tmp_path, res)
    assert (tmp_path / "trace.parquet").exists()
    loaded = load_backtest_result(tmp_path)
    assert loaded.trace is not None
    assert len(loaded.trace) == 1
    assert loaded.trace.n_blocked() == 1


def test_list_jobs_filters_status_and_kind(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.submit("backtest", {"strategy": "a", "days": 1})
    store.submit("sweep", {"strategies": ["a"], "days": 1})
    store.submit("overfit", {"strategy": "a", "days": 1})
    claimed = store.claim()
    assert claimed is not None
    store.finish(claimed.id, "succeeded")
    only_ok = store.list_jobs(status="succeeded")
    assert len(only_ok) == 1
    assert only_ok[0].status == "succeeded"
    only_sweep = store.list_jobs(kind="sweep")
    assert len(only_sweep) == 1
    assert only_sweep[0].kind == "sweep"
    mixed = store.list_jobs(kind=["backtest", "overfit"], status=["queued", "succeeded"])
    kinds = {j.kind for j in mixed}
    assert kinds <= {"backtest", "overfit"}
    assert store.list_jobs(status="failed") == []
    store.close()


def test_save_load_capacity_curve(tmp_path: Path) -> None:
    curve = pd.DataFrame(
        {
            "scale": [1.0, 2.0, 5.0],
            "sharpe": [1.2, 0.9, 0.3],
            "total_return": [0.1, 0.08, 0.01],
            "max_drawdown": [0.05, 0.08, 0.2],
            "impact_bps": [0.0, 2.0, 8.0],
        }
    )
    save_capacity_curve(tmp_path, curve, extra={"saturation_scale": 2.0, "strategy": "mean_reversion"})
    loaded, extra = load_capacity_curve(tmp_path)
    assert extra["saturation_scale"] == pytest.approx(2.0)
    assert list(loaded["scale"]) == [1.0, 2.0, 5.0]


def test_busy_overlay_css_targets_running_widget() -> None:
    from scalper_hft.app_pages._busy import BUSY_OVERLAY_CSS

    assert "stStatusWidgetRunningIcon" in BUSY_OVERLAY_CSS
    assert "Обробка" in BUSY_OVERLAY_CSS
    assert "stHeader" not in BUSY_OVERLAY_CSS
    assert "top: 3.75rem" in BUSY_OVERLAY_CSS


def test_busy_spinner_shows_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from contextlib import nullcontext

    captured: dict[str, object] = {}
    fake = types.ModuleType("streamlit")

    def spinner(text: str, *, show_time: bool = False) -> object:
        captured["text"] = text
        captured["show_time"] = show_time
        return nullcontext()

    fake.spinner = spinner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    from scalper_hft.app_pages._busy import busy

    with busy("Будую графік…"):
        pass
    assert captured["text"] == "Будую графік…"
    assert captured["show_time"] is True


def test_inject_busy_overlay_emits_style(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    html: list[str] = []
    fake = types.ModuleType("streamlit")
    fake.html = lambda body: html.append(body)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    from scalper_hft.app_pages._busy import BUSY_OVERLAY_CSS, inject_busy_overlay

    inject_busy_overlay()
    assert html
    assert "<style>" in html[0]
    assert "stStatusWidgetRunningIcon" in html[0]
    assert BUSY_OVERLAY_CSS.strip() in html[0]


def test_reload_shared_restores_busy_overlay_css() -> None:
    import scalper_hft.app_pages._busy as busy_mod
    from scalper_hft.app_pages import reload_shared

    busy_mod.BUSY_OVERLAY_CSS = "mutated"
    reload_shared()
    import scalper_hft.app_pages._busy as refreshed

    assert "stStatusWidgetRunningIcon" in refreshed.BUSY_OVERLAY_CSS


def test_handle_capacity_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.research.job_handlers import handle_capacity

    canned = pd.DataFrame(
        {
            "scale": [1.0, 2.0],
            "sharpe": [1.0, 0.4],
            "total_return": [0.1, 0.02],
            "max_drawdown": [0.1, 0.2],
            "impact_bps": [0.0, 3.0],
        }
    )
    monkeypatch.setattr("scalper_hft.data.access.ensure_klines", lambda *a, **k: pd.DataFrame({"close": [1.0, 1.1]}))
    monkeypatch.setattr("scalper_hft.data.downloader.download_agg_trades", lambda *a, **k: None)
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", lambda *a, **k: None)
    monkeypatch.setattr("scalper_hft.validation.capacity.capacity_curve", lambda *a, **k: canned)
    monkeypatch.setattr("scalper_hft.validation.capacity.saturation_scale", lambda *a, **k: 1.0)
    job_dir = tmp_path / "cap"
    handle_capacity({"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 7}, job_dir)
    loaded, extra = load_capacity_curve(job_dir)
    assert extra["saturation_scale"] == pytest.approx(1.0)
    assert len(loaded) == 2


def test_apply_research_prefill_max_days_730() -> None:
    from scalper_hft.app_pages._common import apply_shared_research_keys

    state: dict[str, object] = {
        RESEARCH_BT_PREFILL: {
            "strategy": "mean_reversion",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "days": 730,
        }
    }
    apply_research_bt_prefill(state)
    assert state["bt_days"] == 730

    rs_state: dict[str, object] = {}
    apply_shared_research_keys(
        rs_state,
        {"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 730},
    )
    assert rs_state["rs_days"] == 730
