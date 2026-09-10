"""Research integrity: AuditMode, trades coverage, sweep workers, application layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.audit_gate import audit_gate_check
from scalper_hft.validation.verdict_store import record_verdict


def _klines(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx
    )


def test_audit_cell_final_requires_holdout_and_burn(monkeypatch, tmp_path: Path) -> None:
    from scalper_hft import config as cfg
    from scalper_hft.data import access as acc
    from scalper_hft.validation.cell_audit import audit_cell

    df = _klines(120)
    monkeypatch.setattr(acc, "ensure_klines", lambda *a, **k: df)
    settings = cfg.get_settings()
    import dataclasses as dc

    monkeypatch.setattr(
        cfg,
        "get_settings",
        lambda: dc.replace(
            settings,
            enforce_holdout_pct=0.0,
            enforce_oos_burn=False,
            oos_registry_path=tmp_path / "o.md",
        ),
    )
    audit = audit_cell("mean_reversion", "X", "1h", 10, mode="final")
    assert audit.status == "error"
    assert "HOLDOUT" in (audit.error or "")


def test_exploratory_pass_blocks_live_gate(tmp_path: Path) -> None:
    record_verdict("pairs_arb", "BTCUSDT", "1h", "EXPLORATORY_PASS", "", path=tmp_path / "v.jsonl")
    ok, msg = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=tmp_path / "v.jsonl")
    assert not ok
    assert "EXPLORATORY_PASS" in msg


def test_ensure_trades_coverage_fail_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.data import access as acc

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame | None:
            # Кеш починається 30 днів тому (запит — 90 днів) і хвіст свіжий:
            # саме «кеш починається пізніше» і має впасти fail-fast.
            end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=30)
            idx = pd.date_range(end=end, periods=10, freq="1min")
            return pd.DataFrame({"price": 1.0, "amount": 1.0, "side": "buy"}, index=idx)

    # Патчимо САМЕ той символ, який використовує access.py (`from ... import
    # get_store`): патч `store_mod.get_store` не діяв, і тест непомітно читав
    # реальний data/BTCUSDT_aggTrades.parquet — тобто залежав від стану кеша.
    monkeypatch.setattr(acc, "get_store", lambda: _Store())
    with pytest.raises(RuntimeError, match="починається"):
        acc.ensure_trades_coverage("BTCUSDT", days=90)


def test_ensure_trades_coverage_fail_fast_when_cache_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Порожній кеш aggTrades — теж fail-fast (з підказкою про download/vision)."""
    from scalper_hft.data import access as acc

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame | None:
            return None

    monkeypatch.setattr(acc, "get_store", lambda: _Store())
    with pytest.raises(RuntimeError, match="немає даних"):
        acc.ensure_trades_coverage("BTCUSDT", days=90)


def test_sensitivity_uses_research_slice(monkeypatch) -> None:
    from scalper_hft import config as cfg
    from scalper_hft.strategies.mean_reversion import MeanReversionScalper
    from scalper_hft.validation.sensitivity import parameter_sensitivity

    df = _klines(100)
    import dataclasses as dc

    settings = cfg.get_settings()
    monkeypatch.setattr(cfg, "get_settings", lambda: dc.replace(settings, enforce_holdout_pct=20.0))

    seen: list[int] = []

    def _fake_bt(data, strat, **kw):
        seen.append(len(data))
        from scalper_hft.backtest.engine import BacktestResult
        from scalper_hft.backtest.metrics import BacktestMetrics

        m = BacktestMetrics(
            total_return=0.01,
            cagr=0.01,
            ann_vol=0.1,
            sharpe=0.5,
            sortino=0.5,
            calmar=0.1,
            max_drawdown=-0.05,
            win_rate=0.5,
            profit_factor=1.2,
            n_trades=10,
            avg_trade_return=0.001,
            exposure=0.3,
            trades_per_day=1.0,
            turnover=0.1,
            risk_of_ruin=0.0,
        )
        return BacktestResult(
            equity=pd.Series([1.0, 1.01]),
            positions=pd.Series([0.0, 0.0]),
            trades=pd.DataFrame(),
            metrics=m,
        )

    monkeypatch.setattr("scalper_hft.validation.sensitivity.run_backtest", _fake_bt)
    strat = MeanReversionScalper()
    parameter_sensitivity(df, strat, "rsi_period", [10, 12, 14], research_only=True)
    assert seen and all(n == 80 for n in seen)


def test_resolve_sweep_workers_caps_budget() -> None:
    from scalper_hft.validation.sweep import resolve_sweep_workers

    assert resolve_sweep_workers(32, max_workers=2) == 2
    assert resolve_sweep_workers(0, max_workers=4) == 1


def test_application_run_cell_audit(monkeypatch) -> None:
    from scalper_hft.application import RunCellAudit, run_cell_audit
    from scalper_hft.validation.cell_audit import CellAudit

    canned = CellAudit(symbol="A", interval="1h", strategy="x", status="ok")
    monkeypatch.setattr("scalper_hft.validation.cell_audit.audit_cell", lambda *a, **k: canned)
    out = run_cell_audit(RunCellAudit(strategy="x", symbol="A", interval="1h", days=5, mode="exploratory"))
    assert out.status == "ok"


def test_metrics_collector_json(tmp_path: Path) -> None:
    from scalper_hft.live.metrics import MetricsCollector

    m = MetricsCollector()
    m.record_step_latency(12.5)
    m.record_ws_reconnect()
    m.set_recorder_queue_depth(3)
    out = tmp_path / "m.json"
    m.export_json(out)
    assert out.exists()
    assert "step_latency_ms" in out.read_text(encoding="utf-8")


def test_per_job_cpu_budget() -> None:
    from multiprocessing import cpu_count

    from scalper_hft.research.job_worker import per_job_cpu_budget

    assert per_job_cpu_budget(1) == cpu_count()
    assert per_job_cpu_budget(cpu_count()) == 1
