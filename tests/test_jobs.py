"""Черга дослідницьких задач: fingerprint, submit/claim, worker cancel, артефакти."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.research.job_artifacts import load_backtest_result, load_cell_audit, save_backtest_result
from scalper_hft.research.job_handlers import handle_backtest, handle_overfit, payload_from_backtest_cli, run_job
from scalper_hft.research.job_worker import run_claimed_job
from scalper_hft.research.jobs import JobStore, fingerprint


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite")


def test_fingerprint_stable_and_days_matter() -> None:
    a = {"strategy": "mean_reversion", "symbol": "BTCUSDT", "days": 30, "params": {"rsi": 7}}
    b = {"params": {"rsi": 7}, "days": 30, "symbol": "BTCUSDT", "strategy": "mean_reversion"}
    assert fingerprint("backtest", a) == fingerprint("backtest", b)
    c = {**a, "days": 90}
    assert fingerprint("backtest", a) != fingerprint("backtest", c)


def test_fingerprint_sorts_sweep_lists() -> None:
    p1 = {"strategies": ["b", "a"], "symbols": ["ETHUSDT", "BTCUSDT"]}
    p2 = {"strategies": ["a", "b"], "symbols": ["BTCUSDT", "ETHUSDT"]}
    assert fingerprint("sweep", p1) == fingerprint("sweep", p2)


def test_submit_idempotent_queued_and_succeeded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.submit("backtest", {"strategy": "x", "days": 1})
    second = store.submit("backtest", {"strategy": "x", "days": 1})
    assert first.id == second.id
    claimed = store.claim()
    assert claimed is not None and claimed.id == first.id
    third = store.submit("backtest", {"strategy": "x", "days": 1})
    assert third.id == first.id
    assert third.status == "running"
    store.finish(first.id, "succeeded")
    again = store.submit("backtest", {"strategy": "x", "days": 1})
    assert again.id == first.id
    assert again.status == "succeeded"
    store.close()


def test_submit_force_requeues_succeeded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.submit("backtest", {"strategy": "x", "days": 1})
    claimed = store.claim()
    assert claimed is not None
    art = tmp_path / "jobs" / str(job.id)
    art.mkdir(parents=True)
    (art / "metrics.json").write_text("{}", encoding="utf-8")
    store.finish(job.id, "succeeded")
    forced = store.submit("backtest", {"strategy": "x", "days": 1}, force=True)
    assert forced.id == job.id
    assert forced.status == "queued"
    assert not (art / "metrics.json").exists()
    store.close()


def test_failed_resubmit_requeues(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.submit("k", {"n": 1})
    store.claim()
    store.finish(job.id, "failed", error="boom")
    again = store.submit("k", {"n": 1})
    assert again.id == job.id
    assert again.status == "queued"
    store.close()


def test_claim_two_workers_do_not_share_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.submit("k", {"n": 1})
    got: list[int | None] = []

    def _claim() -> None:
        job = store.claim()
        got.append(job.id if job else None)

    threads = [threading.Thread(target=_claim) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    claimed = [i for i in got if i is not None]
    assert len(claimed) == 1
    assert store.claim() is None
    store.close()


def test_reap_stale_running_becomes_queued(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.submit("k", {"n": 1})
    store.claim()
    store._conn.execute(
        "UPDATE jobs SET heartbeat_at=? WHERE id=?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    n = store.reap_stale(max_age_sec=1)
    assert n == 1
    restored = store.get(job.id)
    assert restored is not None and restored.status == "queued"
    store.close()


def test_cancel_stops_child(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.submit("_test_sleep", {"seconds": 60})
    claimed = store.claim()
    assert claimed is not None
    done = threading.Event()

    def _run() -> None:
        run_claimed_job(store, claimed, mp_context="spawn")
        done.set()

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        cur = store.get(job.id)
        if cur is not None and cur.pid:
            break
        time.sleep(0.05)
    store.request_cancel(job.id)
    t.join(timeout=20)
    assert done.is_set()
    finished = store.get(job.id)
    assert finished is not None
    assert finished.status == "cancelled"
    store.close()


def _synth_klines(n: int = 400) -> pd.DataFrame:
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


def test_handler_backtest_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synth_klines()
    monkeypatch.setattr("scalper_hft.data.access.ensure_klines", lambda *a, **k: df)
    monkeypatch.setattr("scalper_hft.data.downloader.download_agg_trades", lambda *a, **k: None)
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", lambda *a, **k: None)
    job_dir = tmp_path / "art"
    handle_backtest(
        {"strategy": "mean_reversion", "symbol": "BTCUSDT", "interval": "1h", "days": 20},
        job_dir,
    )
    res = load_backtest_result(job_dir)
    assert len(res.equity) > 10
    assert res.metrics.n_trades >= 0
    roundtrip = load_backtest_result(job_dir)
    pd.testing.assert_series_equal(res.equity, roundtrip.equity, check_names=False)


def test_handle_overfit_writes_audit_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.validation.cell_audit import CellAudit

    canned = CellAudit(
        symbol="BTCUSDT",
        interval="1h",
        strategy="mean_reversion",
        status="ok",
        avg_oos_sharpe=0.44,
        oos_pos_frac=0.7,
        dsr=0.96,
        smoothness=0.5,
        bt_n_trades=50,
        windows=({"window_idx": 0, "oos_sharpe": 0.4, "is_sharpe": 0.8, "oos_return": 0.01, "n_trades": 5},),
        sensitivity_grid=({"lookback": 10.0, "metric": 1.2},),
        sens_param="lookback",
    )

    monkeypatch.setattr("scalper_hft.validation.cell_audit.audit_cell", lambda *a, **k: canned)
    job_dir = tmp_path / "overfit"
    handle_overfit(
        {
            "strategy": "mean_reversion",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "days": 30,
            "train_bars": 200,
            "test_bars": 50,
        },
        job_dir,
    )
    loaded = load_cell_audit(job_dir)
    assert loaded.status == "ok"
    assert loaded.avg_oos_sharpe == pytest.approx(0.44)
    assert (job_dir / "audit.json").exists()
    assert (job_dir / "wf_windows.parquet").exists()
    assert (job_dir / "sensitivity.csv").exists()


def test_handle_overfit_raises_on_error_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.validation.cell_audit import CellAudit

    canned = CellAudit(symbol="BTCUSDT", interval="1h", strategy="x", status="error", error="немає даних")
    monkeypatch.setattr("scalper_hft.validation.cell_audit.audit_cell", lambda *a, **k: canned)
    with pytest.raises(RuntimeError, match="немає даних"):
        handle_overfit({"strategy": "x", "symbol": "BTCUSDT", "interval": "1h", "days": 7}, tmp_path / "bad")


def test_run_job_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="невідомий kind"):
        run_job("nope", {}, tmp_path / "x")


def test_save_load_backtest_result_roundtrip(tmp_path: Path) -> None:
    from scalper_hft.backtest.engine import BacktestResult
    from scalper_hft.backtest.metrics import BacktestMetrics

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
        n_trades=2,
        avg_trade_return=0.001,
        exposure=0.5,
        trades_per_day=1.0,
        turnover=0.1,
        risk_of_ruin=0.0,
    )
    trades = pd.DataFrame({"entry_ts": idx[:2], "exit_ts": idx[2:4], "ret": [0.01, -0.005], "side": [1, -1]})
    res = BacktestResult(equity=equity, positions=pd.Series(dtype=float), trades=trades, metrics=metrics)
    save_backtest_result(tmp_path, res)
    loaded = load_backtest_result(tmp_path)
    assert loaded.metrics.n_trades == 2
    assert loaded.metrics.sharpe == pytest.approx(1.0)
    assert len(loaded.trades) == 2


def test_payload_from_cli_namespace() -> None:
    class N:
        strategy = "mean_reversion"
        symbol = "BTCUSDT"
        interval = "5m"
        days = 30
        param_dict = {"rsi_period": 7}
        maker = False
        trace = False
        base = "1m"
        breakeven_gate = False

    p = payload_from_backtest_cli(N())
    assert p["strategy"] == "mean_reversion"
    assert p["params"]["rsi_period"] == 7
    assert fingerprint("backtest", p) == fingerprint("backtest", dict(p))


def test_prune_jobs_age_and_active_protection(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from scalper_hft.research.jobs import artifacts_dir

    store = _store(tmp_path)
    # 1. Завершена стара задача (30 днів тому)
    j1 = store.submit("backtest", {"strategy": "old_strat", "days": 1})
    store.claim()
    store.finish(j1.id, "succeeded")
    old_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    store._conn.execute("UPDATE jobs SET finished_at=? WHERE id=?", (old_ts, j1.id))
    art1 = artifacts_dir(store.path, j1.id)
    art1.mkdir(parents=True)
    (art1 / "equity.parquet").write_text("data", encoding="utf-8")

    # 2. Завершена свіжа задача (1 день тому)
    j2 = store.submit("backtest", {"strategy": "fresh_strat", "days": 1})
    store.claim()
    store.finish(j2.id, "succeeded")
    fresh_ts = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store._conn.execute("UPDATE jobs SET finished_at=? WHERE id=?", (fresh_ts, j2.id))
    art2 = artifacts_dir(store.path, j2.id)
    art2.mkdir(parents=True)
    (art2 / "equity.parquet").write_text("data", encoding="utf-8")

    # 3. Активна задача (running/queued), навіть якщо створена давно
    j3 = store.submit("backtest", {"strategy": "running_strat", "days": 1})
    store.claim()
    store._conn.execute("UPDATE jobs SET created_at=? WHERE id=?", (old_ts, j3.id))

    # Виконуємо prune з порогом 14 днів
    stats = store.prune_jobs(days=14, status="all", dry_run=False)
    assert stats.pruned_jobs == 1
    assert stats.deleted_dirs == 1
    assert stats.freed_bytes > 0

    # j1 має бути видалена
    assert store.get(j1.id) is None
    assert not art1.exists()

    # j2 має залишитись
    assert store.get(j2.id) is not None
    assert art2.exists()

    # j3 (running) має залишитись
    assert store.get(j3.id) is not None
    store.close()


def test_prune_jobs_dry_run_and_keep_records(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from scalper_hft.research.jobs import artifacts_dir

    store = _store(tmp_path)
    j1 = store.submit("backtest", {"strategy": "test_strat", "days": 1})
    store.claim()
    store.finish(j1.id, "failed")
    old_ts = (datetime.now(UTC) - timedelta(days=20)).isoformat()
    store._conn.execute("UPDATE jobs SET finished_at=? WHERE id=?", (old_ts, j1.id))
    art = artifacts_dir(store.path, j1.id)
    art.mkdir(parents=True)
    (art / "metrics.json").write_text("{}", encoding="utf-8")

    # 1. Dry run не змінює нічого
    dry_stats = store.prune_jobs(days=10, dry_run=True)
    assert dry_stats.pruned_jobs == 1
    assert store.get(j1.id) is not None
    assert art.exists()

    # 2. keep_records видаляє папку артефактів, але залишає запис у SQLite
    keep_stats = store.prune_jobs(days=10, keep_records=True, dry_run=False)
    assert keep_stats.pruned_jobs == 1
    assert keep_stats.deleted_dirs == 1
    assert not art.exists()
    assert store.get(j1.id) is not None
    store.close()


def test_prune_jobs_cleans_orphaned_directories(tmp_path: Path) -> None:
    store = _store(tmp_path)
    orphan_dir = tmp_path / "jobs" / "99999"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "garbage.log").write_text("hello", encoding="utf-8")

    stats = store.prune_jobs(days=0, dry_run=False)
    assert stats.orphaned_dirs == 1
    assert not orphan_dir.exists()
    store.close()
