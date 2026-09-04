"""Збереження/завантаження результатів бектесту поруч із job (parquet + JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.metrics import BacktestMetrics
from scalper_hft.backtest.pairs import PairsResult


def _series_to_frame(s: pd.Series, name: str) -> pd.DataFrame:
    out = s.rename(name).to_frame()
    out.index.name = "ts"
    return out


def _series_from_parquet(path: Path, name: str) -> pd.Series:
    df = pd.read_parquet(path)
    col = name if name in df.columns else df.columns[0]
    s = df[col]
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        s.index = pd.DatetimeIndex(df.index)
    elif "ts" in df.columns:
        s = df.set_index("ts")[col]
        s.index = pd.DatetimeIndex(s.index)
    s.name = name
    return s


def _metrics_from_dict(raw: dict[str, Any]) -> BacktestMetrics:
    allowed = {f.name for f in fields(BacktestMetrics)}
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    return BacktestMetrics(**kwargs)


def save_backtest_result(
    job_dir: Path,
    res: BacktestResult | PairsResult,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Записати metrics.json + equity/trades/positions parquet."""
    job_dir.mkdir(parents=True, exist_ok=True)
    kind = "pairs" if isinstance(res, PairsResult) else "backtest"
    extra = dict(extra or {})
    if isinstance(res, PairsResult):
        extra.setdefault("funding_pnl", float(res.funding_pnl))
        _series_to_frame(res.spread, "spread").to_parquet(job_dir / "spread.parquet")
    meta = {
        "kind": kind,
        "metrics": asdict(res.metrics),
        "params": res.params,
        "extra": extra,
    }
    (job_dir / "metrics.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
    _series_to_frame(res.equity, "equity").to_parquet(job_dir / "equity.parquet")
    if res.positions is not None and len(res.positions):
        _series_to_frame(res.positions, "position").to_parquet(job_dir / "positions.parquet")
    trades = res.trades if res.trades is not None else pd.DataFrame()
    trades.to_parquet(job_dir / "trades.parquet")


def load_backtest_result(job_dir: Path) -> BacktestResult:
    """Відновити BacktestResult для графіка (pairs теж як BacktestResult + extra у params)."""
    meta = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = _metrics_from_dict(meta.get("metrics") or {})
    equity = _series_from_parquet(job_dir / "equity.parquet", "equity")
    pos_path = job_dir / "positions.parquet"
    if pos_path.exists():
        positions = _series_from_parquet(pos_path, "position")
    else:
        positions = pd.Series(dtype=float)
    trades_path = job_dir / "trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    params = dict(meta.get("params") or {})
    params["_artifact_kind"] = meta.get("kind", "backtest")
    params["_extra"] = meta.get("extra") or {}
    return BacktestResult(equity=equity, positions=positions, trades=trades, metrics=metrics, params=params)


def load_pairs_result(job_dir: Path) -> PairsResult:
    """Відновити PairsResult (equity + spread + funding)."""
    bt = load_backtest_result(job_dir)
    extra = bt.params.get("_extra") or {}
    spread_path = job_dir / "spread.parquet"
    if spread_path.exists():
        spread = _series_from_parquet(spread_path, "spread")
    else:
        spread = pd.Series(dtype=float)
    return PairsResult(
        equity=bt.equity,
        positions=bt.positions,
        spread=spread,
        funding_pnl=float(extra.get("funding_pnl", 0.0)),
        metrics=bt.metrics,
        params=bt.params,
        trades=bt.trades,
    )


def artifact_kind(job_dir: Path) -> str | None:
    path = job_dir / "metrics.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    return str(meta.get("kind") or "backtest")
