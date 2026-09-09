"""Use-cases: делегують до router / cell_audit / pairs_runner без I/O в domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.event_engine import EventBacktestResult
from scalper_hft.validation.cell_audit import AuditMode, CellAudit


@dataclass(frozen=True, slots=True)
class RunBacktest:
    strategy: str
    symbol: str
    interval: str
    days: int
    params: dict[str, Any] = field(default_factory=dict)
    maker: bool = False
    trace: bool = False
    base_interval: str = "1m"


@dataclass(frozen=True, slots=True)
class RunCellAudit:
    strategy: str
    symbol: str
    interval: str
    days: int
    mode: AuditMode = "exploratory"
    train_bars: int | None = None
    test_bars: int | None = None
    with_cscv: bool = False
    explicit_holdout: bool = False
    strategy_params: dict[str, Any] | None = None
    purge_bars: int | None = None
    embargo_bars: int | None = None
    n_trials_floor: int | None = None


@dataclass(frozen=True, slots=True)
class RunPairsPaper:
    leg1: str
    leg2: str
    interval: str = "1h"
    strategy_name: str = "pairs_arb"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    require_audit: bool = True
    audit_path: Path | None = None
    store_path: Path | None = None
    control_path: Path | None = None


def run_backtest(req: RunBacktest) -> BacktestResult | EventBacktestResult:
    """Завантажити дані та прогнати бектест через router."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines, ensure_trades_coverage
    from scalper_hft.data.downloader import download_agg_trades, download_funding
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    strategy = get_strategy(req.strategy, **req.params)
    df = ensure_klines(req.symbol, req.interval, req.days, base_interval=req.base_interval, derive=True)
    if strategy.needs_trades:
        ensure_trades_coverage(req.symbol, req.days)
    trades = download_agg_trades(req.symbol, req.days) if strategy.needs_trades else None
    funding = download_funding(req.symbol, req.days) if strategy.needs_funding else None
    cost = CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
    )
    return run_strategy_backtest(
        df,
        strategy,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        is_maker=req.maker,
        trace=req.trace,
        interval=req.interval,
    )


def run_cell_audit(req: RunCellAudit) -> CellAudit:
    """Повний overfitting-аудит комірки."""
    from scalper_hft.validation.cell_audit import audit_cell

    return audit_cell(
        req.strategy,
        req.symbol,
        req.interval,
        req.days,
        mode=req.mode,
        train_bars=req.train_bars,
        test_bars=req.test_bars,
        with_cscv=req.with_cscv,
        explicit_holdout=req.explicit_holdout,
        strategy_params=req.strategy_params,
        purge_bars=req.purge_bars,
        embargo_bars=req.embargo_bars,
        n_trials_floor=req.n_trials_floor,
    )


def run_pairs_paper(req: RunPairsPaper):
    """Створити PairsPaperRunner (paper-only, DRY_RUN=true за замовчуванням)."""
    from scalper_hft.live.pairs_runner import PairsPaperRunner
    from scalper_hft.live.store import PaperStore
    from scalper_hft.strategies import get_strategy

    strategy = get_strategy(req.strategy_name, **req.strategy_params)
    store = PaperStore(req.store_path) if req.store_path else None
    return PairsPaperRunner(
        req.leg1,
        req.leg2,
        interval=req.interval,
        strategy=strategy,
        store=store,
        require_audit=req.require_audit,
        audit_path=req.audit_path,
        control_path=req.control_path,
    )
