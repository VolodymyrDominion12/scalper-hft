"""Use-cases: делегують до router / cell_audit / pairs_runner без I/O ринку."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scalper_hft.backtest.engine import BacktestResult
from scalper_hft.backtest.event_engine import EventBacktestResult
from scalper_hft.validation.cell_audit import AuditMode, CellAudit

if TYPE_CHECKING:
    import pandas as pd

    from scalper_hft.backtest.micro_price import QueuePositionModel
    from scalper_hft.live.pairs_runner import PairsPaperRunner
    from scalper_hft.overlay.policy import CellPolicy
    from scalper_hft.research.jobs import PruneStats


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
    derive: bool = True


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


def run_backtest(
    req: RunBacktest,
    df: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    overlay: CellPolicy | None = None,
    queue_model: QueuePositionModel | None = None,
    spread_bps: float = 2.0,
    intrabar_exits: bool = False,
    vol_ref: float = 0.0,
) -> BacktestResult | EventBacktestResult:
    """Прогнати бектест через router. Ринкові дані вже завантажив caller (CLI/jobs)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    strategy = get_strategy(req.strategy, **req.params)
    cost = CostModel.from_settings(settings, df=df)
    if vol_ref > 0:
        cost = replace(cost, vol_ref=float(vol_ref))
    return run_strategy_backtest(
        df,
        strategy,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        is_maker=req.maker,
        trace=req.trace,
        overlay=overlay,
        interval=req.interval,
        queue_model=queue_model,
        spread_bps=spread_bps,
        intrabar_exits=intrabar_exits,
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
        derive=req.derive,
    )


def run_pairs_paper(req: RunPairsPaper) -> PairsPaperRunner:
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


def delete_research_jobs(job_ids: Sequence[int], store_path: Path | str | None = None) -> int:
    """Видалити завершені/помилкові задачі разом із їхніми артефактами (папками) на диску."""
    from scalper_hft.research.jobs import JobStore

    with JobStore(store_path) as js:
        return js.delete_jobs(job_ids)


def prune_research_jobs(
    days: int = 14,
    status: str = "all",
    keep_records: bool = False,
    store_path: Path | str | None = None,
) -> PruneStats:
    """Очистити застарілі артефакти задач на диску та з бази."""
    from scalper_hft.research.jobs import JobStore

    with JobStore(store_path) as js:
        return js.prune_jobs(days=days, status=status, keep_records=keep_records)
