"""Paper-моніторинг портфеля ts_momentum 1d long-only (iter10).

Рівноважний портфель: капітал ділиться порівну між символами, кожен символ —
окремий PaperRunner на спільній стратегії. Equity портфеля = сума sub-accounts.
Зберігається в `results/paper_ts_momentum.sqlite`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_engine import PairsPaperResult
from scalper_hft.live.pairs_runner import _paper_loop
from scalper_hft.live.paper_runner import PaperRunner
from scalper_hft.live.store import PaperStore
from scalper_hft.strategies import get_strategy

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_PATH = Path("results") / "control.json"
DEFAULT_STORE_PATH = Path("results") / "paper_ts_momentum.sqlite"
PORTFOLIO_ID = "TSMOM_PORTFOLIO"

# Фіксований універсум з pre-registration (docs/reports/hypothesis_ts_momentum.md)
TS_MOMENTUM_PAPER_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "LTCUSDT",
    "AAVEUSDT",
)

# Параметри pre-registration: long-only, lookback=20
TS_MOMENTUM_PAPER_PARAMS: dict[str, Any] = {
    "lookback": 20,
    "allow_short": False,
    "signal_smooth": 1,
}


@dataclass
class TsMomentumPortfolioPaperRunner:
    """Циклічний paper-прогін рівноважного ts_momentum портфеля."""

    symbols: tuple[str, ...] = TS_MOMENTUM_PAPER_SYMBOLS
    interval: str = "1d"
    strategy_params: dict[str, Any] = field(default_factory=lambda: dict(TS_MOMENTUM_PAPER_PARAMS))
    store: PaperStore | None = None
    control_path: Path = DEFAULT_CONTROL_PATH
    initial_capital: float = 10_000.0
    runners: list[PaperRunner] = field(default_factory=list, init=False)
    _n_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError("TsMomentumPortfolioPaperRunner — paper-only. Використовуйте DRY_RUN=true.")
        n = max(len(self.symbols), 1)
        cap_each = self.initial_capital / n
        strategy = get_strategy("ts_momentum", **self.strategy_params)
        self.runners = []
        for sym in self.symbols:
            acc = PaperAccount(
                initial_capital=cap_each,
                taker_fee=settings.taker_fee,
                maker_fee=settings.maker_fee,
            )
            self.runners.append(PaperRunner(strategy, sym, self.interval, account=acc))

    @property
    def total_equity(self) -> float:
        return sum(r.account.equity for r in self.runners)

    @property
    def total_trades(self) -> int:
        return sum(len(r.account.trades) for r in self.runners)

    def step(self) -> str:
        """Один крок: оновити всі символи, записати equity у store."""
        parts: list[str] = []
        for runner in self.runners:
            try:
                action = runner.step()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s step error: %s", runner.symbol, exc)
                action = f"error:{exc}"
            parts.append(f"{runner.symbol}:{action}")
        self._n_steps += 1
        if self.store is not None:
            ts = pd.Timestamp.now(tz="UTC").tz_convert(None)
            self.store.log_equity(ts, PORTFOLIO_ID, self.total_equity, cash=self.total_equity, realized_pnl=0.0)
            for runner in self.runners:
                self.store.log_equity(
                    ts,
                    runner.symbol,
                    runner.account.equity,
                    cash=runner.account.cash,
                    realized_pnl=runner.account.realized_pnl,
                )
        return "; ".join(parts[:3]) + (f"; +{len(parts) - 3} more" if len(parts) > 3 else "")

    def save_runtime(self) -> None:
        if self.store is None:
            return
        payload = {
            "version": 1,
            "portfolio": PORTFOLIO_ID,
            "interval": self.interval,
            "strategy_params": self.strategy_params,
            "symbols": list(self.symbols),
            "total_equity": self.total_equity,
            "runners": {r.symbol: {"equity": r.account.equity, "trades": len(r.account.trades)} for r in self.runners},
        }
        self.store.save_runtime(payload)

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 3600,
        *,
        daemon: bool = False,
    ) -> PairsPaperResult:
        """Цикл paper-моніторингу (1d → sleep 3600с за замовч.)."""

        class _AggAccount:
            equity = 0.0

        agg = _AggAccount()

        def _step() -> str:
            action = self.step()
            agg.equity = self.total_equity
            return action

        def _save() -> None:
            self.save_runtime()

        return _paper_loop(
            _step,
            _save if self.store is not None else None,
            self.interval,
            agg,  # type: ignore[arg-type]
            PORTFOLIO_ID,
            lambda: self.total_trades,
            lambda: 0,
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            control_path=self.control_path,
        )


__all__ = [
    "DEFAULT_CONTROL_PATH",
    "DEFAULT_STORE_PATH",
    "PORTFOLIO_ID",
    "TS_MOMENTUM_PAPER_PARAMS",
    "TS_MOMENTUM_PAPER_SYMBOLS",
    "TsMomentumPortfolioPaperRunner",
]
