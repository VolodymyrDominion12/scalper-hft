"""Paper-прогін: циклічне виконання стратегії на свіжих даних (симуляція).

На відміну від `paper` (один крок) — `paper-run` крутить цикл:
    fetch свіжих klines/funding → сигнал → виконання через PaperAccount →
    запис equity/угод у results/ → пауза.

Використання:
    python -m scalper_hft.cli paper-run --strategy funding_carry --symbol BTCUSDT \
        --interval 5m --iterations 12 --sleep 300
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.trader import LiveTrader, run_trader_once
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)

_RECENT_BARS = 600  # скільки останніх барів для сигналу


@dataclass
class PaperRunResult:
    equity_points: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    account: PaperAccount | None = None
    symbol: str = ""
    strategy: str = ""

    def summary(self) -> str:
        if not self.equity_points or self.account is None:
            return "Paper-прогін: жодного кроку"
        equity = pd.Series(dict(self.equity_points)).sort_index()
        ret = equity.iloc[-1] / equity.iloc[0] - 1
        return (
            f"Paper-прогін {self.strategy} · {self.symbol}\n"
            f"  кроків: {len(self.equity_points)} | фінальний капітал: {self.account.equity:.2f} "
            f"({ret:+.2%})\n"
            f"  угод: {len(self.account.trades)} | останні дії: {self.actions[-3:] if self.actions else '-'}\n"
            f"  PnL реалізований: {self.account.realized_pnl:+.2f}"
        )


def _fetch_recent(symbol: str, interval: str, limit: int = _RECENT_BARS) -> pd.DataFrame:
    """Останні N свічок напряму через REST (швидко, без повного кешу).

    Увага: `since_ms` мусить бути реальним часом. Історично тут стояло `0`
    (`0 is not None` → ccxt ставить `startTime=0`), і Binance віддавав
    НАЙСТАРІШІ свічки лістингу — див. `recent_since_ms`.
    """
    from scalper_hft.data.client import recent_since_ms

    client = ExchangeClient()  # публічні дані, без ключів
    batch = client.fetch_klines(symbol, interval, since_ms=recent_since_ms(interval, limit), limit=limit)
    if not batch:
        raise RuntimeError(f"Немає даних для {symbol}")
    df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts").sort_index()


class PaperRunner:
    """Циклічний paper-трейдер."""

    def __init__(
        self, strategy: Strategy, symbol: str, interval: str = "5m", account: PaperAccount | None = None
    ) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError(
                "PaperRunner — paper-only: при DRY_RUN=false paper-команди відмовляють. "
                "Live-торгівля single-symbol — лише через свідомий live-запуск, "
                "не через paper/paper-run."
            )
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.account = account or PaperAccount(
            initial_capital=10_000.0,
            taker_fee=settings.taker_fee,
            maker_fee=settings.maker_fee,
        )
        self.trader = LiveTrader(strategy, symbol, interval, account=self.account)

    def step(self) -> str:
        """Один крок: свіжі дані → сигнал на закритому барі → виконання."""
        df = _fetch_recent(self.symbol, self.interval)
        self.trader.maybe_roll_day()
        return run_trader_once(self.trader, df)

    def run(self, iterations: int = 10, sleep_sec: int = 60, out_dir: Path | None = None) -> PaperRunResult:
        """Цикл з iterations кроків і паузою sleep_sec між ними."""
        result = PaperRunResult(account=self.account, symbol=self.symbol, strategy=self.strategy.name)
        out_dir = out_dir or Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(iterations):
                try:
                    action = self.step()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Крок %d помилка: %s", i, exc)
                    action = f"error:{exc}"
                result.actions.append(action)
                result.equity_points.append((pd.Timestamp.utcnow().tz_localize(None), self.account.equity))
                logger.info(
                    "[%d/%d] %s | equity=%.2f | positions=%d",
                    i + 1,
                    iterations,
                    action,
                    self.account.equity,
                    len(self.account.positions),
                )
                if i < iterations - 1:
                    time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("Paper-прогін перервано користувачем (Ctrl+C)")
        finally:
            self.trader.shutdown(reason="runner_stop")
            self._save(out_dir, result)
        return result

    def _save(self, out_dir: Path, result: PaperRunResult) -> None:
        eq = pd.DataFrame(result.equity_points, columns=["ts", "equity"]).set_index("ts")
        eq.to_csv(out_dir / f"paper_equity_{self.symbol}.csv")
        if self.account.trades:
            pd.DataFrame(self.account.trades).to_csv(out_dir / f"paper_trades_{self.symbol}.csv", index=False)
        logger.info("Збережено: %s", out_dir / f"paper_equity_{self.symbol}.csv")
