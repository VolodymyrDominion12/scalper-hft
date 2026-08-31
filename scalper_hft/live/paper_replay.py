"""Paper-replay: відтворення історії через риск-контрольованого трейдера.

Відмінність від звичайного бектесту: replay використовує ТОЙ САМИЙ шар
виконання, що й live (LiveTrader + PaperAccount), включно з:
    - risk-контролем (ліміт позицій, денний ліміт збитків, пауза після серії);
    - комісіями taker на вхід/вихід;
    - funding-платежами для perp-позицій (лонг платить позитивний фандінг);
    - щоденним скиданням day_start_equity.

Це валідує риск-шар на реальній історії та показує, СКІЛЬКИ разів ризик-ліміти
блокували торгівлю (реалістичніша оцінка, ніж чистий бектест без risk-моделі).

Сигнали обчислюються ОДИН раз на весь ряд (без lookahead): сигнал закриття бару
t виконується на барі t+1 — точно як у бектест-рушії.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.config import get_settings
from scalper_hft.live.account import PaperAccount
from scalper_hft.strategies.base import Strategy


@dataclass
class PaperReplayResult:
    equity: pd.Series
    trades: pd.DataFrame
    risk_blocks: list[dict]
    funding_pnl: float
    metrics: BacktestMetrics
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "Paper-replay (risk-контрольований)",
            f"  загальна дохідність: {m.total_return:+.3%}",
            f"  Sharpe (годинний):  {m.sharpe_hourly:+.3f}",
            f"  угод: {m.n_trades} | win rate: {m.win_rate:.0%} | maxDD: {m.max_drawdown:.2%}",
            f"  funding PnL: {self.funding_pnl:+.2f} USDT",
            f"  риск-блокувань: {len(self.risk_blocks)}",
        ]
        kinds: dict[str, int] = {}
        for b in self.risk_blocks:
            kinds[b.get("reason", "?")] = kinds.get(b.get("reason", "?"), 0) + 1
        if kinds:
            lines.append(f"    причини: {kinds}")
        return "\n".join(lines)


def paper_replay(
    df: pd.DataFrame,
    strategy: Strategy,
    funding: pd.DataFrame | None = None,
    account: PaperAccount | None = None,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.01,
    cost: CostModel | None = None,
    decision_every: int = 1,
    symbol: str = "REPLAY",
) -> PaperReplayResult:
    """Відтворення всіх барів df через риск-контрольованого трейдера.

    df: klines (open/high/low/close/volume) у порядку зростання часу.
    funding: DataFrame 'fundingRate' (індекс — час ставки) для funding-платежів.
    decision_every: рішення кожен N-й бар (для 1m можна 1, для 5m теж 1).
    symbol: ключ позиції в PaperAccount (для кількох replay у одному акаунті).
    """
    settings = get_settings()
    account = account or PaperAccount(initial_capital=initial_capital)
    cost = cost or CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
    )
    if len(df) < 50:
        raise ValueError("Замало даних для replay")

    # сигнали один раз на весь ряд (без lookahead: t → виконання t+1)
    if getattr(strategy, "needs_funding", False):
        signals = strategy.generate_signals(df, funding=funding)
    else:
        signals = strategy.generate_signals(df)

    # funding-таймлайн: (ts, rate) — платежі
    funding_events: list[tuple[pd.Timestamp, float]] = []
    if funding is not None and not funding.empty:
        funding_events = list(zip(funding.index, funding["fundingRate"]))

    equity_points: list[tuple[pd.Timestamp, float]] = []
    risk_blocks: list[dict] = []
    funding_pnl = 0.0
    last_day = None
    funding_i = 0

    def _equity(price: float) -> float:
        return account.equity_at({symbol: price})

    for i in range(1, len(df)):
        ts = df.index[i]
        price = float(df["close"].iloc[i])
        account.mark({symbol: price})
        # щоденне скидання day_start_equity
        day = ts.date()
        if last_day is not None and day != last_day:
            account.roll_to_new_day(_equity(price))
        last_day = day

        # funding-платежі за позиції, що тримаються у момент ставки
        while funding_i < len(funding_events) and funding_events[funding_i][0] <= ts:
            fts, frate = funding_events[funding_i]
            if symbol in account.positions:
                pnl = account.apply_funding(symbol, float(frate), fts)
                funding_pnl += pnl
            funding_i += 1

        if i % decision_every != 0:
            equity_points.append((ts, _equity(price)))
            continue

        signal = int(signals.iloc[i - 1]) if i - 1 < len(signals) else 0

        pos = account.positions.get(symbol)
        if signal == 0:
            if pos is not None:
                account.close_position(symbol, price, ts)
        elif signal > 0:
            if pos is not None and pos.side != "long":
                account.close_position(symbol, price, ts)
                pos = None
            if pos is None:
                blocked = _risk_gate(account, settings, price, symbol)
                if blocked:
                    risk_blocks.append({"ts": ts, "reason": blocked, "signal": signal})
                else:
                    size = position_pct * _equity(price) / price
                    account.open_position(symbol, "long", size, price, ts)
        else:
            if pos is not None and pos.side != "short":
                account.close_position(symbol, price, ts)
                pos = None
            if pos is None:
                blocked = _risk_gate(account, settings, price, symbol)
                if blocked:
                    risk_blocks.append({"ts": ts, "reason": blocked, "signal": signal})
                else:
                    size = position_pct * _equity(price) / price
                    account.open_position(symbol, "short", size, price, ts)

        equity_points.append((ts, _equity(price)))

    equity = pd.Series(dict(equity_points)).sort_index()
    trades = pd.DataFrame(account.trades) if account.trades else pd.DataFrame(columns=["type", "symbol"])

    metrics = compute_metrics(equity, exposure=0.0, turnover=0.0)
    return PaperReplayResult(
        equity=equity,
        trades=trades,
        risk_blocks=risk_blocks,
        funding_pnl=funding_pnl,
        metrics=metrics,
        details={"strategy": strategy.name, "symbol": symbol, "bars": len(df)},
    )


def _risk_gate(account: PaperAccount, settings, mark_price: float, symbol: str) -> str | None:
    """Ті самі правила, що в LiveTrader.risk_check; повертає причину блоку або None."""
    account.mark({symbol: mark_price})
    if account.consecutive_losses >= settings.max_consecutive_losses:
        return "max_consecutive_losses"
    if account.equity <= account.day_start_equity * (1 - settings.daily_loss_limit):
        return "daily_loss_limit"
    if len(account.positions) >= settings.max_open_positions:
        return "max_open_positions"
    return None
