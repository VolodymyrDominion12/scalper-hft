"""Live/paper торгівля: виконання через ccxt (Binance USDT-M), ризик-контроль.

Безпека за замовчуванням: DRY_RUN=true (testnet або симуляція). Live —
лише після явного налаштування EXCHANGE=binance та DRY_RUN=false.
Ризик-модель (книга, гл. 4 — Risk Models): ліміт позиції, денний ліміт
збитків, пауза після серії збитків.
"""

from scalper_hft.live.account import PaperAccount
from scalper_hft.live.exit_ladders import LadderRung, OneWayTradingLadder
from scalper_hft.live.pairs_runner import PairsPaperRunner, replay_pairs
from scalper_hft.live.telegram_bot import TelegramBotServer, start_bot_thread
from scalper_hft.live.trader import LiveTrader, SilentAttritionKillSwitch, closed_klines, run_trader_once

__all__ = [
    "PaperAccount",
    "LiveTrader",
    "closed_klines",
    "run_trader_once",
    "PairsPaperRunner",
    "replay_pairs",
    "OneWayTradingLadder",
    "LadderRung",
    "SilentAttritionKillSwitch",
    "TelegramBotServer",
    "start_bot_thread",
]
