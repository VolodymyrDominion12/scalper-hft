"""Live/paper торгівля: виконання через ccxt (Binance USDT-M), ризик-контроль.

Безпека за замовчуванням: DRY_RUN=true (testnet або симуляція). Live —
лише після явного налаштування EXCHANGE=binance та DRY_RUN=false.
Ризик-модель (книга, гл. 4 — Risk Models): ліміт позиції, денний ліміт
збитків, пауза після серії збитків.
"""

from scalper_hft.live.account import PaperAccount
from scalper_hft.live.trader import LiveTrader, run_trader_once

__all__ = ["PaperAccount", "LiveTrader", "run_trader_once"]
