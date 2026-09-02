"""Спільні константи сторінок дашборду.

SYMBOLS — канонічний універсум інструментів (див. `scalper_hft.symbols`),
той самий, що є fallback-списком `DEFAULT_SYMBOLS` у config. Якщо у .env
задано свій `DEFAULT_SYMBOLS`, CLI/sweep/MCP підуть за ним, а дашборд
показує курований список нижче. PAIR_CHOICES — окремий курований список
(лише перевірені пари для pairs_arb).
"""

from __future__ import annotations

from scalper_hft.symbols import CANONICAL_SYMBOLS

SYMBOLS: list[str] = list(CANONICAL_SYMBOLS)

# Пари для pairs_arb: лише ті, що пройшли коінтеграційний скринінг/валідацію.
PAIR_CHOICES = ["XRPUSDT/BTCUSDT", "BTCUSDT/ETHUSDT", "LINKUSDT/BTCUSDT", "LINKUSDT/ETHUSDT"]
