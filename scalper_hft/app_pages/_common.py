"""Спільні константи сторінок дашборду.

SYMBOLS береться з налаштувань (`DEFAULT_SYMBOLS` у .env / config.py), щоб
список інструментів на дашборді не розходився з універсумом CLI, sweep та
MCP-сервера. PAIR_CHOICES — окремий курований список (лише перевірені пари).
"""

from __future__ import annotations

from scalper_hft.config import get_settings

SYMBOLS = list(get_settings().default_symbols)

# Пари для pairs_arb: лише ті, що пройшли коінтеграційний скринінг/валідацію.
PAIR_CHOICES = ["XRPUSDT/BTCUSDT", "BTCUSDT/ETHUSDT", "LINKUSDT/BTCUSDT", "LINKUSDT/ETHUSDT"]
