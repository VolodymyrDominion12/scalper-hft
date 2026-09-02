"""Канонічний універсум інструментів (symbols) scalper-hft.

Модуль навмисно **stdlib-only** (жодних імпортів поза стандартною
бібліотекою), щоб його можна було безпечно імпортувати зі сторінок
дашборду в середовищі без встановленого пакета (тест
`tests/test_dashboard.py::test_streamlit_style_path_can_import_package`).

Єдине джерело істини для списку «за замовчуванням»:
- `config.Settings.default_symbols` використовує його як fallback, коли
  змінна оточення `DEFAULT_SYMBOLS` не задана (а CLI/sweep/MCP поважають
  перевизначення через `DEFAULT_SYMBOLS` у .env);
- сторінки дашборду (`app_pages/_common.py`) показують саме цей список.
"""

from __future__ import annotations

# 15 ліквідних USDT-M перпетуалів Binance (співпадає з DEFAULT_SYMBOLS у .env).
# Порядок = порядок пріоритету на дашборді.
CANONICAL_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "LTCUSDT",
    "AAVEUSDT",
)

__all__ = ["CANONICAL_SYMBOLS"]
