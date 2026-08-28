# AGENTS.md — інструкції для AI-агентів у scalper-hft

## Про проєкт
Високочастотна скальпінг-система на Binance USDT-M futures. Python 3.12, uv.
Архітектура за книгою "Inside the Black Box" (Narang): Alpha → Risk → Transaction Cost
→ Portfolio Construction → Execution поверх Data та Research.

## Правила роботи
1. **Ніколи не друкуй і не коміть секрети**. Ключі лише в `.env` (git-ignored).
   Читай значення через `scalper_hft.config.get_settings()`.
2. **Без lookahead**: сигнали використовують дані лише до закриття бару t;
   виконання — з бару t+1 (рушій робить lag сам). Тест на це: `tests/test_system.py::test_engine_no_lookahead`.
3. **Комісії обов'язкові** у бектестах: Binance maker 0.02% / taker 0.05%, slippage з `.env`.
4. **Анти-перенавчання** перед будь-яким висновком про edge:
   walk-forward → deflated Sharpe → sensitivity. Скіл: `skills/overfitting-audit.md`.
5. **Перед запуском live**: `DRY_RUN=true` за замовчуванням. Live — лише за явним запитом
   користувача і після paper-валідації.
6. **Тести**: `uv run pytest tests/ -q` має бути зеленим після будь-яких змін.

## Структура
- `scalper_hft/strategies/` — альфа-моделі (інтерфейс `Strategy`, реєстр у `__init__.py`).
- `scalper_hft/validation/` — walk-forward, purged CV, Deflated Sharpe, sensitivity, Optuna.
- `scalper_hft/data/` — завантаження/кеш (parquet), клієнт Binance (ccxt).
- `docs/` — RESEARCH.md (web-дослідження), book_notes.md (книга), reports/ (звіти).
- `skills/` — процедури для агентів.
- `.mcp/mcp-config.md` — MCP-сервери.

## Типові команди
```bash
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30
uv run python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1m --days 30
uv run python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 30
uv run python -m scalper_hft.cli report --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 60
```
