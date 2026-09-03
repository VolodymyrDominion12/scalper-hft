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
- `scalper_hft/live/` — paper/live трейдер, pairs runner, reconciliation.
- `scalper_hft/ml/` — triple-barrier, LightGBM, bet sizing, feature importance.
- `scalper_hft/portfolio/` — ERC; risk budget (модуль, не в live-циклі).
- `docs/` — RESEARCH.md (web-дослідження), book_notes.md (книга), reports/ (звіти), DEPLOY_PLAN.md (research локально / робот на VPS).
- `skills/` — процедури для агентів.
- `.mcp/mcp-config.md` — MCP-сервери.

## Типові команди
```bash
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90 --maker
uv run python -m scalper_hft.cli overfit --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli cscv --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli report --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60
```
