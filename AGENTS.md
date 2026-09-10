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
 walk-forward → deflated Sharpe → sensitivity. Скіл: `.agents/skills/overfitting-audit/SKILL.md`.
5. **Перед запуском live**: `DRY_RUN=true` за замовчуванням. Live — лише за явним запитом
   користувача і після paper-валідації.
6. **Дані — лише з LIVE-біржі**. `DATA_EXCHANGE` (дефолт `binanceusdm`) — окремо від
   торгового `EXCHANGE`. Testnet віддає СИНТЕТИЧНУ історію (ціни розходяться на 1–17%,
   рухи +27% за 1m, «плити» з нульовим обсягом, funding обрізаний до ~13 міс) — на ній
   sweep дає фальшивий edge (`ml_strategy` Sharpe 57 на BNBUSDT). Завантаження з testnet
   заблоковано (`require_live_data_exchange`). Перед будь-яким sweep/аудитом:
   `uv run python -m scalper_hft.cli data-audit --days 1095` (exit code 1 = кеш невалідний).
   Увага: `load_dotenv` НЕ перекриває вже експортовані змінні шелу — перевіряйте `echo $EXCHANGE`.
7. **Тести**: `uv run pytest tests/ -q` має бути зеленим після будь-яких змін.

## Структура
- `scalper_hft/strategies/` — альфа-моделі (інтерфейс `Strategy`, реєстр у `__init__.py`).
- `scalper_hft/validation/` — walk-forward, purged CV, Deflated Sharpe, sensitivity, Optuna.
- `scalper_hft/data/` — завантаження/кеш (parquet), клієнт Binance (ccxt).
- `scalper_hft/live/` — paper/live трейдер, pairs runner, reconciliation.
- `scalper_hft/ml/` — triple-barrier, LightGBM, bet sizing, feature importance.
- `scalper_hft/portfolio/` — ERC; risk budget (модуль, не в live-циклі).
- `scalper_hft/research/` — черга задач (`jobs.py`, worker), sweep store, filter trace.
- `docs/` — RESEARCH.md (web-дослідження), book_notes.md (книга), reports/ (звіти), DEPLOY_PLAN.md (research локально / робот на VPS).
- `skills/` — процедури для агентів.
- `.mcp/mcp-config.md` — MCP-сервери.

## Типові команди
```bash
# 0) ПЕРЕД будь-яким дослідженням — перевірити, що кеш реальний (exit 1 = стоп)
uv run python -m scalper_hft.cli data-audit --days 1095
uv run python scripts/download_live_history.py --days 1095          # live-історія (15 символів)
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90 --maker
uv run python -m scalper_hft.cli overfit --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli cscv --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli report --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
uv run python -m scalper_hft.cli job worker --jobs 2
uv run python -m scalper_hft.cli backtest --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90 --enqueue
uv run python -m scalper_hft.cli job list
uv run python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60
```

## Пастки, на які вже наступали
- **Довгоживучий `job worker` тримає старий код**: після змін у конфігу/коді перезапускати
  воркер, інакше джоби падають на застарілій схемі (так згорів WF-матрикс days=730:
  1259 клітинок з `AttributeError` за 4 хвилини, а джоба записалась як `succeeded`).
- **`overfit` за замовчуванням `exploratory`**; `final` вимагає `HOLDOUT_PCT>0` і
  `OOS_ENFORCE_BURN=true` — інакше джоба падає вже після старту.
- **Клітинки-артефакти**: sweep маркує `status="degenerate"` (0 угод, знищений капітал,
  inf profit_factor, |Sharpe|>20) — вони не потрапляють у рейтинги та haircut.
  `mode=backtest` для `ml_strategy`/`ensemble` — внутрішній walk-forward, для решти
  стратегій — in-sample: порівнювати їх в одній таблиці не можна (haircut групує за mode).
