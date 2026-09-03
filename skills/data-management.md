# Скіл: Завантаження та управління ринковими даними

## Мета
Завантаження, кешування та валідація даних біржі Binance USDT-M Futures для бектестів та аналітики.

## Команди
```bash
# Завантаження 1m свічок:
.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 60

# Завантаження свічок з тіками угод (aggTrades) для CVD / order flow:
.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 2 --trades

# Запис L1 BookTicker онлайн:
.venv/bin/python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60
```

## Структура кешу
Дані зберігаються у форматі Apache Parquet у каталозі `data/`:
- `data/klines_<SYMBOL>_<INTERVAL>.parquet`
- `data/aggtrades_<SYMBOL>.parquet`
- `data/funding_<SYMBOL>.parquet`
- `data/bookticker_<SYMBOL>_<DATE>.parquet`

## Валідація
- Перевірка рівномірності часових міток без пропусків (gaps).
- Відсутність аномальних спайків і нульових об'ємів.
- Часові мітки нормалізовані до UTC.
