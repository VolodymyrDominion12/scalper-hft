---
name: data-management
description: >-
  Use this skill when downloading, updating, inspecting, or caching market data for scalper-hft.
  Covers Binance USDT-M Futures klines, aggTrades (CVD/order flow), funding rates, and Parquet cache validation.
---

# Скіл: Управління та валідація ринкових даних

## Мета
Забезпечити швидкий, цілісний та актуальний кеш історичних і потокових ринкових даних (klines, aggTrades, funding rates, L1/L2 book tickers) для бектестингу та моделей машинного навчання.

---

## 1. Завантаження даних через CLI

Система використовує локальний кеш у форматі **Apache Parquet** (`data/`), що забезпечує мікросекундне зчитування та компактне стиснення.

```bash
# Завантаження 1m свічок (найпопулярніший інтервал для скальпінгу):
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 60

# Завантаження свічок з кількома парами:
uv run python -m scalper_hft.cli download --symbol ETHUSDT --interval 1m --days 30
uv run python -m scalper_hft.cli download --symbol XRPUSDT --interval 1h --days 90

# Завантаження тикових угод (aggTrades) для аналізу об'ємів і CVD:
# Увага: Binance обмежує aggTrades глибиною у кілька днів
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 2 --trades

# Запис L1 BookTicker (bid/ask top of the book) у реальному часі:
uv run python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60
```

---

## 2. Структура кешу даних

Кеш зберігається у каталозі `data/`:
- `data/klines_<SYMBOL>_<INTERVAL>.parquet` — свічки OHLCV.
- `data/aggtrades_<SYMBOL>.parquet` — детальні тіки угод покупців/продавців.
- `data/funding_<SYMBOL>.parquet` — історія ставок фінансування (кожні 8 годин).
- `data/bookticker_<SYMBOL>_<DATE>.parquet` — снепшоти найкращого біду та аску.

---

## 3. Перевірка якості та цілісності даних

Перед проведенням бектесту перевірте:
1. **Відсутність прогалин (Gaps)**:
   - Переконайтеся, що часові мітки йдуть суворо з кроком обраного таймфрейму (наприклад, $1\text{m} = 60\,000\text{ ms}$).
2. **Аномальні спайки цін та нульові об'єми**:
   - `high >= low`, `open` та `close` у межах `[low, high]`, `volume >= 0`.
3. **Формат часової зони**:
   - Усі дати нормалізовані до UTC (індекс у форматі `pd.DatetimeIndex` або timestamp ms).

---

## 4. Очищення та скидання кешу
Якщо виникла підозра на пошкоджені файли або неповне завантаження:
```bash
# Перевірити розміри файлів у data/:
ls -lh data/

# Видалити конкретний кеш для перезавантаження:
rm -f data/klines_BTCUSDT_1m.parquet
```
