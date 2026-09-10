---
name: backtest-run
description: >-
  Use this skill when running, analyzing, or troubleshooting strategy backtests in scalper-hft.
  Guides CLI parameter setup, Binance commission and slippage accounting, and metric evaluation.
---

# Скіл: Запуск та інтерпретація бектесту

## Мета
Коректно налаштувати і запустити бектест для перевірки торгової гіпотези, враховуючи реальні транзакційні витрати біржі Binance та оцінити якість торгових сигналів.

---

## 1. Команди запуску

### Крок 0 (обов'язковий): перевірити, що кеш — реальний ринок
```bash
# Звірка з LIVE-біржею + неринкові рухи + покриття funding. Exit code 1 → СТОП.
uv run python -m scalper_hft.cli data-audit --days 1095

# Якщо кеш треба наповнити (лише live-біржа, DATA_EXCHANGE):
uv run python scripts/download_live_history.py --days 1095
```
> [!WARNING]
> Кеш, скачаний з Binance **testnet**, проходить формальну валідацію, але містить
> синтетичну історію: ціни розходяться з ринком на 1–17%, трапляються рухи +27% за 1m
> і «плити» `O=H=L=C` з нульовим обсягом, а історія funding обрізана до ~13 міс.
> На таких даних `ml_strategy` на BNBUSDT 5m показав Sharpe 56.9 і +9478% доходності —
> це запам'ятовані стрибки, а не edge. `EXCHANGE` (торгівля) і `DATA_EXCHANGE` (дані) —
> РІЗНІ змінні; `load_dotenv` не перекриває вже експортовані змінні шелу.

### Завантаження даних перед тестом
```bash
# Для звичайних kline-стратегій (1m/5m/1h):
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30

# Для стратегій з аналізом потоку угод (CVD, OB imbalance):
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 2 --trades
```

### Запуск бектесту
```bash
# Базовий бектест:
uv run python -m scalper_hft.cli backtest --strategy cvd_momentum --symbol BTCUSDT --interval 1m --days 2

# Бектест із передачею кастомних параметрів:
uv run python -m scalper_hft.cli backtest --strategy mean_reversion -p rsi_period=7 -p oversold=40 --days 30

# Бектест парного арбітражу:
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --days 90 --maker
```

---

## 2. Модель комісій та витрат (Cost Model)

> [!WARNING]
> Усі бектести **зобов'язані** враховувати реальні комісії Binance USDT-M Futures та slippage:
> - **Maker fee**: 0.02% (`0.0002`)
> - **Taker fee**: 0.05% (`0.0005`)
> - **Round-trip taker**: ~0.10% (`0.0010`)
> - **Slippage**: береться з конфігурації `.env` (`BACKTEST_SLIPPAGE_BPS` / `CostModel`).

Якщо середній PnL на угоду менший за `round_trip_taker`, стратегія є збитковою на реальному ринку через транзакційні витрати.

---

## 3. Ключові метрики та бенчмарки

| Метрика | Мінімальний поріг | Бажане значення | Коментар |
|---|---|---|---|
| **Sharpe Ratio** | > 1.0 (річний) | > 2.0 | Для HFT/скальпінгу очікується високий шарп через короткий горизонт |
| **Profit Factor** | > 1.25 | > 1.50 | Співвідношення валового прибутку до валового збитку |
| **Win Rate** | > 40% (для трендів) / > 55% (mean-rev) | > 50-65% | Залежить від співвідношення Payoff Ratio (Avg Win / Avg Loss) |
| **Угод на день** | $\ge 5$ для 1m-скальпу | 10–50+ | Менше 0.5 угод/день на 1m — недостатня вибірка |
| **Max Drawdown** | < 10% | < 5% | Максимальна просадка депозиту |
| **P(Ruin)** | ~0% | 0% | Ймовірність банкрутства при поточному розмірі ставок |

---

## 4. Типові пастки та діагностика

1. **Невідповідність таймфреймів**: перевірте, щоб індикатори й сигнал розраховувалися на тому ж барі, що й ціни виконання.
2. **AggTrades ліміти**: Binance обмежує глибину завантаження `aggTrades` API кількома днями. Для довших періодів тестуйте на рівні klines або збережених parquet-дампів.
3. **Over-trading**: якщо кількість угод надто висока, а turnover гігантський — комісії з'їдять весь дохід. Використовуйте фільтри волатильності або режим maker orders (`--maker`).
