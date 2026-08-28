# Дизайн scalper-hft

## 1. Мета
Високочастотна скальпінг-система для Binance USDT-M ф'ючерсів: цикл
**дослідження → реалізація → тестування → аудит → покращення** з жорстким
захистом від перенавчання. Цільовий горизонт стратегій — секунди/хвилини
(реалістичний для retail: латентність з дому 50–200 мс, див. docs/RESEARCH.md).

## 2. Архітектура (за книгою "Inside the Black Box", гл. 2)

```
Дані (Binance: klines, aggTrades, funding → parquet)
   │
   ▼
Alpha Model ────┐
Risk Model ─────┼──► Portfolio Construction ──► Execution (backtest/live)
Cost Model ─────┘
   │
   ▼
Research: walk-forward / purged CV / Deflated Sharpe / sensitivity / Optuna / ML
```

Модулі (`scalper_hft/`):
| Модуль | Відповідальність | Ключові файли |
|---|---|---|
| `data` | ccxt REST, parquet-кеш, розширення кешу | `binance_client.py`, `downloader.py`, `storage.py` |
| `features` | індикатори + мікроструктура (CVD, OB imbalance, spread) | `indicators.py`, `regimes.py` |
| `strategies` | альфа-моделі, єдиний інтерфейс `Strategy` | `base.py`, `mean_reversion.py`, `cvd_momentum.py`, `ob_imbalance.py`, `market_maker.py` |
| `backtest` | рушії (векторизований/подієвий), CostModel, метрики | `engine.py`, `event_engine.py`, `execution.py`, `metrics.py` |
| `validation` | анти-перенавчання | `walk_forward.py`, `cv.py`, `deflated_sharpe.py`, `sensitivity.py`, `optimize.py` |
| `ml` | LightGBM walk-forward класифікатор | `features.py`, `trainer.py` |
| `live` | paper/testnet/live трейдер, ризик-контроль | `account.py`, `trader.py` |
| `cli.py` | CLI-інтерфейс | — |

## 3. Ключові рішення

### 3.1. Виконання без lookahead
Сигнал обчислюється на закритті бару t; позиція застосовується з бару t+1
(`pos = signals.shift(1)`). Комісії — за turnover. Тест: `test_engine_no_lookahead`.
Обмеження: рушій не може перешкодити стратегії читати майбутні рядки всередині
`generate_signals` — це відповідальність автора стратегії (див. `skills/strategy-development.md`).

### 3.2. Модель витрат (гл. 5 книги)
`CostModel`: maker_fee + taker_fee + slippage + impact. `breakeven_move_pct`
показує мінімальний рух для покриття round-trip. Binance: maker 0.02%, taker 0.05%.

### 3.3. Анти-перенавчання
- **Walk-forward**: ковзні IS/OOS вікна, головна метрика — avg OOS Sharpe.
- **Purged K-fold CV** (López de Prado): purging + embargo проти автокореляції.
- **Deflated Sharpe (Bailey & LdP)**: коригування на кількість спроб; DSR > 0.95 = значущий edge.
- **Sensitivity**: smoothness параметрів — плато vs ізольований пік.
- **Правила**: оптимізація лише на train+CV; фінальний вердикт — на недоторканому OOS holdout;
  мінімум ~100 угод для висновків.

### 3.4. Стратегії (гл. 3 книги + дослідження)
1. `mean_reversion` — RSI+BB mean reversion (найменш чутлива до slippage, гл. 5).
2. `cvd_momentum` — потік заявок (CVD) + ціновий моментум (fast alpha, гл. 15).
3. `ob_imbalance` — дисбаланс стакана (потребує bookTicker; поки синтетичний imbalance).
4. `market_maker` — пасивні котирування обох сторін (NCMM, гл. 15; експериментальний,
   потребує L2-даних — поточні філи оптимістичні/песимістичні за параметрами).

### 3.5. Live
Paper/testnet за замовчуванням (`DRY_RUN=true`). Ризик-контроль: ліміт позиції,
денний ліміт збитків, пауза після серії збитків. Live — лише явно.

## 4. Відомі обмеження
- **aggTrades**: Binance REST обмежує 2 доби → CVD-фічі доступні лише для свіжого вікна;
  історичні трейди — через data.binance.vision dumps (безкоштовно).
- **bookTicker/L2**: історично не доступні безкоштовно → OB-стратегії потребують
  власного запису (roadmap) або Tardis.dev.
- **Market maker**: спрощена модель філів (без queue position, спайків глибини).
  Для production — nautilus_trader + L2.
- **testnet**: розріджені стакани, філи нереалістичні — лише інтеграційне тестування.

## 5. Roadmap
1. 1s/5s свічки + запис bookTicker (для OB-стратегій на реальних снапшотах).
2. nautilus_trader як L2-рушій (порівняльний аудит філів).
3. asyncio + ccxt.pro live-цикл, maker-ордери (post-only), телеграм-сповіщення.
4. ML: triple-barrier labeling, hmmlearn regime, DSR для ML.
5. Streamlit-дашборд.
