# Kalman vs OLS — протокол bake-off (pairs_arb 1h maker)

Стан: **2026-09-03**. Дефолт лишається `use_kalman=False`.
CLI: `pairs` / `walkforward` / `overfit` мають `--use-kalman`.

## Правило прийняття (не виконано — дефолт не змінено)

Kalman замінює OLS лише якщо на ≥3 з 4 валідованих пар
(XRP/BTC, LINK/BTC, LINK/ETH, BTC/ETH), 1h, maker, зафіксовані z/lookback
зі STRATEGY_STATUS:

- avg OOS Sharpe ≥ OLS
- DSR не гірший

Інакше Kalman лишається опцією `--use-kalman`.

## Протокол прогону

```bash
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT \
  --interval 1h --days 90 --maker --walkforward --train 1500 --test 500
uv run python -m scalper_hft.cli pairs --strategy pairs_arb --leg1 XRPUSDT --leg2 BTCUSDT \
  --interval 1h --days 90 --maker --walkforward --use-kalman --train 1500 --test 500
```

Повторити для LINK/BTC, LINK/ETH, BTC/ETH. Kalman: `q=1e-5`, `r=1e-3`.
Не крутити Q/R у цьому циклі. Не оптимізувати z/lookback на повній вибірці.

## Результат цього циклу

Повний WF/DSR на 4 парах **не проганявся** тут (потрібен локальний кеш klines
і окремий прогін). Код Kalman і тести збіжності вже є
(`tests/test_kalman_pairs.py`). Після прогону дописати таблицю OOS Sharpe/DSR
у цей файл і лише тоді міняти дефолт.
