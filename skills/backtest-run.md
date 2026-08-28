# Скіл: Запуск та інтерпретація бектесту

## Мета
Коректно запустити бектест і зрозуміти, чи результати варті довіри.

## Команди
```bash
.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30
.venv/bin/python -m scalper_hft.cli backtest --strategy cvd_momentum --symbol BTCUSDT --interval 1m --days 2
# параметри стратегії:
.venv/bin/python -m scalper_hft.cli backtest --strategy mean_reversion -p rsi_period=7 -p oversold=40 --days 30
```

## Що дивитись у метриках
- `Win rate` + `Profit factor`: PF > 1.3 і win rate > 40% — прийнятно для скальпінгу.
- `Угод на день`: 0.1 угоди/день = статистично ніщо; для скальпінгу потрібно 5–50+.
- `Turnover` і комісії: `round_trip_taker` ~0.10% — якщо середній трейд менший за це,
  стратегія вмирає на комісіях, навіть якщо сигнал "працює".
- `Макс. просідання` та `P(розорення)`: звертай увагу при розмірі позиції.

## Пастки
- Дані мають бути свіжими (перевір діапазон дат у логах завантаження).
- Сигнал на 1m з агрегованих 5m даних — mismatch; завжди зіставляй інтервали.
- Для стратегій з `needs_trades=True` (cvd_momentum, ob_imbalance) обов'язково
  передавай `--trades` при download (Binance обмежує aggTrades 2 днями).
