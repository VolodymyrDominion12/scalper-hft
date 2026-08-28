# Скіл: Аудит стратегії на перенавчання

## Мета
Довести (або спростувати), що edge стратегії не є артефактом підгонки під історію.

## Команда
```bash
.venv/bin/python -m scalper_hft.cli overfit --strategy <name> --symbol BTCUSDT --interval 1m --days 30
.venv/bin/python -m scalper_hft.cli cscv --strategy <name> --symbol BTCUSDT --interval 1m --days 90 --variants 30
.venv/bin/python -m scalper_hft.cli report --strategy <name> --symbol BTCUSDT --interval 5m --days 60
```

## Інтерпретація (пороги за замовчуванням)
1. **Walk-forward**: `avg_oos_sharpe > 0.3` І частка вікон OOS>0 ≥ 50% — генералізує.
2. **Deflated Sharpe**: `DSR > 0.95` — edge значущий після коригування на кількість спроб
   (`n_trials` = комбінації параметрів × бектести). `estimate_n_trials` збільшуй,
   якщо дослідник ітеративно перебирав варіанти.
3. **CSCV/PBO**: `pbo < 0.5` — IS-кращі варіанти стабільно працюють на OOS;
   `pbo > 0.5` — ймовірне перенавчання.
4. **Sensitivity**: `smoothness > 0.3` — плато параметрів; ізольований пік = перенавчання.
5. **Мінімальна кількість угод**: для скальпінгу — не менше ~100 угод на аудит, інакше
   метрики нестабільні (перевір `n_trades` у звіті).

## Правила
- Ніколи не тюнінгуй параметри на тому ж періоді, на якому звітуєш фінальний Sharpe.
- Холдаут (останні 20% даних) — недоторканний до фінального рішення.
- Комісії та slippage обов'язкові: `CostModel(maker, taker, slippage)`; перевір
  `breakeven_move_pct` проти середнього руху цілі.
- Порівнюй з базовим buy&hold того ж періоду.

## Критерій завершення
Звіт у `docs/reports/` з усіма 4 секціями (backtest, WF, DSR, sensitivity) і висновком
"готово до paper" або "потребує доопрацювання".
