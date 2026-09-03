# Скіл: Створення нової стратегії

## Мета
Додати нову альфа-модель у `scalper_hft/strategies/` за єдиним стандартом.

## Кроки
1. Наслідуй `Strategy` з `scalper_hft/strategies/base.py`:
   - `name` — унікальний рядок (латиниця, snake_case);
   - `param_space` — словник `{param: (lo, hi, step)}` для Optuna/grid;
   - `needs_trades` — `True`, якщо потрібні aggTrades (CVD, buy_ratio);
   - `generate_signals(df, trades=None) -> pd.Series` — позиції в {-1, 0, +1}.
2. **Без lookahead**: сигнал на барі t використовує лише дані ≤ t (закриття t).
   Виконання з бару t+1 забезпечує рушій — не компенсуй це всередині стратегії.
3. Патерн утримання позиції (вхід тримається до виходу):
   ```python
   sig = pd.Series(float("nan"), index=df.index, dtype=float)
   sig[entry_long] = 1.0
   sig[entry_short] = -1.0
   prev_pos = sig.ffill().shift(1).fillna(0.0)  # стан ПОПЕРЕДНЬОГО бару
   sig[(prev_pos == 1.0) & exit_long_cond] = 0.0  # вихід лише з позиції
   return sig.ffill().fillna(0.0).astype(int)
   ```
4. Додай стратегію в `REGISTRY` у `scalper_hft/strategies/__init__.py`.
5. Напиши тест у `tests/` (мінімум: сигнал без NaN на хвості, no-lookahead сценарій).

## Критерії завершення
- Стратегія реєструється через `get_strategy(name)`.
- `pytest` зелений.
- Бектест на реальних даних показує коректні угоди (n_trades > 0 при розумних параметрах).
