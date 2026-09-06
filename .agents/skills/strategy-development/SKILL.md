---
name: strategy-development
description: >-
  Use this skill when developing, implementing, testing, or refactoring trading strategies
  and alpha models for the scalper-hft system. Enforces project architecture standards,
  no-lookahead guarantees, position holding patterns, and registration in REGISTRY.
---

# Скіл: Створення та розробка нової торгової стратегії (Alpha-моделі)

## Мета
Додати або вдосконалити альфа-модель у `scalper_hft/strategies/` за суворими правилами проекту `scalper-hft` (відповідно до книги "Inside the Black Box" Р. Наранга).

---

## 1. Архітектурні вимоги

Кожна стратегія повинна успадковувати базовий клас `Strategy` з `scalper_hft/strategies/base.py`:
- `name: str` — унікальний ідентифікатор у форматі `snake_case` (наприклад, `cvd_momentum`, `pairs_arb`).
- `param_space: dict[str, tuple[float | int, float | int, float | int]]` — простір параметрів у форматі `{"param": (min, max, step)}` для оптимізації Optuna та grid search.
- `needs_trades: bool` — прапорець `True`, якщо стратегія потребує потоку тікових угод `aggTrades` (наприклад, для Cumulative Volume Delta або Order Flow Imbalance).
- `generate_signals(df: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.Series` — основний метод генерування цільових позицій у множині `{-1, 0, 1}`:
  - `+1` — Long
  - `-1` — Short
  - `0` — Flat (вихід у кеш)

---

## 2. Суворе правило NO-LOOKAHEAD (Без заглядання наперед)

> [!IMPORTANT]
> **Золоте правило бектестингу**: Сигнал на барі з індексом $t$ повинен розраховуватися **виключно** на даних, відомих на момент закриття бару $t$ (тобто $\le t$).
> Виконання ордера відбувається на барі $t+1$. Бектест-рушій (`Engine`) самостійно робить зсув (`lag`) на 1 крок:
> ```python
> # engine.py робить:
> executed_position = signals.shift(1)
> ```
> **Не компенсуйте це всередині методу `generate_signals` додатковим зсувом**, якщо розраховуєте закриті бари.

### Патерн утримання позиції (Stateful Position Holding)
Щоб уникнути хибних входів і зберегти стан між барами:
```python
import pandas as pd
import numpy as np


def generate_signals(self, df: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.Series:
    sig = pd.Series(np.nan, index=df.index, dtype=float)

    # Умови входу
    sig[entry_long] = 1.0
    sig[entry_short] = -1.0

    # Позиція на попередньому барі
    prev_pos = sig.ffill().shift(1).fillna(0.0)

    # Умови виходу тільки з активної відповідної позиції
    sig[(prev_pos == 1.0) & exit_long_cond] = 0.0
    sig[(prev_pos == -1.0) & exit_short_cond] = 0.0

    # Заповнюємо стан утримання і повертаємо цілі числа
    return sig.ffill().fillna(0.0).astype(int)
```

---

## 3. Реєстрація стратегії

1. Додайте клас стратегії у `scalper_hft/strategies/<strategy_name>.py`.
2. Експортуйте та зареєструйте у `scalper_hft/strategies/__init__.py`:
```python
from scalper_hft.strategies.<strategy_name> import <StrategyClass>

REGISTRY["<strategy_name>"] = <StrategyClass>
```

---

## 4. Обов'язкові тести

Створіть або оновіть тести у `tests/test_strategies.py` (або виділеному `tests/test_<strategy_name>.py`):
1. **Тест генерації сигналів**: перевірка відсутності `NaN`, коректність значень (`{-1, 0, 1}`).
2. **Тест на відсутність lookahead**: перевірка інваріантності сигналу $t$ до зміни майбутніх цін $t+1 \dots T$.
3. **Запуск тестів**:
```bash
uv run pytest tests/ -k <strategy_name> -v
```

---

## 5. Чекліст завершення
- [ ] Стратегія наслідує `Strategy` та реалізує `param_space` і `generate_signals`.
- [ ] `get_strategy("<strategy_name>")` коректно повертає інстанс стратегії.
- [ ] Жодних lookahead bias у розрахунках індикаторів (rolling вікна мають закритий правий край).
- [ ] Усі юніт-тести проходять успішно: `uv run pytest tests/ -q`.
- [ ] Наступний крок: запуск бектесту (`backtest-run`) та аудит на перенавчання (`overfitting-audit`).
