---
name: overfitting-audit
description: >-
  Use this skill when auditing trading strategies for data-snooping bias and overfitting.
  Executes walk-forward analysis, Deflated Sharpe Ratio (DSR), Combinatorially Symmetric Cross-Validation (CSCV/PBO),
  and parameter sensitivity tests before approving an alpha model for paper or live trading.
---

# Скіл: Комплексний аудит стратегії на перенавчання (Anti-Overfitting Audit)

## Мета
Довести математично та статистично, що знайдений edge (перевага) стратегії не є артефактом підгонки під історію (data mining bias / p-hacking), згідно з методологією Маркоса Лопеса де Прадо (*Advances in Financial Machine Learning*).

---

## 1. Команди запуску аудиту

```bash
# 1. Загальний аудит (Walk-Forward + Deflated Sharpe + Sensitivity):
uv run python -m scalper_hft.cli overfit --strategy <name> --symbol BTCUSDT --interval 1m --days 30

# 2. CSCV (Combinatorially Symmetric Cross-Validation) та розрахунок PBO:
uv run python -m scalper_hft.cli cscv --strategy <name> --symbol BTCUSDT --interval 1m --days 90 --variants 30

# 3. Комплексний підсумковий звіт:
uv run python -m scalper_hft.cli report --strategy <name> --symbol BTCUSDT --interval 5m --days 60
```

---

## 2. Стандарти та критерії валідації

Стратегія вважається надійною і допускається до фази paper trading **тільки** за виконання таких умов:

| Перевірка | Метрика | Критерій допуску | Що це означає |
|---|---|---|---|
| **Walk-Forward** | `avg_oos_sharpe` | **> 0.3** | Середній Sharpe на позавибіркових (OOS) вікнах вище порогу |
| **Walk-Forward** | `pct_oos_positive` | **$\ge$ 50%** | Більшість позавибіркових періодів є прибутковими |
| **Deflated Sharpe** | `DSR` | **> 0.95** | Sharpe є статистично значущим після врахування кількості спроб підбору (`n_trials`) |
| **CSCV / PBO** | `PBO` (Prob. of Backtest Overfitting) | **< 0.50** (ідеально < 0.25) | Кращі конфігурації на In-Sample не втрачають перевагу на Out-of-Sample |
| **Sensitivity** | `smoothness` | **> 0.30** | Поверхня параметрів утворює плато (стабільність), а не поодинокий гострий пік |
| **Статистична вибірка** | `n_trades` | **$\ge$ 100 угод** (для 1m) | Для високочастотного скальпінгу менше 100 угод на тест — статистичний шум |

---

## 3. Залізні правила аудиту

1. **Недоторканний холдаут (Holdout)**:
   - Останні 20% історичних даних зберігаються як фінальний сліпий тест.
   - Заборонено підбирати параметри чи змінювати логіку сигналів після перегляду результатів на холдауті.
2. **Врахування кількості спроб (`estimate_n_trials`)**:
   - Якщо дослідник чи Optuna протестували 100+ варіантів комбінацій параметрів, показник DSR зобов'язаний враховувати це число. Інакше розрахований коефіцієнт Шарпа буде хибно завищеним.
3. **Порівняння з Benchmark**:
   - Обов'язкове співставлення кумулятивної дохідності стратегії з Buy & Hold того ж активу за однаковий період.

---

## 4. Фіксація результатів
Після виконання аудиту формується звіт у `docs/reports/report_<strategy>_<symbol>_<timestamp>.md` із вердиктом:
- **ГОТОВО ДО PAPER TRADING** (всі критерії виконано)
- або **ВІДХИЛЕНО / ПОТРЕБУЄ ДООПРАЦЮВАННЯ** із зазначенням слабких місць.
