# Iteration 1–2 — «Режим-адаптивна сутність» vs одиночні стратегії (1h)

Дата: 2026-09-06/07 · Скрипт: `experiments/iter1_wf_compare.py` · Стан: ОБНОВЛЕНО (1a+1b)

## Контекст
Останні досліди (sweep 2026-09-05/06, 90 клітинок WF з overlay) дали максимум
avg OOS Sharpe ~0.29 (regime_supervisor XRPUSDT 1h) при DSR≈0. Мета ітерації 1:
чи мета-сутність, що **обирає стратегію за режимом ринку**, дає стабільніший OOS
за одиночні стратегії, і чи зниження churn допомагає.

## Зміни коду
- Коміт 4604e30: `min_dwell_bars` (гістерезис структури) + `blend_mode="best_prior"`
  (жорсткий вибір однієї суб-стратегії з найвищим taxonomy-пріором у режимі).
- Коміт 627a364: `vol_high_veto` (флет у high-vol) + `trend_direction_gate`
  (без контр-трендових входів) — опції supervisor, вимкнені за замовчуванням.

## Методика
WF на 1h: 365 днів (~8760 барів), train=2500/test=500 (≈12 вікон), ФІКСОВАНІ
параметри (без IS-оптимізації). Символи: XRPUSDT, AVAXUSDT, UNIUSDT, LINKUSDT,
AAVEUSDT, ADAUSDT. Діти: official = mean_reversion,supertrend,hmm_reversion;
split = supertrend,cross_momentum,stoch_rsi (тренд vs рейндж за taxonomy).
Результати: `results/iter1_wf_compare.csv` (maker), `results/iter1_wf_compare.md`.

## Результат 1a — taker
| variant | avg OOS SR | pos frac | n_trades |
|---|---|---|---|
| single:supertrend | −0.058 | 51% | 311 |
| sup:regime_soft:d3 (split) | −0.355 | 40% | 6286 |
| sup:best_prior:d3 (split) | −0.402 | 33% | 683 |
| sup:contextual_hedge (split) | −0.666 | 31% | 23578 |

## Результат 1b — maker (модель maker-філів рушія)
| variant | avg OOS SR | pos frac | n_trades | кращі символи |
|---|---|---|---|---|
| single:supertrend | −0.027 | 53% | 311 | AAVE +0.141, LINK +0.141, UNI +0.051, XRP +0.091 |
| sup:official:regime_soft:d3 | −0.188 | 44% | 855 | LINK +0.032 |
| sup:split:regime_soft:d3 | −0.174 | 43% | 6251 | AVAX +0.046 |
| sup:official:best_prior:d3 | −0.205 | 39% | 455 | — |

## Висновки ітерацій 1a/1b (чесні)
1. **На 1h (taker або maker, без overlay) жоден напрямковий варіант не дає
   стабільно позитивного avg OOS по всіх символах.** Taker вбиває все;
   maker покращує, але супервізор не перевершує кращого сингла (supertrend).
2. **Churn реальний:** contextual_hedge дає 23–24k угод на 8760 барів
   (дробові зважені позиції змінюються щобара). `best_prior` ріже угоди в
   35–50 разів при порівнянному OOS — жорсткий per-regime вибір ефективний
   як анти-churn механізм, але не рятує слабкі діти.
3. **Трендовий шар на 1h позитивний лише на окремих символах** (AAVE/LINK/UNI/XRP,
   maker) — це найперспективніший напрямковий кандидат; range-MR діти
   (mean_reversion/hmm_reversion/stoch_rsi) тягнуть supervisor вниз.
4. Це узгоджується з research: «спершу підтверди edge кожного sleeve, потім
   диверсифікуй»; на цьому таймфреймі підтвердженого напрямкового edge поки немає.

## Експеримент 2 — режимні гейти (running)
sup (official/split) × (regime_soft/best_prior, dwell=3) × гейти {none, vhv, tdg, both},
5 символів, maker. Результати: `results/iter2_gates.csv` (запущено).

## Діагностика per-regime (maker, 365д, IS-орієнтовна)
Повна таблиця: stdout jobs (XRPUSDT/AVAXUSDT/LINKUSDT/AAVEUSDT). Інсайти:
1. **Режимна структура символ-специфічна і не збігається з taxonomy**:
   - XRP: supertrend дає майже весь PnL у trend_up|normal (+1.85, PF 1.27);
     у trend_down|normal/low — глибокий мінус (шорти не відкриваються, але лонг
     тримається під час локальних «downtrend»-міток на тлі річного аптренду).
   - AVAX: supertrend позитивний лише у trend_up|high; cross_momentum стабільно
     додатний у trend_down|low/normal.
   - stoch_rsi на XRP/AVAX позитивний у trend_up (купівля провалів в аптренді).
2. **Висновок для сутності**: 1h-структура (EMA 9/50) запізнюється відносно
   старшого тренду → напрямковий гейт потрібен на СТАРШОМУ ТФ (4h/1d), інакше
   «trend_down»-мітки на тлі аптренду ріжуть правильні лонги.
3. Гіпотеза ітерації 3: multi-timeframe direction gate (старший ТФ визначає
   дозволений напрямок; 1h-режим визначає інтенсивність/активність).

## План ітерації 3 (наступний крок)
1. Multi-timeframe тренд-фільтр для сутності (4h/1d EMA-напрямок поверх 1h).
2. Перевірка на 4h/1d горизонтах (тренд працює на довших горизонтах — research).
3. Per-symbol (інструмент×ТФ) профіль режимів — «стратегія, що краща на
   інструменті/режимі» — з OOS-валідацією кожної клітинки (DSR з n_trials =
   кількість перебраних (режим×стратегія×символ) варіантів).
4. Окремо: валідований напрямок = pairs_arb 1h maker; regime-шар — risk-гейт
   поверх валідованих пар (поза цим порівнянням).
