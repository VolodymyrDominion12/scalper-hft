# Iteration 1 — «Режим-адаптивна сутність» vs одиночні стратегії (1h)

Дата: 2026-09-06/07 · Скрипт: `experiments/iter1_wf_compare.py` · Стан: RUNNING/ЧЕРНЕТКА

## Контекст
Останні досліди (sweep 2026-09-05/06, 90 клітинок WF з overlay) дали максимум
avg OOS Sharpe ~0.29 (regime_supervisor XRPUSDT 1h) при DSR≈0 — тобто без
статистично підтвердженого edge. Мета ітерації: з'ясувати, чи мета-сутність,
що **обирає стратегію за режимом ринку**, дає кращий/стабільніший OOS за
одиночні стратегії, і чи зниження churn (гістерезис `min_dwell_bars`,
жорсткий вибір `best_prior`) допомагає.

## Зміни коду (коміт 4604e30)
- `features/regimes.apply_min_dwell` — каузальний гістерезис категоріального ряду.
- `features/regime_detector` — опційний `min_dwell_bars` для structure.
- `strategies/regime_supervisor` — параметр `min_dwell_bars` + новий `blend_mode="best_prior"`
  (одна суб-стратегія з найвищим taxonomy-пріором у поточному режимі; нативні {-1,0,1}).

## Методика
Walk-forward на 1h, 365 днів (~8760 барів), train=2500 / test=500 (≈12 вікон),
фіксовані параметри (БЕЗ IS-оптимізації — це тест гіпотези архітектури).
Символи: XRPUSDT, AVAXUSDT, UNIUSDT, LINKUSDT, AAVEUSDT, ADAUSDT.
Діти supervisor: official = mean_reversion,supertrend,hmm_reversion;
split = supertrend,cross_momentum,stoch_rsi (тренд vs рейндж за taxonomy).

## Результат 1a — taker (maker=False)

| variant | avg OOS SR | pos win frac | n_trades OOS |
|---|---|---|---|
| single:supertrend | −0.058 | 51% | 311 |
| single:cross_momentum | −0.178 | 50% | 3183 |
| sup:regime_soft:d3 (split) | −0.355 | 40% | 6286 |
| sup:best_prior:d3 (split) | −0.402 | 33% | 683 |
| sup:contextual_hedge:d0 | −0.666 | 31% | 23578 |

**Висновки 1a:**
1. На taker 1h все від'ємне — taker 0.05%+slippage вбиває короткі 1h-позиції.
2. `contextual_hedge` генерує ~24k угод на 8760 барів — **дробовий churn**:
   зважені float-позиції змінюються щобара → комісії з'їдають усе. `best_prior`
   ріже угоди в ~35 разів (683) при порівнянному OOS — гістерезис/жорсткий
   вибір працює, але діти слабкі на taker.
3. Кращий одиночний = supertrend (51% позитивних вікон, позитивний OOS на
   AAVE/LINK/UNI/XRP) — відповідає «трендовий шар на 1h життєздатний лише maker».

## Результат 1b — maker (очікується)
Запущено з `--maker` (модель maker-філів рушія з adverse selection) — див.
`results/iter1_wf_compare.csv`. Порівняння з офіційним overlay-sweep (де
supertrend XRPUSDT 1h мав OOS 0.164, regime_supervisor XRP 0.294).

## План ітерації 2 (залежить від 1b)
1. Vol-high veto (не торгувати в режимі high-vol) як параметр supervisor.
2. Trend-direction gate (у trend_up лише long-сигнали, trend_down лише short).
3. Breakeven/cost-gate для supervisor (expected move > round-trip).
4. Повторний WF-прогін + DSR на конкатенованих OOS-дохідностях.
