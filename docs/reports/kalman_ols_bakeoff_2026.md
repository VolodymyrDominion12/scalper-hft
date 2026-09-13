# Kalman vs OLS Bake-off: pairs_arb LINK/BTC (2026-09-13)

Результат H15-A з [hypothesis_iter15.md](hypothesis_iter15.md).

## Методологія

Two-way bake-off на однакових даних:
- **OLS**: rolling OLS hedge ratio (дефолт, validated config)
- **Kalman**: rolling Kalman filter (CLI `--use-kalman`)

Дані: LINKUSDT/BTCUSDT 1h, 1095 days (2023-09-14 → 2026-09-13), live Binance USDT-M.
Комісії: maker 0.02%, slippage 2 bps, `regime_scale=0.25`.

## Результати

| Метрика | OLS ✅ | Kalman ❌ | Поріг PASS |
|---|---|---|---|
| Return 3y | **+2.09%** | −10.64% | — |
| CAGR | **+0.69%** | −3.68% | — |
| Sharpe (річний) | **+0.360** | −2.911 | — |
| WF OOS SRh avg | **+0.010** | −0.028 | ≥ baseline |
| WF pos вікон (49) | **55%** | 22% | ≥ 60% |
| maxDD | **−2.93%** | −10.74% | ≤ −30% |
| Win rate | **70.8%** | 34.9% | — |
| Угоди 3y | 72 | **525** | ≥ 80 |
| Profit factor | **1.354** | 0.447 | — |
| Time-decay | **PASS** | FAIL | — |
| P(розорення) | **0.000** | 1.000 | — |

## Аналіз причин провалу Kalman

Kalman генерує **525 угод** проти 72 OLS — в 7.3× більше.
Механізм: rolling Kalman оновлює β(LINK/BTC) щогодини і дуже чутливо реагує
на шум → спред «мерехтить» навколо порогів z → надмірна кількість входів →
round-trip fee-drag (0.04% × 525 = 21% total) знищує PnL.

OLS з фіксованим rolling window виступає природним lowpass-фільтром:
β оновлюється плавно, спред стійкіший, лише 72 угоди за 3 роки.

Kalman може мати теоретичну перевагу в ринках з **повільним** дрейфом β,
але LINK/BTC демонструє **стрибкоподібні** зміни β (alt-season,
BTC dominance),  що робить Kalman гіршим саме тут.

## Вердикт H15-A: **FAIL** (Kalman)

**Дефолт `use_kalman=False` підтверджено.** Kalman не просуватиметься.

Документ:
- `use_kalman` залишається `False` в `VALIDATED_PAIRS` та `pairs_arb` дефолті
- Наступний bake-off лише якщо з'явиться нова незалежна вісь (наприклад,
  структурний break-point detector який перемикає Kalman↔OLS по режимах)

*Дата: 2026-09-13 | Iter: 15 | Agent: Antigravity*
