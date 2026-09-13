# Pre-registration: Iteration 15 (2026-09-12)

> **Правило:** усі критерії PASS/FAIL зафіксовані ДО запуску бектестів.
> Будь-яка зміна критеріїв після погляду на результати = data-snooping.

Посилання: [STRATEGY_STATUS.md](../STRATEGY_STATUS.md) · [ROADMAP.md](../ROADMAP.md) ·
[iter14_improvement_cycle.md](iter14_improvement_cycle.md)

---

## Контекст

Після iter9→14 простір «дешевих» покращень вичерпано:
- `ts_momentum` CORE_15 1d/4h у monitoring (Port Sharpe +1.365 / +1.006, t_NW +2.61 / +2.49)
- `pairs_arb` LINK/BTC validated (WF pos 65%, PBO=0.000)
- Нові пари (H14-C), CS-momentum (H14-A), skip-month (H14-B) — всі FAIL gate

Залишились **структурно незалежні** вісі:

---

## H15-A: Kalman vs OLS hedge-ratio (W1-K)

**Гіпотеза:** Rolling Kalman filter для β(LINK/BTC) адаптивно слідкує за дрейфом
hedge-ratio краще, ніж OLS rolling window. Механізм: Kalman швидше реагує на зміну β
під час режимних зрушень → менша дисперсія спреду → більше WF-вікон з позитивним OOS.

**Базова лінія (OLS 2026-09-12):**
- WF pos = 65% (32/49 вікон), WF avg OOS Sharpe = +0.0064, maxDD ≈ −27%, n_trades ≈ 127

**Критерії PASS (pre-registered):**
| Метрика | Поріг |
|---|---|
| WF avg OOS Sharpe (Kalman) | ≥ OLS baseline (+0.0064) |
| WF pos_windows | ≥ 60% |
| maxDD 3y | ≤ −30% |
| n_trades | ≥ 80 |
| CSCV PBO | < 0.5 |

**Команди:**
```bash
uv run python -m scalper_hft.cli data-audit --days 1095
uv run python -m scalper_hft.cli pairs --strategy pairs_arb \
  --leg1 LINKUSDT --leg2 BTCUSDT --interval 1h --days 1095 --maker
uv run python -m scalper_hft.cli pairs --strategy pairs_arb \
  --leg1 LINKUSDT --leg2 BTCUSDT --interval 1h --days 1095 --maker --use-kalman
uv run python -m scalper_hft.cli overfit --strategy pairs_arb \
  --leg1 LINKUSDT --leg2 BTCUSDT --interval 1h --days 1095
```

**Результат:** `docs/reports/kalman_ols_bakeoff_2026.md`

---

## H15-B: ts_momentum — attribution і скорочений universe

**Гіпотеза:** Bottom-3 (LTC, DOT, LINK) мають від'ємний/нульовий внесок.
Видалення підвищить t_NW без суттєвої втрати WF pos frac.

**Базова лінія (CORE_15):** Port Sharpe +1.365, t_NW +2.61, symbols pos frac 73.3%

**Критерії PASS:**
| Метрика | Поріг |
|---|---|
| Port Sharpe (CORE_12) | ≥ +1.365 |
| t_NW | ≥ +2.61 |
| symbols pos frac | ≥ 65% |
| maxDD 3y | ≤ baseline × 1.2 |

**Two-stage:** Selection 2019–22 → Validation 2023–26 (held-out half)

---

## H15-C: AVAX/NEAR + Kalman (умова: H15-A PASS)

**Гіпотеза:** AVAX/NEAR (WF pos 63% gate PASS, val ret −5.2%) програло через дрейф β.
Kalman hedge може виправити.

**Критерії PASS:** WF pos ≥ 55%, val_ret > 0, PBO < 0.5, n_trades ≥ 50

---

## H15-D: Funding carry + high-funding filter (3σ)

**Гіпотеза:** Вхід лише при funding > 3× rolling std (90d) + maker only.
Менше угод, але вищий P&L per trade, менший fee-drag.

**Критерій PASS:** WF avg OOS Sharpe > 0.3, ≥ 20 угод OOS, DSR > 0.95

---

## H15-E: sparse_basket OOS аудит

**Гіпотеза:** Lasso-вибраний субкошик CORE_15 ≥ рівноважний ts_momentum.

**Критерій PASS (cell_audit standard):** WF OOS Sharpe ≥ 0.3, pos frac ≥ 50%, DSR ≥ 0.95, PBO < 0.5

---

## Правила циклу

1. Результати → `results/iter15/results.json`
2. Лідерборд оновлюється лише після аудиту
3. Жодної реоптимізації після FAIL
4. Paper стартує незалежно від H15

*Дата: 2026-09-12 | Agent: Antigravity*
