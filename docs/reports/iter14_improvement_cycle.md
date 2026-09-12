# iter14 — CS-momentum, 12-1 TSMOM, друга пара (two-stage)

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter14.md](hypothesis_iter14.md)
Дані: LIVE `binanceusdm`, нативні 1d CORE_15 + 1h validation пар.

## TL;DR

| Гіпотеза | Результат | Деталі |
|---|---|---|
| H14-A CS momentum long-only | FAIL | lookback=10d; validation SR +0.814, t_NW +1.46, CI [-0.27, +1.84] |
| H14-B 12-1 skip-month TSMOM | FAIL | validation SR +0.420, t_NW +0.82, CI [-0.55, +1.44] |
| H14-C друга пара | FAIL | AVAXUSDT/NEARUSDT WF pos=63% ret=-5.2% |

## H14-A: selection half (2019–2022)

| lookback | selection Sharpe | t_NW | днів |
|---:|---:|---:|---:|
| 10 | +1.564 | +2.45 | 1149 |
| 20 | +1.559 | +2.53 | 1149 |
| 60 | +1.562 | +2.55 | 1149 |

Обрано lookback = **10** днів.

## H14-A: validation half (2023–2026)

| Метрика | long-only | long-short контроль |
|---|---:|---:|
| Sharpe | +0.814 | +0.020 |
| t_Newey–West(20) | +1.46 | +0.04 |
| bootstrap 95% CI | [-0.27, +1.84] | [-1.07, +0.98] |
| maxDD | -58.74% | -44.27% |

Рішення: validation Sharpe>0 AND t_NW≥2.0 AND CI без 0 → **НЕ промотувати**.

## H14-B: 12-1 skip-month (validation 2023–2026)

| Варіант | Sharpe | t_NW | CI | maxDD |
|---|---:|---:|:---|---:|
| 12-1 (skip=21) | +0.420 | +0.82 | [-0.55, +1.44] | -71.43% |
| контроль skip=0 | +0.569 | +1.05 | [-0.46, +1.62] | -71.19% |

Рішення: **НЕ промотувати**.

## H14-C: нові пари

| Пара | ADF p (IS) | HL 1d | WF pos | вікон | val ret | угод | maxDD | гейт |
|---|---:|---:|---:|---:|---:|---:|---:|:---|
| AVAXUSDT/NEARUSDT | 0.015 | 24.5 | 63% | 49 | -5.2% | 245 | -6.7% | FAIL |
| ADAUSDT/UNIUSDT | 0.015 | 41.1 | 31% | 49 | -5.9% | 223 | -6.4% | FAIL |
| AAVEUSDT/LINKUSDT | 0.016 | 37.3 | 43% | 49 | -7.7% | 226 | -8.4% | FAIL |

PASS → candidate (не `VALIDATED_PAIRS`; CSCV перед будь-якою зміною списку).

## H14-D: value-added vs ts_momentum 1d

- **cs**: corr=+0.60; standalone SR +1.12 vs ts +1.36; combined +1.12; adds=False
- **skipmom**: corr=+0.49; standalone SR +0.79 vs ts +1.36; combined +0.79; adds=False

## Вердикт для paper

Жодна нова гіпотеза не пройшла гейт. Paper стартує на наявному наборі: **pairs_arb LINK/BTC 1h maker** (validated) + **ts_momentum 1d/4h long-only CORE_15** (monitoring). Новий research-цикл — лише з новою pre-registration.

⚠ Вікна після цього прогону спалені: див. `docs/reports/oos_usage.md`.
