# iter13 — ts_momentum 1w (тижневий momentum), two-stage protocol

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter13.md](hypothesis_iter13.md)
Дані: LIVE `binanceusdm`, нативні 1w klines CORE_15 (перше використання 1w у проєкті).

## TL;DR

| Гіпотеза | Результат | Деталі |
|---|---|---|
| H13-A ts_momentum 1w long-only CORE_15 | FAIL | lookback=13w; validation SR +0.578, t_NW +1.11, CI [-0.30, +1.39] |
| контроль long-short | діагностика | validation SR +0.377, t_NW +0.88 |
| H13-B value-added 1w⊕1d | PASS | corr=+0.09; combined SR +1.51 |

## H13-A: selection half (2019–2022)

| lookback (тижні) | selection Sharpe | t_NW | тижнів |
|---:|---:|---:|---:|
| 4 | +1.048 | +1.22 | 123 |
| 8 | +1.156 | +1.27 | 123 |
| 13 | +1.387 | +1.54 | 123 |
| 26 | +1.155 | +1.50 | 123 |

Обрано lookback = **13** тижнів.

## H13-A: validation half (2023–2026, невидима під час selection)

| Метрика | Значення |
|---|---:|
| Sharpe (bpy=52) | +0.578 |
| t_Newey–West(8) | +1.11 |
| bootstrap 95% CI | [-0.30, +1.39] |
| maxDD | -0.34% |
| OOS-тижнів | 186 |
| (довідково, повний OOS) Sharpe / t_NW | +0.962 / +1.85 |

Рішення (заздалегідь): validation Sharpe>0 AND t_NW≥2.0 AND CI без 0 → **НЕ промотувати**.

## Контроль long-short (той самий lookback)

| Варіант | validation Sharpe | t_NW | повний OOS Sharpe | t_NW |
|---|---:|---:|---:|---:|
| long-only | +0.578 | +1.11 | +0.962 | +1.85 |
| long-short | +0.377 | +0.88 | +0.606 | +1.32 |

## H13-B: value-added 1w ⊕ 1d

Спільний span: 2209 днів; кореляція 1w↔1d = **+0.09**.

| Портфель | Sharpe | maxDD |
|---|---:|---:|
| ts_momentum 1w CORE_15 | +0.95 | -0.34% |
| ts_momentum 1d CORE_15 | +1.30 | -0.34% |
| combined 50/50 | +1.51 | -0.29% |

Вердикт: **диверсифікація додає вартість** (portfolio-construction evidence, не paper-gate).

## Per-symbol OOS Sharpe (обраний lookback, повний OOS)

| symbol | Sharpe |
|---|---:|
| BNBUSDT | +1.14 |
| BTCUSDT | +0.91 |
| ADAUSDT | +0.71 |
| NEARUSDT | +0.61 |
| ETHUSDT | +0.56 |
| LINKUSDT | +0.41 |
| SOLUSDT | +0.41 |
| XRPUSDT | +0.36 |
| ATOMUSDT | +0.30 |
| LTCUSDT | +0.29 |
| AVAXUSDT | +0.18 |
| DOTUSDT | +0.15 |
| AAVEUSDT | +0.07 |
| DOGEUSDT | +0.06 |
| UNIUSDT | -0.02 |

## Вердикт для paper

- `ts_momentum 1w` **не промотується** (гейт H13-A не пройдено) — monitoring-набір iter11 лишається без змін.

⚠ 1w-вікно після цього прогону спалене: див. `docs/reports/oos_usage.md`.