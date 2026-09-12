# iter12 — improvement loop: two-stage vol-target + value-added

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter12.md](hypothesis_iter12.md)
Дані: LIVE `binanceusdm`, перевикористання спаленого OOS iter11 (CORE_15 1d).

## TL;DR

| Гіпотеза | Результат | Деталі |
|---|---|---|
| H12-A two-stage vol-target | FAIL | НЕ промотувати (post-hoc лишається); обране σ=60 |
| H12-B value-added pairs⊕ts | FAIL | без доданої вартості; corr=-0.08 |
| H12-C attribution | діагностика | топ: ['DOGEUSDT', 'NEARUSDT', 'XRPUSDT'] | worst: ['DOTUSDT', 'LINKUSDT', 'LTCUSDT'] |

Портфель ts_momentum 1d CORE_15 (повний OOS): Sharpe **+1.36**, t_NW **+2.61**.

## H12-A: two-stage vol-target overlay

Сітка σ-вікон (заздалегідь): [20, 40, 60].
Selection half 2019-01-01→2022-12-31 обирає вікно з макс. Sharpe; validation half 2023-01-01→2026-12-31 — невидиме під час selection.

| σ-вікно | selection Sharpe |
|---:|---:|
| 20 | +1.160 |
| 40 | +1.152 |
| 60 | +1.167 |

Обране σ-вікно: **60**

| Метрика | plain | vol-target |
|---|---:|---:|
| validation Sharpe | +1.108 | +0.994 |
| validation t_NW | +1.69 | +1.55 |
| validation CI | — | [-0.17, +2.06] |
| validation maxDD | -0.34% | -0.44% |

Рішення (заздалегідь): vt Sharpe > plain AND t_NW(vt) ≥ 2.0 на validation → **НЕ промотувати**.

## H12-B: value-added pairs_arb ⊕ ts_momentum

Спільний span: 1095 днів; кореляція pairs↔ts = **-0.08**.

| Портфель | Sharpe | maxDD |
|---|---:|---:|
| pairs_arb LINK/BTC 1h | +0.04 | -4.58% |
| ts_momentum 1d CORE_15 | +1.06 | -0.34% |
| combined 50/50 risk-equal | +0.16 | -2.07% |

Вердикт: **без доданої вартості** (portfolio-construction evidence, не paper-gate).

## H12-C: per-symbol attribution

| symbol | contribution | Sharpe | corr→port |
|---|---:|---:|---:|
| DOGEUSDT | +0.1238 | +0.88 | +0.55 |
| NEARUSDT | +0.0918 | +0.85 | +0.60 |
| XRPUSDT | +0.0889 | +0.94 | +0.48 |
| SOLUSDT | +0.0857 | +0.99 | +0.54 |
| AVAXUSDT | +0.0670 | +0.72 | +0.61 |
| ADAUSDT | +0.0613 | +0.66 | +0.59 |
| BNBUSDT | +0.0584 | +0.66 | +0.45 |
| ETHUSDT | +0.0564 | +0.83 | +0.64 |
| AAVEUSDT | +0.0500 | +0.59 | +0.56 |
| BTCUSDT | +0.0477 | +1.04 | +0.54 |
| UNIUSDT | +0.0374 | +0.39 | +0.52 |
| ATOMUSDT | +0.0339 | +0.38 | +0.55 |
| DOTUSDT | +0.0222 | +0.30 | +0.71 |
| LINKUSDT | +0.0208 | +0.25 | +0.59 |
| LTCUSDT | -0.0186 | -0.26 | +0.58 |

## Висновки для paper

- `pairs_arb LINK/BTC 1h` — validated, без змін.
- `ts_momentum 1d/4h long-only CORE_15` — monitoring (iter11), без змін.
- vol-target overlay → лишається пост-хок (two-stage FAIL на validation).

⚠ OOS перевикористано з iter11 (two-stage protocol); нового снупінгу немає.