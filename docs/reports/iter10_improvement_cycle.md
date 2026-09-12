# iter10 — цикл покращення та лідерборд (2026-09-12)

## Стан проєкту

| Перевірка | Результат |
|---|---|
| `uv run pytest tests/ -q` | **1547 passed**, 9 skipped |
| `data-audit --days 1095` | **15/15 символів OK** (live binanceusdm) |
| Системні дефекти iter9 (S1–S7) | виправлені раніше, тести зелені |

## Висновки iter9 (база)

- **Таймфрейм**: edge лише на **4h/1d**; 1h і нижче — fee-drag.
- **Стратегія**: `ts_momentum` 1d — єдиний directional кандидат; `pairs_arb` LINK/BTC — validated.
- **Жодна single-cell** не проходить гейт `avg_oos_sharpe > 0.3` + DSR + PBO.

## iter10: гіпотези та результати

Тестовано 6 варіантів портфеля `ts_momentum` (15 перпів, нативні 1d klines, 2500 днів, WF train=250/test=125, maker):

| Варіант | Зміна | Port Sharpe | t_Newey–West | Vol-target Sharpe | t_NW (VT) |
|---|---|---:|---:|---:|---:|
| baseline_lb20 | дефолт | +0.758 | +1.87 | +1.086 | +2.54 |
| smooth3_lb20 | signal_smooth=3 | +0.815 | +2.01 | +1.137 | +2.72 |
| **long_only_lb20** | allow_short=False | **+1.365** | **+2.61** | +1.034 | +2.10 |
| smooth3_long_lb20 | smooth=3 + long-only | +1.356 | +2.67 | +1.044 | +2.13 |
| top10_lb20 | 10 найкращих символів | +1.007 | +2.30 | +1.293 | +2.97 |
| smooth3_top10 | smooth=3, top-10 | +1.104 | +2.49 | **+1.366** | **+3.17** |

### Причина покращення long-only

1. **Шорти на крипто-перпах 2019–2026** часто збиткові: bull-режим домінує, funding на шортах додатковий drag.
2. **Менше угод** (1315 vs 2644 OOS) → менше round-trip витрат при тій самій альфі.
3. Портфель long-only **б'є buy&hold** (+0.86) на цьому вікні; baseline — ні.

⚠️ Long-only — **період-специфічна** гіпотеза (bull market). Потрібна pre-registration перед paper.

### Аудит single-cell (ts_momentum 1d, 3y)

| Символ | avg_oos_sharpe | oos_pos_frac | Вердикт |
|---|---:|---:|---|
| ADAUSDT | +0.256 | 50% | FAIL (DSR=0) |
| DOTUSDT | +0.065 | 75% | FAIL |
| AVAXUSDT | +0.059 | 50% | FAIL |
| DOGEUSDT | +0.035 | 38% | FAIL |
| BTCUSDT | −0.327 | 38% | FAIL |
| BNBUSDT | −0.385 | 38% | FAIL |

Жодна комірка не проходить directional гейт — **портфель обов'язковий**.

## Рекомендації для paper-трейдингу

| # | Комірка | Тіер | Дія |
|---|---|---|---|
| 1 | **pairs_arb LINK/BTC 1h** | ✅ Validated | Paper на `paper-v0.2.0` (вже в плані) |
| 2 | **ts_momentum 1d long-only портфель** | 🟡 Monitoring | Pre-register гіпотезу → paper-моніторинг разом з pairs |
| 3 | ts_momentum 1d baseline (short+long) | 🔵 Candidate | t_NW=1.87, CI містить 0 — недостатньо для live |
| 4 | Усі 1h/4h directional | ⛔ Rejected | fee-drag / низький OOS |

## Артефакти

- `results/iter10/variants.csv` — порівняння варіантів
- `results/iter10/audit_ts_momentum_1d.csv` — аудит 8 комірок
- `docs/reports/LEADERBOARD.md` — лідерборд (оновлюється `cli leaderboard`)
- `experiments/iter10_research_cycle.py` — скрипт циклу

## Наступні кроки

1. Pre-registration `docs/reports/hypothesis_ts_momentum.md` (long-only портфель).
2. Розширення універсуму до 30–50 перпів (підняти t-стат).
3. Paper-моніторинг `ts_momentum` long-only поруч із `pairs_arb`.
