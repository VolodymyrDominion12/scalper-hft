# iter11 — незалежне підтвердження long-only (розширений універсум + нативний 4h)

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter11_universe.md](hypothesis_iter11_universe.md)  
Дані: LIVE `binanceusdm`, нативні **1d** (42 перпи, ~6.8y) і нативні **4h** (CORE_15)  
Скрипт: `experiments/iter11_expansion_cycle.py` · Артефакти: `results/iter11/variants.csv`

Параметри альфи **заморожені** (lookback=20, smooth=1, maker). Не підганялись після метрик.

---

## TL;DR

| Гіпотеза | Результат | Port Sharpe | t_NW | Bootstrap 95% CI | Вердикт |
|---|---|---:|---:|---|---|
| H0 реплікація CORE_15 1d long-only | **PASS** | +1.365 | +2.61 | [+0.49, +2.25] | monitoring → paper-книга |
| H1 NEW_27 1d long-only (незалежні імена) | **FAIL** | +0.839 | +1.70 | [−0.03, +1.67] | не додавати нові імена в paper |
| H1 контроль NEW short+long | FAIL | +0.185 | +0.46 | [−0.57, +0.94] | шорти знову гірші |
| H2 FULL_42 1d long-only | PASS* | +1.143 | +2.28 | [+0.27, +1.97] | *несе CORE; не незалежне підтвердження NEW |
| H3 CORE_15 **нативний 4h** long-only | **PASS** | +1.006 | +2.49 | [+0.25, +1.77] | новий ТФ-рукав для monitoring |
| H3 контроль 4h short+long | FAIL | +0.316 | +0.88 | [−0.41, +1.05] | шорти вбивають і 4h |
| H4 періоди CORE 1d | PASS | 2019–22 **+1.68** / 2023–26 **+1.11** | — | обидва > 0 |
| H4 періоди CORE 4h | PASS | 2019–22 **+1.18** / 2023–26 **+0.88** | — | обидва > 0 |

**Причина (підтверджена на нових іменах і новому ТФ):** шорти на крипто-перпах 2019–2026 з'їдають edge (funding + структурний бичий дрейф). Long-only зменшує turnover (~2× менше угод) і лишає додатний портфельний Sharpe.

**Що НЕ робити:** не котити NEW_27 у paper — H1 не пройшов t-гейт. FULL_42 проходить лише тому, що CORE тягне t-стат.

---

## Відібрані комірки для paper

| # | Комірка | Тіер | Чому |
|---|---|---|---|
| 1 | **pairs_arb LINK/BTC 1h maker** (`regime_scale=0.25`) | Validated | єдиний PASS pair-гейту; не чіпали в цьому циклі |
| 2 | **ts_momentum 1d long-only CORE_15** | Monitoring | H0 реплікація; CI без нуля; обидва підперіоди > 0 |
| 3 | **ts_momentum 4h long-only CORE_15** | Monitoring | H3 незалежний нативний ТФ; ті самі параметри |

Запуск:

```bash
DRY_RUN=true MAKER_EXECUTION=true \
  uv run python -m scalper_hft.cli paper-run-pairs --portfolio --daemon

DRY_RUN=true MAKER_EXECUTION=true \
  uv run python -m scalper_hft.cli paper-run-ts-momentum --daemon --sleep 3600
```

4h-рукав: той самий runner з `interval="4h"` (не дефолт; окремий sqlite). Не live.

---

## Деталі

- NEW: 27/30 імен (пропуск EOS — немає USDT-M; FTM/MKR — валідатор 1d, 3 спайки). Зафіксовано **до** повторного прогону.
- Середня парна кореляція CORE 1d = 0.27; NEW = 0.25; 4h = 0.32.
- Per-symbol mean WF OOS на NEW = **−0.012** при портфельному Sharpe +0.84 — диверсифікація маскує слабкі одинаки; саме тому H1 t_NW < 2.
- Vol-target overlay (діагностика, не гейт): NEW t_NW 1.96 — майже поріг, лишається пост-хок.

## Виправлення в цьому циклі

- pandas 3: Binance `1d` → offset `1D` (`pandas_resample_rule`); `Timestamp.utcnow` замінено.
- `ts_momentum` signal_smooth — векторизовано (без `Series.apply`).
- Лідерборд: дедуп LINK/BTC, pairs на одному тікері ≠ paper, top10/smooth3 (post-hoc) не monitoring.

⚠️ OOS цього прогону зареєстровано в `oos_usage.md` (purpose `iter11:universe/native-4h`).
