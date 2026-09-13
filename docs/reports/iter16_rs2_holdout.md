# iter16 / RS-2 — holdout на свіжих символах + risk-off overlay

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter16_rs2.md](hypothesis_iter16_rs2.md)
Попередній раунд: [iter15_regime_supervisor.md](iter15_regime_supervisor.md)
Harness: `experiments/iter15_regime_supervisor_cycle.py`, вердикт: `experiments/iter15c_verdict.py`

## TL;DR

| Гіпотеза | Вердикт | Числа |
|---|---|---|
| **H16-A** заморожені політики на свіжих символах | ✅ **підтверджено** (обидві частини) | *селектор* `argmax@det_rule` (1d): mean sym SR **+0.078** (<0.3) — не переноситься; *risk-scaling* `riskoff_anchor@det_vol` (4h): port SR **+1.156**, t **+2.01**, CI **[+0.05, +2.15]**, 91% символів > 0 — **переноситься** |
| **H16-B** асиметричний гістерезис (dwell) краще | ❌ **фальсифіковано** | fresh 4h: `riskoff_gate` 0.912 < `riskoff_anchor` **1.156**; CORE 4h: 0.676 < 0.963; turnover падає 67.8→36.9 (і 110.8→59.7), але Sharpe втрачається більше |
| **H16-C** `gap_dwell@det_mkt` був артефактом | ✅ **підтверджено** | на свіжих: 4h **−0.140**, 1d **+0.388** (на CORE-валідації було +1.53) |
| **H16-D** cost-реалізм | ✅ **підтверджено** | taker (свіжі 4h): `riskoff_anchor@det_vol` port SR **+0.857**, t 1.49, 82% > 0, Calmar 0.489 проти incumbent +0.490 / Calmar 0.284 → **ΔSR +0.37** (знак зберігається) |
| **Tier-1 (promotion)** | ❌ не досягнуто | DSR = 0 (потрібен t ≈ 3.4 при n_trials=85; фактичний t = 2.01) |
| **Tier-2 (risk-overlay)** | ✅ **досягнуто на справжньому holdout** | ΔSR **+0.55**, ΔCalmar **+0.37**, ΔmaxDD +0.17 п.п. проти incumbent |

**Головний висновок циклу RS:** режимний шар працює **не як перемикач стратегій, а як
масштабування експозиції**. Селектор (argmax/gap_dwell по станах) не переноситься ні на
свіжі символи, ні навіть у межах CORE (RS-1); risk-off scaling 0.25× — переноситься і дає
статистично відчутну перевагу над incumbent'ом на 11 символах, які жодного разу не
використовувались для вибору політики.

---

## 1. Дизайн (див. pre-registration)

- **FRESH_11** (holdout): APT, ARB, OP, INJ, SUI, FIL, ETC, TRX, ALGO, RUNE, SAND — 4h 7200
  барів (2023-06 → 2026-09), 1d 1200–2433 барів; завантажено з LIVE 2026-09-12.
- CORE_15 — контекст (вибір політик робився тут у RS-1).
- Політики заморожені: `gap=0.5`, `dwell=5`, `tau=1.0`, `shrink=0.5`, risk-off scale 0.25.
- Нові елементи раунду: детектор `det_btcvol` (терцилі vol-перцентиля BTC) і політика
  `riskoff_gate` (асиметричні dwell: `off_dwell` у risk-off, `on_dwell` назад).
- n_trials для DSR = **85** (63 з RS-1 + 22 нових).

---

## 2. Свіжі символи (4h, maker) — ключова таблиця

| Варіант | mean sym SR (sel) | mean sym SR (val) | port SR (val) | t_NW | %симв >0 | Calmar | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| **incumbent** (ts_long) | +0.041 | +0.295 | +0.610 | 1.06 | 73% | 0.362 | 1.0 |
| equal (5 рукавів) | −0.046 | +0.205 | +0.326 | 0.65 | 82% | 0.238 | 1.0 |
| best_single_train | −0.200 | +0.018 | +0.177 | 0.32 | 45% | 0.087 | 3.0 |
| `riskoff_anchor@det_vol` (RS-1 4h winner) | **+0.017** | **+0.697** | **+1.156** | **2.01** | **91%** | **0.731** | 67.8 |
| `riskoff_gate@det_vol` (dwell) | +0.054 | +0.572 | +0.912 | 1.74 | 82% | 0.672 | 36.9 |
| `argmax@det_rule` (RS-1 1d winner) | −0.222 | +0.123 | +0.339 | 0.63 | 55% | 0.237 | 569.5 |
| `gap_dwell@det_mkt` (RS-1 «зірка» 1d) | −0.227 | −0.166 | −0.140 | −0.26 | 27% | −0.055 | 37.9 |
| oracle (ex-post межа) | +0.162 | +1.282 | +2.877 | 5.38 | 100% | 3.775 | 0 |

- `riskoff_anchor@det_vol`: **t = +2.01**, stationary-bootstrap CI **[+0.05, +2.15]** (без нуля),
  maxDD −0.29% проти −0.46% в incumbent'а.
- `argmax@det_rule` робить 569 одиниць turnover (перемикання майже щокварталу на кожному
  символі) і не дає нічого.

**Cost-стресс (taker, свіжі 4h):** incumbent port SR +0.490 (Calmar 0.284),
`riskoff_anchor@det_vol` **+0.857** (t 1.49, 82% символів > 0, Calmar 0.489),
`riskoff_gate@det_vol` 0.650, `gap_dwell@det_mkt` −0.440. Перевага overlay'я
(ΔSR +0.37) зберігається при тейкерських витратах → H16-D ✅.

## 3. Свіжі символи (1d, maker)

| Варіант | mean sym SR (val) | port SR (val) | t_NW | %симв >0 |
|---|---:|---:|---:|---:|
| incumbent | +0.179 | +0.633 | 0.95 | 73% |
| `argmax@det_rule` | +0.078 | +0.649 | 1.06 | 36% |
| `gap_dwell@det_mkt` | +0.104 | +0.388 | 0.75 | 64% |
| `riskoff_anchor@det_vol` | +0.044 | +0.171 | 0.35 | 64% |

На 1d risk-off scaling не працює (0.171 < incumbent 0.633) — режимний шар чутливий до ТФ,
і це узгоджується з тим, що на 1d власний vol-перцентиль повільний, а даних для валу
менше. **Цільовий ТФ для overlay — 4h.**

## 4. CORE_15 (4h) — контекст і порівняння з RS-1

| Варіант | mean sym SR (sel) | port SR (val) | t_NW | %симв >0 | turnover |
|---|---:|---:|---:|---:|---:|
| incumbent | −0.112 | +0.721 | 1.24 | 93% | 1.0 |
| `riskoff_anchor@det_vol` | +0.194 | **+0.963** | 1.72 | 93% | 110.8 |
| `riskoff_gate@det_vol` | +0.177 | +0.676 | 1.29 | 80% | 59.7 |
| `riskoff_gate@det_btcvol` | +0.082 | +0.027 | 0.05 | 53% | 20.5 |

На CORE перевага anchor'а +0.24 SR (ΔCalmar +0.16 — Tier-2 формально не дотягує 0.20),
на FRESH +0.55 SR (ΔCalmar +0.37 — дотягує). Обидва універсуми дають **той самий знак і
той самий порядок величини**, що і є головним аргументом «це не шум одного вікна».

## 5. Що це означає практично

1. **Селекторний режим supervisor'а закривається.** 16 політик × 3 ТФ × 26 символів: жодна
   селекторна політика не б'є incumbent'а на даних, які не використовувались для вибору.
2. **Живий механізм — risk-scaling**: рівноважний портфель рукавів, експозиція ×0.25 у
   режимі високої власної волатильності (4h). Це масштабування, а не зміна «куди» —
   рівно те, що дали iter6 (`regime_scale`, PBO = 0.000) і `entity_recommendation`.
3. **Чого overlay НЕ дає:** DSR ≥ 0.95 (потрібен t ≈ 3.4, є 2.0). Тому статус — Tier-2
   (risk-overlay) → **paper-моніторинг, не live**, і тільки разом із наявним validated-ядром
   (`pairs_arb` LINK/BTC).
4. **Dwell-підтвердження не потрібне** на 4h: воно зменшує turnover удвічі, але з'їдає
   більше Sharpe (H16-B фальсифіковано). Практичний висновок: risk-off реагує одразу.

## 6. Вердикт раунду

- **Tier-1:** ❌ (не досягнуто; обмежувач — статистична потужність 3.3 року 4h-даних).
- **Tier-2:** ✅ **`riskoff_anchor@det_vol` на 4h** (risk-on = рівновага 5 рукавів,
  risk-off = 0.25 × incumbent-рукав, детектор = власний vol-перцентиль).
- **Tier-3:** усі селекторні політики; `riskoff_gate` з dwell; 1d-версія overlay.

## 7. Наступний раунд (RS-3)

1. **Реалізувати механізм у коді** (`blend_mode: risk_overlay` у `RegimeSupervisor`,
   spec → код → тести; параметри заморожені: детектор vol-перцентиль, scale 0.25, 4h).
2. **Кошик рукавів зафіксувати** як `strategies=ts_momentum(long-only),ts_momentum(ls),
   cross_momentum,funding_carry,supertrend` (саме цей пул дав Tier-2).
3. Перевірити, що стратегія через свій інтерфейс відтворює ті самі числа (інтеграційний
   тест проти `results/iter15/*`), і лише потім — paper-моніторинг поруч із pairs.
4. Meta-labeling (HB-8/R2) як окремий напрям підвищення Tier-1-потужності.

⚠ Спалено: FRESH_11 4h/1d (2023-06 → 2026-09 / 2020-01 → 2026-09), CORE_15 4h/1d —
записи в `docs/reports/oos_usage.md` (`iter16:rs-holdout/*`, `iter16:rs-overlay/*`).

## 8. Артефакти

- `results/iter15/verdict_table_{4h,1d}_maker_{fresh,core}.csv`, `gate_*.json`
- `results/iter15/portfolio_returns_4h_maker_fresh.parquet` (політики на свіжих символах)
- `results/iter15/sleeves/4h_maker_*.parquet` (11 свіжих + 15 CORE)
- `experiments/rs2_download_fresh_universe.py` — завантаження holdout-універсуму
