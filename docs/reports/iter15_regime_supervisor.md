# iter15 / RS-1 — Regime Supervisor: економіка ТФ, політики перемикання, детектори

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter15_rs.md](hypothesis_iter15_rs.md)
План циклу: [regime_supervisor_research.md](regime_supervisor_research.md) · Harness: `experiments/iter15_regime_supervisor_cycle.py`
Аудит: `experiments/iter15b_policy_sensitivity.py`, `experiments/iter15c_verdict.py`

## TL;DR

| Гіпотеза | Вердикт | Ключове число |
|---|---|---|
| **H15-A** економіка ТФ (1h мертвий, 4h/1d живий) | ✅ **підтверджено** | 1h: winner val SR **−0.92** (incumbent −0.56); 1d: incumbent **+1.05**, 4h: +0.72 |
| **H15-B** гістерезис+розрив краще argmax | ⚠️ **частково** | PBO: gap_dwell **0.029** vs argmax **0.20**; але чесний переможець 1d — `argmax@det_rule` |
| **H15-C** мало станів (risk-off anchor) краще 9-коміркової карти | ❌ **фальсифіковано** як PBO-гіпотеза / ✅ як єдиний near-miss | PBO riskoff **0.371** > argmax 0.20; але 4h `riskoff_anchor@det_vol` — єдина політика, додатна в **обох** половинах |
| **H15-D** IC детектора передбачає ранжування OOS | ❌ **фальсифіковано** | \|IC\| ≈ 0.02; ранжування збігається лише на 4h (1 з 3 ТФ) |
| **Tier-1 (promotion)** | ❌ **не досягнуто на жодному ТФ** | найкращий near-miss: 4h ΔSR +0.24, t=1.72, DSR=0 |
| **Tier-2 (risk-overlay)** | ❌ **не досягнуто** (ΔCalmar +0.16 < 0.20) | — |

**Головний результат раунду — методологічний.** «Зірка валідації» 1d `gap_dwell@det_mkt`
(port SR **+1.53**, t 2.14, CI [0.36, 2.60], PBO 0.071) на **selection half** має
Sharpe **−0.09** (mean per-symbol −0.09). Тобто поки ми вибираємо політику лише на
минулих даних (як і має бути), ця політика **не вибирається** — її перевага існує
тільки на тій половині, де ми її знайшли. Раніший «успіх» iter7 (+2.58) був тією ж
природою, лише з lookahead; тут — коректний лаг, і все одно артефакт відбору.

---

## 1. Що саме тестувалося

- **5 рукавів:** `ts_long` (ts_momentum lb=20 long-only), `ts_ls` (контроль), `cs_mom`
  (cross_momentum lb=10), `carry` (funding_carry), `supertrend`.
- **3 детектори:** `det_rule` (власна EMA 9/50-структура), `det_vol` (власний vol-перцентиль),
  `det_mkt` (ринковий risk-on: BTC close > EMA200 і перцентиль rv(60) < 0.8).
- **7 політик + oracle:** incumbent, equal, best_single_train, argmax, gap_dwell,
  soft_shrink, riskoff_anchor.
- **Протокол:** walk-forward (train=250/test=125, purge/embargo=2) → NET OOS-дохідності
  рукавів; далі політика з каузальним **лагом на рішенні** (мітка t−1 → позиція t),
  фолди 5 (фолд 0 — burn-in), вибір переможця **лише на selection half**, гейт — на
  validation half. Вартість re-weighting політики віднімається (|Δw|₁ × pct × cost).
- Дані: LIVE `binanceusdm`, кеш перевірено `data-audit --days 1095` (15/15 ok).

**Mutation-тест лага** (`--selfcheck`): синтетика, де мітка бару ідеально знає переможця
того ж бару → лагнутий PnL **−0.019** (≈0), без лага **+1.43**. PASS.

---

## 2. Результати (maker, CORE_15, 15 символів)

| ТФ | winner за selection half | val port SR | incumbent val | Δ | t_NW | CI 95% | %симв >0 | maxDD | Calmar | PBO | DSR(n=63) | Tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1d** | `argmax@det_rule` | +0.457 | **+1.046** | −0.59 | 0.65 | [−0.87, 1.60] | 73% | −0.4% | 0.35 | 0.043 | 0.00 | 3 |
| **4h** | `riskoff_anchor@det_vol` | **+0.963** | +0.721 | **+0.24** | 1.72 | [−0.14, 1.99] | **93%** | −0.3% | 0.53 | 0.071 | 0.00 | 3 (near-2) |
| **1h** | `gap_dwell@det_rule` | −0.915 | −0.561 | −0.35 | −1.07 | [−2.63, 0.72] | 27% | −0.4% | −0.44 | 0.014 | 0.00 | 3 |

**Cost-стресс (taker, 1d):** winner `argmax@det_rule` val SR +0.681, incumbent **+1.001**,
t 0.95 → той самий висновок, деградація ~0.05 Sharpe (як в audit §M3).

**Порівняння з incumbent (окремо, 4h):** incumbent `ts_long`-портфель: sel −0.112 / val
+0.416, Calmar 0.371, 14/15 символів > 0. `riskoff_anchor@det_vol`: sel +0.194 / val
+0.963, Calmar 0.529, 14/15 > 0. Унікальність: **єдина політика, додатна в обох половинах**
на 4h.

**Oracle (ex-post верхня межа)** — 1d 2.42, 4h 1.94, 1h 1.63: тобто навіть ідеальний
ex-post вибір рукава дає Sharpe < 2.5 — стеля самого пулу рукавів невисока.

---

## 3. Чому «перемикання» не дало Tier-1

1. **Оцінка на selection half — шум.** Mean per-symbol Sharpe за ~1000 барів (1d) має
   стандартну похибку ≈ 1/√(1000/365) ≈ 0.6; розкид між політиками (0.04…0.12) у межах
   похибки. Переможець за таким критерієм майже випадковий — саме тому він і провалив
   валідацію.
2. **Detector instability:** `det_mkt` домінує у валідації на всіх ТФ, але на selection
   half він то додатний (1h +0.199), то від'ємний (1d −0.092, 4h −0.125). Мітка, яка не
   має сталої передбачувальної сили, не може бути основою вибору.
3. **IC ≈ 0:** метрика «стан передбачає *відносний* PnL рукавів» дала |IC| 0.008–0.024.
   Це узгоджується з фальсифікацією H15-D: детектори не несуть інформації про те, який
   рукав кращий.
4. **Множинність:** 16 варіантів × 3 ТФ × 15 символів. DSR = 0 для всіх кандидатів
   (потрібен t ≈ 3.3 при n_trials=63; фактичний t ≤ 2.14).
5. **Turnover перемикання:** `argmax` на 4h робить до 6763 змін ваг на 15 символів,
   `riskoff_anchor` — 32 486; після віднімання вартості re-weighting його Sharpe і Calmar
   просідають саме настільки, щоб Tier-2 перестав виконуватись (ΔCalmar 0.16 < 0.20).

---

## 4. Sensitivity (G8) — де саме «плато»

`gap_dwell@det_mkt` (1d), validation Sharpe по сітці:

| gap \ dwell | 3 | 5 | 8 | 12 |
|---|---|---|---|---|
| 0.3 | 1.483 | 1.534 | 1.535 | 1.465 |
| 0.5 | 1.475 | 1.525 | 1.546 | 1.445 |
| 0.6 | 1.453 | 1.484 | 1.512 | 1.451 |
| 1.0 | 1.175 | 1.177 | 1.171 | 1.134 |

- **Плато є** (1.42–1.55 для gap ≤ 0.6 і dwell 3–12; деградація лише при gap ≥ 0.75) —
  формально G8 виконано для цієї політики, але плато не компенсує провал на selection half.
- `det_vol`/`det_rule` на тому ж ТФ: 0.5–0.9 — тобто «магія» була саме в `det_mkt`, а не в механіці.

`soft_shrink`: монотонно кращий при **більшому** шринкеджу (shrink 0.7 краще за 0.3) —
прямий доказ, що умовні Sharpe-оцінки по станах шумні й потребують стискання до безумовного.

**PBO по сім'ях (1d):** gap_dwell **0.029** < argmax 0.20 < riskoff_anchor 0.371 < soft_shrink 0.386.

---

## 5. Cross-symbol transfer (proxy для HB-11)

Навчання карти на 10 символах → застосування до 5 held-out (1d maker):

| detector | policy | fit | port SR (val, held-out) | mean sym SR | %позитивних |
|---|---|---|---|---|---|
| det_mkt | gap_dwell | per_symbol | 1.021 | 0.483 | 80% |
| det_mkt | gap_dwell | pooled_map | 0.699 | 0.429 | 80% |
| det_vol | gap_dwell | per_symbol | 0.257 | 0.090 | 60% |
| det_vol | gap_dwell | pooled_map | 0.605 | 0.311 | 80% |
| det_rule | gap_dwell | pooled_map | 0.676 | 0.397 | 80% |

Пулінг символів **покращує** слабкі детектори (det_vol: 0.26 → 0.61; det_rule: 0.28 → 0.68)
і лише частково шкодить сильному (det_mkt: 1.02 → 0.70). Це вказує на реальний важіль для
RS-2: оцінювати умовну статистику на **пулі символів**, а не на одному (збільшує n і зменшує
дисперсію вибору).

---

## 6. Вердикт для циклу

- **Tier-3** для селекторного напряму в поточному вигляді (усі ТФ).
- **Єдиний живий слід — risk-scaling на 4h** (`riskoff_anchor@det_vol`): ΔSR +0.24,
  ΔCalmar +0.16, 93% символів > 0, але t=1.72 і DSR=0. Це рівно та картина, яку описує
  література (R1/R3) і власна історія проєкту (iter6 `regime_scale`, PBO=0.000;
  `entity_recommendation`: «regime-шар = експозиція, не селектор»).
- **Наступний раунд (RS-2)** — [hypothesis_iter16_rs2.md](hypothesis_iter16_rs2.md):
  (H16-A) заморожені політики на **11 свіжих символах**; (H16-B) `riskoff_gate` з
  асиметричними порогами і dwell; (H16-C) прямий тест, чи `gap_dwell@det_mkt` був артефактом;
  (H16-D) cost-реалізм на taker.

⚠ Спалено: CORE_15 1d/4h/1h 2019-11 → 2026-09 (purpose `iter15:rs/*`) — запис додано
в `docs/reports/oos_usage.md`.

## 7. Артефакти

- `results/iter15/verdict_table_<tf>_<mode>.csv` — усі варіанти × metric
- `results/iter15/gate_<tf>_<mode>.json` — гейт G1..G10
- `results/iter15/sensitivity_*.csv`, `sensitivity_report_1d_maker.md` — плато G8
- `results/iter15/transfer_1d_maker.csv`, `pbo_families_1d_maker.csv`
- `results/iter15/sleeves/*.parquet` — NET OOS-дохідності 5 рукавів (45 матриць)
