# Regime Supervisor: дослідження «як зробити перемикання стратегій робочим»

Дата: 2026-09-12 · Автор: research-цикл `RS` (раунди RS-1, RS-2, …)
Статус: **дослідницький план + pre-registered гейт** (жодних висновків про edge без прогону)
Компаньйон: [regime_switching_literature.md](regime_switching_literature.md) — зовнішній огляд методів і пасток.

---

## 1. Постановка задачі

Користувацька вимога: сутність, яка **сама змінює стратегію (або експозицію) залежно від
стану ринку** — бичий / ведмежий / флет — і робить це **прибутково після витрат**,
а не «на папері».

Формально: є пул рукавів (sleeves) `S = {s_1..s_n}` з власними сигналами, є каузальний
детектор стану `r_t ∈ R`, і потрібна політика `π: (історія до t-1) × R → ваги w_t ∈ Δ^n`
(або скаляр експозиції), яка максимізує **net-of-cost** Sharpe портфеля на OOS,
переживає аудит на перенавчання (DSR/PBO/sensitivity) і **перевершує incumbent**
(те, що вже в paper/monitoring: `pairs_arb` LINK/BTC 1h maker + `ts_momentum` 1d/4h
long-only CORE_15).

Три можливі фінали циклу — усі три прийнятні як результат, але тільки перший — як edge:
1. **Promotion**: supervisor проходить гейт §7 → candidate → аудит → paper-моніторинг.
2. **Partial**: supervisor як risk-overlay (масштаб експозиції) покращує Calmar/DD без
   втрати Sharpe → окремий, слабший продуктовий статус.
3. **Negative**: доведено з числами, що regime-switching у цьому універсумі не має edge →
   напрям закривається, ресурс іде на L2/maker та paper-гейт. Це теж «результат»,
   і він теж вимагає такого ж аудиту (DSR/PBO на selector-політиці), а не «нам здалося».

---

## 2. Що вже встановлено в проєкті (факти, не гіпотези)

| # | Факт | Числа | Джерело |
|---|---|---|---|
| F1 | **Пастка lookahead у селекторі** | «правильний» варіант зі `shift(1)` давав Sharpe **+2.58**, варіант без лага **+2.59**, а лаг на **ВИБІР** — **+0.41** | `experiments/iter7_regime_analysis.py:128-157` |
| F2 | **Чесний тест перемикання (1h, 10 символів, 3y, maker)** | switch **+0.365** (медіана +0.19) vs best-single-H1 **−0.453** vs рівноважний бленд **−0.767**; oracle-single **+0.479**, oracle-switch **+0.827** | `results/iter7_switch_summary.csv` |
| F3 | **Аудит селектора** | DSR селектора = **0.00** на 9/10 символів (макс 0.578 DOGE); PBO(CSCV) **0.44–0.91** при порозі 0.5 | `results/iter7_dsr_pbo.csv` |
| F4 | **Свіжі символи (iter8, 730d fit)** | `map.v2` **−0.288**, taxonomy v1.4 −0.322, equal-weight −0.375, найкращий сингл (funding_carry) −0.124; **0/4** символів позитивні | `results/iter8_validation_summary.csv` (з застереженнями: інша cost-база + інша ревізія коду — `docs/reports/computation_audit_2026-09-11.md` §M3–M5) |
| F5 | **Економіка ТФ — головна вісь** | 1h і нижче збиткові після round-trip витрат (середній Sharpe −0.2…−1.5); невід'ємний edge лише на 4h/1d | `docs/STRATEGY_STATUS.md` §iter9 |
| F6 | **Режимні гейти для пар шкодять** | `hmm_vol_gate` на LINK/BTC: 3y +86.5% → +12.3% (PF 1.74 → 1.33) | `docs/reports/entity_recommendation.md` |
| F7 | **Regime-шар працює як ЕКСПОЗИЦІЯ, не як селектор** | `regime_scale` (фактор 0.25 у high-vol): maxDD −44%→−22%, LINK/ETH −19%→+12%; CSCV PBO = **0.000** | `docs/reports/iter6_regime_scale.md` |
| F8 | **Живі рукави на цей момент** | `pairs_arb` LINK/BTC 1h validated; `ts_momentum` 1d long-only CORE_15 monitoring (SR +1.36, t_NW +2.61) | `docs/reports/LEADERBOARD.md` |
| F9 | **Диверсифікація ТФ реальна** | corr 1w↔1d = **+0.09**, combined 50/50 SR **+1.51** vs 1.30/0.95 | `docs/STRATEGY_STATUS.md` §iter13 |
| F10 | **Чесний гейт проєкту** | OOS SR > 0.3, pos ≥ 50%, DSR > 0.95, PBO < 0.5, n_trades ≥ 100 (1m) / ≥ 30 (3y) | `.agents/skills/overfitting-audit/SKILL.md` |

Кеш даних перевірено перед циклом: `uv run python -m scalper_hft.cli data-audit --days 1095`
→ 15/15 символів `ok`, funding 100%, exit 0.

---

## 3. Діагноз: чому поточна реалізація не працює

1. **Вона торгує там, де немає економіки.** Supervisor за замовчуванням живе на 1h, де
   базовий рівень усіх рукавів ≤ 0 (F5). Перемикання на збитковій базі може лише
   перерозподіляти збиток; у iter7 meta-варіанти зробили **4 000–85 000 угод** проти
   2 200–5 000 у singles — витрати множаться, альфа не додається.
2. **Argmax без порога значущості.** Вибір «найкращого у комірці» без вимоги
   *мінімального розриву* і без гістерезису ловить шум: комірки по 30–100 барів дають
   Sharpe зі стандартною похибкою ≳ 1, тому «найкращий» у комірці — переважно випадковий.
3. **Замала вибірка на комірку.** `fit_days=730` на 9 комірок (3 структури × 3 vol) —
   це ~80 барів на комірку на 1d і ~750 на 1h; при порозі `min_bars=30` це статистично
   порожньо.
4. **Множинність не врахована в рішенні.** 10 символів × 20 варіантів × 9 комірок:
   портфельний `n_trials=270` давав DSR 0.666 при per-symbol 0.821 — тобто навіть
   «обнадійливий» результат помирав саме від множинності (F3).
5. **Мітка режиму не пов'язана з PnL стратегій.** `structure|vol` описує *напрямок і
   волатильність ціни*, а не *умовну дисперсію продуктивності рукавів*. Прямий доказ:
   `supertrend` мав у taxonomy тег `trend_down` — свій найгірший режим (Sharpe −4.82).
6. **Статична карта, навчена на одному вікні.** iter8 на свіжих символах дав 0/4
   позитивних (F4). Карта «режим → стратегія» не переноситься між символами і періодами.
7. **Об'єктив не включає вартість перемикання.** Відбір іде за gross Sharpe; turnover і
   комісії з'являються вже після рішення (звідси 85k угод у `ctx_hedge`).
8. **Немає «якоря».** Політика `flat` у high-vol (1/3 комірок) вимикає і той edge, який є;
   incumbent (ts_momentum 1d long-only) при цьому не має бути вимкнений — тільки
   масштабований (F6, F7).

---

## 4. Що каже зовнішнє знання (короткий синтез, повний огляд — у literature-файлі)

- **Перемикання провалюється не через детектор, а через витрати й множинність.** Класичне
  спостереження практиків — «market-specific strategies» рідко відтворюються, а false
  regime signals і whipsaw з'їдають edge ([QuantConnect: Rage Against the Regimes](https://www.quantconnect.com/forum/discussion/14818/rage-against-the-regimes-the-illusion-of-market-specific-strategies/p1/comment-43614#1), [AhaSignals: False Regime Signals](https://ahasignals.com/learn/regime/false-regime-signals/)).
- **HMM/режимні моделі дають згладжені (smoothed) ймовірності, які не можна
  використовувати онлайн** — тільки filtered/forward, інакше це lookahead
  ([HMM regime detection](https://rstudio-pubs-static.s3.amazonaws.com/1426256_bfdd92ab2e8c4979aa73c18c91d62eec.html), [Blanchard & Goffard, HMM](https://hal.science/hal-04608937/file/Blanchard_Goffard_HMM.pdf)).
- **Volatility-managed portfolios** (Moreira & Muir, *JoF* 2017) — масштабування експозиції
  обернено до волатильності дає значно більш робастний ефект, ніж перемикання стратегій
  ([Volatility-Managed Portfolios](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513)).
  Це прямо узгоджується з F7 проєкту.
- **TSMOM** (Moskowitz, Ooi, Pedersen) — те, що реально реплікується в крипті: long-only
  time-series momentum + vol-targeting ([Time Series Momentum, AQR](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)).
- **Deflated Sharpe / PBO** (Bailey & López de Prado) — обов'язковий інструмент, коли
  політик/комірок десятки ([Min backtest length and deflated SR](https://cocalc.app/github/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/08_ml4t_workflow/01_multiple_testing/README.md)).
- **Meta-labeling** (López de Prado, AFML гл. 3): primary-модель дає *напрям*, secondary —
  *чи брати ставку*. Це, як правило, робастніше за regime-switching, бо не вимагає
  вгадувати стан, а лише фільтрувати якість сигналу.
- **Бектест із режимними мітками** має власні граблі: мітка, обчислена на тому ж барі, що
  й дохідність, дає фальшивий edge ([Regime Blog: backtesting with regime data](https://getregime.com/blog/backtesting-with-regime-data)) — саме F1.

---

## 5. Принципи дизайну v3 (які саме зміни робимо, і чому)

| # | Принцип | Як реалізується |
|---|---|---|
| P1 | **Економіка ТФ понад усе** | цільові ТФ — **4h і 1d**; 1h — тільки як контроль; sub-1h — заблоковано (F5) |
| P2 | **Перемикання лише з підтвердженням** | `min_dwell` (гістерезис) + `min_gap` (розрив Sharpe між кандидатами) + бюджет перемикань на квартал |
| P3 | **М'яке зважування + шринкедж** | ваги ∝ softmax(умовний Sharpe/τ), зширені до безумовного (рівноважного) розподілу (empirical-Bayes/James–Stein) |
| P4 | **Мало станів** | 2 (risk-on / risk-off) + опційно 1 нейтральний; жодних 9 комірок × N стратегій |
| P5 | **Net-of-cost об'єктив** | відбір за Sharpe *після* комісій/slippage, з штрафом за turnover у самому критерії |
| P6 | **Режим має передбачати PnL рукавів, а не ціну** | детектор відбирається за |IC| між станом на t і *майбутнім відносним* результатом рукавів на t+1..t+h (train-only) |
| P7 | **Якір (anchor)** | incumbent (ts_momentum 1d long-only CORE_15) ніколи не вимикається; supervisor лише масштабує експозицію в [0.25, 1.0] або додає рукав |
| P8 | **Meta-labeling як альтернатива** | secondary-модель (triple-barrier + LightGBM) вирішує «брати/не брати», замість вгадування режиму |
| P9 | **Чесна оцінка** | лаг на **РІШЕННЯ**, walk-forward, purge/embargo, DSR з чесним `n_trials`, CSCV/PBO, sensitivity-плато, свіжі символи |
| P10 | **Pre-registration + burn-реєстр** | гіпотеза фіксується ДО прогону; кожне нове вікно додається в `oos_usage.md`; 2 раунди без покращення → закриття напряму |

---

## 6. Backlog гіпотез (порядок = пріоритет)

| ID | Гіпотеза (механізм) | Фальсифіковане передбачення | Раунд |
|---|---|---|---|
| **HB-1** | **Економіка ТФ**: перемикання має edge на 4h/1d, бо там базові рукави не збиткові | На 4h/1d mean OOS SR політики > +0.5 і > best-single на тих самих фолдах; на 1h — ≤ 0 | RS-1 |
| **HB-2** | **Підтвердження замість argmax**: `min_dwell` + `min_gap` зменшують churn і дають вищий net Sharpe, ніж argmax | switch-per-quarter падає ≥ 50%, Sharpe не гіршає; turnover-витрати < 20% gross | RS-1 |
| **HB-3** | **Мало станів**: політика risk-on/risk-off б'є 9-коміркову карту (менша множинність) | PBO risk-on/off < PBO карти і DSR вище при тому ж n_trials | RS-1 |
| **HB-4** | **Детектор ≠ ціна**: стан, що краще передбачає *відносний* PnL рукавів (IC), дає кращий net Sharpe | Ранжування детекторів за \|IC\| на train збігається з ранжуванням на OOS Sharpe | RS-1 |
| **HB-5** | **Шринкедж**: ваги, зширені до рівноважних, робастніші за чисті argmax/softmax | Sharpe(shrink) ≥ Sharpe(softmax) ≥ Sharpe(argmax) на OOS; розкид по фолдах менший | RS-2 |
| **HB-6** | **Якір**: overlay-масштабування incumbent'а (0.25–1.0) покращує Calmar ≥ 20% без втрати Sharpe | ΔSharpe ≥ −0.05, ΔCalmar ≥ +0.2, ΔmaxDD ≤ −15% | RS-2 |
| **HB-7** | **Cost-aware відбір**: включення turnover-штрафу в критерій змінює вибір і покращує net PnL | Net Sharpe(cost-aware) > Net Sharpe(gross-optimal) при тому ж gross | RS-2 |
| **HB-8** | **Meta-labeling** на incumbent'і (triple-barrier, purged CV) додає Sharpe | OOS SR(meta) > SR(primary), DSR ≥ 0.95, p_trade стабільний по фолдах | RS-3 |
| **HB-9** | **Режим дисперсії/кореляції** (absorption-ratio-подібний: частка варіації в 1-й PC портфеля рукавів) кращий за `structure|vol` | IC і net Sharpe вищі за F4-базову карту | RS-4 |
| **HB-10** | **HMM/BOCPD як детектор** (filtered probability + dwell) кращий за rule-based | Net Sharpe і стабільність вищі; PBO ≤ rule-based | RS-4 |
| **HB-11** | **Свіжі символи**: політика, навчена на CORE_15, переноситься на 10 імен поза CORE | ≥ 60% свіжих символів позитивні, mean SR ≥ +0.3 | RS-5 |
| **HB-12** | **Комбінація з pairs** (market-neutral якір) покращує портфельний Calmar | corr ≤ 0.3 і combined SR ≥ max(компонент) + 0.1 | RS-5 |

**Kill-критерії для напряму:** якщо у двох послідовних раундах найкраща honest-політика
не досягає навіть Partial-tier (HB-6) і PBO ≥ 0.5 → напрям закривається з фіксацією
негативного результату (F-style таблиця + оновлення `STRATEGY_STATUS.md`).

---

## 7. Гейт прийнятності (pre-registered; фіксується ДО прогону)

**Tier 1 — Promotion (candidate → аудит → paper-моніторинг):**

| # | Критерій | Порог |
|---|---|---|
| G1 | Mean OOS Sharpe політики (net, 4h або 1d, CORE_15) | ≥ **+0.50** |
| G2 | Частка символів з OOS SR > 0 | ≥ **70%** |
| G3 | ΔSharpe проти incumbent (`ts_momentum` 1d long-only CORE_15) | ≥ **+0.20** |
| G4 | ΔSharpe проти best-single (обраний на тих самих train-фолдах) і проти equal-weight | ≥ **+0.20** |
| G5 | t_Newey–West(20) + stationary-bootstrap 95% CI | t ≥ **2.0**, CI **без 0** |
| G6 | PBO (CSCV) | < **0.25** |
| G7 | DSR при чесному `n_trials` (реєструється у pre-reg) | ≥ **0.95** |
| G8 | Sensitivity: dwell/gap/τ ±20% і комісії ±20% | Sharpe ≥ 0.6 × базового (плато) |
| G9 | Свіжі символи (не використані для навчання політики) | ≥ 60% позитивні |
| G10 | Lookahead: лаг на рішення + mutation-тест у pytest | інваріантність до майбутніх барів |

**Tier 2 — Partial (risk-overlay, окремий статус, НЕ paper-гейт):**
ΔSharpe ≥ −0.05 при ΔCalmar ≥ +0.20 і ΔmaxDD ≤ −15% відносно incumbent'а.

**Tier 3 — Reject:** усе інше. Кожен reject фіксується з числами та причиною.

---

## 8. Протокол циклу (як саме крутиться loop)

```
┌─ RS-k ────────────────────────────────────────────────────────────────────┐
│ 0. PRE-REGISTER  docs/reports/hypothesis_iter<K>_rs.md                    │
│    гіпотези (≤3), метрики, гейт, kill-критерії, n_trials, які вікна палимо│
│ 1. IMPLEMENT     код: політика/детектор + pytest (no-lookahead mutation)   │
│ 2. TEST          experiments/iter<K>_regime_supervisor_cycle.py (WF, folds)│
│ 3. AUDIT         DSR (чесний n_trials) + CSCV/PBO + sensitivity            │
│ 4. ANALYZE       атрибуція: turnover-витрати, switch count, per-fold, per-sym│
│ 5. REPORT        docs/reports/iter<K>_regime_supervisor.md + оновлення     │
│                  STRATEGY_STATUS.md / spec / oos_usage.md (burn registry)  │
│ 6. NEXT          ≤3 нові гіпотези з §6 або нові, народжені з аналізу;      │
│                  якщо Tier-1 досягнуто → аудит повного гейту → paper       │
└───────────────────────────────────────────────────────────────────────────┘
```

Правила циклу:
- **Одна гіпотеза — один раунд.** Не підбирати політику після перегляду OOS: усі
  варіанти фіксуються в pre-reg; OOS-метрії звітуються для ВСІХ варіантів, а не для
  найкращого.
- **Чесний `n_trials`**: у pre-reg рахуємо кількість політик × комірок × порогів і
  записуємо число; DSR рахується від нього.
- **Burn registry**: кожен новий span/символ, використаний у тесті, додається в
  `docs/reports/oos_usage.md` з purpose `iter<K>:rs/<tf>`.
- **Сумісність cost-бази**: усі прогони раунду — з одним `CostModel` (maker/taker
  фіксується в pre-reg), інакше порівняння заборонене (граблі iter7↔iter8, §M3 audit).
- **Лаг на рішення**, а не на серію дохідностей: `decision[t] = f(label[t-1])`,
  `pnl[t] = ret[реалізація decision[t]]` — саме так, як у `apply_choice_map` (F1).

---

## 9. Ризики та запобіжники

| Ризик | Запобіжник |
|---|---|
| Вигоряння даних: усі 15 символів 1h/1d уже використані під інші гіпотези | Для RS: 1d/4h CORE_15 + **10 символів поза CORE** як свіжі (G9); фіксація в oos_usage |
| «Поліпшення» на перевикористаному OOS | Two-stage: selection half (2019–2022) → validation half (2023–2026), як в iter12–14 |
| Множинність політик | Мало політик (≤6), фіксований реєстр, DSR від чесного n_trials, PBO |
| Шум на 4h (коротка історія ~6 років) | Портфельна агрегація + t_NW + bootstrap CI; мінімум 30 угод на клітинку |
| Автокореляція PnL | t_Newey–West + stationary bootstrap (блоки 10) |
| Overlay підігнаний пост-хок (як в iter12 vol-target) | Overlay перевіряється на held-out половині (validation half), інакше → Tier-3 |

---

## 10. Артефакти циклу

| Артефакт | Шлях |
|---|---|
| Цей документ (план, гейт, backlog) | `docs/reports/regime_supervisor_research.md` |
| Зовнішній огляд знань | `docs/reports/regime_switching_literature.md` |
| Pre-registration раунду | `docs/reports/hypothesis_iter<K>_rs.md` |
| Harness раунду | `experiments/iter<K>_regime_supervisor_cycle.py` |
| Результати | `results/iter<K>/*.csv`, `*.md` |
| Звіт раунду | `docs/reports/iter<K>_regime_supervisor.md` |
| Оновлення статусу | `docs/STRATEGY_STATUS.md`, `specs/strategies/regime_supervisor.yaml` |
| Реєстр спаленого OOS | `docs/reports/oos_usage.md` |
