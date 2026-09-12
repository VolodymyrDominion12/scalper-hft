# Статус стратегій scalper-hft (консолідовано; оновлено 2026-09-12)

Оновлюється після кожного аудиту. Методологія: повний цикл інвестігейт →
реалізація → тест → аудит (walk-forward + Deflated Sharpe + CSCV/PBO + stress/cohort + paper-replay).

## 🔄 Iteration 14 — CS-momentum / 12-1 TSMOM / друга пара (2026-09-12)

Повний звіт: [reports/iter14_improvement_cycle.md](reports/iter14_improvement_cycle.md).
Pre-registration: [reports/hypothesis_iter14.md](reports/hypothesis_iter14.md).
Три нові механізми (не реоптимізація `ts_momentum` CORE_15). Усі **FAIL** гейт.

- **H14-A true CS-momentum 1d long-only CORE_15 — FAIL.** Selection 2019–22:
  lookback 10/20/60 плато (SR 1.56). Validation 2023–26: SR **+0.81**, t_NW
  **+1.46** < 2.0, CI **[−0.27, +1.84]** містить 0. Long-short контроль ≈ 0.
  MaxDD **−59%** (концентрований alt-beta, не overlay). Не промотується.
- **H14-B 12-1 skip-month TSMOM — FAIL.** Validation SR **+0.42**, t_NW **+0.82**,
  CI містить 0, maxDD **−71%**. Контроль без skip слабко кращий (+0.57 / 1.05) —
  skip не додає. Академічна специфікація на крипті не тримається.
- **H14-C друга пара з IS coint-scan — FAIL.** Топ-3 нових (AVAX/NEAR, ADA/UNI,
  AAVE/LINK): усі validation ret **−5…−8%**. AVAX/NEAR єдиний з WF pos 63%
  (гейт вікон PASS), але cumulative PnL від'ємний → не candidate.
- **H14-D value-added** vs ts_momentum 1d: corr CS **+0.60** / skip **+0.49**;
  combined не б'є standalone ts.

**Висновок:** нових комірок немає. Paper стартує на наявному наборі
(`pairs_arb` LINK/BTC 1h validated + `ts_momentum` 1d/4h monitoring).
Дешеві осі альфи вичерпано (iter9→14); наступний крок — **8 тижнів paper-gate**,
не ще один sweep.

## 🔄 Iteration 13 — ts_momentum на тижневому ТФ (1w), two-stage (2026-09-12)

Повний звіт: [reports/iter13_weekly_momentum.md](reports/iter13_weekly_momentum.md).
Pre-registration: [reports/hypothesis_iter13.md](reports/hypothesis_iter13.md).
Перше використання **нативних 1w klines** (2019–2026, CORE_15) — остання
неспалена вісь за рекомендацією iter9 §7.

- **H13-A ts_momentum 1w long-only CORE_15 — FAIL (не промотується).**
  Selection half (2019–22) обрала lookback=13w (плато: усі 4–26w позитивні,
  SR 1.05–1.39). Validation half (2023–26): SR **+0.58**, t_NW **+1.11** < 2.0,
  CI **[−0.30, +1.39]** містить 0 → за pre-registered правилом НЕ monitoring.
  Повний OOS: SR +0.96, t_NW +1.85; 14/15 символів > 0.
- **Контроль long-short слабший** (validation +0.38 vs +0.58) — підтверджує
  механізм iter11: edge = відмова від шортів.
- **H13-B value-added 1w⊕1d — PASS як діагностика**: corr 1w↔1d лише **+0.09**,
  combined 50/50 SR **+1.51** (t_NW 2.61) проти 1.30 (1d) і 0.95 (1w), maxDD
  кращий. Це portfolio-construction evidence, **не paper-gate** — у лідерборді
  тіер candidate.
- Дорогою виправлено: підтримка `1w` в interval-мапах (downloader/engine/
  trader_bars), WF-вікно 1w (50,25)→(50,30) (мінімум рушія 30 барів), а також
  кеш LINKUSDT 1m (дірка 48 діб 2023-07→09 + halt-плита 2021-03-02) і
  калібрування flat-run валідатора (нульовий обсяг = біржовий halt, не «плита»).

**Висновок циклу iter9→13:** простір «дешевих» покращень вичерпано —
monitoring-набір (**pairs_arb LINK/BTC 1h** validated + **ts_momentum 1d/4h
long-only CORE_15** monitoring) лишається оптимальним; подальша підгонка на
тих самих даних = data-snooping. Наступний крок — paper-гейт 8 тижнів
(`paper-v0.2.0`) і накопичення L2 для maker-напряму.

## 🔄 Iteration 12 — improvement loop: two-stage vol-target + value-added (2026-09-12)

Повний звіт: [reports/iter12_improvement_cycle.md](reports/iter12_improvement_cycle.md).
Pre-registration: [reports/hypothesis_iter12.md](reports/hypothesis_iter12.md).

**Цикл «аналіз → покращення → тест» на перевикористаному OOS iter11 (two-stage protocol):**

- **H12-A two-stage vol-target overlay — FAIL.** Сітка σ={20,40,60}; selection half
  2019-22 обрав σ=60 (майже плоско: 1.15-1.17); validation half 2023-26: vt SR **+0.99**
  < plain **+1.11**, t_NW 1.55 < 2.0. → overlay лишається **пост-хок**, НЕ monitoring.
  Причина: overlay з iter11 був підігнаний під повну вибірку; на held-out половині не узагальнюється.
- **H12-B value-added pairs⊕ts — FAIL.** Спільний span 1095d, corr pairs↔ts = **−0.08**.
  pairs_arb LINK/BTC на свіжих 3y: SR **+0.04** (edge деградував), ts_momentum 1d: SR **+1.06**.
  Combined 50/50: SR +0.16, DD −2.07% — слабка нога pairs тягне вниз; диверсифікація зменшує
  DD vs pairs (−4.58%) але не б'є ts standalone.
- **H12-C attribution.** Портфель ts_momentum 1d CORE_15 тягнуть DOGE/NEAR/XRP/SOL
  (contribution +0.09…+0.12); найгірші — LTC (−0.02, єдиний негативний), DOT/LINK (слабкі).

**Висновок:** нових прибуткових комірок не виявлено; monitoring-набір (pairs_arb LINK/BTC +
ts_momentum 1d/4h long-only CORE_15) лишається оптимальним. Покращення не додають вартості
на held-out — це саме та перевірка на overfitting, яку вимагав цикл.

## 🔄 Iteration 11 — незалежне підтвердження long-only (2026-09-12)

Повний звіт: [reports/iter11_improvement_cycle.md](reports/iter11_improvement_cycle.md).
Pre-registration: [reports/hypothesis_iter11_universe.md](reports/hypothesis_iter11_universe.md).
Лідерборд: [reports/LEADERBOARD.md](reports/LEADERBOARD.md).

**Комірки для paper:**
1. **pairs_arb LINK/BTC 1h maker** — validated (без змін).
2. **ts_momentum 1d long-only CORE_15** — monitoring. Реплікація H0: Port Sharpe **+1.37**, t_NW **+2.61**, bootstrap CI **[+0.49, +2.25]** (без нуля), обидва підперіоди > 0.
3. **ts_momentum 4h long-only CORE_15** — monitoring. Нативний 4h (не ресемпл): Sharpe **+1.01**, t_NW **+2.49**, CI **[+0.25, +1.77]**.

**Не paper:** 27 нових імен (H1 t_NW +1.70, CI містить 0). Причина edge — відмова від шортів (контроль short+long: 1d +0.19 / 4h +0.32).

## 🔄 Iteration 9 — відбір стратегій/інструментів/таймфреймів (2026-09-12)

Повний звіт: [reports/iter9_strategy_selection.md](reports/iter9_strategy_selection.md).
Скрин 300 клітинок (3 роки, 15 символів, 1h/4h/1d) → аудит 17 вибіркових комірок
(WF+DSR+CSCV+sensitivity+extensions) → довга історія (нативні **1d klines**, 6.0–6.8 року)
→ портфельний тест.

**Головне:**
- **Економіка ТФ — головна вісь відбору**: 1h і нижче збиткові після round-trip
  витрат (середній Sharpe −0.2…−1.5); невід'ємний edge лише на 4h/1d.
- **Жодна з 17 аудитованих комірок не проходить гейт** (макс. `avg_oos_sharpe`
  0.256; 8/17 з PBO>0.5; DSR=0 всюди). Деградація IS→OOS 60–93% — in-sample
  Sharpe скрину (до +0.93) є артефактом відбору.
- **Новий кандидат: `ts_momentum` 1d (time-series momentum)**, виділений з
  single-symbol гілки `cross_momentum` (spec + реєстр). На 6-річній нативній
  денній історії: середній OOS Sharpe **+0.13** (12/15 символів > 0), позитивний
  в обох підперіодах (2019–22 +0.16, 2023–26 +0.02), плато за `lookback` 20–60.
  Рівноважний портфель 15 символів: **Sharpe +0.76** (maker), середня парна
  кореляція 0.36, **але t_Newey–West 1.87 і bootstrap CI [−0.01, +1.49]**.
  Vol-targeting overlay (та сама альфа, σ-вікно 20–90 днів) піднімає портфель
  до **Sharpe +0.85…+1.09** з `t_NW 2.05…2.54` і CI без нуля →
  **кандидат для моніторингу, НЕ validated** (overlay обрано пост-хок).
- **Відхилено в iter9**: `funding_carry` (усі ТФ, середній SR −0.58…−0.74),
  `stoch_rsi` (4h/1d), `supertrend` 4h як standalone, `mean_reversion` (1–2 угоди
  на 3 роки в дефолті).
- **Виправлено 6 системних дефектів** дослідницького контуру + розв'язано
  суперечність «код ↔ spec ↔ тест» у `cross_momentum` (TSMOM винесено в
  `ts_momentum`). Дефекти (застаріла схема
  `sweep.db`, кеш даних sweep без урахування `needs_funding` → 47/49 carry-клітинок
  падали, завищена на 4–5 порядків множинність DSR, вироджений квінтильний тест
  на дискретному сигналі, відсутність WF-вікон для 1d, валідатор, що відкидав
  справжню денну історію) — деталі у звіті §2.

⚠️ OOS 3-річної сітки спалено (`docs/reports/oos_usage.md`, purpose `audit_cell/cscv`).

## 🔄 Iteration 7 — рейтинг стратегій + regime-перемикання (2026-09-11, 1h maker, 3y, 10 символів)

Повний звіт: [reports/strategy_rating_regime.md](reports/strategy_rating_regime.md).
200 клітинок (20 варіантів × 10 символів), WF train=2500/test=500, фіксовані дефолти.

**Ключове (ранжування за 10 символах, не за найкращим):**
- Жоден варіант (single чи meta) не проходить гейт `cell_audit` (SR>0.3 & pos≥50%).
- **Режимна атрибуція OOS**: у бичому режимі (`trend_up`) моментум/flow-стратегії
  позитивні на **10/10 символах** (`supertrend` +3.96, `smc_fvg` +3.36, `stoch_rsi` +2.77);
  у флеті та ведмежому — **9/10 позитивних у `funding_carry`** (+1.42 / +2.78), а
  моментум там −4.8…−5.3 на всіх 10.
- **Усі реалізовані мета-стратегії негативні** (`ensemble` −0.15…−0.20,
  `regime_supervisor` −0.15…−0.20, `exp3` −0.47) і не б'ють найкращий сингл.
  Причини: (1) taxonomy-пріори суперечать емпіриці (`supertrend` має `trend_down`
  у preferred, а це його найгірший режим); (2) порожній `preferred_regimes`
  = вага 1.0 у всіх режимах → `best_prior` вироджується в «перша універсальна»;
  (3) у пулі дітей немає carry-рукава.
- **Чесний тест перемикання** (карта режим→стратегія з фолду k → фолд k+1, лаг 1 бар):
  selector **+2.64** vs best-single −0.25 vs рівноважний бленд −0.76; rolling —
  +2.58, **30/30 фолд×символ > 0**, **PBO (CSCV) = 0.004**, DSR 0.821 per-symbol
  (6/10 ≥ 0.95) але 0.666 при портфельному n_trials=270 → **обнадійливо, не валідовано**.
- Неактивні на 1h (0 угод за 3 роки): `basis_reversion`, `cvd_momentum`,
  `ob_imbalance`, `ens_vote`; `mean_reversion` — 2, `hmm_reversion` — 1 угода.

**Пріоритет розвитку:** (1) `pairs_arb` LINK/BTC + `regime_scale` — валідоване ядро
(без змін); (2) regime-selector на **емпіричній** OOS-карті (`RegimeStrategyMap`,
Phase 2B) + `funding_carry` у пулі дітей; (3) sensitivity порогів режиму; (4) paper.

⚠️ OOS цього вікна (10 символів × 1h, 2023-09-12 → 2026-09-11) **спалено**:
див. `docs/reports/oos_usage.md`, purpose `iter7:strategy_rating/wf:1h`.

## ✅ Перевалідація pairs на поточному коді (2026-09-07, 3y 1h maker, 49 WF-вікон)

| Пара | WF avg OOS SRh | WF pos вікон | 3y ret | 3y PF | 400д ret | Статус |
|---|---|---|---|---|---|---|
| **pairs_arb LINK/BTC** (z=2.0/lb=240) | +0.0064 | **65%** | **+54.9%** | 1.95 | −9.65% | ✅ найсильніший на 3y (rolling-моніторинг) |
| pairs_arb XRP/BTC (z=2.0/lb=480) | +0.0054 | 43% | −63.0% | 0.77 | **+3.49%** | ⚠ позитивний лише у свіжому вікні |
| pairs_arb LINK/ETH (z=2.0/lb=240) | −0.0009 | 51% | −17.3% | 1.32 | −5.66% | ⚠ потребує доопрацювання |
| pairs_arb BTC/ETH (z=2.0/lb=240) | −0.0028 | 45% | −51.0% | 0.47 | −2.24% | ⚠ потребує доопрацювання |

> Цифри аудиту 2026-08-30 (розділ нижче) отримані ДО фіксів моделі виконання
> (code-review раунди 2–4, Phase B chase/unwind, adverse-selection maker-модель):
> на тому ж вікні (2025-09-01→2026-08-31) поточний код дає XRP/BTC +5.17% (PF 2.12),
> LINK/BTC +1.43% (PF 1.15) — кількість угод та сама, PnL нижчий. Деталі:
> [reports/iter4_directional.md](reports/iter4_directional.md).

## ✅ Iteration 6 — regime_scale overlay (2026-09-08, 3y 1h maker, A/B)

[reports/iter6_regime_scale.md](reports/iter6_regime_scale.md). Масштабування
експозиції до 0.5 у high-vol/trend leg2 (BTC). 0 нових числових параметрів.

| Пара (lb) | 3y ret F→T | 400d ret F→T | 3y maxDD F→T | WF pos F→T |
|---|---|---|---|---|
| LINK/BTC lb=120 | +86%→+51% | −3.0%→−0.4% | −44%→−27% | 65%→61% |
| LINK/BTC lb=240 | +60%→+57% | −9.0%→−6.2% | −41%→−22% | 59%→63% |
| XRP/BTC lb=480 | −63%→−32% | +3.5%→+1.8% | −69%→−38% | 43%→47% |
| LINK/ETH lb=240 | **−19%→+12%** | −5.9%→−0.5% | −43%→−25% | 55%→57% |
| BTC/ETH lb=240 | −51%→−28% | −2.2%→−1.1% | −55%→−33% | 47%→49% |

**Висновок:** maxDD зменшено вдвічі на всіх 5 парах; LINK/ETH перевернуто з
від'ємного у прибуткове; 400d повернення покращено всюди. Очікує CSCV/PBO +
sensitivity перед validated-статусом.

### CSCV/PBO + sensitivity (iter6b, 2026-09-08)

[reports/iter6_regime_scale.md](reports/iter6_regime_scale.md). LINK/BTC lb=240:
- **CSCV PBO = 0.000** для regime_scale=True та baseline (PASS — data-snooping відсутній).
- Sensitivity factor ∈ {0.25,0.5,0.75,1.0}: Sharpe плато (різниця <7%); **Calmar пік при 0.25** (3.12, монотонно спадає).
- Фінальна A/B (lb=120): **400d +0.5%** при fac=0.25 (vs −3.0% baseline), maxDD −3.7% (4× менше).

**Вердикт: PASS ✅** — regime_scale робастний за PBO. Дефолт `regime_scale=True,
regime_scale_factor=0.25` впроваджено у `VALIDATED_PAIRS`.

## ✅ Валідовані кандидати (станом на аудит 2026-08-30 — застарілі, див. перевалідацію)

| Стратегія | Результат | Умови | Статус |
|---|---|---|---|
| **pairs_arb XRP/BTC** | **+15.9%/рік**, 28 угод, maxDD −6.5%, WF OOS>0, 80% вікон, 9/13 міс. >0 | 1h, maker, z=2.0/lb=480 | ✅ найсильніший* |
| **pairs_arb LINK/BTC** | **+12.9%/рік**, 46 угод, maxDD −4.6%, 80% вікон | 1h, maker, z=2.0/lb=240 | ✅ валідований* |
| **pairs_arb LINK/ETH** | **+14.9%/рік**, 51 угода, maxDD −5.7%, 80% вікон | 1h, maker, z=2.0/lb=240 | ✅ валідований* |
| **pairs_arb BTC/ETH** | **+8.3%/рік**, 47 угод, maxDD −4%, 60% вікон | 1h, maker, z=2.0/lb=240 | ✅ валідований* |
| **Портфель 3–4 пар** | **+12.5%/рік, maxDD −3.6%** (XRP/BTC+BTC/ETH+LINK/BTC) | 1h, maker, ERC / Equal Weight | ✅ диверсифікований* |
| pairs_arb ETH/SOL | +5.6%/рік, WF OOS +0.013, 40% вікон | 1h, maker, z=3.0/lb=720 | ⚠ кандидат (12 угод — мала вибірка) |

\* — цифри до фіксів виконання; актуальні — у блоці «Перевалідація» вище.

---

## 🧪 Нові стратегії та мета-моделі (Спринти 1–5)

| Стратегія | Опис | Статус дослідження |
|---|---|---|
| **sparse_basket** | Мульти-активний арбітраж кошика активів (Lasso sparse weights + PCA) | Додано в кодову базу (Спринт 5). Потребує OOS аудиту на портфелі з 5+ монет. |
| **bandit** (`Exp3Bandit`) | Онлайн Multi-Armed Bandit вибір інструментів та стратегій | Додано в кодову базу (Спринт 5). Знижує regret при концепт-дрейфі. |
| **blend / ensemble** | Hedge no-regret блендінг ваг, voting та regime-gating | Реалізовано. Дозволяє динамічно адаптувати ваги суб-моделей. |
| **ml_strategy** | LightGBM meta-labeling з triple-barrier та bet sizing | Інтегровано з мікроструктурними, HMM та GARCH фічами. Вимагає строгих DSR перевірок. |

---

## ⚠ Відхилено (з причинами)

| Стратегія | Причина |
|---|---|
| mean_reversion (+regime) | OOS Sharpe < 0, DSR = 0 |
| cvd_momentum | fee-drag (PF 0.30); надмірна частота угод на 1m барах |
| funding_carry | облік funding виправлено (був баг 480×); тертя > фандінг |
| funding_arb (перп+спот) | фандінг-режим 2025–26 низький (0% періодів >36% річних) |
| basis_reversion | basis надто вузький (1–2.4 bps) проти 2-leg витрат |
| pairs_arb BTC/SOL | відносний тренд SOL (спред не ревертується) |
| pairs_arb на 1m (будь-які) | овертрейдинг → fee-drag |
| **hmm_reversion** (HMM-гейтований MR) | 1 угода/30д — гейт майже повністю вимикає входи; OOS SR −0.05, DSR=0 |
| supertrend 1d (allow_short) — iter4 | OOS +0.50 на 3y, але **−0.15 на 5y** (2021–26): edge період-специфічний (2023–26 альт-аптренд), не робастний |
| regime-гейти на 1h структурі (vhv/tdg) — iter2 | погіршують OOS: 1h EMA-структура запізнюється відносно старшого тренду, гейт ріже правильні лонги |

---

## ⏳ Очікують даних/інфраструктури

| Стратегія | Чого чекає |
|---|---|
| ob_imbalance (depth-weighted) | тривале накопичення depth5 (systemd-сервіс активний) |
| market_maker | L2-дані для точного моделювання черги та adverse selection |

---

## 🔬 Дослідницькі висновки (важливо для нових стратегій)

1. **Фандінг-режим**: високий лише у 2024 Q1; з квітня 2024 — структурно низький. Funding-стратегії — режимно-сплячі ([docs/funding_regimes.md](funding_regimes.md)).
2. **Basis**: перп стійко дешевший за спот (−4 bps) — надто вузький для трейдингу після комісій.
3. **Top-of-book imbalance**: вкрай біполярний (83% |imb|>0.5); depth-weighted стабільніший (43–46%).
4. **Комісії вирішують**: maker-виконання (post-only) обов'язкове для будь-якої carry/arb структури; taker-скальпінг на 1m не виживає.
5. **Micro-price**: використання зваженого стакана для maker-котирувань знижує adverse selection.

---

## Правила розгортання кандидата (pairs_arb)

**Гейт пари** (не directional Sharpe 0.3): комірка `LEG1/LEG2` у
`results/audit_verdicts.jsonl`. `paper-run-pairs` не стартує без свіжого PASS.
Пороги pairs: WF pos ≥ 0.55, PBO < 0.5, n_trades ≥ 20 (`validation/pairs_gate.py`).
Не плутати з `cell_audit.OOS_SHARPE_MIN = 0.3` для односимвольних стратегій.

1. **Paper-режим** з maker post-only (`DRY_RUN=true`).
2. **Моніторинг місячної концентрації**: стоп на пару після 2 місяців поспіль збитку.
3. **Щотижневий аудит**: переоцінка walk-forward на свіжих даних.
4. **Ліміт ноціоналу**: на пару ≤ 30% капіталу; сумарно на весь портфель пар ≤ 60%.
