# План: Gap-аналіз дослідження «Стратегії MFT Криптоторгівлі 2026» → scalper-hft

> Статус: **Active gap-план** (2026-09-13) | Джерело: `Стратегії MFT Криптоторгівлі 2026.md`
> Супровідні: [ROADMAP.md](../ROADMAP.md) · [IMPROVEMENT_PLAN_2026.md](../IMPROVEMENT_PLAN_2026.md) · [analysis_hft_2026.md](../analysis_hft_2026.md) · [STRATEGY_STATUS.md](../STRATEGY_STATUS.md)

## Контекст і невід'ємні правила

Проєкт — **MFT pairs_arb LINK/BTC 1h maker**, не субмілісекундний HFT. Дослідження
описує ландшафт MFT-2026 загалом; багато його тез **вже реалізовано** в scalper-hft
(частково глибше, ніж у дослідженні). Цей план — не «впровадити все з дослідження»,
а **закрити конкретні gap'и**, не порушуючи поточної дисципліни:

1. **Paper Gate — P0, не блокувати новими альфами.** Жодних нових стратегій у
   `VALIDATED_PAIRS` і жодного `DRY_RUN=false` до 8 тижнів paper на `paper-v0.2.0`.
2. **Без lookahead. Без зміни z/lookback/regime_scale на повній вибірці.**
3. **Один PR = одна задача.** Червоний тест → фікс → `uv run pytest tests/ -q` →
   `uv run ruff check --fix && uv run ruff format`.
4. **Дослідження локально, робот на VPS через git-тег.** Демон на VPS не чіпати.
5. **Дані лише з LIVE-біржі** (`data-audit --days 1095` exit 1 = стоп).

---

## 1. Зведена карта: дослідження ↔ проєкт

| § | Тема | У проєкті | Стан | Gap |
|---|---|---|---|---|
| 1.1 | Процеси Хоукса | `features/hawkes.py` | Feature, не у live | **G1** |
| 1.2 | VPIN | `features/microstructure.py` | Feature + MM shield | **G1** |
| 2.1 | LightGBM/XGBoost | `ml/trainer.py` (тільки LightGBM) | Research | **G2** |
| 2.1 | DeepLOB (LSTM/CNN) | `ml/lob_models.py`, `train_lob.py` | Stub | **G3** |
| 2.2 | Purged K-Fold / PBO | `validation/cv.py`, `cscv.py` | **Готово** | — |
| 3.1 | Коінтеграція (ADF/Johansen) | `validation/coint_scan.py`, `hedge_ratio.py` | **Готово** | — |
| 3.1 | Kalman hedge | `pairs_arb` `--use-kalman` (off) | W1-K bake-off | **G4** |
| 3.2 | O-U process / half-life | `sparse_basket.py`, `pairs_gate.py` | **Готово** | — |
| 3.3 | Z-score + time stop | `pairs_arb` (time stop ⏳) | Частково | **G5** |
| 4.1 | Avellaneda-Stoikov / GLFT | `strategies/market_maker.py` | Backtest only | **G6** |
| 4.2 | DRL (PPO) для γ | — | Не реалізовано | **G7** |
| 5.1 | Funding rate arb | `funding_arb.py`, `funding_carry.py`, `basis_reversion.py` | Research (спить) | **G8** |
| 5.2 | Трикутний арбітраж (Bellman-Ford) | — | Не реалізовано | **G9** |
| 6.1 | EGARCH / GJR-GARCH / HAR-RV | `features/volatility.py` | Feature, не у portfolio | **G10** |
| 6.2 | Fractional Kelly | `portfolio/risk_budget.py::fractional_kelly()` | Є, **не підключено** | **G11** |
| 6.2 | VaR / CVaR | Hist VaR у `pairs_runner`; CVaR нема | Частково | **G12** |
| 6.2 | Kill switches | `reconcile.KillSwitch`, `SyncEngine` | **Готово** | — |
| 7.1 | VIP tiers / BNB знижка | `config.py` maker/taker 0.02/0.05 | Базові | **G13** |
| 7.2 | WebSocket + Ping/Pong | `ws_user_stream.py`, `bookticker_recorder.py` | **Готово** | — |
| 7.2 | API weight tracking | — | **Не реалізовано** | **G14** |
| 7.2 | Колокація AWS Tokyo | `docs/DEPLOY_PLAN.md` (план) | Не впроваджено | **G15** |
| 7.2 | hftbacktest-стиль queue replay | `backtest/micro_price.py` (спрощений) | Частково | **G16** |
| 8 | Податки UA (Закон 10225-д) | — | Не реалізовано | **G17** |

**Висновок:** 17 gap'ів. З них **6 вже плановано в ROADMAP** (G4, G6, G8, G10,
G15, G16), **5 — «мертве дрітництво»** (G1, G11, G12, G13, G14), **3 — нові
дослідження** (G2, G7, G9), **3 — інфра/ops** (G15, G16, G17). **Жоден не блокує
Paper Gate.**

---

## 2. Виправлення «мертвого дрітництва» (P1, паралельно з paper, без демонів)

Ці речі **вже написані в коді, але не підключені** до рекомендованого контуру.
Аудит 2026-09-12 (ROADMAP §6.6) вже фіксував цю проблему; тут конкретизуємо.

### G1 — Wire VPIN / Hawkes як regime filter для pairs_arb (опційно) ✅

**Навіщо.** Дослідження §1.1–1.2: VPIN + Hawkes — мікроструктурний фільтр
токсичного потоку, перемикає mean-reversion ↔ momentum. У проєкті фічі є, але
живуть лише в `market_maker` shield і `features/`. Для 1h pairs це **опційний**
додатковий gate, не заміна HMM.

**Файли:**
- `scalper_hft/strategies/pairs_arb.py` — додати опційний `vpin_gate` /
  `hawkes_gate` (через `Settings.pairs_vpin_filter`, дефолт `false`)
- `scalper_hft/config.py` — `PAIRS_VPIN_FILTER`, `PAIRS_VPIN_THRESHOLD`
- `tests/test_pairs_arb_vpin_gate.py` (новий) — синтетика: високий VPIN блокує вхід
- `docs/DESIGN.md` §pairs gates — задокументувати як **off за замовчуванням**

**Готово коли:** фіча опційна, дефолт off, тест зелений, документ каже «off».
**Не робити:** не вмикати в `VALIDATED_PAIRS` без окремого OOS PASS + DSR.

**Статус:** зроблено. `PairsArb._apply_flow_toxicity_gate(sig, trades, ...)`: блокує
нові входи при VPIN > поріг АБО |Hawkes-дисбаланс| > поріг; виходи/утримання
зберігає (консистентно з іншими гейтами). Каузально (ffill на kline-індекс).
Параметри `flow_toxicity_gate`/`vpin_threshold`/`hawkes_imbalance_threshold`
(дефолт off). Конфіг `PAIRS_VPIN_FILTER`/`PAIRS_VPIN_THRESHOLD`/
`PAIRS_HAWKES_THRESHOLD`. Spec YAML оновлено. 9 тестів у
`tests/test_pairs_arb_flow_toxicity.py` (блокування, holding, виходи,
no-lookahead, збалансований потік). DESIGN.md оновлено.

### G11 — Fractional Kelly у live sizing (опційно, off за замовчуванням)

**Навіщо.** Дослідження §6.2: фракційний Kelly (0.25–0.5) для sizing.
`portfolio/risk_budget.py::fractional_kelly()` вже є, але **не імпортується** ніде
поза portfolio/tests. Це класичне dead wire.

**Файли:**
- `scalper_hft/live/pairs_engine.py` — опційний `kelly_size` режим поруч із
  vol-target (через `Settings.sizing_mode` у `{fixed, vol_target, kelly}`)
- `scalper_hft/config.py` — `SIZING_MODE`, `KELLY_FRACTION` (дефолт 0.25)
- `tests/test_pairs_engine_kelly.py` — sizing monotone у win-rate
- `docs/DESIGN.md` §risk — «kelly off у VALIDATED_PAIRS, лише після PASS»

**Готово коли:** функція підключена опційно, дефолт `fixed`, тест зелений.
**Не робити:** не вмикати Kelly у paper-gate конфіг (скине годинник).

### G12 — CVaR / Expected Shortfall у portfolio risk

**Навіщо.** Дослідження §6.2: VaR 99% + CVaR. У проєкті лише hist VaR 95% у
`pairs_runner._portfolio_var_ok()`. CVaR (середня втрата за хвостом) відсутній —
а для «товстих хвостів» крипто це критично.

**Файли:**
- `scalper_hft/portfolio/risk_budget.py` — `historical_cvar(returns, alpha=0.99)`
- `scalper_hft/live/pairs_runner.py` — `_portfolio_cvar_ok()` поруч із VaR
  (через `Settings.portfolio_cvar_limit`, дефолт `None` = off)
- `scalper_hft/config.py` — `PORTFOLIO_CVAR_LIMIT`, `PORTFOLIO_CVAR_ALPHA`
- `tests/test_portfolio_cvar.py` — важкий хвіст дає CVaR > VaR
- `docs/reports/cvar_addition_2026.md` — звіт

**Готово коли:** CVaR рахується, опційний gate, тест зелений.
**Не робити:** не вмикати CVaR gate у paper-gate конфіг без окремого PASS.

### G13 — VIP tier config для комісій ✅

**Навіщо.** Дослідження §7.1: VIP 4–9 на Binance дає 0% maker; Bybit Supreme VIP
0.000% maker. У проєкті `config.py` має лише базові 0.02/0.05. Для MFT з 50
угод/день це визначає прибутковість.

**Файли:**
- `scalper_hft/config.py` — `MAKER_FEE_BPS`, `TAKER_FEE_BPS` вже є; додати
  `FEE_TIER` у `{vip0, vip4, vip9, supreme}` + таблиця в новому `fees.py`
- `scalper_hft/backtest/execution.py::CostModel.from_settings` — читати tier
- `tests/test_cost_model_tiers.py` — vip9 maker = 0
- `.env.example` — коментарі про BNB знижку 25% і tier requirements

**Готово коли:** tier обирається, CostModel коректно рахує, тест зелений.
**Не робити:** не змінювати дефолт `vip0` без реального статусу акаунту.

**Статус:** зроблено. `scalper_hft/data/fees.py` — таблиці BINANCE_USDTM_TIERS
(vip0..vip9, vip9=0% maker) та BYBIT_DERIVATIVES_TIERS (vip0..supreme,
supreme=0% maker); `resolve_fees(tier, exchange, bnb_discount=)`; BNB-дисконт
25%. `CostModel.from_settings` читає `fee_tier` і перекриває maker/taker з
таблиці (дефолт vip0 = лишає явні). Конфіг `FEE_TIER`/`FEE_TIER_BNB_DISCOUNT`.
`.env.example` оновлено. 29 тестів у `tests/test_fee_tiers.py`.

### G14 — API weight tracking (`X-MBX-USED-WEIGHT-1M`) ✅

**Навіщо.** Дослідження §7.2: Binance ліміт 6000 ваги/хв; HTTP 429 = backoff,
418 = бан. У проєкті **нема** парсингу заголовка ваги — ризик бану IP при
інтенсивному sweep/live.

**Файли:**
- `scalper_hft/data/client.py` — `WeightBudget` обгортка навколо ccxt; парсити
  `X-MBX-USED-WEIGHT-1M` з response headers; exponential backoff при ≥5500
- `scalper_hft/live/trader_loop.py` — пауза перед order-submit при high weight
- `tests/test_weight_budget.py` — мок headers, backoff спрацьовує
- `docs/OB_RECORDER_RUNBOOK.md` — секція про weight hygiene

**Готово коли:** weight парситься, backoff тестується, документ має розділ.
**Пріоритет:** P1 — безпека інфра, не чіпає альфу.

**Статус:** зроблено. `scalper_hft/data/weight_budget.py::WeightBudget`
(пасивний лічильник, парсить `X-MBX-USED-WEIGHT-1M` з
`exchange.last_response_headers` після кожного ccxt-виклику);
`ExchangeClient._track_weight()` + `throttle_if_needed()` (0.2–5.0 с лінійно);
`create_order` throttle перед submit; конфіг `API_WEIGHT_LIMIT`/`API_WEIGHT_THROTTLE_PCT`;
32 тести в `tests/test_weight_budget.py`; рунбук оновлено. `trader_loop` не
змінювався — throttle вже вбудовано в `create_order` (одна точка).

---

## 3. Дослідження нових підходів (P2, після Paper Gate, з OOS-дисципліною)

### G2 — XGBoost vs LightGBM bake-off

**Навіщо.** Дослідження §2.1: XGBoost і LightGBM — обидва еталони. У проєкті лише
LightGBM. Bake-off — не «заміна», а перевірка чи XGBoost дає ΔSharpe > 0 на
тій самій OOS-вибірці (Narang гл.9 value-added).

**Файли:**
- `scalper_hft/ml/trainer.py` — `ModelBackend` у `{lightgbm, xgboost}` (опційно,
  `xgboost` у `[optim]` extras)
- `scalper_hft/cli/research_backtest.py` — `--ml-backend xgboost`
- `docs/reports/xgboost_lightgbm_bakeoff_2026.md` — звіт за шаблоном W1-K
- `tests/test_ml_backend_xgboost.py` — skip якщо не встановлено

**Критерій PASS:** WF avg OOS Sharpe XGBoost ≥ LightGBM **і** DSR > 0.95 **і**
CSCV PBO < 0.5. Інакше — залишити LightGBM дефолтом.
**Не робити:** не міняти дефолт без PASS. Не додавати XGBoost у hard deps.

### G5 — Time stop для pairs_arb (2× half-life)

**Навіщо.** Дослідження §3.3: якщо позиція утримується довше 2 періодів
напіврозпаду — коінтеграція зламана, примусово ліквідувати. У `pairs_arb`
time stop відсутній (лише z-score exit).

**Файли:**
- `scalper_hft/strategies/pairs_arb.py` — `exit_levels()` з опційним
  `max_hold_bars = 2 * half_life` (через `Settings.pairs_time_stop`, дефолт off)
- `scalper_hft/validation/hedge_ratio.py` — `rolling_half_life(spread)`
- `tests/test_pairs_arb_time_stop.py` — синтетика зламаної коінтеграції
- `docs/reports/pairs_time_stop_2026.md` — OOS PASS/FAIL

**Критерій PASS:** ΔSharpe > 0 на 3y OOS; maxDD не гірший. Інакше — off.
**Не робити:** не вмикати в paper-gate конфіг (скине годинник).

### G7 — DRL (PPO) для γ в Avellaneda-Stoikov (дослідження, P3)

**Навіщо.** Дослідження §4.2: гібрид AS + DRL (PPO) динамічно підбирає γ
(неприйняття ризику) → Sharpe 2.9–4.2. У проєкті DRL відсутній. Це **окремий
дослідницький трек**, не production без L2.

**Файли:**
- `scalper_hft/ml/drl_mm.py` (новий) — PPO агент на `gym`-симуляторі LOB
- `scalper_hft/backtest/lob_sim.py` (новий) — event-driven LOB simulator
- `tests/test_drl_mm_smoke.py` — 1 episode без crash
- `docs/reports/drl_mm_research_2026.md` — звіт; **не** production gate

**Критерій PASS (research):** DRL γ-adaptive Sharpe > статичного AS на L2 OOS.
**Не робити:** не підключати до live без повного L2 + окремого Paper Gate.
**Залежність:** G6 (live MM) + G16 (queue replay) мають бути готові раніше.

### G9 — Графовий трикутний арбітраж (Bellman-Ford) (P3)

**Навіщо.** Дослідження §5.2: орієнтований граф курсів, negative-weight cycle =
арбітраж; Bellman-Ford 0.002 ms. У проєкті відсутній. Це **HFT-трек**, не MFT
pairs — потребує колокації + L2.

**Файли:**
- `scalper_hft/strategies/triangular_arb.py` (новий) — `Strategy` з `requires={l2}`
- `scalper_hft/data/graph.py` (новий) — `BellmanFord` на лог-курсах
- `tests/test_triangular_arb.py` — синтетичний negative cycle
- `docs/reports/triangular_arb_research_2026.md`

**Критерій PASS (research):** виявлення ≥90% синтетичних вікон; PnL > fees на L2 OOS.
**Не робити:** не реєструвати в `REGISTRY` production без колокації + L2 archive.
**Залежність:** G15 (колокація) + G16 (queue replay) + `quality_ok` depth5.

---

## 4. Інфраструктура та операції (P2–P3)

### G3 — DeepLOB revival gate (опційно, P3)

**Навіщо.** Дослідження §2.1: LSTM/GRU + TFT для послідовностей LOB. У проєкті
`ml/lob_models.py::DeepLOB` є, але experimental, потребує PyTorch + L2 tensors.
Revival лише після `quality_ok` depth5 (див. ROADMAP §6.5 HFT-1).

**Файли:**
- `scalper_hft/ml/lob_models.py` — рев'ю інференс-шляху; `requires={l2}`
- `scalper_hft/ml/train_lob.py` — walk-forward + DSR на L2 OOS
- `docs/reports/deeplob_revival_2026.md` — PASS/FAIL
- `tests/test_deeplob_smoke.py` — skip без torch

**Критерій PASS:** WF OOS Sharpe > LightGBM baseline на тих самих фічах + DSR>0.95.
**Не робити:** не додавати torch у hard deps. Не production без L2 archive.

### G6 — Live market making (AS/GLFT) (P3, окремий продукт)

**Навіщо.** Дослідження §4.1: GLFT для perpetuals без термінального часу. У
проєкті `strategies/market_maker.py` має `reservation_price`, `glft_quotes`,
VPIN shield — але **backtest only**. Live MM потребує real L2 queue.

**Файли:**
- `scalper_hft/live/mm_runner.py` (новий) — live MM daemon
- `scalper_hft/live/mm_engine.py` (новий) — AS/GLFT quote loop
- `docs/reports/mm_live_gate_2026.md` — окремий Paper Gate для MM
- `tests/test_mm_engine.py`

**Критерій PASS:** окремий 8-тижневий Paper Gate для MM; не блокує pairs Gate.
**Не робити:** не запускати MM live без `quality_ok` L2 + колокації.
**Залежність:** G15 + G16 + HFT-1 (ROADMAP).

### G10 — EGARCH/HAR-RV у portfolio risk (wire) (P2)

**Навіщо.** Дослідження §6.1: EGARCH/GJR-GARCH для асиметрії шоків; HAR-RV для
1-денних прогнозів. У проєкті вони у `features/volatility.py`, але **не у
portfolio** risk engine. ROADMAP §6.2 W1-V вже це планує.

**Файли:**
- `scalper_hft/portfolio/risk_budget.py` — `egarch_vol_forecast()` для vol-target
- `scalper_hft/live/pairs_engine.py` — опція `vol_method ∈ {realized, egarch, har_rv}`
- `tests/test_portfolio_egarch.py`
- `docs/reports/egarch_vol_target_2026.md` — A/B vs realized

**Критерій PASS:** vol-target з EGARCH дає ΔSharpe > 0 vs realized на 3y OOS.
**Не робити:** не міняти дефолт `realized` без PASS.

### G15 — Колокація AWS Tokyo (ops, P3)

**Навіщо.** Дослідження §7.2: Binance matching у AWS ap-northeast-1; локальна
латентність 200–250 ms, колокація 0.6–1.5 ms. У `docs/DEPLOY_PLAN.md` це план.
Для 1h pairs **не критично**, але для MM/triangular — обов'язково.

**Файли:**
- `docs/DEPLOY_PLAN.md` — секція AWS Tokyo VPS provisioning
- `scripts/deploy_paper.sh` — опція `--region tokyo`
- `docs/reports/latency_benchmark_2026.md` — ping до Binance

**Критерій:** ping < 5 ms до Binance API. **Не робити:** не мігрувати paper-gate
VPS без скидання годинника (новий sqlite + новий тег).

### G16 — hftbacktest-стиль queue replay (P3)

**Навіщо.** Дослідження §7.2: hftbacktest (Numba) моделює queue position +
latency на наносекунді. У проєкті `backtest/micro_price.py::QueuePositionModel`
спрощений. Повний replay потрібен лише для MM/triangular.

**Файли:**
- `scalper_hft/backtest/queue_replay.py` (новий) — L2 event replay
- `tests/test_queue_replay.py`
- `docs/reports/queue_replay_2026.md`

**Критерій:** fill parity vs live MM ≥ 95% на L2 sample.
**Залежність:** G6 + `quality_ok` depth5.

### G17 — Податковий аудит-лог для UA (Закон 10225-д) (P2)

**Навіщо.** Дослідження §8: 23% ПДФО+ВЗ на прибуток лише при виході у фіат;
крипто-обмін не оподатковується. Для фінмоніторингу потрібен повний аудит
API-логів (документальне підтвердження витрат на придбання).

**Файли:**
- `scalper_hft/live/tax_audit.py` (новий) — експорт trades у CSV/JSON з
  cost-basis tracking (FIFO); окремо «обмін» vs «вихід у фіат»
- `scalper_hft/cli/ops.py` — `cmd_tax_report --year 2026`
- `tests/test_tax_audit.py`
- `docs/reports/tax_audit_ua_2026.md` — інструкція для бухгалтера

**Критерій:** повний лог угод з cost-basis; окрема подія «вихід у фіат».
**Не робити:** не давати податкові поради — лише аудит-дані.

---

## 5. Порядок виконання та пріоритети

```text
P0 (зараз) — Paper Gate closure (ROADMAP §6.0). Цей план НЕ блокує Gate.
P1 (паралельно з paper, без демонів):
   G14 API weight tracking      ← безпека інфра, 1–2 д
   G13 VIP tier config           ← комісії, 0.5–1 д
   G12 CVaR у portfolio           ← risk, 1–2 д
   G11 Fractional Kelly wire      ← risk, 1 д
   G1  VPIN/Hawkes gate (опція)   ← research, 1–2 д
P2 (після Paper Gate, з OOS PASS):
   G5  Time stop pairs_arb        ← 1–2 д + bake-off
   G2  XGBoost bake-off           ← 3–4 д
   G10 EGARCH/HAR-RV wire         ← 2–3 д
   G17 Tax audit-log UA           ← 2–3 д
P3 (окремий HFT-трек, не pairs):
   G3  DeepLOB revival            ← після L2 archive
   G6  Live MM (AS/GLFT)          ← після L2 + колокація
   G7  DRL γ-adaptive MM          ← після G6
   G9  Triangular arb             ← після G15 + G16
   G15 Колокація AWS Tokyo        ← ops
   G16 Queue replay               ← після L2
```

**Тверде правило:** жоден P2/P3 не змінює дефолт `VALIDATED_PAIRS` без
`cell_audit --audit-mode final` PASS + DSR > 0.95 + CSCV PBO < 0.5.

---

## 6. Зведена таблиця задач

| ID | Завдання | Пріоритет | Дні | Блокує Gate? | Залежності |
|---|---|---|---|---|---|
| G14 | API weight tracking | P1 | 1–2 | ні | ✅ |
| G13 | VIP tier config | P1 | 0.5–1 | ні | ✅ |
| G12 | CVaR portfolio | P1 | 1–2 | ні | ✅ |
| G11 | Fractional Kelly wire | P1 | 1 | ні | ✅ |
| G1 | VPIN/Hawkes gate (опція) | P1 | 1–2 | ні | ✅ |
| G5 | Time stop pairs_arb | P2 | 1–2 | ні | ✅ |
| G2 | XGBoost bake-off | P2 | 3–4 | ні | Paper Gate |
| G10 | EGARCH/HAR-RV wire | P2 | 2–3 | ні | ✅ |
| G17 | Tax audit-log UA | P2 | 2–3 | ні | — |
| G3 | DeepLOB revival | P3 | — | ні | L2 archive |
| G6 | Live MM (AS/GLFT) | P3 | — | ні | G15, G16, L2 |
| G7 | DRL γ-adaptive MM | P3 | — | ні | G6 |
| G9 | Triangular arb | P3 | — | ні | G15, G16, L2 |
| G15 | Колокація AWS Tokyo | P3 | ops | ні | — |
| G16 | Queue replay | P3 | — | ні | L2 archive |

---

## 7. Що категорично НЕ робити з цього плану

- Не вмикати жоден опційний gate (VPIN, Kelly, CVaR, time stop, EGARCH) у
  `paper-v0.2.0` конфіг без окремого OOS PASS — це скине 8-тижневий годинник.
- Не додавати XGBoost / torch у hard deps — лишити в `[optim]` extras.
- Не реєструвати `triangular_arb` / live MM у `REGISTRY` production без
  `quality_ok` L2 + колокації + окремого Paper Gate.
- Не змінювати `MAKER_FEE_BPS` / `TAKER_FEE_BPS` дефолти без реального статусу
  акаунту (інакше бектест буде брехати).
- Не давати податкові поради (G17) — лише аудит-дані для бухгалтера.
- Не мігрувати paper-gate VPS на AWS Tokyo без скидання sqlite + нового тегу.

---

## 8. Відкриті питання для користувача

1. **G7 DRL / G9 triangular** — це окремий HFT-продукт, не MFT pairs. Чи
   взагалі включати в цей репозиторій, чи окремий форк?
2. **G17 податки** — чи потрібен зараз, чи після першого live-прибутку?
3. **G15 колокація** — чи є бюджет на AWS Tokyo VPS (~$50/міс), чи спочатку
   довести pairs Gate на поточному VPS?
4. **G2 XGBoost** — чи варто додавати другу ML-залежність, чи LightGBM
   достатньо для MFT horizons?

---

*Наступний перегляд: після тегу `paper-v0.2.0` або закриття всіх P1 задач.*
