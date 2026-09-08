# План змін: здоров’я платформи → одна чесна комірка

Стан на 2026-09-08. Попередній цикл ([CHANGE_PLAN.md](CHANGE_PLAN.md), фази A–E)
закритий у коді. Деплой paper: [DEPLOY_PLAN.md](DEPLOY_PLAN.md) (тег
`paper-v0.1.0` є; 8 тижнів на VPS — ні).

Мета цього плану — **не нові альфи**. Закрити дірки, через які демон може
торгувати після аварії, paper стартує без аудиту, а документи продають ширший
edge, ніж є в `VALIDATED_PAIRS`. Валідований кандидат лишається
**pairs_arb 1h maker, LINK/BTC, regime_scale=0.25**.

Контекст аудиту: огляд 2026-09-08 (live/backtest/validation vs STRATEGY_STATUS).

## Принципи

1. Один PR = одна фаза. Не змішувати KillSwitch з правками STRATEGY_STATUS.
2. Кожна кодова фаза: спочатку червоний тест, потім фікс →
   `uv run pytest tests/ -q` → `uv run ruff check --fix && uv run ruff format`.
3. Live з реальними коштами — **поза цим планом**. Код live-адаптера можна
   робити fail-closed, але CLI/systemd live не з’являються.
4. `use_kalman` і `legging_mode=chase` лишаються не-дефолтом (правило A–E).
5. Без lookahead. Без зміни z/lookback на повній вибірці.
6. Спеки стратегій (`specs/strategies/`) не чіпати, поки фаза не змінює
   логіку альфи. Цей план альфу не змінює.

## Поза скоупом (свідомо не чіпати)

- Нові 1m/5m taker-скальпи, VWAP/EMA/ORB.
- Live `market_maker` / `ob_imbalance` без L2 (див. [L2_DATA_PLAN.md](L2_DATA_PLAN.md)).
- Увімкнення відхилених: `mean_reversion`, `cvd_momentum`, `funding_*`,
  `basis_reversion`, `hmm_reversion`, `supertrend` як directional.
- Kalman як дефолт; chase як дефолт.
- ERC у live, поки в `VALIDATED_PAIRS` одна пара.
- Підключення `exit_ladders` / `portfolio/risk_budget.py` у paper-цикл
  (фаза L, лише після gate і другої пари).
- `DRY_RUN=false`, тег `live-v*`, systemd live-юніт.
- Перейменування репозиторію / пакету (HFT у назві) — окреме рішення, не цей цикл.

## Що вже є і не треба переписувати

| Шар | Є | Не чіпати |
|---|---|---|
| Paper pairs persist/daemon/control.json | `live/store.py`, `_paper_loop`, `deploy/scalper-paper-pairs.service` | протокол SIGTERM |
| Live-адаптер ніг (мок) | `PairsLiveAdapter` / `PairsLiveRunner` | не додавати CLI |
| IntentStore | працює в `LiveTrader` | розширити на pairs, не замінювати формат |
| Chase = taker, ADF-kill, WS `/public`/`private` | фази A–E | регресійні тести |
| `VALIDATED_PAIRS` | лише LINK/BTC lb=120, regime_scale | не повертати XRP/ETH без нового PASS |

---

## Порядок

```text
H1  KillSwitch fail-closed у демоні          ← P0, блокер будь-якого live
H2  Hydrate без ціни 1.0                     ← P0
H3  IntentStore на pairs live (той самий coid) ← P0, до CLI live
H4  Audit-гейт пари (не двох ніг окремо)     ← P1, блокер чесного paper
H5  Документи = код                           ← P1, можна паралельно з H1–H3
H6  IS → підказка CostModel у paper-audit     ← P1 TCA
H7  Jobs: max-attempt / dead-letter           ← P2
H8  Книга стратегій: усі імена REGISTRY      ← P2 поверхня
Ops Paper-gate 8 тижнів на VPS               ← [DEPLOY_PLAN.md](DEPLOY_PLAN.md) фаза 2
L   Пізніше: ERC, risk_budget, L2, live-v*   ← не починати, доки Ops відкритий
```

H1–H3 — послідовно (той самий `pairs_live` / `_paper_loop`).
H5 можна паралельним PR.
H4 залежить від формату `verdict_store` — окремий PR після або поруч з H5.
H6 після H1 (не чіпати loop, поки KillSwitch не зелений).

Оцінка коду H1–H8: **5–8 робочих днів** при одному PR/фазу. Ops — 8 тижнів
календаря, не коду.

---

## Фаза H1 — KillSwitch не ковтається демоном (P0)

**Навіщо.** `PairsLiveRunner.step` ловить `KillSwitch` лише навколо
`reconcile_runtime()`. `_unwind_live_leg` кидає `KillSwitch` з `on_bar` →
`_paper_loop` робить `except Exception: logger.warning` і крутить далі.
Однонога позиція після невдалого unwind лишається, демон не ставить
`pause+flatten`.

**Файли**

- `scalper_hft/live/pairs_runner.py` — `_paper_loop`, `PairsLiveRunner.step`
- `scalper_hft/live/control.py` — без змін API; виклик `save_control`
- `tests/test_pairs_live.py` — нові кейси
- `tests/test_paper_persist.py` / `tests/test_pairs_paper.py` — `_paper_loop`

**Поведінка**

| Подія | Зараз | Стати |
|---|---|---|
| `KillSwitch` з `reconcile_runtime` | pause+flatten, `killed:killswitch`, цикл живий | без змін |
| `KillSwitch` з `on_bar` / unwind | warning, `error:…`, наступний бар | як reconcile: `save_control(pause=True, flatten=True)`, action `killed:…` |
| `_paper_loop` бачить `killed:` або `KillSwitch` | ігнорує, sleep, наступна ітерація | `halt.set()`, більше `step` немає |
| Інший `Exception` (мережа klines) | warning, цикл живий | **без змін** (не перетворювати всі помилки на kill) |

`_paper_loop` має ловити `KillSwitch` **окремо** від `Exception`, навіть якщо
`step` пропустить. Для paper-only `PairsPaperRunner` `KillSwitch` майже не
буде (reconcile no-op), тож поведінка paper не змінюється, крім тесту на
синтетичний raise.

**Тести (червоні спочатку)**

1. `test_live_step_unwind_kill_pauses_control` — мок `fail_market_orders=True`
   після одноногого філу → `step()` повертає `killed:…`, `control.json`
   має `pause=true`, `flatten=true`.
2. `test_paper_loop_halt_on_killswitch` — `step` кидає `KillSwitch` → loop
   робить рівно 1 ітерацію, `on_stop` викликано, немає другої спроби `step`.
3. Регресія: звичайний `Exception("boom")` у paper loop як і раніше дає
   `error:boom` і **не** ставить pause (щоб дрібна помилка klines не flatten-ила
   паперовий рахунок).

**Готово коли:** unwind-KillSwitch не може пережити наступний бар; pytest
зелений; paper daemon без live-адаптера не змінює частоту ітерацій.

---

## Фаза H2 — гідратація без ціни 1.0 (P0)

**Навіщо.** `PairsLiveAdapter._fetch_last_price` при помилці REST повертає
`1.0`. Гідратація після рестарту відкриває локальну позицію за фіктивною
ціною → UPNL/розмір брешуть, reconcile може «зійтись» за стороною й розміром,
але не за mark.

**Файли**

- `scalper_hft/live/pairs_live.py` — `_fetch_last_price`, `_hydrate_from_exchange`
- `tests/test_pairs_live.py` — мок без klines / `fail_fetch_klines`

**Поведінка**

- Немає ціни (порожній batch, exception, `close<=0`) → `KillSwitch`, не `1.0`.
- Успіх: last close з `fetch_klines` як зараз.
- `start_live` гідратація: якщо хоч одна нога без ціни — не відкривати
  частковий рахунок (або все, або kill).

**Тести**

1. `test_hydrate_missing_price_is_killswitch` — `klines={}`, порожній локальний
   + позиції на біржі → `KillSwitch`, `account.positions` порожній.
2. Регресія: нормальні klines 100/50 як у поточному моку → гідратація ok.

**Готово коли:** жоден шлях гідратації не викликає `open_position(..., 1.0)`
через fallback; тест на відсутність klines червоний на старому коді.

---

## Фаза H3 — ідемпотентний coid на pairs live (P0)

**Навіщо.** `LiveTrader` уже має `IntentStore` (JSON, retry = той самий id).
`PairsLiveAdapter._place_one_leg` щоразу робить `next_client_order_id("shp")`.
Таймаут після accept біржі + повтор → другий ордер на ту саму ногу.

**Файли**

- `scalper_hft/live/pairs_live.py` — `_place_one_leg`, chase (`shc`), unwind (`shu`)
- `scalper_hft/live/intent_store.py` — без зміни формату; інжект path у тестах
- `tests/test_pairs_live.py`
- опційно `tests/test_intent_store.py` (якщо ще немає кейсу pairs-ключа)

**Ключ наміру (стабільний між retry, унікальний між барами)**

```text
{pid}:{kind}:{symbol}:{side}:{bar_ts_iso}:{want}
kind ∈ {entry, chase, unwind}
```

`pid` уже є на engine. `bar_ts` — час закритого бара, не `now()`.

**Поведінка**

| Подія | coid |
|---|---|
| Перший `create_order` наміру | `next_client_order_id`, `IntentStore.put` |
| Retry того ж наміру | `IntentStore.get` → той самий coid |
| Філл / cancel / unwind завершено | `IntentStore.pop` |
| Новий бар | новий ключ (інший `bar_ts`) |

Не міняти `LiveTrader` шлях, окрім спільного store, якщо захочете один файл
`results/intent_coids.json` (прийнятно).

**Тести**

1. `test_place_leg_retry_reuses_coid` — перший `create_order` кидає після того,
   як клієнт уже записав ордер з coid A; повтор → `client_order_id is A`,
   не новий UUID. Мок рахує унікальні coid (має бути 1).
2. `test_new_bar_gets_new_coid` — два бари, два різні coid.
3. Регресія: rollback другої ноги як зараз (скасувати першу).

**Готово коли:** повтор place тієї ж ноги на тому ж барі не може народити
другий клієнтський id; Binance-дедуп працює за контрактом тесту.

---

## Фаза H4 — audit-гейт для пари, не двох directional-комірок (P1)

**Навіщо.** `require_audit_pass(strategy, [leg1, leg2], interval)` шукає PASS
для `pairs_arb × LINKUSDT` і `pairs_arb × BTCUSDT` окремо. Це не та комірка,
яку досліджують (`pairs` CLI, iter6). Paper при `REQUIRE_AUDIT_PASS=false`
лише логує warning — суперечить AGENTS.md.

`cell_audit.OOS_SHARPE_MIN = 0.3` відсік би поточний pairs SRh ≈ 0.006.
Потрібен **окремий pairs-гейт**, а не підганяти Sharpe.

**Файли**

- `scalper_hft/validation/verdict_store.py` — поле `pair` (напр. `LINKUSDT/BTCUSDT`)
- `scalper_hft/live/audit_gate.py` — `audit_gate_check_pair`, `require_pair_audit_pass`
- `scalper_hft/validation/cell_audit.py` — **не** знижувати `OOS_SHARPE_MIN`;
  додати `pairs_pass_reasons` або окремий `validation/pairs_gate.py`
- `scalper_hft/live/pairs_runner.py` — `PairsPaperRunner` / portfolio: fail-closed
  на пару, ігноруючи «два символи»
- `scalper_hft/cli/pairs.py` — запис вердикта після `overfit`/`pairs` WF
- `.env.example` — коментар: VPS paper = гейт завжди; локальний research paper
  може лишити env, але `paper-run-pairs` fail-closed у коді
- `tests/test_audit_gate.py`

**Правило PASS для pairs (зафіксувати в коді + STRATEGY_STATUS)**

Не Sharpe 0.3. Мінімум для `pairs_arb` 1h maker:

| Метрика | Поріг | Звідки |
|---|---|---|
| Частка WF-вікон з OOS > 0 | ≥ 0.55 | STATUS LINK/BTC 65% |
| CSCV PBO | < 0.50 (краще 0.20) | iter6b PBO=0.000 |
| n_trades на вікні аудиту | ≥ 20 | рідкі входи |
| 3y maxDD | документується, не авто-PASS | iter6 |
| Параметри | збіг з `VALIDATED_PAIRS` (z, lb, regime_scale) | інакше FAIL |

Вердикт ключ: `(strategy, pair, interval)`, не `(strategy, symbol)`.

**Поведінка старту**

- `paper-run-pairs` / `PairsPortfolioRunner`: завжди `require_pair_audit_pass`
  (не залежить від `REQUIRE_AUDIT_PASS`).
- Немає запису / FAIL / вік > `AUDIT_MAX_AGE_DAYS` → `RuntimeError`, процес
  не входить у loop.
- `PairsLiveRunner` уже кличе `require_audit_pass` — перевести на pair-гейт.
- Одноразовий bootstrap: скрипт або CLI
  `uv run python -m scalper_hft.cli overfit --strategy pairs_arb …` має
  вміти записати pair-вердикт. Поки запису немає — **не** крутити paper
  «щоб не простоював»; це і є гейт.

Міграція: перед мерджем H4 прогнати audit LINK/BTC і покласти рядок у
`results/audit_verdicts.jsonl` (файл git-ignored). У тестах — tmp_path.

**Тести**

1. PASS на `LINKUSDT/BTCUSDT` + відсутній PASS на `BTCUSDT` як символ → paper
   **стартує** (старе правило б впало).
2. Немає pair-вердикта → `RuntimeError` у `PairsPaperRunner.__init__`.
3. Протухлий вердикт (> max_age) → FAIL.
4. Directional `cell_audit` як і раніше ріже Sharpe ≤ 0.3 (регресія).

**Готово коли:** неможливо стартувати `paper-run-pairs` без свіжого pair-PASS;
два пороги (directional vs pairs) описані в STRATEGY_STATUS одним абзацом.

---

## Фаза H5 — документи збігаються з кодом (P1)

**Навіщо.** TODO/ROADMAP досі цитують Aug-30 (+15.9% XRP, портфель +12.5%).
ROADMAP: «23 файли тестів»; README — 65; диск — 79. DESIGN обіцяє «повний LOB»
при `prepare_lob_tensors = pass`. `strategy_book` LINK headline `lb=120: +86%`
без regime_scale (iter6: +86%→+51% на 3y, live factor=0.25).

**Файли (лише текст, без логіки альфи)**

- `docs/TODO.md` — прибрати Aug-30 таблицю з «зроблено»; посилання на STATUS
- `docs/ROADMAP.md` — дата, лічильник тестів, «валідований кандидат = LINK/BTC»,
  посилання на цей план
- `docs/DESIGN.md` — LOB = stub; універсум пар = одна комірка в live-конфігу
- `docs/CHANGE_PLAN.md` — шапка «цикл A–E закритий → CHANGE_PLAN_HEALTH.md»
- `README.md` — лічильник тестів не хардкодити або оновити + «не HFT»
- `scalper_hft/research/strategy_book.py` — headline LINK з regime_scale 0.25
  і 3y +51% (або формулювання «див. STATUS», без +86% як факт live)
- `docs/DEPLOY_PLAN.md` блок «Стан коду» — тег уже є (2026-09-08)

Не переписувати `docs/reports/iter6_*.md` (історичні звіти). На початку
STATUS уже є примітка про застарілі Aug-30 цифри — лишити, стиснути дубль
у TODO.

**Готово коли:** grep по репо на `+15.9%` і `23 файли` не знаходить їх як
поточний факт (лише в `docs/reports/` як історія); `strategy_book` не
суперечить `VALIDATED_PAIRS`.

---

## Фаза H6 — замикання TCA: IS → підказка, не тихий CostModel (P1)

**Навіщо.** `calibrate_from_is` / `CostModel.with_is_slippage` покриті
`tests/test_cost_calibration.py`, але жоден paper/live runner їх не викликає.
Документи фази D казали «зроблено». Бектест лишається з `.env` slippage, paper
пише IS у журнал.

**Не робити:** автоматично підставляти новий `slippage_frac` у наступний бар
paper (змінює облік посеред прогону, ламає paper-gate).

**Файли**

- `scalper_hft/validation/paper_audit.py` — блок «рекомендований slippage з IS
  (maker / chase), поточний CostModel, дельта»
- `scalper_hft/cli/paper.py` або `cli/pairs.py` — `paper-audit` уже є; розширити
- `scalper_hft/live/is_log.py` — якщо треба читати sqlite, не новий формат
- `tests/test_cost_calibration.py` або `tests/test_paper_audit.py`

**Поведінка**

- `paper-audit` друкує median IS bps maker vs chase vs поточний `SLIPPAGE_BPS`.
- Якщо |дельта| > 1 bp — статус `warn` (не падати).
- Опційно: `uv run python -m scalper_hft.cli cost-calibrate --dry-run` показує
  що було б у `.env`; запис у `.env` лише явною командою (не цей PR, якщо
  зайве).

**Готово коли:** на фікстурі з IS-журналом тест бачить різні maker/chase
медіани; runner paper **не** змінює комісії на льоту.

---

## Фаза H7 — poison jobs не крутяться вічно (P2)

**Навіщо.** `JobStore.reap_stale` переводить `running` → `queued` і
`attempt+1` без стелі. Воркер, що падає на одній задачі, зациклює її.

**Файли**

- `scalper_hft/research/jobs.py` — `reap_stale`, константа `MAX_ATTEMPTS = 3`
- `scalper_hft/app_pages/jobs.py` — показати `failed` + attempt
- `tests/` існуючі job-тести (знайти `test_jobs` / sprint)

**Поведінка**

- Після `attempt >= MAX_ATTEMPTS` stale → `status=failed`,
  `error='stale heartbeat (max attempts)'`, не `queued`.
- `force_requeue` як зараз дозволяє ручний rerun (CLI `job rerun`).
- `MAX_ATTEMPTS` не чіпає успішні задачі.

**Тести**

1. `attempt=2`, stale reap → queued, attempt=3.
2. `attempt=3`, stale reap → failed, не queued.

**Готово коли:** отруйна задача не з’являється знову в `queued` без `rerun`.

---

## Фаза H8 — книга стратегій покриває REGISTRY (P2)

**Навіщо.** У REGISTRY є `smc_fvg`, `stoch_rsi`, `cross_momentum`; у
`STRATEGY_BOOK` їх немає → `lane_for` = `"research"` мовчки. Dashboard не
відрізняє «можна крутити paper» від «порт з від’ємним OOS».

**Файли**

- `scalper_hft/research/strategy_book.py` — рядки + lane
- `docs/STRATEGY_STATUS.md` — короткі рядки в «нові / відхилені»
- `tests/test_strategy_specs.py` або новий `tests/test_strategy_book.py` —
  кожне ім’я REGISTRY (крім утиліт) має запис у книзі
- дашборд: не торкати торгівлю; лише підпис смуги (уже `select_label`)

**Lane (за звітами, не нові прогони)**

| Ім’я | Lane | Підстава |
|---|---|---|
| `smc_fvg` | research | spec candidate; iter3 4h OOS −0.39 |
| `stoch_rsi` | research | iter3 4h −0.24 |
| `cross_momentum` | research | немає PASS у STATUS |
| `supertrend` | rejected | уже в книзі |
| `regime_supervisor` | research | уже в книзі |

Не додавати їх у `VALIDATED_PAIRS`. Не міняти `generate_signals`.

**Готово коли:** тест `test_strategy_book_covers_registry` зелений;
невідоме ім’я в UI не маскується під «можна в paper-gate».

---

## Ops — Paper-gate (не код цього плану)

Це [DEPLOY_PLAN.md](DEPLOY_PLAN.md) фаза 2. Тут лише чеклист, щоб не забути
залежність: **H4 має бути в тегу, з якого стартує 8 тижнів**, інакше 8 тижнів
paper без PASS.

1. Злити H1–H5 (мінімум H1+H4+H5) у main, новий мінор `paper-v0.1.1` (hotfix
   від `paper-v0.1.0`, не latest experiment).
2. VPS: `scripts/deploy_paper.sh paper-v0.1.1`, `DRY_RUN=true`,
   `.env` не з ноута.
3. Щотижня `paper-audit` vs бектест; maxDD ≤ BT × 1.5.
4. Не міняти z/lb/regime_scale під час прогону (клас C hotfix заборонений).

**Готово коли:** 8 тижнів журналу без обнулення sqlite. Це **не** дозвіл на live.

---

## Фаза L — пізніше (не цей цикл)

| ID | Що | Умова старту |
|---|---|---|
| L1 | `risk_budget` у `PairsPortfolioRunner` | ≥2 пари в VALIDATED_PAIRS **або** явний запит |
| L2 | ERC ваги в live/paper portfolio | друга пара з pair-PASS |
| L3 | CLI `live-run-pairs` + systemd live | Paper-gate закритий + явний запит live |
| L4 | Tardis / черга MM | архів depth5 ≥ 3 міс. |
| L5 | ML meta-label як фільтр угод pairs | окремий spec + DSR; не 1m |
| L6 | mypy blocking на `pairs_live.py` | після H1–H3 |

---

## Критерій «план виконано»

- [x] H1: KillSwitch з unwind зупиняє daemon; звичайний Exception — ні
- [x] H2: гідратація без ціни = KillSwitch, не `open_position(..., 1.0)`
- [x] H3: retry ноги = той самий coid
- [x] H4: `paper-run-pairs` не стартує без pair-PASS; directional Sharpe 0.3 живий
- [x] H5: TODO/ROADMAP/DESIGN/book не суперечать VALIDATED_PAIRS
- [x] H6: `paper-audit` показує IS vs CostModel; fees у loop не пливуть
- [x] H7: третя stale-спроба → `failed`
- [x] H8: REGISTRY ⊂ STRATEGY_BOOK
- [x] `uv run pytest tests/ -q` зелений після кожної кодової фази
- [x] Немає `DRY_RUN=false` у юнітах; немає live CLI
- [x] Kalman/chase не ввімкнені в `deploy/scalper-paper-pairs.service`

## Що вважати успіхом продукту (не цього PR)

Одна комірка LINK/BTC 1h maker на VPS 8 тижнів з `paper-audit` у межах
допусків. Все інше в REGISTRY — дослідження, поки не пройде той самий
pair-гейт (H4).
