# План змін: огляд 2026-09-09 → чесний paper-gate

> **Наступний цикл (2026-09-11):** [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md)
> (Wave 0: TCA + R5–R7 + тег `paper-v0.2.0`). Цей файл — R1–R8; R1–R3 і R8 закриті.

Стан на 2026-09-09. Попередні цикли закриті в коді:
[CHANGE_PLAN.md](CHANGE_PLAN.md) (A–E), [CHANGE_PLAN_HEALTH.md](CHANGE_PLAN_HEALTH.md) (H1–H8).
Деплой: [DEPLOY_PLAN.md](DEPLOY_PLAN.md). L2: [L2_DATA_PLAN.md](L2_DATA_PLAN.md).
Джерело пріоритетів: огляд коду 2026-09-09.

Мета цього плану — **не нові альфи**. Зробити paper LINK/BTC порівнюваним з
бектестом, закрити дірки деплою API, розділити профілі research/VPS і лише
потім стартувати (або перезапустити) 8-тижневий Paper-Gate. Валідований
кандидат лишається **pairs_arb 1h maker, LINK/BTC, lb=120, regime_scale=0.25**.

## Принципи

1. Один PR = одна фаза. Не змішувати модель філу з systemd API.
2. Кожна кодова фаза: спочатку червоний тест, потім фікс →
   `uv run pytest tests/ -q` → `uv run ruff check --fix && uv run ruff format`.
3. Live з реальними коштами — **поза цим планом**. Немає `live-run-pairs`,
   немає `DRY_RUN=false` у юнітах, немає тегу `live-v*`.
4. `use_kalman` і `legging_mode=chase` лишаються не-дефолтом.
5. Без lookahead. Без зміни z/lookback/`regime_scale` на повній вибірці.
6. Зміна моделі філу (R1) — **клас B (execution)**. Якщо на VPS уже крутиться
   `paper-v0.1.0`, після R1 потрібен новий тег і **скидання 8-тижневого годинника**.
   Не котити R1 на живий sqlite посеред прогону.

## Поза скоупом (свідомо не чіпати)

- Нові 1m/5m taker-скальпи, VWAP/EMA/ORB.
- Revival `market_maker` / `ob_imbalance` у paper без `quality_ok` L2 і окремого OOS.
- Увімкнення відхилених: `mean_reversion`, `cvd_momentum`, `funding_*`,
  `basis_reversion`, `hmm_reversion`, `supertrend` як directional.
- Kalman як дефолт; chase як дефолт.
- ERC / `risk_budget` / `exit_ladders` **у циклі** (лишаються бібліотекою;
  дроти — фаза L після Gate і другої пари).
- Повний rewrite на `domain/` + `infrastructure/` (R6 — тонкий фасад, не CA).
- Перейменування репозиторію / пакету (HFT у назві).
- Переписування numpy-бектесту на `Decimal`.
- Повернення XRP/ETH/BTC у `VALIDATED_PAIRS` без нового pair-PASS.

## Що вже є і не треба переписувати

| Шар | Є | Не чіпати |
|---|---|---|
| Pair-гейт, KillSwitch, persist, daemon | H1–H8 | протокол SIGTERM / `control.json` |
| `decide_fill` + vector touch/probability | `live/fills.py`, `backtest/pairs.py` | формула `k=0.08` без окремого OOS |
| `require_safe_api_bind` | CLI `api serve` | контракт localhost vs ключ ≥32 |
| Depth recorder на VPS | `scalper-record@.service` | один писач на символ |
| `VALIDATED_PAIRS` | лише LINK/BTC lb=120 | не розширювати цим планом |

---

## Порядок

```text
R1  Fill-parity paper ↔ backtest     ← P0, блокер чесного Paper-Gate
R2  API bind fail-closed у systemd   ← P0, можна паралельно з R1
R3  Профілі .env: research vs VPS    ← P1, у тег разом з R1
R4  Чесний dead wiring (ladders)     ← P1, можна паралельно; НЕ вмикати ladders
R5  sync_depth.sh + quality-звіт     ← P1, паралельно, paper не чіпає
R6  I/O геть з application/          ← P2, без зміни поведінки
R7  Тести trader_loop + вузькі except ← P1, не pairs-демон
R8  Документи = цей план             ← P2, після R1–R3 або паралельно
Ops Paper-gate 8 тижнів на VPS       ← після тегу з R1+R2+R3
L   Пізніше: ERC, Decimal ledger, live-v*, Tardis, CA
```

R1 блокує Ops: 8 тижнів на оптимістичному paper vs інша модель у BT — даремно.
R2 не блокує paper-pairs (інший юніт), але блокує будь-який публічний API.
R3 входить у той самий paper-тег, що R1 (`ENABLE_VOL_TARGET` змінює сайзинг —
теж клас C/B; не вмикати посеред прогону).

Оцінка коду R1–R8: **6–9 робочих днів** при одному PR/фазу.
Ops — 8 тижнів календаря після тегу.

---

## Фаза R1 — Одна модель філу paper і бектест (P0)

**Навіщо.** Зараз **дві різні** політики, попри коментар у
`_maker_pair_positions` («та сама модель, що й paper»):

| Шлях | mid | rng | Наслідок після touch |
|---|---|---|---|
| Paper `PairsEngine._resolve_pending` | рахує mid1/mid2, **не передає** | немає | `decide_fill` ставить mid=limit → **P=1** |
| BT `_maker_pair_positions` | mid з high/low | `run_pairs_backtest` **не передає rng** | draw=1.0; філ лише якщо `1.0 ≤ p(mid)` тобто **p=1 (ліміт≈mid)** |
| BT `_maker_pair_positions_reference` | mid + опційний rng | якщо передати | чесна Bernoulli |

Paper-gate порівнює fill-rate і tracking error з бектестом. При поточних
дефолтах paper заповнює майже кожен touch, BT — майже лише котирування на mid.
Аудит бреше в будь-який бік.

**Файли**

- `scalper_hft/live/pairs_engine.py` — `_resolve_pending`, `to_snapshot` / `apply_snapshot`
- `scalper_hft/backtest/pairs.py` — `run_pairs_backtest` передає той самий Generator
- `scalper_hft/live/fills.py` — без зміни формули `k`; лише споживачі
- `scalper_hft/config.py` + `.env.example` — `MAKER_FILL_SEED` (int, дефолт `42`)
- `tests/test_pairs_fill_parity.py` — новий; регресія `tests/test_pairs_paper.py`,
  `tests/test_pairs_maker_vector.py`

**Поведінка**

| Подія | Зараз | Стати |
|---|---|---|
| Paper pending на барі | `decide_fill(..., high, low)` | `decide_fill(..., high, low, mid=mid, rng=self._fill_rng)` |
| BT maker path | `_maker_pair_positions(..., rng=None)` | `rng = np.random.default_rng(settings.maker_fill_seed)` |
| Рестарт демона | rng заново | snapshot: `fill_rng_state` (bit_generator.state); той самий seed якщо знімка немає |
| `rng is None` у бібліотеці | paper P=1 / BT p≥1 | лишити для юнітів `decide_fill` без mid (сумісність); **прод-шляхи завжди передають mid+rng** |

Seed однаковий у paper і `run_pairs_backtest`. Не читати `/dev/urandom` на VPS.

**Тести (червоні спочатку)**

1. `test_paper_resolve_uses_mid_probability` — ліміт buy далеко нижче mid
   (dist великий, `p ≪ 1`), бар торкається ліміту, `rng` зі seed, що дає
   `random() > p` → `unfilled_prob`, не `filled`.
2. `test_pairs_engine_backtest_fill_decisions_match` — той самий OHLC, той
   самий seed, `strict_both`: послідовність filled/unfilled збігається з
   `_maker_pair_positions` (або з reference) на фікстурі ≥200 барів.
3. `test_fill_rng_survives_snapshot` — `step` → `to_snapshot` → новий engine
   `apply_snapshot` → наступний `random()` той самий, що без рестарту.
4. Регресія: без mid у голому `decide_fill` тести `test_pairs_paper` /
   `test_roadmap` лишаються P=1 після touch (контракт бібліотеки).

**Готово коли:** прод-шляхи paper і BT maker використовують mid+seeded rng;
pytest зелений; у `paper-audit` можна вказати той самий seed, що в BT.
Не деплоїти на running `paper-v0.1.0` без нового тегу.

---

## Фаза R2 — systemd API не обходить bind-захист (P0)

**Навіщо.** `require_safe_api_bind` працює лише в `cli/ops.py` (`api serve`).
`deploy/scalper-api.service` стартує
`uvicorn … --host 0.0.0.0` — порожній `API_SECRET_KEY` на публічному інтерфейсі.

**Файли**

- `deploy/scalper-api.service`
- `scalper_hft/api/server.py` — lifespan
- `scalper_hft/config.py` — без зміни контракту `require_safe_api_bind`
- `deploy/README.md` — API лише localhost + SSH-тунель
- `tests/test_api.py` / `tests/test_p0.py` — lifespan/host з settings

**Поведінка**

| Шлях | Зараз | Стати |
|---|---|---|
| Юніт ExecStart | `--host 0.0.0.0 --port 8000` | `--host ${API_HOST} --port ${API_PORT}` з `EnvironmentFile=.env`; у `.env.example` `API_HOST=127.0.0.1` |
| Lifespan FastAPI | не перевіряє bind | `require_safe_api_bind(settings.api_host, settings)` на старті; падіння процесу, якщо host публічний і ключ слабкий |
| Документація | немає застереження | README: не біндити 0.0.0.0 без ключа ≥32; дашборд — те саме через `require_dashboard_password` |

Не відкривати Streamlit/pgAdmin в інтернет (вже правило DEPLOY_PLAN).

**Тести**

1. Існуючі `test_require_safe_api_bind_*` лишаються.
2. `test_api_lifespan_rejects_public_weak_key` — `api_host=0.0.0.0`, порожній
   секрет → `RuntimeError` (можна викликати той самий хелпер, що lifespan).
3. Регресія: `127.0.0.1` + порожній ключ — ок.

**Готово коли:** юніт за замовчуванням localhost; uvicorn не стартує на
`0.0.0.0` зі слабким ключем навіть якщо CLI обійшли.

---

## Фаза R3 — Два профілі конфігу: research і vps-paper (P1)

**Навіщо.** Дефолти в коді м’які (`REQUIRE_AUDIT_PASS=false`, `HOLDOUT_PCT=0`,
`OOS_ENFORCE_BURN=false`, `ENABLE_VOL_TARGET=false`), щоб не ламати sweeperи.
ROADMAP Paper-Gate хоче vol-target. `.env.example` не документує HOLDOUT/OOS.
Один `.env` з ноута на VPS змішує режими.

**Файли**

- `.env.example` — коментарі HOLDOUT_PCT, OOS_ENFORCE_BURN, MAKER_FILL_SEED
- `docs/env/vps-paper.env.example` — **новий**, без секретів, копіювати на VPS
- `docs/env/research.env.example` — **новий**, для локальної машини
- `docs/DEPLOY_PLAN.md` — крок «не копіювати ноут; взяти vps-paper профіль»
- `tests/test_config_profiles.py` — опційно: парсинг прикладів не містить
  `DRY_RUN=false`

**Контракт профілів (без секретів)**

| Змінна | research (ноут) | vps-paper |
|---|---|---|
| `DRY_RUN` | true | **true** |
| `EXCHANGE` | testnet / порожні ключі | testnet або public-only |
| `ENABLE_VOL_TARGET` | false (експерименти) | **true** (як ROADMAP gate) |
| `MAKER_FILL_SEED` | 42 | **той самий 42**, що BT-аудит |
| `REQUIRE_AUDIT_PASS` | false | true (пари і так fail-closed) |
| `HOLDOUT_PCT` | 0 exploratory / 20 перед final | не використовується ботом |
| `OOS_ENFORCE_BURN` | false / true для final | не використовується ботом |
| `API_HOST` | 127.0.0.1 | 127.0.0.1 |
| `USE_EXIT_LADDERS` | false | false |

`ENABLE_VOL_TARGET=true` змінює ноціонал → **разом з R1 у новому paper-тегу**,
не hotfix на живий прогін.

**Готово коли:** два приклади в git; `.env.example` пояснює HOLDOUT/OOS;
DEPLOY_PLAN посилається на vps-paper профіль; жоден приклад не ставить
`DRY_RUN=false`.

---

## Фаза R4 — Прибрати фальшивий дріт exit_ladders (P1)

**Навіщо.** `LiveTrader` робить `getattr(settings, "use_exit_ladders", False)`.
Поля в `Settings` немає → гілка в `trader_loop` мертва. Документи кажуть
«модуль не в циклі», код удає підключення. **Не вмикати** ladders у paper
(план L / CHANGE_PLAN_HEALTH).

**Файли**

- `scalper_hft/config.py` — поле `use_exit_ladders: bool = False` **або**
  видалити гілку (перевага: явне поле + `.env.example=false`, щоб не
  з’явився третій стан)
- `scalper_hft/live/trader.py` — читати `settings.use_exit_ladders`
- `scalper_hft/live/trader_loop.py` — без зміни логіки
- `.env.example` — `USE_EXIT_LADDERS=false` + коментар «не для pairs-gate»
- `docs/DESIGN.md` §3.5 — «вимкнено прапорцем, дефолт false»
- `tests/test_trader_notional_cap.py` або вузький тест: за замовчуванням
  `trader.use_exit_ladders is False`

**Не робити в цій фазі:** виклики `OneWayTradingLadder` у `PairsEngine`.

**Готово коли:** немає `getattr(..., False)` на неіснуючому полі; дефолт false;
pytest зелений; DESIGN не обіцяє live-ladders.

---

## Фаза R5 — Автосинк depth5 VPS → research (P1)

**Навіщо.** [L2_DATA_PLAN.md](L2_DATA_PLAN.md) Фаза 2 обіцяє `scripts/sync_depth.sh`;
файлу немає. Без архіву на research-машині MM/OBI не аудитити. Paper-pairs
не залежить від цього; робимо паралельно, щоб Фаза C огляду мала дані.

**Файли**

- `scripts/sync_depth.sh` — rsync `data/*depth5*.parquet` + `*bookTicker*`
- `scalper_hft/data/validate.py` — уже є `validate_depth` / `validate_bookticker`
- CLI: `python -m scalper_hft.cli` підкоманда або виклик зі скрипта після rsync
- `docs/OB_RECORDER_RUNBOOK.md` — Крок sync замість «ручний rsync»
- `docs/L2_DATA_PLAN.md` — позначити Фазу 2 як код цього циклу
- Тест: `tests/test_data_validate.py` уже покриває валідатори; скрипт —
  `bash -n` або pytest на dry-run шлях (мокнутий rsync не обов’язковий)

**Поведінка**

```text
sync_depth.sh [--dry-run] [user@host:path]
  rsync -avz --partial
  після копії: validate_depth / validate_bookticker → results/quality_depth.md
  ненульовий exit, якщо quality_ok=false
```

Не комітити parquet. Не відкривати рекордер ключами.

**Готово коли:** скрипт у репо; ранбук містить одну команду; валідатор
fail-closed на битих файлах (уже так для save path).

---

## Фаза R6 — Use-cases без завантаження ринку (P2)

**Навіщо.** ROADMAP H-ARCH позначено ✅, але `run_backtest()` викликає
`ensure_klines` / `download_*` / `get_settings()`. Це фасад, не порти.
Повний `domain/` зараз дорожчий за цінність. Тонкий крок: I/O лишається в
CLI/jobs.

**Файли**

- `scalper_hft/application/use_cases.py` — `run_backtest(req, df, trades=, funding=)`
  або окремий `load_*` не тут
- `scalper_hft/cli/research_backtest.py`, `research/job_handlers.py` — composition
- `scalper_hft/strategies/ml_strategy.py` — `data_dir` інжектити / передавати df;
  прибрати `get_settings().data_dir_abs` з стратегії, якщо торкаєтесь файлу
- `tests/test_health_contracts.py` — якщо є перевірка H-ARCH, послабити claim
  або перевіряти «use_cases не імпортує downloader на рівні модуля»

**Не робити:** новий пакет `infrastructure/`, протокол `OrderBroker`.

**Готово коли:** `use_cases.py` не імпортує `data.downloader` / `data.access`;
CLI як і раніше качає дані; pytest CLI/jobs зелений.

**Статус:** зроблено. `run_backtest(req, df, …)` + `CostModel.from_settings`; I/O в CLI/jobs.

---

## Фаза R7 — Тести `trader_loop` і вузькі винятки (P1)

**Навіщо.** Directional `run_trader_once` покритий у `test_p0.py`, але
`trader_loop.py` не згадується в тестах. `macro_recorder.py` — 0 іменних тестів.
Широкі `except Exception` у `pairs_live` / `pending_orders` / `trader` маскують
помилки стану. Не звужувати мережеві except у WS-рекордері (там цикл reconnect
свідомий).

**Файли**

- `tests/test_trader_loop.py` — новий
- `scalper_hft/live/trader_loop.py` — без зміни семантики, якщо тести зелені
- `scalper_hft/live/sync_engine.py` — не ковтати fetch позицій порожнім `pass`
  (якщо ще є); лог + проброс у reconcile
- `tests/test_macro_recorder.py` — мінімум: flush/reconnect хелпер без мережі
- Точково `pairs_live.py` / `pending_orders.py`: `except (NetworkError, ExchangeNotAvailable)`
  там, де зараз BLE001 на submit; стан ордера — fail-closed

**Тести**

1. `test_trader_loop_pause_skips_signal` — control.pause → немає submit.
2. `test_trader_loop_dd_breaker_flatten` — якщо breaker уже в `test_trader_dd_breaker.py`,
   додати виклик через `trader_loop`, не дублювати всю логіку.
3. `test_sync_fill_symbols_does_not_swallow` — мок fetch кидає → не `pass`.
4. Регресія: мережевий збій klines у paper-loop **не** ставить KillSwitch
   (контракт H1).

**Готово коли:** є `tests/test_trader_loop.py`; немає голого `except Exception: pass`
на fetch позицій; pytest зелений.

---

## Фаза R8 — Документи вказують на цей цикл (P2)

**Навіщо.** ROADMAP/TODO досі «поточний цикл = HEALTH H1–H8». Без покажчика
план не знайдуть.

**Файли**

- `docs/ROADMAP.md` — рядок «поточний цикл коду» → цей файл; Ops лишається ⏳
- `docs/TODO.md` — секція backlog: чекбокси R1–R8; 1b реліз тегу після R1
- `docs/CHANGE_PLAN.md`, `docs/CHANGE_PLAN_HEALTH.md` — шапка «наступний цикл»
- `docs/DESIGN.md` «Наступні кроки» — спочатку R1+Ops, не «портфель 4 пар»
- `docs/STRATEGY_STATUS.md` — не чіпати цифри; лише лінк «не плутати Aug-30 портфель
  з VALIDATED_PAIRS» якщо ще двозначно

**Готово коли:** з ROADMAP/TODO один клік сюди; немає обіцянки «поточний цикл H1–H8»
як відкритої роботи.

---

## Ops — Paper-gate (не код цього плану)

Чеклист після злиття **мінімум R1+R2+R3** (R4–R8 бажано в тому ж тегу, не обов’язково).

1. Тег `paper-v0.2.0` (не котити R1 на `paper-v0.1.0` без скидання годинника).
2. VPS: `scripts/deploy_paper.sh paper-v0.2.0`, профіль `vps-paper.env.example`,
   `.env` не з ноута, `DRY_RUN=true`.
3. Порожній або новий sqlite, якщо модель філу/vol-target змінили облік.
4. Щотижня `paper-audit` vs BT з **тим самим `MAKER_FILL_SEED`**:
   fill-rate, tracking error, maxDD ≤ BT × 1.5, MAE/MFE.
5. Клас C hotfix (z/lb/regime_scale) заборонений під час 8 тижнів.

Деталі операцій: [DEPLOY_PLAN.md](DEPLOY_PLAN.md) фаза 2,
[reports/paper_v0.1.0_vps_runbook.md](reports/paper_v0.1.0_vps_runbook.md)
(оновити тег у ранбуку окремим комітом після R1).

**Готово коли:** 8 тижнів журналу без обнулення sqlite на конфігурації
LINK/BTC 1h maker + regime_scale(0.25) + vol-target + fill-parity.
Це **не** дозвіл на live.

---

## Фаза L — пізніше (не цей цикл)

Умова старту кожної — **Ops закритий** або явне рішення, що пункт не чіпає paper sqlite.

| ID | Що | Умова старту |
|---|---|---|
| L1 | `risk_budget` у `PairsPortfolioRunner` | ≥2 пари в `VALIDATED_PAIRS` або явний запит |
| L2 | ERC ваги в paper portfolio | друга пара з pair-PASS |
| L3 | CLI `live-run-pairs` + systemd live | Paper-gate + **явний запит live** |
| L4 | Decimal через close-path `PaperAccount` (float лишається в numpy BT) | після R1, окремий PR; новий snapshot version |
| L5 | Tardis / черга MM | архів depth5 ≥ 4–8 тиж. **і** R5 працює |
| L6 | Порти `OrderBroker` / `MarketData` | якщо з’являється live CLI (L3), не раніше |
| L7 | Увімкнути `USE_EXIT_LADDERS` | не pairs-gate; окремий spec directional |

---

## Критерій «план виконано»

- [x] R1: paper і BT maker — mid + той самий seed; snapshot rng; тести parity
- [ ] R2: API-юніт localhost; lifespan fail-closed на слабкий ключ + публічний host
- [ ] R3: `docs/env/vps-paper.env.example` + `research.env.example`; HOLDOUT у `.env.example`
- [ ] R4: `use_exit_ladders` явне поле, дефолт false; немає getattr-магії
- [ ] R5: `scripts/sync_depth.sh` + quality-звіт
- [x] R6: `use_cases` без downloader
- [ ] R7: `tests/test_trader_loop.py`; немає swallow на fetch позицій
- [ ] R8: ROADMAP/TODO покажчик на цей файл — зроблено разом із публікацією плану; тримати в синхроні при зміні фаз
- [ ] `uv run pytest tests/ -q` зелений після кожної кодової фази
- [ ] Немає `DRY_RUN=false` у юнітах; немає live CLI
- [ ] Kalman/chase не в `deploy/scalper-paper-pairs.service`

## Що вважати успіхом продукту (не цього PR)

Одна комірка LINK/BTC 1h maker на VPS 8 тижнів на тегу з R1–R3, `paper-audit`
у межах допусків проти **тієї ж** моделі філу. Все інше в REGISTRY —
дослідження, поки не пройде той самий pair-гейт (H4).
