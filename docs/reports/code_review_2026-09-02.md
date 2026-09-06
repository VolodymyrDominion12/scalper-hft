# Code Review scalper-hft — 2026-09-02

> Повний незалежний аудит коду. Стан: **349 passed, 3 skipped** (pytest),
> **ruff — чисто**, **mypy — 18 помилок** у 5 файлах.
> Усі «критичні» знахідки нижче перевірені емпірично (відтворено), не лише статично.

---

## 0. Загальний вердикт

Проєкт — зріла, добре структурована квант-платформа (Alpha → Risk → T-Cost →
Portfolio → Execution), тести зелені, архітектура відповідає книзі Narang.
**Головна проблема — не «стиль», а точність моделей виконання/ризику в
live/event-контурах**: кілька місць систематично завищують PnL або можуть
зламатися на реальних ринкових переходах. Нижче — від критичного до дрібного.

Попередній звіт `analysis_report.md` застарів частково: пункти 2 (annualization)
і 3 (spread fallback) **вже виправлені**; пункт 1 (`risk_of_ruin`) змінений на
експоненційне наближення, але досі має проблеми з одиницями (див. B3).

---

## 1. 🔴 CRITICAL (відтворено)

### C1. `PairsEngine` — прямий реверс сигналу (+1→−1 без 0) кидає ValueError і «вічно клинить» пару
- **Файли:** `scalper_hft/live/pairs_runner.py:260-329` (`_quote`), `account.py:79-80`
- **Суть:** `_quote` захищає лише `want == have` і вхід з `have==0`. Якщо
  `PairsArb` стрибає +1 → −1 за один бар без проміжного 0 (можливо: вихід
  спрацьовує лише при `|z| < exit_z`, див. `pairs_arb.py:78-86`), рушій котирує
  **новий відкриваючий** ордер на ключі, де позиція вже відкрита. При філлі обох
  ніг `_apply_fills` → `open_position()` → `ValueError` (позиція вже існує) —
  до очищення `pending`, тож помилка повторюється кожен бар.
- **Відтворення:** синтетичний ряд з переходом z +2 → −2 → `replay_pairs`
  падає: `ValueError: позиція AAUSDT/BBUSDT:AAUSDT вже відкрита — спочатку close`.
  У `run()` помилка лише логується — пара залишається заклиненою (можлива й
  однонога позиція після crash у середині `_apply_fills`).
- **Фікс:** при `want != have` і `want != 0` — спершу reduce-only закриття обох
  ніг (all-or-none), і лише після підтвердженого закриття котирувати дзеркальний
  вхід. `_apply_fills` зробити атомарною щодо `pending`/`have`.

### C2. `LiveTrader` — живий maker-ордер: локальна позиція брониться як миттєвий повний філл, ордер не відстежується
- **Файл:** `scalper_hft/live/trader.py:336-395`
- **Суть:** `_submit_order` ставить limit post_only і повертає True при
  *прийомі*; `_open`/`_close` одразу бронять повну позицію за ціною запиту.
  Статус ордера не читається, `clientOrderId` не зберігається/не запитується,
  cancel-on-timeout немає. Maker-ордер, що «висить» без філла → локальна позиція
  є, біржової немає → наступна звірка `reconcile` кидає `KillSwitch` (або
  навпаки: ордер виконався, а локально позицію закрили раніше).
- **Фікс:** журнал відкритих ордерів (`clientOrderId → ордер`), філ-букинг лише
  з реальних fill-звітів (`fetch_order`/user-data stream), скасування за таймаут,
  обробка `PARTIALLY_FILLED`/`EXPIRED`/`REJECTED`. Live maker сьогодні або
  заповнюється за один REST-цикл, або самогальмується.

### C3. `event_engine` — перевернутий знак PnL для шорт-покриттів + фантомні ціни філів «краще за ринок»
- **Файли:** `scalper_hft/backtest/event_engine.py:86-87, 106-112, 119-123, 149-151`
- **Суть (3 дефекти, що підсумовуються у фантомний прибуток MM):**
  1. Закриття шорта: `pnl = _pnl(avg_cost, f, closed)` = `(f − avg_cost)/avg_cost`
     — для прибуткового покриття (`f < avg_cost`) дає **від'ємний** PnL;
     збиткове покриття дає додатний. Лонг-гілка правильна.
  2. Ціни філів: buy = `prev_bid·(1 − h·s)` (нижче за bid!), sell = `prev_ask·(1 + h·s)`
     (вище за ask) — це **покращення** ціни для MM на ~1 bp з кожного боку,
     тоді як докстрінґ каже «ціна погіршується». Плюс філл може бути кращим за
     high/low бару — ціни, якої на ринку не було.
  3. Mark-to-market відкритого інвентаря — за *попереднім* mid (`prev_mid`),
     тож геп-бар, що заповнив наш bid, бронить миттєвий прибуток.
- **Відтворення:** сценарій «шорт у ріст, покриття у падіння» → у trade `ret`
  додатний при збитковому покритті (перевернуто).
- **Фікс:** `pnl = −_pnl(...)` для шорт-покриттів; філ за ціною котирування
  (bid/ask, не всередині); mark за поточний close бару.

### C4. `run_event_backtest` ніколи не викликає `strategy.generate_signals()` — стратегії-«подієві» мертві
- **Файли:** `backtest/event_engine.py:45-175`, `backtest/router.py:10, 26-33`
- **Суть:** роутер надсилає `ob_imbalance` і `market_maker` в event-рушій, але
  рушій не читає жодного параметра стратегії (лише `strategy.name`); усі
  параметри MM (`spread_offset_mult`, `inventory_cap`, `adverse_sel_haircut`,
  `quote_size_pct`, пороги ob_imbalance) не впливають на результат. Свіпи/Optuna
  за цими стратегіями повертають константу.
- **Фікс:** або споживати `generate_signals()`/`strategy.params`, або прибрати
  ці стратегії з `EVENT_STRATEGIES` і backtest-поверхні до реалізації.

### C5. `frac_diff_expanding()` — нескінченний цикл (hard-coded `threshold=0.0`)
- **Файл:** `scalper_hft/ml/frac_diff.py:86-111`
- **Суть:** `_get_weights_ffd(d, threshold=0.0)` — умова `|w_k| < 0.0` ніколи не
  істинна (поліноміальний розпад ваг не досягає 0), цикл не завершується.
- **Відтворення:** виклик на 50 точках завис (timeout 15 с).
- **Фікс:** передавати реальний поріг (аргумент функції) або капіт window.

---

## 2. 🟠 MAJOR

### M1. Funding на грубих барах: кілька ставок в одному барі — враховується лише остання
- **Файли:** `backtest/engine.py:222-230`, `backtest/pairs.py:199-206`, `backtest/delta_neutral.py:85-92`
- **Відтворення:** 1d-бари, фандінг кожні 8h → враховано **33%** очікуваного
  (решта «зливається» через `searchsorted` + присвоєння в дублікат-бари).
- **Фікс:** групувати ставки за баром і сумувати (`rates.groupby(bars).sum()`).

### M2. Maker-бектест пар ≠ paper-модель філів (розбіжність BT↔paper)
- **Файли:** `backtest/pairs.py:68-103, 131` vs `live/pairs_runner.py:213-214`; `live/fills.py:48-68`
- **Суть:** `_maker_pair_positions` викликає `decide_fill` з `mid` і `rng=None` →
  філ вимагає `p == 1.0`, тобто mid бару має пройти крізь ліміт (детермінований
  «адверсний» фільтр). Paper-ранер кличе без `mid` → філ на будь-який touch.
  Докстрінґ каже «та сама модель, що й paper» — не та сама. Результати
  валідованих пар (maker) і paper-репліки непорівнянні; на синтетиці: paper
  28 філів vs backtest-maker 8 угод.
- **Фікс:** єдиний seeded `rng` в обох шляхах або спільна семантика touch+prob;
  регрес-тест «backtest fill-rate ≈ paper fill-rate».

### M3. Відсутній живий order-lifecycle для пар у `DRY_RUN=false`
- **Файли:** `live/pairs_runner.py:484-530, 568-606`, `live/reconcile.py`
- **Суть:** під `DRY_RUN=false` раннери створюють auth-клієнт, але реальні
  ордери не ставлять; звірка порівнює paper-ноги з реальною біржею → `KillSwitch`
  кожен цикл, який `run()` ковтає (нескінченні warning-и). Або провести реальні
  ордери, або жорстко заблокувати `DRY_RUN=false` для цих класів.
- **Плюс:** у `PairsEngine` паралельний лічильник `consecutive_pair_losses`
  не скидається на день (`pairs_runner.py:139, 247-251, 363-365`) — на відміну
  від `LiveTrader`, пара може «назавжди» загальмувати після 3 збитків за 3 дні.

### M4. Live-розмір і ризик-гейти рахуються з локального симульованого рахунку, не з реального
- **Файли:** `live/trader.py:142-149, 439`; `binance_client.py:126-127` (`fetch_balance` ніде не викликається)
- **Суть:** `PaperAccount(initial_capital = position_pct·100_000)` = **$1000** за
  замовчуванням навіть у live; розмір позиції й денний ліміт збитків — проти
  цієї фікції. Перша розбіжність з реальним рахунком → невірні розміри без
  перевірки реальної маржі. Leverage/margin-mode не налаштовуються.
- **Фікс:** сідити live-стан з `fetch_balance`/`fetch_positions` щоциклу і рахувати
  від реального equity, або відмовити live до появи live-адаптера рахунку.

### M5. Немає обробки точності біржі (LOT_SIZE / tick / MIN_NOTIONAL) перед ордером
- **Файли:** `live/trader.py:353-364`, `live/orders.py`
- **Суть:** сирий float-розмір (`0.0012345…`) і ціна close без
  `amount_to_precision`/`price_to_precision` — Binance відхилить більшість
  live-ордерів або виконає несподівані розміри.
- **Фікс:** `load_markets()` → округлення qty до `LOT_SIZE`, ціни до tick,
  перевірка `MIN_NOTIONAL`/`MAX_QTY` перед сабмітом.

### M6. `sparse_basket` — повно-семпльний fit ваг (lookahead) + недосяжний шлях
- **Файл:** `scalper_hft/strategies/sparse_basket.py:146-163`
- **Суть:** ваги `compute_sparse_basket_weights(basket_df.tail(lookback*2))` —
  на **хвості всього** датасету, потім застосовуються до всіх барів; а
  `basket_df` рушієм взагалі не передається → безшумний fallback на
  single-series під тим самим ім'ям.
- **Фікс:** каузальний expanding fit (або фіксовані ваги поза бектестом) +
  явний wire `basket_df` або видалення з REGISTRY.

### M7. Triple-barrier: викидання timeout-подій (label=0) ДО walk-forward + purge не підключений
- **Файли:** `ml/features.py:232-233`, `ml_strategy.py:163-170`, `ml/trainer.py:165-171`
- **Суть:** `events[events["label"] != 0]` — OOS-передбачення/угоди існують лише
  на барах, чий PT/SL-результат був відомий ex-post (subset-selection bias);
  purge за `t1` реалізований у тренері, але `MLStrategy`/`train_from_ohlcv`
  `t1` не передають → останні `holding_bars` тренувальних лейблів «протікають»
  у тест.
- **Фікс:** повертати `t1` з `label_from_ohlcv`, застосовувати purge+embargo у
  кожному вікні; timeout-події не видаляти до розбиття (або документувати bias).

### M8. DSR/PBO — математичні відхилення
- **Файл:** `scalper_hft/validation/deflated_sharpe.py:104-109, 128-156`; `cli.py:306-310`
- **Суть:**
  - `pd.Series.kurt()` — **надлишковий** ексцес, а формула variance потребує
    `(γ4−1)/4` зі *звичайним* ексцесом; у PSR `+3.0` додають (`:178`), у DSR —
    ні → DSR завищений.
  - `cli.py` передає `n_trials = len(param_space)` (кількість параметрів), а не
    кількість комбінацій/спроб → корекція на множинне тестування занижена.
  - `probability_of_backtest_overfitting` — не канонічний PBO (marginal bootstrap
    + ad-hoc log2-скейл); CSCV теж спрощений.
- **Фікс:** виправити ексцес у DSR; `n_trials` = добуток розмірностей
  param_space (або фактичні trials з Optuna/sweep); позначити PBO як евристику.

### M9. Walk-forward/optimize: `trades` ріжуться бар-офсетами, а не часом
- **Файли:** `validation/walk_forward.py:102-103`, `validation/optimize.py:64, 129`
- **Суть:** `trades.iloc[start:start+train_bars]` для `needs_trades` стратегій —
  невірні часові вікна (funding ріжеться часом — отже, це oversight).
- **Фікс:** timestamp-маски, як для funding.

### M10. LiveTrader реверс — окремий ризик: `run_trader_once` котирує close без перевірки «а чи заповнився maker-close»
Див. C2 — той самий корінь: локальний стан оновлюється миттєво, реальні філи не
перевіряються. У `pairs_runner` unfilled-модель є (wait_bars); у `LiveTrader` її
немає зовсім, хоча `MAKER_FILL_WAIT_BARS` існує в конфігу.

---

## 3. 🟡 MINOR / покращення

- **`bandit.py:116-127`** — `exp3_select_signals`: обрана рука не записується в
  `history` до `update()`; `prev_arm` завжди 0 (fixed point) → бандит не вчиться.
- **`metrics.py:91-96`** — Sortino = 0 при ≤1 від'ємному значенні (guard
  `len(downside) > 1`), хоч Sharpe високий; нестандартне downside-відхилення.
  **Відтворено:** 200 виграшних + 1 програшний бар → Sharpe≈1200, Sortino=0.
- **`metrics.py:80-81, 125-126`** — CAGR експлодує на вікнах <1 року;
  `trades_per_day` ділить на календарний span.
- **`metrics.py:85-88`** — дублікати індексу → `delta_s=0` → `bars_per_year`
  ≈ 31.5 млн (×5616 завищення Sharpe/vol); варто валідувати унікальність індексу.
- **`metrics.py:141-149`** — `risk_of_ruin`: `edge` = середній прибуток угоди на
  *весь* капітал, а формула очікує edge на одиницю ставки; `f=0.01` захардкоджено.
  Для скальпінгових edge результат ≈ константа.
- **`engine.py:61-94` / `pairs.py:209-231` / `delta_neutral.py:114-137`** —
  per-trade `ret` у `_extract_trades` не включає exit-комісію бару закриття
  (equity-метрики OK, але `avg_trade_return`/`win_rate`/PF завищені на ~одну
  сторону витрат).
- **`engine.py:170-171, 186-196`** — maker-адверс у векторному рушію: flat
  `adverse_bps=1e-4` та `prob_touch=0.5` без залежності від реального руху/
  distance-to-mid → maker-edge калібрується константами, не даними.
- **`execution.py:267`** — `calibrate_cost_model` має floor 1 bp slippage навіть
  при ідеальних лімітних філах (консервативно, але систематично).
- **`pairs_arb.py:108-114`** — внутрішній breakeven-gate використовує дефолтний
  `CostModel()` замість переданого (поріг ігнорує .env slippage).
- **`binance_client.py:44`** — `exchange_id="binance-testnet"` не існує як клас
  ccxt → **мовчазний fallback на mainnet** `ccxt.binance` (перевірено:
  `hasattr(ccxt,'binance-testnet') == False`). Публічні дані — байдуже; auth/live —
  небезпечно. **Фікс:** `set_sandbox_mode(True)` для `*testnet*` + whitelist id.
- **`reconcile.py:38-93`** — точна set-звірка відкритих позицій: false-trip на
  сторонні позиції того ж рахунку; не бачить resting-ордерів і дрейфу cash/equity.
- **`risk_gate.py:86-93` / `pairs_runner.py:271-273`** — `CORR_NOTIONAL_CAP`
  рахується від номінального size_pct, без урахування `_entry_size_mult` і того,
  що корелює лише спільна нога (половина пари).
- **`live/telegram.py:20-25`** — `os.getenv` без імпорту config → standalone
  використання мовчки вимикає алерти.
- **`paper_replay.py`** — replay risk-gate не дорівнює live risk-gate (немає
  cooldown-вікна/half-size/тижневих лімітів) → валідаційна фідельність.
- **`mcp_trading.py:165-184`** — `paper_step` покладається лише на config-гейт;
  додати явний `if not settings.dry_run: raise` у хендлер.
- **`bookticker_recorder.py`** — depth-рекордер без reconnect-циклу; обидва
  рекордери втрачають буфер при Ctrl+C (немає `finally` flush).
- **`coint_scan.py:59-90`** — вибір пар (ADF/half-life) на повному семплі перед
  WF-заявками в docs → selection leakage для парного буклету.
- **`erc.py:39-43`** — zero-variance обробляється лише якщо ВСІ активи flat;
  один «мертвий» актив може захопити ERC-портфель.
- **`deflated_sharpe` sweep default** — `sweep.default_strategies()` ганяє 7
  docs-відхилених стратегій за замовчуванням (registry має «сміття»).
- **`market_maker.py`** — `generate_signals` завжди повертає 0; стратегія жива
  лише через event-роутер (який її не читає — див. C4).
- **`cross_momentum.py`** — у робочому дифі прибрано внутрішній `shift(1)`
  (правильно — рушій лагає сам); переконатися, що всі клієнти (CLI/paper)
  використовують його лише через рушій з lag-1.
- **Документація:** `docs/DESIGN.md`, `ROADMAP.md`, `analysis_report.md`
  застаріли щодо кількох пунктів (кількість тестів, статус виправлень,
  «339 passed»). `analysis_report.md` варто оновити/архівувати.

---

## 4. ✅ Перевірено і КОРЕКТНО

- **No-lookahead ядро:** `engine.py:157` `signals.shift(1)` — сигнал на close t →
  позиція з t+1; тест `test_engine_no_lookahead` зелений; funding каузальний
  (searchsorted side='right'−1, ставка за бар, що покриває fts).
- **База комісій:** turnover = частка капіталу, fee-ставки тієї ж розмірності;
  реверс (+→−) сплачує обидві сторони рівно один раз; maker без slippage, taker =
  fee+slippage+impact.
- **Funding-знаки** (long платить, short отримує) і раз-на-період — згідно з
  тестами `test_funding_charged_once_per_block`.
- **Funding-стратегії** (`funding_arb`, `funding_carry`) діють за *попередньою*
  опублікованою ставкою (`fr.shift(1)`) — без lookahead, консервативно.
- **`closed_klines`** відкидає формуючий бар; paper/pairs-цикл оцінює філи лише
  за закритими барами; дедуп через `_last_ts`.
- **Fail-closed:** при `create_order` exception локальний стан не мутують;
  `reduce_only=True` на всіх закриттях; ризик-гейти до розміщення; закриття
  ніколи не блокуються; `DRY_RUN` гейт у `_submit_order` + `require_live_credentials`
  (без виводу значень ключів).
- **PurgedKFold** purge/embargo логіка у `validation/cv.py` коректна (коли t1 є).
- **Resampling** (open-time, label/closed left, UTC-вирівнювання), storage
  round-trip (naive UTC), парсинг ключів `pair:symbol` у `pos_pair_id`/`pair_legs`,
  ERC-математика, OOS-слайсинг walk-forward — коректні.
- **Секрети:** `.env` у gitignore; значення ключів ніде не друкуються.

---

## 5. Пріоритети (рекомендований порядок робіт)

1. **C2 + C4 + M4 + M5** — live/event контур: order-lifecycle, реальний рахунок,
   точність біржі. Без цього live maker-торгівлю запускати не можна.
2. **C1 + M3** — pairs: реверс close-before-flip, звірка/блок `DRY_RUN=false`.
3. **C3** — event_engine: знак PnL шортів, ціни філів, mark.
4. **C5 + M1 + M2 + M7** — data/ML: нескінченний цикл FFD, funding на грубих
   барах, єдина fill-модель BT↔paper, purge t1 у ML-потоці.
5. **M6, M8, M9, bandit** — валідаційна чесність (lookahead у sparse_basket,
   DSR-формули, slicing trades, bandit).
6. **Minor** — Sortino, risk_of_ruin, tick-size floor, `binance-testnet` fallback,
   реконсіляція, оновлення docs.
7. Після кожної зміни: `uv run pytest tests/ -q` (349+ тестів) має бути зеленим.

---

## Додаток (раунд 2: C2 + C4) — 2026-09-02, виправлено

### C2 — LiveTrader order lifecycle (виправлено)
`live/trader.py` — тепер:
- Live maker-ордер (limit post_only) реєструється у `pending_orders` (clientOrderId +
  exchange order id); локальна позиція НЕ брониться до підтвердження філа.
- `poll_pending_orders()` на початку кожного кроку (`run_trader_once`) питає
  `fetch_order`: closed/filled → `_book_pending_fill` за фактичною ціною/об'ємом;
  частковий філ → брониться заповнене, решта скасовується; dead статуси
  (canceled/expired/rejected) → drop без позиції; таймаут
  (`maker_fill_wait_bars` × інтервал) → cancel.
- Нові входи блокуються, поки є pending (hold:open_pending / hold:closing_pending);
  сигнал 0/реверс скасовує resting-ордер; close теж чекає філа (позиція лишається,
  доки біржа не підтвердить закриття).
- `_submit_order` повертає (ok, status); market/paper шлях — миттєвий філ, як раніше
  (тести paper/testnet не змінились).
- `BinanceClient`: додано `fetch_order`/`fetch_open_orders`.
- Регресії: `tests/test_review_fixes.py::TestLiveMakerLifecycle` (5 тестів, фейковий
  live-клієнт: open не брониться до філа, fill брониться за ціною філа, таймаут
  скасовує без позиції, сигнал 0 скасовує pending, maker-close тримає позицію до філа).

### C4 — event-роутер (виправлено)
`backtest/router.py`:
- `EVENT_STRATEGIES = {"market_maker"}` — `ob_imbalance` виключено: це напрямкова
  стратегія з власним `generate_signals`, для неї коректний векторний рушій.
- Параметри `market_maker` (quote_size_pct, inventory_cap, spread_offset_mult,
  adverse_sel_haircut) тепер реально передаються в `run_event_backtest` —
  sweep/Optuna по MM більше не дають константу.
- Регресії: `TestRouterParamFlow` (MM-параметри міняють результат; ob_imbalance йде
  у векторний рушій і торгує).
- `tests/test_roadmap.py` оновлено під новий склад `EVENT_STRATEGIES`.

### Перевірено
`uv run pytest tests/ -q` → зелено (375 passed, 3 skipped у фінальному прогоні),
ruff — чисто, mypy на змінених модулях — без помилок.

---

## Додаток (раунд 3: M3 + M4 + M5) — 2026-09-02, виправлено

### M5 — точність біржі перед live-ордером (виправлено)
`data/binance_client.py`:
- `BinanceClient.sanitize_order(symbol, side, amount, price)` → `(qty, price, error)`: округлює
  об'єм до LOT_SIZE і ціну до tick через ccxt `amount_to_precision`/`price_to_precision`,
  валідує `minQty`/`maxQty`/`MIN_NOTIONAL`.

`live/trader.py::_submit_order`: у live-гілці перед `create_order` викликає `sanitize_order`;
при error — **fail-closed**: ордер не летить на біржу (раніше сирий float-розмір/ціна →
Binance відхиляв би більшість ордерів або виконував несподівані розміри).

### M4 — live equity з біржі (виправлено)
`data/binance_client.py`:
- `fetch_usdt_equity()` — реальний equity USDT-M: wallet balance + unrealized PnL
  (`info.totalWalletBalance` + `info.totalUnrealizedProfit`, fallback на ccxt `total`).

`live/trader.py`:
- `LiveTrader.sync_live_equity()` — у live ребейзить `account.cash` так, що локальний
  equity ≈ реальний баланс біржі, і фіксує `day_start_equity` від реального equity при
  першому sync; поля `last_live_equity` / `_live_equity_seeded`.
- `run_trader_once`: sync перед обчисленням сигналу → sizing (`base_size`) і денний ліміт
  збитків рахуються від РЕАЛЬНОГО балансу, а не від фіктивного
  `PaperAccount(position_pct × 100_000)`. Paper — no-op.

### M3 — pairs paper-ранери блокують live (виправлено)
`live/pairs_runner.py`: `PairsPaperRunner` і `PairsPortfolioRunner` кидають
`RuntimeError("…paper-only…")` при `DRY_RUN=false`. Вони лише симулюють maker-філи на
локальному PaperAccount і НЕ ставлять реальні ордери ніг — live-режим був некоректним
(звірка локальних ніг з реальною біржею → KillSwitch). Live-pairs потребує окремого
адаптера з реальними ордерами.

### Регресії (tests/test_review_fixes.py)
`TestPairsRunnersLiveBlock` (2 тести), `TestLiveEquitySync` (2), `TestOrderSanitize` (1).

### Перевірено
ruff — чисто; повний pytest — зелений.

---

## Додаток (раунд 4: дрібні фікси) — 2026-09-02, виправлено

### m — telegram.py читає .env через config
`live/telegram.py`: `_creds()` тепер бере `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` з
`scalper_hft.config.get_settings()` (який робить load_dotenv). Раніше голий
`os.getenv` — standalone-виклик мовчки втрачав алерти, якщо config ще не імпортували.
Поля додано в `config.Settings`.

### m — mcp_trading._paper_step: явний dry_run-guard
`mcp_trading.py`: хендлер кидає RuntimeError при `DRY_RUN=false` (belt-and-braces
понад config-гейт) — «paper_step дозволений лише у DRY_RUN=true».

### metrics — CAGR/дублікати/sortino_hourly
`backtest/metrics.py`:
- CAGR на вікнах < 1 року більше не експлодує (повертає total_return);
- дублікати/нульові дельти індексу → fallback на хвилинний масштаб замість
  bars_per_year = 31.5M (Sharpe/vol більше не завищуються ×5616);
- `sortino_hourly` — стандартне downside deviation (RMS min(r,0)), не std збитків.

### m — exit-комісія у per-trade ret
`backtest/engine.py` (`_extract_trades`): fee бару зміни позиції розщеплюється
пропорційно розмірам сторін — exit-частка йде закритій позиції, entry — новій.
Раніше fee бару закриття губилась → avg_trade_return був завищений (~одна сторона
витрат). `backtest/pairs.py` і `backtest/delta_neutral.py` — аналогічно для
flat-exit (strat_ret бару закриття додається до ret угоди). Equity не змінилась.

### m — erc.py: мертвий актив
`portfolio/erc.py`: актив із ≈нульовою дисперсією отримує нульову вагу (решта —
ERC на живих); раніше нульова дисперсія одного активу спотворювала коваріацію й
могла захопити ERC-портфель.

### m — bookticker_recorder
`live/bookticker_recorder.py`: `_record_depth` отримав reconnect-цикл (як у
`_record_symbol`); обидва рекордери скидають буфер у `finally` (Ctrl+C не втрачає
рядки); `ts` = час ПОДІЇ від біржі (`E`/`T`, мс → naive UTC), не час прийому.

### Регресії (tests/test_review_fixes.py)
`TestTelegramConfig`, `TestMcpPaperStepGuard`, `TestMetricsMinor` (2),
`TestExitFeeInTradeRet` (3), `TestErcDeadAsset`, `TestRecorderEventTs`.

### Перевірено
ruff — чисто; mypy на змінених модулях — без помилок; повний pytest — зелений.
