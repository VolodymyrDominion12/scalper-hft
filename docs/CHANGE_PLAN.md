# План змін: практики 2026 → scalper-hft

Стан на 2026-09-03. Мета — закрити дірки, через які paper/live міряють не той ринок,
а не додавати нові альфи. Валідований кандидат лишається **pairs_arb 1h maker**.

Повний аудит практик: [RESEARCH.md](RESEARCH.md), [STRATEGY_STATUS.md](STRATEGY_STATUS.md).
Коротко: maker pairs на 1h уже правильна гра; цей план — дроти live-шару і чесні філи.

## Принципи

1. Один PR = одна фаза. Не змішувати WS-інфру з Kalman.
2. Кожна фаза: тести → `uv run pytest tests/ -q` зелений → `uv run ruff check --fix && uv run ruff format`.
3. `use_kalman` і `legging_mode=chase` **не** стають дефолтом, поки фаза B+C не зелені.
4. Live з реальними коштами — поза цим планом (Phase 1 paper-gate 8 тижнів).
5. Без lookahead; chase тарифікується як taker (0.05%).

## Поза скоупом (свідомо не чіпати)

- Нові 1m скальп-стратегії (VWAP/EMA/ORB) — fee-drag уже вбив `cvd_momentum`.
- Live `market_maker` / `ob_imbalance` без L2-черги (Phase 4 / Tardis).
- Увімкнення відхилених: mean_reversion, funding_carry, basis_reversion.
- Оптимізація z/lookback на повній вибірці.
- Зміна `DEFAULT_INTERVAL` глобально на 1h (pairs CLI уже `--interval 1h`).

---

## Фаза A — WebSocket USDT-M (P0)

**Навіщо.** Binance вимкнув legacy `wss://fstream.binance.com/ws` 2026-04-23.
Рекордер глибини і user-stream на старих URL мовчки глухі.

**Файли**

- `scalper_hft/live/ws_user_stream.py`
- `scalper_hft/live/bookticker_recorder.py`
- `tests/test_ws_user_stream.py` (+ новий `tests/test_bookticker_urls.py` якщо треба)

**Зміни**

| Стрім | Було | Стати |
|---|---|---|
| User data | `/ws/{listenKey}` | `/private/ws?listenKey=…&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE` |
| bookTicker | `/ws/{symbol}@bookTicker` | `/public/ws/{symbol}@bookTicker` |
| depth5 | `/ws/{symbol}@depth5@100ms` | `/public/ws/{symbol}@depth5@100ms` |
| Testnet | `stream.binancefuture.com/ws/{key}` | той самий split `/private` `/public` на testnet-хості |

Додати:

- keep-alive `PUT /fapi/v1/listenKey` кожні 30 хв (зараз немає — ключ помирає за 60 хв);
- регенерація listenKey на reconnect;
- константи URL в одному місці (не дублювати в recorder і user stream).

**Тести (без мережі)**

- `ws_url` prod містить `/private/ws` і `listenKey=` як query, не path.
- `ws_url` testnet — новий хост/шлях.
- bookTicker/depth URL містять `/public/ws`.
- парсер `ORDER_TRADE_UPDATE` без регресії.

**Готово коли:** юніт-тести зелені; ручна перевірка `record-bookticker --minutes 1` отримує тіки (опційно, не в CI).

---

## Фаза B — Чесні філи pairs: chase = taker, unwind = flatten (P1)

**Навіщо.** `resolve_legging` уже є, але `_apply_fills` завжди бере `self.is_maker`.
Chase по mid без taker 5 bps занижує вартість розсинхрону. У paper-OHLC unwind
коректний як «не брати pending»; у моделі з уже виконаною ногою — ні.

**Файли**

- `scalper_hft/live/fills.py` — `LeggingResolution` + прапорці `leg1_maker` / `leg2_maker`
- `scalper_hft/live/pairs_runner.py` — `_apply_fills(..., maker1, maker2)`; unwind
- `scalper_hft/live/account.py` — без змін API (вже є `is_maker` на open/close)
- `tests/test_legging_manager.py`

**Поведінка**

| Дія | Нога 1 | Нога 2 | Комісія |
|---|---|---|---|
| `both_filled` | maker | maker | 0.02% + 0.02% |
| `chase_leg2` | maker | **taker** на mid | 0.02% + 0.05% |
| `chase_leg1` | **taker** на mid | maker | 0.05% + 0.02% |
| `unwind_leg1` | taker close вже заповненої | не відкривати | flatten, не silent unfilled |
| `strict_both` | як зараз: all-or-nothing, без chase | | без змін дефолту |

Paper-OHLC: поки обидві ноги ще в `pending` (філ не застосований до акаунта),
unwind = скасувати pending (поточна поведінка). Якщо колись одна нога вже в
`account.positions` — unwind **обов’язково** `close_position(..., is_maker=False)`.

Дефолт рушія лишається `legging_mode="strict_both"`. `chase` — явний прапорець.

**Тести**

- chase_leg2: комісія другої ноги = `taker_fee * notional`, не maker.
- unwind при вже відкритій нозі: позиція закрита, cash зменшився на taker.
- `strict_both` регресія: одноногий філл не відкриває позицію.
- існуючі `test_resolve_legging_*` оновити під нові поля.

**Готово коли:** chase не може виглядати дешевшим за maker-maker.

---

## Фаза C — Kalman vs OLS bake-off (P1)

**Навіщо.** Код Kalman + OU half-life є (`use_kalman=False`). Увімкнути дефолт
без OOS = новий параметрний ступінь свободи на тих самих парах.

**Файли**

- `scalper_hft/strategies/pairs_arb.py` — без зміни дефолтів
- `scalper_hft/features/signal_processing.py` — already `KalmanHedgeRatio`
- `tests/test_kalman_pairs.py` — розширити
- CLI: `pairs` / `overfit` / `walkforward` з `--use-kalman` (якщо ще немає)
- звіт: `docs/reports/kalman_vs_ols.md` (після прогону, не до)

**Протокол (анти-перенавчання)**

На кожній валідованій парі (XRP/BTC, LINK/BTC, LINK/ETH, BTC/ETH), 1h, maker:

1. База: поточний OLS, зафіксовані z/lookback зі STRATEGY_STATUS.
2. Kalman: ті самі z/lookback; `q=1e-5, r=1e-3` (BTC-кластер); альти окремо не крутити в цьому PR.
3. Метрики: walk-forward avg OOS Sharpe, DSR, % вікон > 0, maxDD, n trades, fill-rate paper-replay.
4. Half-life gate: якщо `estimate_half_life` → inf / > N барів — сигнал 0 (окремий прапорець, default off до OOS).

**Правило прийняття:** Kalman замінює OLS як дефолт **лише якщо** avg OOS Sharpe ≥ OLS
і DSR не гірший, на ≥3 з 4 пар. Інакше лишається опцією `--use-kalman`.

**Готово коли:** звіт у `docs/reports/` + тести збіжності/half-life без зміни дефолту
(або зміна дефолту, якщо правило прийняття виконане).

---

## Фаза D — T-cost з філів + risk budget у циклі (P1)

**Навіщо.** `IsJournal` пишеться, `CostModel` і `risk_budget` — бібліотека.
Без зворотного зв’язку slippage лишається константою 2 bps; денний ліміт
портфеля не ріже нові входи.

**Файли**

- `scalper_hft/backtest/execution.py` — `calibrate_from_is(records) -> slippage_frac`
- `scalper_hft/live/is_log.py` — агрегат mean/p50/p90, split maker vs chase
- `scalper_hft/live/pairs_runner.py` (`PairsPortfolioRunner.step`)
- `scalper_hft/portfolio/risk_budget.py` — виклик, не перепис формул
- `scalper_hft/live/store.py` — якщо треба читати IS з SQLite
- тести: `tests/test_sprint2.py` / новий `tests/test_cost_calibration.py`,
  `tests/test_pairs_paper.py`

**Зміни**

1. Калібровка: median IS bps (окремо maker / chase) → `CostModel.slippage_frac`.
   Не підганяти на тому ж вікні, що звітуємо як OOS.
2. У `PairsPortfolioRunner.step`: якщо денний PnL ≤ `daily_loss_limit` або
   тижневий ≤ `weekly_loss_limit` — блок **нових** входів; flatten дозволений
   (як у `risk_gate.decide_entry(flattening=True)`).
3. Vol-target: опційний `size_mult = vol_target_scale(...)`, clip як у trader
   `[0.25, 3]`. Default off, прапорець.

**Готово коли:** paper-audit показує IS mean; halt по daily-loss на портфелі
покритий тестом.

---

## Фаза E — Дрібниці конфігу / сесій (P2)

- CLI `pairs` / dashboard: явний дефолт 1h уже є; у generic `backtest` додати
  попередження, якщо `strategy=pairs_arb` і interval ∈ {1m,5m}.
- `paper-audit`: розбивка fill-rate / PnL по годині UTC (модуль
  `research/session_analysis.py` уже є — підключити до audit CLI).
- Лічильник API weight / backoff на 429 — лише коли з’явиться REST у гарячому
  циклі (зараз pairs paper на закритих барах).

---

## Фаза F — пізніше (Phase 4, не цей цикл)

- Архів depth5 + Tardis для черги ліміток.
- Avellaneda live після моделі черги/markout.
- Nautilus як зовнішній L2-бенчмарк.
- Реальні ключі Binance після 8 тижнів paper.

---

## Порядок PR

```text
A  WS URL + listenKey keepalive          ← блокер live/recorder
B  chase taker + unwind flatten          ← блокер чесного PnL
C  Kalman bake-off (звіт, дефолт off)    ← можна паралельно з A після B-тестів
D  IS→CostModel + daily/weekly halt      ← після B (потрібні maker/chase мітки)
E  session audit + pairs interval warn   ← за бажанням
F  L2 / MM                               ← не починати, доки paper-gate відкритий
```

Paper-моніторинг 8 тижнів іде **паралельно** і не чекає C–F.
Після B варто перезапустити paper з чистою метрикою комісій.

## Критерій «план виконано»

- [ ] Фаза A: нові WS URL у тестах; recorder не на `/ws/{symbol}@`
- [ ] Фаза B: chase не може пройти як maker; `strict_both` без регресії
- [ ] Фаза C: звіт Kalman vs OLS; дефолт змінено лише за правилом прийняття
- [ ] Фаза D: портфельний daily-loss halt у циклі; IS у paper-audit
- [ ] `uv run pytest tests/ -q` зелений
- [ ] Kalman/chase не ввімкнені «тихо» в paper systemd-юніті без прапорців
