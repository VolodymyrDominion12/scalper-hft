# План змін: практики 2026 → scalper-hft

Стан на 2026-09-03 (цикл реалізації). Мета — закрити дірки, через які paper/live
міряють не той ринок, а не додавати нові альфи. Валідований кандидат лишається
**pairs_arb 1h maker**.

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

## Фаза A — WebSocket USDT-M + listenKey keepalive (P0)

**Навіщо.** Binance вимкнув legacy `wss://fstream.binance.com/ws` 2026-04-23.
Рекордер глибини і user-stream на старих URL мовчки глухі. Docstring обіцяв
`PUT /fapi/v1/listenKey` кожні 30 хв; без нього ключ помирає за 60 хв.

**Файли**

- `scalper_hft/live/ws_urls.py` — спільні константи `/public` `/private`
- `scalper_hft/live/ws_user_stream.py`
- `scalper_hft/live/bookticker_recorder.py`
- `tests/test_ws_user_stream.py`, `tests/test_bookticker_urls.py`

**Зміни**

| Стрім | Було | Стати |
|---|---|---|
| User data | `/ws/{listenKey}` | `/private/ws?listenKey=…&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE` |
| bookTicker | `/ws/{symbol}@bookTicker` | `/public/ws/{symbol}@bookTicker` |
| depth5 | `/ws/{symbol}@depth5@100ms` | `/public/ws/{symbol}@depth5@100ms` |
| Testnet | `stream.binancefuture.com/ws/{key}` | той самий split `/private` `/public` на testnet-хості |

Додати:

- keep-alive `PUT /fapi/v1/listenKey` кожні 30 хв (інжектований колбек, без мережі в тестах);
- регенерація listenKey на reconnect і `listenKeyExpired`;
- константи URL в одному місці (не дублювати в recorder і user stream).

**Готово коли:** юніт-тести зелені; prod `ws_url` містить `/private/ws` і `listenKey=` як query.

---

## Фаза B — Чесні філи pairs: chase = taker, unwind = flatten (P1)

**Навіщо.** `resolve_legging` уже є, але `_apply_fills` завжди бере `self.is_maker`.
Chase по mid без taker 5 bps занижує вартість розсинхрону.

**Поведінка**

| Дія | Нога 1 | Нога 2 | Комісія |
|---|---|---|---|
| `both_filled` | maker | maker | 0.02% + 0.02% |
| `chase_leg2` | maker | **taker** на mid | 0.02% + 0.05% |
| `chase_leg1` | **taker** на mid | maker | 0.05% + 0.02% |
| `unwind_leg1` | taker close вже заповненої | не відкривати | flatten, не silent unfilled |
| `strict_both` | як зараз: all-or-nothing, без chase | | без змін дефолту |

Paper-OHLC: поки обидві ноги ще в `pending`, unwind = скасувати pending.
Якщо одна нога вже в `account.positions` — `close_position(..., is_maker=False)`.

Дефолт рушія лишається `legging_mode="strict_both"`. `chase` — явний прапорець.

**Готово коли:** chase не може виглядати дешевшим за maker-maker.

---

## Фаза B2 — rolling ADF / half-life kill у paper (P1)

**Навіщо.** Ринок глушить пару, щойно спред перестає бути стаціонарним.
Стоп після 2 збиткових місяців запізнюється. `coint_scan._adf_pvalue` /
`_half_life` уже є; paper/live їх не викликали.

**Поведінка**

- На закритому барі t (без lookahead): ADF p і OU half-life на lookback спреду.
- Якщо `p > 0.05` або HL = inf / > 200 барів — **блок нових входів**; виходи
  дозволені. Не авто-flatten (шумний ADF).
- Дефолт у paper: увімкнено. У бектест-сигнал за замовчуванням не чіпати.

**Готово коли:** тест на random-walk спред блокує entry; коінтегрований OU — ні.

---

## Фаза C — Kalman vs OLS bake-off (P1)

Код Kalman + OU half-life є (`use_kalman=False`). CLI `--use-kalman`.
Увімкнути дефолт без OOS = новий параметрний ступінь свободи.

**Правило прийняття:** Kalman замінює OLS як дефолт **лише якщо** avg OOS Sharpe ≥ OLS
і DSR не гірший, на ≥3 з 4 пар. Інакше лишається опцією `--use-kalman`.

Звіт: `docs/reports/kalman_vs_ols.md` **після** прогону (або протокол, якщо кеш даних
неповний).

---

## Фаза D — T-cost з філів + risk budget у циклі (P1)

- `calibrate_from_is`: median IS bps окремо maker / chase → `slippage_frac`.
- У `PairsPortfolioRunner.step`: денний/тижневий PnL ≤ ліміту — блок **нових**
  входів; flatten дозволений.
- Vol-target: опційний, default off.

**Готово коли:** paper-audit показує IS mean; halt по daily-loss на портфелі
покритий тестом.

---

## Фаза E — сесії, markout, warn 1m (P2)

- `paper-audit`: розбивка fill-rate / PnL по годині UTC.
- Warn у generic `backtest`, якщо `pairs_arb` і interval ∈ {1m, 5m}.
- Мінімальний markout: mid@fill vs mid наступного бара в IS-журналі.

---

## Фаза F — пізніше (Phase 4, не цей цикл)

- Архів depth5 + Tardis для черги ліміток.
- Avellaneda live після моделі черги/markout.
- Nautilus як зовнішній L2-бенчмарк.
- Реальні ключі Binance після 8 тижнів paper.

---

## Порядок

```text
A   WS URL + listenKey keepalive          ← блокер live/recorder
B   chase taker + unwind flatten          ← блокер чесного PnL
B2  rolling ADF / half-life kill          ← блокер хворих пар
C   Kalman bake-off (звіт, дефолт off)
D   IS→CostModel + daily/weekly halt      ← після B
E   session audit + markout + 1m warn
F   L2 / MM                               ← не починати, доки paper-gate відкритий
```

Paper-моніторинг 8 тижнів іде **паралельно**. Після B варто перезапустити paper
з чистою метрикою комісій.

## Критерій «план виконано»

- [x] Фаза A: нові WS URL у тестах; recorder не на `/ws/{symbol}@`; PUT keepalive
- [x] Фаза B: chase не може пройти як maker; `strict_both` без регресії
- [x] Фаза B2: ADF-kill блокує RW, не чіпає здорову пару
- [x] Фаза C: CLI `--use-kalman`; звіт/протокол Kalman vs OLS; дефолт off
- [x] Фаза D: портфельний daily-loss halt у циклі; IS у paper-audit
- [x] Фаза E: session UTC; markout наступний бар; warn pairs 1m/5m
- [x] `uv run pytest tests/ -q` зелений (перевіряється після реалізації)
- [x] Kalman/chase не ввімкнені «тихо» в paper systemd-юніті без прапорців
