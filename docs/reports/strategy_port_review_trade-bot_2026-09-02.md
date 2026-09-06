# Огляд: які стратегії з `trade-bot-main` варто перенести в `scalper-hft`

Дата: 2026-09-02 · Автор: огляд коду (AI-агент) · Статус: рекомендація, чекає рішення

## 0. Резюме

- **Універсум інструментів вже синхронізовано** (див. §6): 15 інструментів
  (BTC, ETH, SOL, BNB, XRP, LINK, ADA, DOGE, AVAX, NEAR, DOT, ATOM, UNI, LTC, AAVE)
  тепер є канонічним списком `scalper_hft/symbols.py`, у `DEFAULT_SYMBOLS` (.env),
  на дашборді та в MCP.
- **Стратегії**: прямого «перенести все» не рекомендується — більшість із ~37
  стратегій `trade-bot-main` належать до родин, які в `scalper-hft` вже
  покриті (mean-reversion) або є взаємозамінними варіаціями одного й того ж
  (trend + SL/TP). Кандидатів, що додають **нову родину альфи**, — обмежений
  набір (SMC, часова структура сесій, trailing-trend), див. §4.
- **Важливий контекст**: `trade-bot-main` — спотова система 1h/4h з GA та
  комісією 0.1%; `scalper-hft` — ф'ючерсна (maker 0.02% / taker 0.05%),
  інтерфейс стратегій інший (Series позицій, без lookahead), і тут уже
  відхилено більшість single-symbol стратегій після чесного аудиту
  (див. `docs/STRATEGY_STATUS.md`). Тому будь-який порт має проходити
  штатний валідаційний цикл scalper-hft, а не переносити «перемоги» з іншого
  ринку/комісійного режиму.

---

## 1. Контекст: чому це не copy-paste

| | trade-bot-main | scalper-hft |
|---|---|---|
| Ринок | Спот (perp-пари у результатах, 0.1% fee в backtest) | Binance USDT-M ф'ючерси, maker/taker 0.02%/0.05% + slippage |
| Таймфрейми | 1h / 4h / 8h | 1m…4h (ресемплінг з 1m-бази) |
| Інтерфейс стратегії | `generate_signals(df) → DataFrame` з `signal/stop_loss/take_profit/entry_price`; `get_last_signal()` для live | `generate_signals(df[, trades, funding]) → pd.Series` позицій в {-1,0,+1}; опційно `exit_levels()`, `param_space` |
| Вихід з позиції | Stop-loss/take-profit як окремі рівні (спрацьовують за ціною) | Позиція тримається, поки сигнал не зміниться на закритті бару; SL/TP — атрибути угод для візуалізації |
| Оптимізація | GA (`GENE_SPACE`) | Optuna (`param_space`) + walk-forward / DSR / CSCV |
| Long/short | Long-only за замовчуванням, `allow_short` опційно | Завжди ±1 (шорт — першокласний) |

Порт = **переписати логіку входу/виходу у форму «Series бажаної позиції»**,
перенісши індикаторні розрахунки у `scalper_hft.features` або всередину
модуля стратегії. Зареєструвати у `strategies/__init__.py::REGISTRY` — після
цього стратегія автоматично з'являється на дашборді, у sweep/ensemble/звітах.

---

## 2. Інвентар trade-bot-main (~37 модулів) за родинами

**R1. Mean-reversion / осцилятори:** `bb_mr`, `cci_mr`, `stoch_rsi`, `williams_r`,
`zscore_mr`, `rsi_ema` (мультифільтр RSI+EMA+ADX+MACD+MFI), `mfi`, `cmf`,
`divergence`, `vwap`, `vwap_stdev`, `volume_profile`.

**R2. Trend-following + SL/TP:** `aroon`, `chandelier`, `chop_trend`, `di_cross`,
`donchian`, `ema_adx`, `heikin_ashi`, `hull_ma`, `ichimoku`, `obv_trend`, `psar`,
`roc_mom`, `supertrend`, `keltner` (squeeze-breakout), `macd_bb`, `ttm_squeeze`,
`shock_fade` (fade вибухового бару), `orb` / `session_orb` (часова структура).

**R3. SMC / структурні (цінові рівні):** `smc_ob` (order blocks), `smc_fvg`
(fair value gaps), `smc_bos` (BOS/CHoCH), `liquidity_sweep` (збір ліквідності).

**R4. Інше:** `btc_residual` (alt-мінус-BTC beta residual MR — потребує
референсного ряду BTC), `ensemble_strategy` (голосування суб-стратегій).

## 3. Що вже є в scalper-hft (REGISTRY, 13 стратегій)

`pairs_arb` (валідований), `cross_momentum`, `mean_reversion`, `cvd_momentum`,
`ob_imbalance`, `market_maker`, `funding_carry`, `funding_arb`, `basis_reversion`,
`ml_strategy`, `ensemble`, `hmm_reversion`, `sparse_basket` + хелпери
`bandit` (Exp3), `blend` (Hedge).

**Перекриття з trade-bot-main:**
- R1 (MR-осцилятори) ≈ `mean_reversion` / `hmm_reversion`. `bb_mr` = практично
  той самий RSI+BB MR, що вже є.
- R2 (trend на OHLCV) ≈ `cross_momentum` (time-series proxy у single-symbol
  режимі) та `cvd_momentum` (дельтова версія momentum, потребує aggTrades).
- R4 ensemble ≈ `ensemble` / `blend` / `bandit` (у scalper є свої, новіші).
- SMC (R3), сесійні (ORB/VWAP/volume_profile), trailing-trend — **прогалини**.

---

## 4. Рекомендації «що переносити»

Оцінки effort: **S** — ~0.5–1 день, **M** — 2–4 дні, **L** — тиждень+.
Критерії: (a) нова родина альфи проти поточного портфеля; (b) сумісність з
інтерфейсом Series-позицій; (c) наявні дані (лише OHLCV — дешево; aggTrades/
funding — додаткові вимоги); (d) попередній досвід у trade-bot-main.

### Tier A — варто перенести (нова альфа, помірний effort)

| # | Стратегія | Родина | Чому | Effort |
|---|---|---|---|---|
| 1 | `smc_fvg` + `smc_ob` | SMC | Структурні зони (FVG/order blocks) — окрема від осциляторів/трендів альфа, у scalper-hft відсутня; працює на 1h/4h maker. Дані — лише OHLCV | M |
| 2 | `smc_bos` (BOS/CHoCH) | SMC | Те саме; у trade-bot-main — найближчий до промоції кандидат Wave 4 (BNB/USDT 4h: OOS Sharpe ≈1.24) | M |
| 3 | `supertrend` | trend | ATR state-machine → ідеально лягає у Series-позицій (тримаємо напрямок, вихід на фліпі); репрезентант родини trailing-trend | S |
| 4 | `stoch_rsi` | MR | Єдиний robust-кандидат власного скринінгу trade-bot (BTC/USDT 1h, дефолти): OOS Sharpe 1.48; як *другий* MR-представник у scalper (після наявного mean_reversion) | S |
| 5 | `liquidity_sweep` | SMC | Торгівля після збору ліквідності з підтвердженням — добре стикується з наявним research (session/volume) | M |

### Tier B — можна, але нижчий пріоритет (дублікати родин або слабкі докази)

| Стратегії | Коментар |
|---|---|
| `cci_mr`, `williams_r`, `zscore_mr`, `mfi`, `cmf`, `divergence`, `vwap`, `vwap_stdev`, `volume_profile` | Усі — варіації R1. Портувати **1–2 представники** (напр. `zscore_mr`, `vwap_stdev`) достатньо для перевірки гіпотези «ще один осцилятор дає value-added»; не всі 9 |
| `aroon`, `di_cross`, `ema_adx`, `heikin_ashi`, `hull_ma`, `obv_trend`, `roc_mom`, `donchian`, `psar`, `macd_bb`, `keltner`, `ttm_squeeze`, `chop_trend` | Взаємозамінні trend-тригери. Обирати **1–2** (найпростіші у Series-формі: `psar`, `donchian`), не «родину цілком»; `ichimoku` — великий обсяг індикаторів, лише якщо конче треба |
| `orb`, `session_orb` | Окрема «часова структура» (UTC-сесії). Цікаво, але потребує денного ресету (у даних є ts) і перевірки через наявний session-analysis; спершу — research-прототип, не REGISTRY |
| `shock_fade`, `chandelier` | Оригінальні ідеї (fade шоку / chandelier-exit) — S–M; портувати, якщо потрібна ширина в research-скринінгу |

### Tier C — не переносити

| Стратегія | Причина |
|---|---|
| `bb_mr` | Прямий дублікат наявного `mean_reversion` (RSI + BB) |
| `ensemble_strategy` | У scalper вже є `ensemble` (mean/vote/hedge), `blend`, `bandit` |
| `btc_residual` | Потребує референс-ряду BTC та бета-регресії — немає відповідної інфраструктури у single-symbol рушії; за змістом перекривається `pairs_arb` (валідований) |
| `rsi_ema` (мультифільтр) | Роль «комбінації фільтрів» у scalper виконують `ensemble` + `ml_strategy` |
| Будь-що з `needs_trades/funding` із trade-bot | У trade-bot таких немає; зворотний напрямок (funding/basis/OB/CVD/pairs) уже scalper-специфічний |

> Докази з trade-bot-main слабкі для scalper-hft: його phase-5 скринінг
> (BTC/USDT 1h, дефолтні параметри, **спот 0.1% fee**) показав robust лише
> `stoch_rsi`; більшість стратегій — негативний OOS Sharpe. Це інший ринок і
> інші витрати, тому остаточний арбітр — валідаційний цикл scalper-hft
> (walk-forward → Deflated Sharpe → sensitivity), як у `docs/STRATEGY_STATUS.md`.

---

## 5. Як портувати (рецепт для Tier A/B)

1. **Новий модуль** `scalper_hft/strategies/<name>.py`:
   - успадкувати `scalper_hft.strategies.base.Strategy`, задати `name`,
     `param_space` (lo, hi, step) для Optuna, класові прапори
     (`needs_trades` / `needs_funding` / `use_breakeven_gate` — за потреби);
   - індикатори перенести в `scalper_hft/features/indicators.py` (або
     лишити локально в модулі, якщо специфічні);
   - реалізувати `generate_signals()` → Series у {-1, 0, +1}: входи/виходи
     **на закритті бару t** (рушій сам лагає виконання на t+1 — lookahead
     заборонено, див. `tests/test_system.py::test_engine_no_lookahead`);
   - SL/TP-логіку (якщо була рівневою в trade-bot) переписати у стан-машину
     позицій (вихід, коли close перетинає рівень) + опційно `exit_levels()`
     для графіка;
   - обов'язково врахувати комісії — їх додає рушій, але «кількість угод»
     (frequency) має бути реалістичною для maker/taker режиму.
2. **Зареєструвати** у `strategies/__init__.py` (import + REGISTRY) — стратегія
   одразу з'явиться на дашборді, у `sweep`, `ensemble`, звітах і MCP.
3. **Прогнати**: `download` даних → `backtest` (з комісіями) →
   `walkforward` → `overfit` (DSR) → `cscv` → `capacity`/`time-decay`.
4. **Тести**: `uv run pytest tests/ -q` зелений; юніт-тест на нову стратегію
   (детермінований сигнал на синтетичному df, перевірка відсутності lookahead).

Приклад команди після порту:
```bash
./.venv/bin/python -m scalper_hft.cli backtest --strategy supertrend --symbol BTCUSDT --interval 1h --days 90
./.venv/bin/python -m scalper_hft.cli walkforward --strategy supertrend --symbol BTCUSDT --interval 1h --days 180
./.venv/bin/python -m scalper_hft.cli overfit --strategy supertrend --symbol BTCUSDT --interval 1h --days 180
```

---

## 6. Інструменти: що вже зроблено (супутній запит)

Канонічний універсум розширено з 4–5 до **15** інструментів
(список користувача, включно з DOTUSDT, якого немає в trade-bot-main):

`BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, LINKUSDT, ADAUSDT, DOGEUSDT,
AVAXUSDT, NEARUSDT, DOTUSDT, ATOMUSDT, UNIUSDT, LTCUSDT, AAVEUSDT`

Змінені файли:
- `scalper_hft/symbols.py` — **новий** stdlib-only модуль `CANONICAL_SYMBOLS`
  (єдине джерело істини; імпортується з дашборду без site-packages);
- `scalper_hft/config.py` — fallback `DEFAULT_SYMBOLS` = `CANONICAL_SYMBOLS`;
- `.env` / `.env.example` — `DEFAULT_SYMBOLS` = 15 інструментів;
- `scalper_hft/app_pages/_common.py` — `SYMBOLS` = канонічний список
  (дашборд «Моніторинг»/«Бектест»/«Дослідження» показує всі 15);
- `scalper_hft/mcp_trading.py` — `market_status` читає
  `settings.default_symbols`.

Примітки:
- `PAIR_CHOICES` для `pairs_arb` навмисно не розширювали: лише пари, що
  пройшли коінтеграційний скринінг (XRP/BTC, BTC/ETH, LINK/BTC, LINK/ETH).
- Решта інструментів універсуму trade-bot-main (SUI, TRX, XLM, HBAR, APT)
  у список не входили — можна додати однією зміною у `symbols.py`, якщо треба.
- Дані для нових символів підтягуються командою:
  `./.venv/bin/python -m scalper_hft.cli download --symbol BNBUSDT --interval 1m --days 30 --trades`

---

## 7. Статус: що портовано (2026-09-02, «швидкий старт»)

За рішенням користувача портовано першу хвилю (Tier A, «швидкий старт»):

| Стратегія | Модуль | Родина | Статус |
|---|---|---|---|
| `supertrend` | `scalper_hft/strategies/supertrend.py` | trailing-trend (ATR) | ✅ реалізовано, зареєстровано, юніт-тести |
| `stoch_rsi` | `scalper_hft/strategies/stoch_rsi.py` | stochastic RSI MR | ✅ реалізовано, зареєстровано, юніт-тести |
| `smc_fvg` | `scalper_hft/strategies/smc_fvg.py` | SMC (Fair Value Gaps) | ✅ реалізовано, зареєстровано, юніт-тести |

Допоміжні зміни:
- `scalper_hft/features/smc.py` — новий: `fair_value_gaps()` (перенесено з
  trade-bot-main/smc/liquidity.py; без lookahead);
- `scalper_hft/strategies/__init__.py` — REGISTRY розширено (тепер 16);
- `tests/test_ported_strategies.py` — новий: реєстрація, форма/діапазон
  сигналів, детермінізм, **відсутність lookahead** (зріз-тест), smoke-бектест
  з комісіями, робота в `ensemble`.

Тести: `uv run pytest tests/ -q` — зелений.

### Адаптації при порті (важливо для розуміння результатів)

- Сигнали повертаються як **Series позицій** ∈ {−1, 0, +1} (конвенція
  scalper-hft), а не як DataFrame з SL/TP-колонками trade-bot;
- SL/TP з оригіналів переписані у стан-машину позицій: вихід спрацьовує на
  **дотику бару** (low ≤ SL або high ≥ TP), виконання — з наступного бару
  (рушій лагає сам; lookahead-тест підтверджує каузальність);
- `allow_short=False` за замовчуванням (спотова поведінка оригіналів);
  `allow_short=True` — дзеркальні шорти зі своїми SL/TP;
- `use_mtf` (мульти-таймфрейм фільтр оригіналу) не переносився — у scalper
  немає MTF-інфраструктури; таймфрейм задається бектестом;
- у `supertrend` ATR рахується локально через `ewm(span=…)`, як в оригіналі —
  `features.indicators.atr` (min_periods) лишає рекурсивні стрічки NaN.

### Наступні кроки (валідація — після наявності даних)

```bash
./.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 90 --trades
./.venv/bin/python -m scalper_hft.cli backtest  --strategy supertrend --symbol BTCUSDT --interval 1h --days 90
./.venv/bin/python -m scalper_hft.cli walkforward --strategy smc_fvg --symbol BTCUSDT --interval 1h --days 180
./.venv/bin/python -m scalper_hft.cli overfit --strategy stoch_rsi --symbol BTCUSDT --interval 1h --days 180
```

Перш ніж вважати стратегію кандидатом — повний цикл за
`docs/STRATEGY_STATUS.md` (walk-forward → DSR → sensitivity → paper).
