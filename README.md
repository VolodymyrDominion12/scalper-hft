# scalper-hft — високочастотна скальпінг-система (Binance USDT-M futures)

Проєкт створено за моделлю книги **"Inside the Black Box" (R. Narang)**:
Alpha → Risk → Transaction Cost → Portfolio Construction → Execution, на фундаменті
Data та Research (бектест + анти-перенавчання). Дослідження ринку/фреймворків/бірж —
у [docs/RESEARCH.md](docs/RESEARCH.md), конспект книги — [docs/book_notes.md](docs/book_notes.md).

## Що всередині

```
scalper_hft/
├── data/        Binance klines / aggTrades / funding → кеш: parquet або PostgreSQL (ccxt)
│                store.py (бекенди) · resample.py (1m → 5m/15m/1h/...) · access.py (ensure_klines)
├── features/    RSI, EMA, BB, ATR, VWAP, волатильність + CVD, OB-imbalance, spread
├── strategies/  mean_reversion · cvd_momentum · ob_imbalance · market_maker (експерим.)
├── backtest/    векторизований рушій + подієвий (maker) + метрики + CostModel
├── validation/  walk-forward · purged CV · Deflated Sharpe + PBO · sensitivity · Optuna · sweep
├── ml/          LightGBM walk-forward класифікатор напрямку (FreqAI-стиль)
├── live/        paper/testnet/live трейдер з ризик-контролем
└── cli.py       CLI: download / backtest / walkforward / optimize / overfit / ml / paper / report / sweep
```

## Кеш даних: parquet або PostgreSQL у Docker

Дані завантажуються з Binance **один раз** і живуть у кеші. Два бекенди:

- `DATA_BACKEND=parquet` (за замовчуванням) — файли у `data/*.parquet`;
- `DATA_BACKEND=postgres` — **PostgreSQL у Docker** (зручно, коли інструментів багато):

```bash
docker compose up -d postgres          # піднімає scalper_postgres:16 на порту 5433
# у .env:
#   DATA_BACKEND=postgres
#   POSTGRES_PORT=5433                  # (на цій машині 5432/5433 зайняті → 5440)
cp .env.example .env
```

Схема створюється автоматично: `klines(symbol, interval, ts, ohlcv)`,
`agg_trades(symbol, trade_id, ts, price, amount, side)`, `funding(symbol, ts, funding_rate)`.
Дані переживають перезапуск контейнера (docker volume `postgres_data`).

## Ресемплінг: 1m → 5m / 15m / 30m / 1h / 4h / 1d (без повторних запитів до API)

Хвилинні дані покривають усі старші таймфрейми: завантажуємо базу один раз,
решту агрегуємо локально (OHLCV: open=first, high=max, low=min, close=last, volume=sum;
неповний останній бар відкидається):

```bash
# база качається один раз (1m)
.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 90

# бектест на 5m/15m/1h — ресемплінг з 1m-кешу, нуль API-дзвінків
.venv/bin/python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 15m --days 90 --base 1m --derive
```

Логіка `data/access.py::ensure_klines`: кеш цільового таймфрейму → якщо немає і
`derive=True` → ресемплінг із бази → результат зберігається в кеш. Будь-який
нестандартний інтервал (3m, 45m, 2h, ...) автоматично виводиться з бази.

## Матричний прогон: всі стратегії × таймфрейми × інструменти

```bash
# всі single-symbol стратегії × 8 символів × 6 таймфреймів × 30 днів
.venv/bin/python -m scalper_hft.cli sweep --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,LINKUSDT,BNBUSDT,DOGEUSDT,ADAUSDT \
    --intervals 1m,5m,15m,30m,1h,4h --days 30 --workers 4
# результат: results/sweep.csv + results/sweep.md

# ML-стратегія та ensemble — повільно, окремим прогоном на старших таймфреймах
.venv/bin/python -m scalper_hft.cli sweep --all --strategies ml_strategy,ensemble \
    --symbols BTCUSDT,ETHUSDT --intervals 15m,1h,4h --days 30

# walk-forward режим замість бектесту
.venv/bin/python -m scalper_hft.cli sweep --mode walkforward --train 2000 --test 500 --days 60
```

База (1m) качається один раз на символ, решта таймфреймів — ресемплінг.
Кожна клітинка — незалежний бектест з повними комісіями; помилки однієї клітинки
не зупиняють прогон (status/error). Пари (`pairs_arb`, `sparse_basket`) і
delta-neutral `funding_arb` потребують двох ніг — у пер-символьний sweep не входять.

## Швидкий старт

```bash
# 1. ключі (лише для live; дані та бектести працюють без них)
cp .env.example .env   # встав BINANCE_API_KEY / BINANCE_API_SECRET

# 2. середовище
uv venv .venv && uv pip install -e ".[optim,ml,dev]"

# 3. дані (публічні, без ключів)
.venv/bin/python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30 --trades

# 4. бектест
.venv/bin/python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 1m --days 30

```bash
# 5. повний аудит на перенавчання (WF + sensitivity + Deflated Sharpe)
.venv/bin/python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 30

# 5b. бектест з breakeven-гейтом (не торгуємо, якщо ATR < round-trip витрат)
.venv/bin/python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 60 --breakeven-gate

# 5c. cohort decay + децильний lift-аналіз фіч
.venv/bin/python -m scalper_hft.cli cohort --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90
.venv/bin/python -m scalper_hft.cli lift --strategy pairs_arb --symbol BTCUSDT --interval 1h --days 90 --top 10

# 6. markdown-звіт → docs/reports/
.venv/bin/python -m scalper_hft.cli report --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 60
```

## Команди CLI

| Команда | Призначення |
|---|---|
| `download` | klines / aggTrades / funding у кеш (`--trades-days` для aggTrades; `--vision` — data.binance.vision) |
| `backtest` | бектест стратегії з комісіями та slippage (funding PnL для funding_carry); `--base 1m --derive` — ресемплінг |
| `plot` | **інтерактивний HTML-графік бектесту**: свічки + індикатори + точки входу/виходу + рівні SL/TP (Plotly, standalone, `--bars/--start/--end/--out`) |
| `sweep` | **матричний прогон: всі стратегії × символи × таймфрейми** → `results/sweep.csv` (+`--mode walkforward`, `--workers`) |
| `walkforward` | ковзні IS/OOS вікна — середній OOS Sharpe |
| `optimize` | Optuna-пошук параметрів з purged CV цільовою функцією |
| `overfit` | аудит: WF + sensitivity (плато vs пік) + Deflated Sharpe |
| `cscv` | **PBO через Combinatorial Purged CV** (López de Prado) |
| `ml` | walk-forward LightGBM класифікатор напрямку (`--trades` micro, `--hmm`, `--garch`) |
| `cohort` | **деградація edge за когортами угод** (Predictive Marketing: silent attrition) |
| `lift` | **децильний lift-аналіз фіч** — які фічі реально зсувають PnL (uplift-концепт) |
| `featimp` | **MDI/MDA/SFI feature importance** (AFML Ch.8) + PCA-перевірка |
| `stress` | **стрес-тест**: crash / liquidity / vol_spike / funding_shock (Narang гл. 10) |
| `capacity` | **capacity-тест**: Sharpe при масштабуванні позицій (share of wallet) |
| `survival` | **Kaplan–Meier**: медіанний час утримання позиції (+ за бінами фічі) |
| `mcp` | **MCP-сервер для трейдінгу** (stdio, JSON-RPC) — бектест/аналіз як інструменти для AI-асистента |
| `paper` | один крок paper-торгівлі на останньому барі |
| `paper-run` | **циклічний paper-прогін** (N кроків, збереження equity/угод у `results/`) |
| `paper-replay` | відтворення історії через risk-трейдера (валiдація risk-шару) |
| `arb` | **delta-neutral funding arb** (перп+спот): бектест + walk-forward, `--maker` |
| `pairs-portfolio` | портфель валідованих пар; `--method erc` (Equal Risk Contribution), `--turnover-rate` |
| `report` | повний markdown-звіт у `docs/reports/` |
| `record-bookticker` | запис best bid/ask (WS) у parquet — для OB-стратегій (`--depth` — 5 рівнів) |

## Дашборд (Streamlit)

```bash
uv pip install -e ".[dashboard]"
.venv/bin/streamlit run scalper_hft/dashboard.py
```
Мультисторінка (`st.navigation`, сторінки у `scalper_hft/app_pages/`):
- **Моніторинг** — кеш даних по символах, paper pairs (SQLite), paper-run CSV;
- **Бектест** — запуск бектесту, інтерактивний графік угод (свічки +
  індикатори + точки входу/виходу + SL/TP, **клік по маркеру → деталі
  угоди**), таблиця угод, Deflated Sharpe, діагностика (cohort/stress/capacity).

## Візуалізація бектестів (`scalper_hft/visualization/`)

Багатопанельний інтерактивний графік (Plotly, без UI-залежностей):
**свічки + індикатори (BB/EMA/VWAP) + точки входу/виходу (▲/▼/×) +
пунктирні рівні SL/TP + об'єм + позиція + equity/просадка**; внизу range
slider для зуму, легенда вмикає/вимикає шари. `trade_detail_figure` —
детальний графік однієї угоди (вікно навколо входу).

```bash
# standalone HTML (відкривається у браузері без сервера)
.venv/bin/python -m scalper_hft.cli plot --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 30
# вікно + даунсемплінг
.venv/bin/python -m scalper_hft.cli plot --strategy mean_reversion --symbol BTCUSDT --interval 1m --days 7 --bars 2000 --start "2026-08-20" --end "2026-08-22" --out docs/plots/mr_bt.html
```

Як це працює:
- рушій бектесту збагачує кожну угоду цінами входу/виходу (`entry_price`/`exit_price`)
  та, якщо стратегія реалізує `Strategy.exit_levels(df)`, рівнями `sl_price`/`tp_price`
  на барі входу. Реалізації:
  - `mean_reversion` / `hmm_reversion` — ціль = середина BB, стоп = середина ∓ ATR×mult
    (рівні відносні середини смуг, а не ціни входу — це фактична логіка виходу);
  - `ml_strategy` — бар'єри triple-barrier: tp/sl = close ± множник × `_daily_vol(span=100)`
    (та сама волатильність, що в навчальних мітках);
- у дашборді (сторінка «Бектест») — повна фігура з вікном, тумблерами
  (маркери/SL-TP/індикатори), **кліком по маркеру угоди** (деталі: свічки
  навколо входу, SL/TP-сегмент, позиція, equity) та таблицею угод (PnL, ціни, SL/TP);
- для великих даних — даунсемплінг `max_bars`, бари входу/виходу угод завжди
  зберігаються на графіку.

## Telegram-сповіщення

Ключі `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` у `.env` (скопійовані з
trade-bots/.env; валідність перевірена через getMe). Використання:
`paper-run --notify` або `scalper_hft.live.telegram.send_telegram("...")`.

## Анти-перенавчання (validation/)

- **Walk-forward**: середній OOS Sharpe по ковзних вікнах; деградація IS→OOS.
- **Purged CV**: K-fold з purging + embargo (López de Prado) — без витоку між сусідніми зрізами.
- **Deflated Sharpe (DSR)**: коригування Sharpe на кількість спроб (Bailey & López de Prado).
  `DSR > 0.95` — edge статистично значущий; `estimate_n_trials` рахує ефективні спроби.
- **Sensitivity**: smoothness сітки параметрів — плато = здорово, ізольований пік = перенавчання.
- **Правило**: оптимізуєш на train+CV, фінальний вердикт — лише на OOS holdout.

## Комісії та витрати (критично для скальпінгу!)

Binance USDT-M: **maker 0.02% / taker 0.05%** (BNB-дисконт → 0.018/0.045).
Round-trip taker ≈ **0.10%** ноціоналу — це ~10 повних угод на день, щоб спалити 1% капіталу.
Поріг беззбитковості для taker-стратегії = `CostModel.breakeven_move_pct`.
Висновок дослідження: серйозний скальпінг = **maker-підхід** (post-only) або направлені
рухи > 0.1–0.2%. Деталі: [docs/RESEARCH.md](docs/RESEARCH.md).

## Live / paper

За замовчуванням `DRY_RUN=true` — paper або testnet. Live лише після явної зміни:
`.env`: `DRY_RUN=false`, `EXCHANGE=binance` + робочі ключі. Ризик-контроль
(ліміт позиції, денний ліміт збитків, пауза після серії збитків) — у `live/trader.py`.

⚠ Ключі з `trade-bots/.env`, скопійовані у цей проєкт, на момент створення **невалідні**
(ccxt: `Invalid Api-Key ID`). Публічні дані працюють без ключів; для live оновіть ключі.

## Тести

```bash
uv run pytest tests/ -q   # 238 тестів у 15 тестових сюїтах
```

## Аудит стратегій (90 днів 1m-даних, комісії + slippage)

> ⚠ **Виправлення обліку funding (ітерація 4)**: попередній "позитивний" результат
> funding_carry був артефактом багу — ставка нараховувалася кожен бар замість
> одного разу за 8h-блок (~480× завищення). Після виправлення стратегія відхилена.
> Деталі: [docs/audit_findings.md](docs/audit_findings.md).
>
> 📉 **Фандінг-режим 2025–2026 (ітерація 6)**: ставки структурно низькі (BTC 2.3%,
> ETH 1.3%, SOL −1.2% річних; жодного періоду >36% річних з квітня 2024). Funding-
> стратегії режимно-сплячі. Високий режим був лише у 2024 Q1 (9–19% точок >36%).
> Деталі: [docs/funding_regimes.md](docs/funding_regimes.md).

| Стратегія | Результат (виправлено) | Вердикт аудиту |
|---|---|---|
| mean_reversion (з regime-фільтрами) | від'ємний | ⚠ відхилено (OOS<0, DSR=0) |
| cvd_momentum (реальні aggTrades) | 1040 угод/тиждень, fee-drag | ⚠ відхилено (PF 0.30) |
| funding_carry (збір фандінгу) | BTC −0.10%, ETH −0.05%, SOL −0.54% | ⚠ відхилено (OOS −0.35, DSR=0) |
| funding_arb (delta-neutral) | −0.4…−2.2% (maker/taker) | ⚠ відхилено (тертя > фандінг за поточних ставок) |
| basis_reversion (1m) | овертрейдинг, −31…−85% | ⚠ відхилено (fee-drag) |
| **pairs_arb (XRP/BTC, LINK/BTC, BTC/ETH)** | **+8.3…+15.9%/рік**, maxDD −3.6…−6.5% | ✅ **валідовані кандидати** (деталі: docs/pairs_audit.md) |
| market_maker (спрощена модель) | adverse selection > спред | ⚠ потребує L2-даних |
| ob_imbalance (depth-weighted) | даних замало | ⏳ накопичення іде (systemd-сервіс активний) |

**Чесний висновок**: жодна 1m-скальпінг стратегія не пройшла аудит через комісійне тертя. Єдиний валідований напрямок — **портфель 1h пар з maker-виконанням (post-only)**.
Наступні кроки: безперервний paper-прогін (≥8 тижнів), накопичення стакана для OB, моніторинг фандінг-режиму.

## Roadmap (цикл інвестігейт → реалізація → тест → аудит → покращення)

1. **Дані**: історичні aggTrades через `data.binance.vision` (✅ реалізовано),
   генерація tick/volume/dollar/imbalance барів (✅ реалізовано), накопичення bookTicker (✅ рекордер).
2. **Тіки та L2**: підключення Tardis.dev / nautilus_trader для моделювання черги заявок у маркет-мейкінгу.
3. **Live**: paper-моніторинг pairs портфеля (≥8 тижнів), звірка reconciliation та kill-switch (✅ реалізовано).
4. **ML та синтез методологій** ([docs/book_approaches_synthesis.md](docs/book_approaches_synthesis.md)):
   ✅ Спринт 1: bet sizing із імовірностей, мета-лейблінг, Hedge-блендінг, breakeven-гейт, cohort/lift аналіз.
   ✅ Спринт 2: мікроструктурні фічі VPIN/Kyle λ/Roll/Amihud, HMM-режими, GARCH(1,1), емпіричні витрати (Square-Root impact у `CostModel`), ERC-алокація, MDI/MDA/SFI.
   ✅ Спринт 3: стрес-тест (`validation/stress.py`), портфельний risk budget (`portfolio/risk_budget.py`), сигмоїдний sizing + лімітна ціна (`ml/bet_sizing.py`), capacity-тест, Kaplan–Meier survival.
   ✅ Спринт 4: MCP-сервер для трейдінгу (`scalper_hft/mcp_trading.py`, CLI `mcp`), alpha-гіпотеза `hmm_reversion`, live-інтеграція (vol-scaled sizing + HMM-блок), дашборд §5.
   ✅ Спринт 5: micro-price котирування, price ladder exits (`live/exit_ladders.py`), Clustered Feature Importance (CLI `cfi`), Sparse Basket Arbitrage (`strategies/sparse_basket.py`), Exp3 онлайн-бандит (`strategies/bandit.py`).
5. **Dashboard**: Streamlit для моніторингу стратегій і параметрів у реальному часі (✅ реалізовано).
