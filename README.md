# scalper-hft — високочастотна скальпінг-система (Binance USDT-M futures)

Проєкт створено за моделлю книги **"Inside the Black Box" (R. Narang)**:
Alpha → Risk → Transaction Cost → Portfolio Construction → Execution, на фундаменті
Data та Research (бектест + анти-перенавчання). Дослідження ринку/фреймворків/бірж —
у [docs/RESEARCH.md](docs/RESEARCH.md), конспект книги — [docs/book_notes.md](docs/book_notes.md).

## Що всередині

```
scalper_hft/
├── data/        Binance klines / aggTrades / funding → parquet-кеш (ccxt)
├── features/    RSI, EMA, BB, ATR, VWAP, волатильність + CVD, OB-imbalance, spread
├── strategies/  mean_reversion · cvd_momentum · ob_imbalance · market_maker (експерим.)
├── backtest/    векторизований рушій + подієвий (maker) + метрики + CostModel
├── validation/  walk-forward · purged CV · Deflated Sharpe + PBO · sensitivity · Optuna
├── ml/          LightGBM walk-forward класифікатор напрямку (FreqAI-стиль)
├── live/        paper/testnet/live трейдер з ризик-контролем
└── cli.py       CLI: download / backtest / walkforward / optimize / overfit / ml / paper / report
```

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
| `download` | klines / aggTrades / funding у parquet-кеш (`--trades-days` для aggTrades) |
| `backtest` | бектест стратегії з комісіями та slippage (funding PnL для funding_carry) |
| `walkforward` | ковзні IS/OOS вікна — середній OOS Sharpe |
| `optimize` | Optuna-пошук параметрів з purged CV цільовою функцією |
| `overfit` | аудит: WF + sensitivity (плато vs пік) + Deflated Sharpe |
| `cscv` | **PBO через Combinatorial Purged CV** (López de Prado) |
| `ml` | walk-forward LightGBM класифікатор напрямку |
| `cohort` | **деградація edge за когортами угод** (Predictive Marketing: silent attrition) |
| `lift` | **децильний lift-аналіз фіч** — які фічі реально зсувають PnL (uplift-концепт) |
| `featimp` | **MDI/MDA/SFI feature importance** (AFML Ch.8) + PCA-перевірка |
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
Секції: кеш даних по символах, бектест з equity-кривою та Deflated Sharpe,
результати paper-run.

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
.venv/bin/python -m pytest tests/ -q   # 9 тестів: no-lookahead, метрики, DSR, CV, кеш, WF
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
| **pairs_arb (BTC/ETH, 1h, maker)** | **+8.3%/рік**, 47 угод, maxDD −4%, WF OOS>0 | ✅ **перший валідований кандидат** (деталі: docs/pairs_audit.md) |
| market_maker (спрощена модель) | adverse selection > спред | ⚠ потребує L2-даних |
| ob_imbalance (depth-weighted) | даних замало | ⏳ накопичення іде (systemd-сервіс активний) |

**Чесний висновок**: жодна стратегія не пройшла аудит за поточних ринкових умов —
це правильний результат циклу. Система відхиляє edge, якого немає, а валідаційні
шари (бектест ↔ paper-replay ↔ юніт-тести) ловлять баги обліку (funding ~480×).
Наступні кроки: maker post-only виконання (готово в клієнті), накопичення стакана
для OB, моніторинг фандінг-режиму (weekly-audit).

## Roadmap (цикл інвестігейт → реалізація → тест → аудит → покращення)

1. **Дані**: історичні aggTrades через `data.binance.vision` (✅ реалізовано),
   1s/5s свічки, накопичення bookTicker (✅ рекордер), Tardis.dev для L2.
2. **Тіки**: підключення nautilus_trader як альтернативного L2-рушія (порівняльний аудит філів).
3. **Live**: asyncio + ccxt.pro WebSocket-цикл, maker-ордери (post-only), телеграм-сповіщення.
4. **ML**: triple-barrier labeling, hmmlearn regime, deflated Sharpe для ML-моделей.
   ✅ Спринт 1 (docs/book_approaches_synthesis.md): bet sizing із імовірностей,
   мета-лейблінг, confidence-фільтр через proba, Hedge-блендінг (ensemble mode='hedge'),
   breakeven-гейт, cohort/lift аналіз.
   ✅ Спринт 2: мікроструктурні фічі VPIN/Kyle λ/Roll/Amihud/Corwin–Schultz
   (`features/microstructure.py`), HMM-режими (`features/hmm_regime.py`), GARCH σ_{t+1}
   (`features/volatility.py`), емпіричні витрати (vol-scaled slippage + Square-Root
   impact у `CostModel`), ERC-алокація (`portfolio/erc.py`, `pairs-portfolio --method erc`),
   MDI/MDA/SFI feature importance (`ml/feature_importance.py`, CLI `featimp`).
5. **Dashboard**: Streamlit для моніторингу стратегій і параметрів у реальному часі.
