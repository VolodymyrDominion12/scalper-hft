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

# 5. повний аудит на перенавчання (WF + sensitivity + Deflated Sharpe)
.venv/bin/python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 30

# 6. markdown-звіт → docs/reports/
.venv/bin/python -m scalper_hft.cli report --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 60
```

## Команди CLI

| Команда | Призначення |
|---|---|
| `download` | klines / aggTrades / funding у parquet-кеш |
| `backtest` | бектест стратегії з комісіями та slippage |
| `walkforward` | ковзні IS/OOS вікна — середній OOS Sharpe |
| `optimize` | Optuna-пошук параметрів з purged CV цільовою функцією |
| `overfit` | аудит: WF + sensitivity (плато vs пік) + Deflated Sharpe |
| `ml` | walk-forward LightGBM класифікатор напрямку |
| `paper` | один крок paper-торгівлі на останньому барі |
| `report` | повний markdown-звіт у `docs/reports/` |

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

## Roadmap (цикл інвестігейт → реалізація → тест → аудит → покращення)

1. **Дані**: 1s/5s свічки + tick-запис bookTicker для OB-стратегій; data.binance.vision dumps.
2. **Тіки**: підключення nautilus_trader як альтернативного L2-рушія (порівняльний аудит філів).
3. **Live**: asyncio + ccxt.pro WebSocket-цикл, котирування maker-ордерів, телеграм-сповіщення.
4. **ML**: triple-barrier labeling, hmmlearn regime, deflated Sharpe для ML-моделей.
5. **Dashboard**: Streamlit для моніторингу стратегій і параметрів у реальному часі.
