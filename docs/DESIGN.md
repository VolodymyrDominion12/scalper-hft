# Дизайн scalper-hft

## 1. Мета
Високочастотна скальпінг-система для Binance USDT-M ф'ючерсів: цикл
**дослідження → реалізація → тестування → аудит → покращення** з жорстким
захистом від перенавчання. Цільовий горизонт стратегій — секунди/хвилини/години
(з урахуванням реалістичної для retail латентності з дому 50–200 мс, див. [docs/RESEARCH.md](RESEARCH.md)).

## 2. Архітектура (за книгою "Inside the Black Box", гл. 2)

```
Дані (Binance: klines, aggTrades, funding, vision dumps → parquet / PostgreSQL)
   │
   ▼
Alpha Model ────┐
Risk Model ─────┼──► Portfolio Construction (ERC / Risk Budget) ──► Execution (backtest/live)
Cost Model ─────┘
   │
   ▼
Research & Validation:
  - Walk-forward / Purged K-fold CV / Deflated Sharpe / CSCV (PBO) / Sensitivity
  - Stress-testing / Cohort Decay / Feature Lift / Capacity / Survival / Time-decay / Quintiles
  - ML: Triple-barrier labeling / Meta-labeling / Bet sizing / CFI / LOB models
```

### Модулі (`scalper_hft/`):
| Модуль | Відповідальність | Ключові файли |
|---|---|---|
| `data` | REST/ccxt, Binance Vision dumps, parquet/PostgreSQL кеш, tick/volume/dollar/imbalance бари, валідація | `access.py`, `bars.py`, `binance_client.py`, `binance_vision.py`, `downloader.py`, `storage.py`, `store.py`, `validate.py` |
| `features` | ТА індикатори, мікроструктура (VPIN, Kyle λ, Roll, Amihud), HMM-режими, GARCH(1,1), FFD, DSP | `indicators.py`, `microstructure.py`, `hmm_regime.py`, `volatility.py`, `fractional_diff.py`, `signal_processing.py`, `regimes.py` |
| `strategies` | Альфа-моделі (єдиний інтерфейс `Strategy`, реєстр у `__init__.py`) | `mean_reversion.py`, `cvd_momentum.py`, `ob_imbalance.py`, `market_maker.py`, `funding_carry.py`, `funding_arb.py`, `basis_reversion.py`, `pairs_arb.py`, `ml_strategy.py`, `ensemble.py`, `hmm_reversion.py`, `sparse_basket.py`, `cross_momentum.py`, `bandit.py`, `blend.py` |
| `portfolio` | Конструювання портфеля: Equal Risk Contribution (ERC), risk-parity, vol-targeting, VaR, loss-budget | `erc.py`, `risk_budget.py` |
| `backtest` | Векторизований/подієвий рушії, парний/портфельний бектест, емпіричний CostModel, micro-price, router | `engine.py`, `event_engine.py`, `pairs.py`, `pairs_portfolio.py`, `execution.py`, `micro_price.py`, `metrics.py`, `router.py` |
| `validation` | Анти-перенавчання, статистична та сценарна валідація | `walk_forward.py`, `cv.py`, `deflated_sharpe.py`, `cscv.py`, `sensitivity.py`, `stress.py`, `cohort.py`, `lift.py`, `capacity.py`, `survival.py`, `time_decay.py`, `quintile.py`, `coint_scan.py`, `hedge_ratio.py`, `optimize.py`, `oos_registry.py` |
| `ml` | LightGBM meta-labeling, bet sizing, sample weights, MDI/MDA/SFI, CFI, LOB models | `labeling.py`, `trainer.py`, `bet_sizing.py`, `features.py`, `feature_importance.py`, `clustered_importance.py`, `sample_weights.py`, `frac_diff.py`, `lob_models.py`, `train_lob.py` |
| `live` | Paper/testnet/live трейдер, bookTicker/depth recorder, exit ladders, reconciliation & kill-switch, SQLite | `account.py`, `trader.py`, `pairs_runner.py`, `paper_runner.py`, `paper_replay.py`, `reconcile.py`, `bookticker_recorder.py`, `exit_ladders.py`, `fills.py`, `is_log.py`, `store.py`, `telegram.py` |
| `mcp_trading.py` | MCP (Model Context Protocol) сервер для AI-асистентів | `mcp_trading.py` |
| `research` | Sweep store, черга дослідницьких задач (SQLite + worker-процеси, без брокера) | `jobs.py`, `job_worker.py`, `job_handlers.py`, `sweep_store.py` |
| `dashboard.py` | Streamlit аналітичний та моніторинговий дашборд | `dashboard.py` |
| `cli.py` | Повний CLI-інтерфейс (30+ команд) | `cli.py` |

---

## 3. Ключові рішення

### 3.1. Виконання без lookahead
Сигнал обчислюється на закритті бару t; позиція застосовується з бару t+1
(`pos = signals.shift(1)`). Комісії — за turnover. Тест: `test_engine_no_lookahead`.
Обмеження: рушій не може перешкодити стратегії читати майбутні рядки всередині
`generate_signals` — це відповідальність автора стратегії (див. `skills/strategy-development.md`).

### 3.2. Модель витрат (гл. 5 книги)
`CostModel`: maker_fee (0.02%) + taker_fee (0.05%) + vol-aware slippage + Square-Root market impact (`k_imp * sigma * sqrt(Q/ADV)`).
`breakeven_move_pct` та `breakeven_gate` запобігають входам, коли очікуваний рух менший за round-trip витрати.

### 3.3. Анти-перенавчання (гл. 9 книги + AFML)
- **Walk-forward**: ковзні IS/OOS вікна, головна метрика — avg OOS Sharpe.
- **Purged K-fold CV & CPCV** (López de Prado): purging + embargo проти витоку інформації та автокореляції.
- **Deflated Sharpe (Bailey & LdP)**: коригування на кількість спроб (DSR > 0.95 = значущий edge).
- **Sensitivity & Monotonicity**: стабільність параметрів (плато vs пік) та квінтильні тести монотонності.
- **Stress-Testing & Cohorts**: сценарні шоки (crash −30%, liquidity, funding spike) та когортне відстеження згасання edge.

### 3.4. Стратегії
1. `pairs_arb` — парний статистичний арбітраж на 1h барах (z-score log-ratio, maker post-only). **Єдиний валідований кандидат**.
2. `sparse_basket` — кошиковий мульти-активний арбітраж на основі Lasso/PCA.
3. `ml_strategy` — LightGBM з потрійним бар'єром (triple-barrier), meta-labeling та ймовірнісним bet-sizing.
4. `ensemble` — ансамбль з динамічними вагами (Hedge no-regret, voting, regime-gating).
5. `cross_momentum` — time-series momentum (CLI) / крос-секційний ранг (мультиколонки; 1D-сигнал першого символу). Лаг t+1 робить рушій.
6. `hmm_reversion`, `mean_reversion`, `cvd_momentum`, `funding_carry`, `funding_arb`, `basis_reversion`, `ob_imbalance`, `market_maker` — досліджені та відхилені або законсервовані (див. [docs/STRATEGY_STATUS.md](STRATEGY_STATUS.md)).

*Примітка: `blend` та `bandit` — це алгоритми агрегації та утиліти, а не окремі підкласи `Strategy`.*

### 3.5. Live та ризик-контроль
Paper/testnet за замовчуванням (`DRY_RUN=true`).
Багаторівневий ризик-контроль:
- Hard limits: ліміт позиції, денний ліміт збитків на ногу та портфель, зупинка після серії збитків;
- Динамічне масштабування: vol-scaled sizing (GARCH/EWMA) та блокування нових входів за HMM-режимом (помилка моделі → блок входу);
- Звірка (reconciliation): `fetch_positions` + kill-switch у `run_trader_once` та `PairsPortfolioRunner.step` (paper/`DRY_RUN=true` — no-op);
- Exit ladders (`live/exit_ladders.py`) та portfolio risk budget (`portfolio/risk_budget.py`) — бібліотечні модулі з тестами, **ще не** в paper/live циклі.

---

## 4. Поточний статус та Roadmap

### Реалізовано:
1. ✅ Повна інфраструктура збору та кешування даних (Parquet, Postgres, Binance Vision, tick/dollar bars).
2. ✅ Рекордер bookTicker/depth5 на systemd user-юніті.
3. ✅ Валідований універсум пар (XRP/BTC, LINK/BTC, LINK/ETH, BTC/ETH) та мульти-парний портфель з ERC-алокацією.
4. ✅ Повний ML-пайплайн (мета-лейблінг, bet sizing, MDI/MDA/SFI/CFI, LOB моделі).
5. ✅ Розширені валідаційні тести (stress, capacity, survival, lift, cohort, quintiles, time-decay).
6. ✅ MCP-сервер для трейдінгу та Streamlit дашборд.

### Наступні кроки (Roadmap):
1. **Paper-Gate**: 8 тижнів безперервного paper-прогону портфеля пар (`paper-run-pairs`) для верифікації відсутності розходжень з бектестом.
2. **VPS / research split**: демон + persist стану, реліз git-тегом, paper на сервері паралельно з локальним дослідженням — [DEPLOY_PLAN.md](DEPLOY_PLAN.md).
3. **L2 Order Book**: накопичення тривалого масиву depth-даних та інтеграція Tardis.dev для моделювання черги лімітних ордерів у маркет-мейкінгу.
4. **Live Execution**: перехід на реальний рахунок (лише за явним запитом і після успішного Paper-Gate + live-адаптера ніг).
