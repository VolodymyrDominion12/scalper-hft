# TODO / Backlog (стан після Спринту 5, 2026-08-30)

## ✅ Зроблено (ітерації 7–11 та Спринти 1–5)
- [x] **Статистичний арбітраж пар**: `pairs_arb` (z-score log-ratio), рушій `backtest/pairs.py` (2 перп-ноги, funding обох ніг, 2-leg maker комісії), портфель пар `pairs_portfolio.py` з Equal Risk Contribution (ERC) алокацією.
- [x] **Валідований універсум пар**:
  - XRP/BTC 1h (+15.9%/рік, 80% WF OOS вікон);
  - LINK/BTC 1h (+12.9%/рік, 80% WF OOS вікон);
  - LINK/ETH 1h (+14.9%/рік, 80% WF OOS вікон);
  - BTC/ETH 1h (+8.3%/рік, 60% WF OOS вікон);
  - Портфель 3 пар: +12.5%/рік при maxDD −3.6%.
- [x] **Дані та кешування**:
  - `data.binance.vision` dumps інтеграція для історичних aggTrades та klines;
  - Семплінг специфічних барів: tick bars, volume bars, dollar bars, imbalance bars (`data/bars.py`);
  - Підтримка бекендів Parquet та PostgreSQL (`data/store.py`, `data/storage.py`);
  - systemd user-юніт `scalper-record.service` з активним linger для збору depth5/bookTicker.
- [x] **Анти-перенавчання та валідація** (16 модулів у `validation/`):
  - Walk-forward, Purged K-fold CV + embargo, Deflated Sharpe (DSR), CSCV/PBO;
  - Стрес-тести: crash, liquidity, vol-spike, funding-shock (`validation/stress.py`);
  - Когортний аналіз decay edge (`validation/cohort.py`) та децильний lift фіч (`validation/lift.py`);
  - Capacity (масштабування капіталу), Survival (Kaplan–Meier час утримання), Time-decay (лаг входу);
  - Квінтильний тест монотонності (`validation/quintile.py`), Johansen/OLS hedge-ratio, Cointegration scanner (`coint_scan.py`).
- [x] **Машинне навчання та мікроструктура**:
  - Мета-лейблінг + bet sizing (сигмоїда / ймовірності) (`ml/labeling.py`, `ml/bet_sizing.py`);
  - MDI, MDA, SFI feature importance + Kendall PCA перевірка (`ml/feature_importance.py`);
  - Clustered Feature Importance — CFI (`ml/clustered_importance.py`);
  - Мікроструктурні фічі (VPIN, Kyle λ, Roll, Amihud, Corwin–Schultz) (`features/microstructure.py`);
  - HMM-виявлення режимів (`features/hmm_regime.py`) та GARCH(1,1) волатильність (`features/volatility.py`).
- [x] **Виконання та Execution**:
  - Емпіричний `CostModel`: vol-aware slippage + Square-Root impact (`backtest/execution.py`);
  - Micro-price розрахунок та котирування fair value (`backtest/micro_price.py`);
  - Price ladder exit каскадні рівні (`live/exit_ladders.py`);
  - Live reconciliation та kill-switch при розходженні позицій (`live/reconcile.py`);
  - SQLite persistence для paper/live (`live/store.py`).
- [x] **Мульти-активні та онлайн-стратегії**:
  - `SparseBasketArb` (кошиковий арбітраж на Lasso/PCA);
  - `Exp3Bandit` (multi-armed bandit для вибору інструментів/моделей);
  - `EnsembleStrategy` з режимами mean, vote, hedge, regime-gated;
  - `HedgeBlend` (no-regret онлайн навчання).
- [x] **Інтерфейси**:
  - MCP-сервер для AI-асистентів (`scalper_hft/mcp_trading.py`, CLI `mcp`);
  - Streamlit дашборд (`scalper_hft/dashboard.py`);
  - Telegram сповіщення.
- [x] **Тести**: 238 тестів у `tests/` — усі проходять (`pytest` зелений).

---

## ⏳ Відкриті задачі (актуальний backlog)

### 1. Paper-валідація та моніторинг (Phase 1 Gate)
- [ ] Безперервний моніторинг paper pairs портфеля (`results/paper_pairs.sqlite`) протягом **≥8 тижнів** без розходження з бектестом.
- [ ] Щотижневий звіт tracking error: порівняння фактичного fill-rate post-only ордерів з моделлю бектесту.

### 2. Накопичення даних глибини стакана (L2 / Depth)
- [ ] Накопичення архіву depth5 снапшотів для повноцінної перевірки `ob_imbalance` на мікроструктурних рівнях.
- [ ] Інтеграція Tardis.dev / L2 historical feeds для бектесту HFT маркет-мейкінгу (`market_maker`).

### 3. Production Hardening & Live (Phase 2)
- [ ] Оновлення Binance API ключів у `.env` (після успішного проходження 8-тижневого paper-гейту).
- [ ] Фінальна верифікація WebSocket потоків через `ccxt.pro` під високим навантаженням.
