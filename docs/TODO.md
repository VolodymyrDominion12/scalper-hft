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
  - Price ladder exit (`live/exit_ladders.py`) та portfolio risk budget — модулі з тестами, ще не в live-циклі;
  - Live reconciliation в циклі (`run_trader_once`, `PairsPortfolioRunner.step`) та kill-switch;
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
- [x] **Тести**: 23 файли в `tests/` — `uv run pytest tests/ -q` має бути зеленим (Postgres-сюїта skip без `TEST_POSTGRES_DSN`).

---

## ⏳ Відкриті задачі (актуальний backlog)

Детальний план з файлами, тестами і порядком PR: [CHANGE_PLAN.md](CHANGE_PLAN.md).

### 1. Paper-валідація та моніторинг (Phase 1 Gate)
- [ ] Безперервний моніторинг paper pairs портфеля (`results/paper_pairs.sqlite`) протягом **≥8 тижнів** без розходження з бектестом.
- [x] Щотижневий звіт tracking error: CLI `paper-audit` (`validation/paper_audit.py`) — fill-rate, maxDD vs BT×1.5, MAE/MFE forensics.

### 1b. Research локально / робот на VPS
План з фазами, файлами і протоколом hotfix: [DEPLOY_PLAN.md](DEPLOY_PLAN.md).
- [x] **Фаза 0:** знімок `PaperAccount` у SQLite, `--daemon`, `control.json`, SIGTERM save.
- [x] **Фаза 1 (код):** systemd `scalper-paper-pairs.service`, `scripts/deploy_paper.sh` (не `git pull` на VPS).
- [ ] **Фаза 1 (реліз):** тег `paper-v0.1.0` на VPS.
- [ ] **Фаза 2:** 8 тижнів paper на VPS.
- [ ] **Фаза 3:** live-адаптер ніг pairs під моком; дефолт `DRY_RUN=true`.
- [ ] **Фаза 4:** `live-v*` лише після Gate і явного запиту.

### 2. План змін 2026-09 (практики vs код)
- [x] **A (P0)** Міграція USDT-M WS на `/public` і `/private` + PUT listenKey keepalive.
- [x] **B (P1)** Chase-нога = taker; unwind flatten вже виконаної ноги; дефолт `strict_both`.
- [x] **B2 (P1)** Rolling ADF / half-life kill у paper (блок нових входів).
- [x] **C (P1)** CLI `--use-kalman`; протокол OLS vs Kalman; дефолт `use_kalman=False` до прийняття.
- [x] **D (P1)** Калібровка CostModel з IsJournal; daily/weekly halt у `PairsPortfolioRunner`.
- [x] **E (P2)** Session-розбивка в `paper-audit`; markout наступний бар; попередження pairs на 1m/5m.

### 3. Накопичення даних глибини стакана (L2 / Depth) — Phase 4, не цей цикл
- [ ] Накопичення архіву depth5 снапшотів для повноцінної перевірки `ob_imbalance` на мікроструктурних рівнях.
- [ ] Інтеграція Tardis.dev / L2 historical feeds для бектесту HFT маркет-мейкінгу (`market_maker`).

### 4. Production Hardening & Live (після paper-gate)
- [ ] Оновлення Binance API ключів у `.env` (після успішного проходження 8-тижневого paper-гейту).
- [ ] Фінальна верифікація WebSocket потоків через `ccxt.pro` під високим навантаженням.
