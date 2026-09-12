# TODO / Backlog (стан після Спринту 5, 2026-08-30)

## ✅ Зроблено (ітерації 7–11 та Спринти 1–5)
- [x] **Статистичний арбітраж пар**: `pairs_arb` (z-score log-ratio), рушій `backtest/pairs.py` (2 перп-ноги, funding обох ніг, 2-leg maker комісії), портфель пар `pairs_portfolio.py` з Equal Risk Contribution (ERC) алокацією.
- [x] **Валідований універсум пар** (актуально: лише LINK/BTC у `VALIDATED_PAIRS`; Aug-30 цифри застарілі — див. [STRATEGY_STATUS.md](STRATEGY_STATUS.md)):
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
  - Price ladder exit (`live/exit_ladders.py`) — вимкнено прапорцем `USE_EXIT_LADDERS=false` (не для pairs-gate); portfolio risk budget — модуль з тестами, ще не в live-циклі;
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
- [x] **Тести**: ~120 файлів у `tests/` — `uv run pytest tests/ -q` має бути зеленим (Postgres-сюїта skip без `TEST_POSTGRES_DSN`).

---

## ⏳ Відкриті задачі (актуальний backlog)

**Останній research-цикл:** iter9 ([reports/iter9_strategy_selection.md](reports/iter9_strategy_selection.md)) —
відбір стратегій/ТФ/інструментів + 6 виправлених дефектів контуру.
**Поточний цикл:** [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md) (Wave 0 → paper-v0.2.0).
R1–R8: [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md). H1–H8:
[CHANGE_PLAN_HEALTH.md](CHANGE_PLAN_HEALTH.md). Архів A–E: [CHANGE_PLAN.md](CHANGE_PLAN.md).
VPS: [DEPLOY_PLAN.md](DEPLOY_PLAN.md).

### 0. Цикл R1–R8 (хвіст) + Wave 0
Деталі R: [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md). Wave 0 (TCA, quintile пар):
[IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md). R1 змінює модель філу —
не котити на живий `paper-v0.1.0` без нового тегу і скидання 8-тижневого годинника.
- [x] **R1** Fill-parity paper ↔ backtest (mid + `MAKER_FILL_SEED`, snapshot rng)
- [x] **R2** `scalper-api.service` localhost + lifespan `require_safe_api_bind`
- [x] **R3** Профілі `docs/env/vps-paper.env.example` і `research.env.example`
- [x] **R4 / W0-R4** Тест + docs `use_exit_ladders=false` (поле в Settings уже є)
- [x] **R5 / W0-R5** `scripts/sync_depth.sh` + quality-звіт
- [x] **R6 / W0-R6** I/O геть з `application/use_cases.py` + `CostModel.from_settings`
- [x] **R7 / W0-R7** Тести `trader_loop` + вузькі except на fetch позицій
- [x] **R8** Покажчики ROADMAP/TODO/DESIGN на CHANGE_PLAN_REVIEW
- [x] **W0-TCA** IS: mid ≠ fill + opportunity cost unfilled + секція в `paper-audit`
- [x] **W0-Q** Quintile/time-decay пар на −Δspread, не на `close` ноги
- [ ] **W0-OPS** Тег `paper-v0.2.0` + 8 тижнів paper LINK/BTC (див. ROADMAP Phase 6.0)
- [x] **Phase 6.1** `portfolio_var_limit` regression fix + docs sync (analysis_hft, ROADMAP)
- [x] **L0** Shadow-TCA chase у paper (`strict_both` не змінюється; `paper-audit` / `is-report`)

### 0b. Після iter9 (research backlog)
- [x] Pre-registration гіпотези `ts_momentum` 1d (`docs/reports/hypothesis_ts_momentum.md`) — iter10 2026-09-12
- [x] Розширення універсуму momentum-портфеля (27 нових перпів, H1 FAIL t_NW 1.70 — імена не в paper) — iter11
- [x] Нативна 4h-історія CORE_15; long-only PASS (Sharpe +1.01, t_NW +2.49) — iter11
- [x] **iter12 improvement loop**: two-stage vol-target overlay (FAIL — post-hoc підтверджено) +
  value-added pairs⊕ts (FAIL — pairs деградував на свіжих 3y) + per-symbol attribution — [iter12](reports/iter12_improvement_cycle.md) 2026-09-12
- [x] **iter13 weekly momentum**: ts_momentum 1w CORE_15 (H13-A FAIL — validation t_NW 1.11<2.0,
  CI містить 0; H13-B value-added 1w⊕1d PASS як construction evidence, corr +0.09, combined SR +1.51) —
  [iter13](reports/iter13_weekly_momentum.md) 2026-09-12
- [x] **iter14 CS / 12-1 / друга пара**: H14-A/B/C усі FAIL (CS val t_NW 1.46, skip-month 0.82,
  нові пари −5…−8% ret). Paper на наявному наборі — [iter14](reports/iter14_improvement_cycle.md) 2026-09-12

### 1. Paper-валідація та моніторинг (Phase 1 Gate)
- [ ] Безперервний моніторинг paper pairs (`results/paper_pairs.sqlite`) протягом **≥8 тижнів** без розходження з бектестом — **після тегу з R1+R3**.
- [x] Щотижневий звіт tracking error: CLI `paper-audit` (`validation/paper_audit.py`) — fill-rate, maxDD vs BT×1.5, MAE/MFE forensics.

### 1b. Research локально / робот на VPS
План з фазами, файлами і протоколом hotfix: [DEPLOY_PLAN.md](DEPLOY_PLAN.md).
- [x] **Фаза 0:** знімок `PaperAccount` у SQLite, `--daemon`, `control.json`, SIGTERM save.
- [x] **Фаза 1 (код):** systemd `scalper-paper-pairs.service`, `scripts/deploy_paper.sh` (не `git pull` на VPS).
- [x] **Фаза 1 (реліз):** теги `paper-v0.1.0` і `paper-v0.2.0` у git (VPS крутить тег, не `main`).
- [ ] **Фаза 2:** 8 тижнів paper на VPS на `paper-v0.2.0` (Gate; SHA тега ≠ HEAD `main`).
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
