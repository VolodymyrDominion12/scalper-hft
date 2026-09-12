# Роадмап scalper-hft

Стан на 2026-09-12 (після Wave 0, Phase 5 та аудиту HFT/MFT). Цей документ — план розвитку після
аудиту коду, стратегій і live-шару. Детальний статус стратегій:
[STRATEGY_STATUS.md](STRATEGY_STATUS.md), пари — [pairs_audit.md](pairs_audit.md), синтез підходів — [book_approaches_synthesis.md](book_approaches_synthesis.md).
Порівняння з практиками HFT/MFT: [analysis_hft_2026.md](analysis_hft_2026.md).

> **Поточний цикл (2026-09-12):** [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md) —
> Wave 0 (код) ✅ → **W0-OPS** (тег `paper-v0.2.0` + 8 тижнів paper) ⏳. Не нові альфи.
> Код-цикл R1–R8: [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md). Phase 6 — closure нижче.

## Чесний вердикт

Проєкт — зріла **квант-платформа** (Alpha → Risk → T-Cost → Portfolio → Execution, бектест, walk-forward, DSR, CSCV, ML meta-labeling, стрес-тести, MCP) з валідованим кандидатом: **pairs_arb LINK/BTC 1h maker** (`VALIDATED_PAIRS`).
Це **не** субмілісекундний тіковий HFT-скальпер. Taker-скальпінг на 1m відхилено через fee-drag;
funding/basis сплять у низькому режимі 2025–26. Phase 0 закрито повністю.
Поточний цикл коду: [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md) (R1–R8, fill-parity / API bind / профілі .env).
H1–H8 закриті: [CHANGE_PLAN_HEALTH.md](CHANGE_PLAN_HEALTH.md).
Юніт-тести: `uv run pytest tests/ -q`.

**Головне правило розгортання:** жоден live з реальними коштами, доки paper pairs не пройде **≥8 тижнів безперервного моніторингу** без розходження з бектестом.

Розділення research (локально) і робота (VPS, git-тег): [DEPLOY_PLAN.md](DEPLOY_PLAN.md).
Доповнення 2026-09-12: [Phase 6.6](#66--honesty-live-parity-hygiene-аудит-2026-09-12-p0p1)
(chase-cost vs paper, job integrity, quarantine REGISTRY).

---

## Phase 0 — P0 баги (закрито 2026-08-28) ✅

`pairs_arb` у REGISTRY; ф'ючерсний облік `PaperAccount` (cash + UPNL, без
подвоєння PnL); сигнал/філл з закритого бару; close-before-flip + limit/post-only;
paper-replay реверсує і рахує daily-loss на mark-to-market. Тести: `tests/test_live.py`, `tests/test_p0.py`.

| ID | Проблема | Статус |
|---|---|---|
| P0-1 | `pairs_arb` немає в REGISTRY | ✅ |
| P0-2 | `PaperAccount.equity` подвоює PnL | ✅ |
| P0-3 | Сигнал з формуючого бару | ✅ |
| P0-4 | Market-ордери + flip без close | ✅ |
| P0-5 | Replay не реверсує; risk на зламаному equity | ✅ |

---

## Phase 1 — Paper pairs (код закрито; триває 8-тижневий моніторинг) ⏳

Зроблено в репозиторії:
1. ✅ `PairsEngine` / `PairsPaperRunner` / `PairsPortfolioRunner` — дві ноги, z-score на закритому барі, maker all-or-none, рівні ноціонали.
2. ✅ Модель філла: post-only на OHLC; unfilled + wait_bars; відсутність одноногих позицій.
3. ✅ CLI: `paper-run-pairs`, `paper-replay-pairs`, `pairs-portfolio` з підтримкою `--method erc` (Equal Risk Contribution); `paper-audit` — tracking error vs бектест + MAE/MFE forensics.
4. ✅ Ризик: `PAIR_NOTIONAL_PCT` / `PORTFOLIO_NOTIONAL_PCT`, стоп після `MAX_LOSING_MONTHS`; cooldown size×0.5 перед halt; кап корельованого ноціоналу (`CORR_NOTIONAL_CAP`).
5. ✅ SQLite `results/paper_pairs.sqlite` (equity, orders, trades, months).
6. ✅ Дашборд: Streamlit секція pairs + SQLite аналітика.
7. ✅ Weekly-audit: валідовані пари + портфель; Telegram сповіщення.
8. ✅ Linger: активний `loginctl enable-linger $USER` для depth-рекордера.

**Критерій виходу (Paper Gate):** ≥8 тижнів paper-прогону, CLI `paper-audit` (tracking error vs бектест, fill-rate, maxDD ≤ бектест × 1.5, MAE/MFE forensics).

Стартовий портфель для Gate: **лише LINK/BTC** (`lookback=120`, `regime_scale=0.25`)
у `VALIDATED_PAIRS`. XRP/ETH-рукави — поза книгою, доки не буде нового pair-PASS + ADF у paper.

---

## Phase 2 — Production hardening & ризик-інфраструктура ✅

1. ✅ **Reconciliation**: `reconcile_exchange_state` у `run_trader_once` та `PairsPortfolioRunner.step` (`fetch_positions` + kill-switch; paper — no-op).
2. ✅ **Execution правила**: Reduce-only на закритті; GTX/post-only на вході; ніколи market на pairs.
3. ✅ **Telegram інтеграція**: fill, reject, daily PnL, risk-block, kill-switch (`scalper_hft/live/telegram.py`).
4. ✅ **Risk budget** (модуль, не в live-циклі): денний/тижневий ліміт збитків, vol-targeting (`scalper_hft/portfolio/risk_budget.py`).
5. ✅ **Hedge-ratio**: rolling OLS та Johansen вектори коінтеграції (`scalper_hft/validation/hedge_ratio.py`).
6. ✅ **Емпіричний CostModel & пропущені філи**: vol-aware slippage, Square-Root impact, micro-price та ймовірність виконання (`scalper_hft/backtest/execution.py`, `micro_price.py`).
7. ✅ **Exit ladders** (модуль; `USE_EXIT_LADDERS=false` за замовчуванням, не pairs-gate): драбини рівнів виходу (`scalper_hft/live/exit_ladders.py`).

---

## Phase 3 — Дослідження альф та ML-стек ✅

1. ✅ **Коінтеграційний сканер**: систематичний скан пар активів за Johansen/ADF (`validation/coint_scan.py`, CLI `coint-scan`).
2. ✅ **ML-пайплайн**: triple-barrier labeling, sample weights (uniqueness + time decay), meta-labeling з F1/log-loss оптимізацією, bet sizing сигмоїдою (`ml/labeling.py`, `ml/trainer.py`, `ml/bet_sizing.py`).
3. ✅ **Feature Importance**: MDI, MDA, SFI та Clustered Feature Importance (CFI) з PCA-перевіркою (`ml/feature_importance.py`, `ml/clustered_importance.py`).
4. ✅ **Мікроструктурні та режимні фічі**: VPIN, Kyle λ, Roll, Amihud, Corwin–Schultz, каузальний HMM-режим, GARCH(1,1) та фракційне диференціювання FFD (`features/`).
5. ✅ **Мульти-активні структури**: `SparseBasketArb` (Lasso/PCA кошики), `Exp3Bandit` (онлайн-вибір інструментів), `HedgeBlend` (no-regret блендінг).
6. ✅ **Розширені валідатори**: `stress.py`, `cohort.py`, `lift.py`, `capacity.py`, `survival.py`, `time_decay.py`, `quintile.py`.

---

## Phase 1.5 — Чесне виконання і live-інфра (план 2026-09-03)

Деталі: [CHANGE_PLAN.md](CHANGE_PLAN.md). Не нові альфи; дроти того, що вже є.

| ID | Що | Статус |
|---|---|---|
| A | USDT-M WS `/public` `/private` + PUT listenKey keepalive 30 хв | код у цьому циклі |
| B | Chase = taker; unwind = flatten; дефолт `strict_both` | код у цьому циклі |
| B2 | Rolling ADF / half-life kill у paper (блок входів, не flatten) | код у цьому циклі |
| C | Kalman vs OLS OOS bake-off; CLI `--use-kalman`; дефолт off | код у цьому циклі |
| D | IS → CostModel; daily/weekly halt у portfolio runner | код у цьому циклі |
| E | Session UTC у paper-audit; markout наступний бар; warn 1m/5m | код у цьому циклі |

**Головне правило розгортання** не змінюється: жоден live з реальними коштами,
поки paper pairs не пройде ≥8 тижнів. Після фази B — перезапуск paper, бо
комісії chase змінять tracking error.

## Phase 1.6 — Research локально / paper на VPS (план 2026-09-03)

Деталі: [DEPLOY_PLAN.md](DEPLOY_PLAN.md). Не нові альфи; демон, persist стану,
реліз тегом.

| ID | Що | Статус |
|---|---|---|
| 0 | Persist `PaperAccount` + `--daemon` + `control.json` + SIGTERM | код |
| 1 | systemd `scalper-paper-pairs` + теги `paper-v0.1.0` / `paper-v0.2.0` | теги в git; 8 тижнів на v0.2.0 — ops |
| 2 | 8 тижнів paper на VPS (операції; Paper-Gate) | план |
| 3 | Live-адаптер ніг pairs (код, `DRY_RUN=true` за замовчуванням) | план |
| 4 | Тег `live-v*` + ключі IP-whitelist | лише після Gate і явного запиту |

## Phase 2.5 — Execution Intelligence & Empirical Cost Modeling (MFT Standard)

1. **Implementation Shortfall (IS) Feedback Loop**: збір метрик з `live/fills.py` для оновлення `CostModel`.
2. **Dynamic Market Impact Model**: Заміна константного slippage на функцію Square-Root Law (Almgren-Chriss).

## Phase 3.5 — Structural Risk & Cointegration Defense

1. **Cointegration Break Detector**: Інтеграція динамічного (rolling) ADF/Johansen тесту в live-рушій `pairs_arb` для паузи при втраті коінтеграції.
2. **Portfolio-Level Risk & ERC**: Portfolio-level VaR та динамічний Vol-Targeting (Equal Risk Contribution) безпосередньо в циклі `PortfolioRunner`.

## Phase 4 — HFT та L2 Order Book інфраструктура (наступний етап)

1. **Глибина стакана**: тривале накопичення depth5 снапшотів через активний рекордер.
2. **L2 Backtesting**: інтеграція Tardis.dev / L2 історичних даних для точного моделювання черги заявок у маркет-мейкінгу (`market_maker`).
3. **Nautilus Trader інтеграція**: підключення як зовнішнього бенчмарк-рушія для L2 філів.
4. **Live з реальними коштами**: підключення перевірених ключів Binance лише після виконання критерію Paper Gate (Phase 1).

---

## Phase 5 — Аудит 2026-09-09: «dead wiring» і чесна валідація

Зовнішній аудит показав: платформа зріла, але багато правильних механізмів
**існують, проте вимкнені за замовчуванням або не підключені** до
рекомендаційного контуру. План усунення (пріоритет зверху вниз).

### 5.1 — Чесна валідація (код цього циклу) ✅

| ID | Що | Статус |
|---|---|---|
| V1 | `audit_cell`: purge/embargo за замовчуванням `max(1, 1% test-вікна)` (AFML Ch.7/11); явний 0 = відтворення старих прогонів; прапорці `--purge-bars/--embargo-bars` у CLI `overfit`/`cscv`; purge/embargo проброшено у `pbo_cscv` | ✅ код |
| V2 | `cmd_report`: DSR на конкатенованих OOS-дохідностях WF (як `audit_cell`), а не на IS-забрудненій full-sample equity | ✅ код |
| V3 | `sweep_winners_haircut()` + секція у `save_sweep_report`: переможець per interval з Bailey–LdP selection haircut (`survives=False` → випадковий максимум) | ✅ код |
| V4 | `pairs_arb` v1.3: дефолт = валідована конфігурація iter6b (`regime_scale=True`, `factor=0.25`; CSCV PBO=0.000); spec YAML синхронізовано; engine-тести ізольовані явним `regime_scale=False` | ✅ код |

### 5.2 — Живий ризик-шар (код цього циклу) ✅

| ID | Що | Статус |
|---|---|---|
| R1 | `ENABLE_VOL_TARGET` → `PairsEngine` (масштаб ноціоналу `clip(target/realized σ спреду, 0, 1)`). Прапорець читається в `PairsPaperRunner` / `PairsPortfolioRunner` / `PairsLiveRunner`. ERC (`erc_vol_target_sizes`) лишається модулем: у `VALIDATED_PAIRS` одна пара | ✅ код |
| R2 | `max_leverage` cap + опційний vol-target у `run_backtest`; `run_strategy_backtest` підставляє `MAX_LEVERAGE` / `VOL_TARGET_ANN` з settings (CLI/jobs/WF/sweep) | ✅ код |
| R3 | Partial-fill політика cancel/wait (`PARTIAL_FILL_POLICY`); IntentStore ключ містить бар/ts | ✅ код |
| R4 | Capability contract: `needs_trades`/`needs_funding` + `requires = {basket, l2, multi_symbol}` → `MissingDataError`. `sparse_basket` без кошика, `cross_momentum`/`pairs_arb` на 1 символі, векторний `market_maker` без L2 — fail-fast. Обхід: `strict_data=False` | ✅ код |

### 5.3 — Мультифакторна режимна система ✅

| ID | Що | Статус |
|---|---|---|
| M1 | `preferred_regimes` з OOS `RegimePerfMatrix` (`preferred_regimes_from_matrix` / `apply_oos_preferred_regimes`). Клас лишає порожній frozenset як «невалідовано», доки немає матриці; `RegimeSupervisor(perf_matrix_path=…)` накладає теги на інстанси | ✅ код |
| M2 | `haircut_roster` + `RegimeSupervisor.from_haircut_roster` — 3–5 стратегій зі `survives_haircut`, не всі 17 і не відхилений дефолт MR/ST/HMM | ✅ код |
| M3 | Meta-OOF: expanding TS-split + t1-purge (каузальний AFML purge; повний K-fold тренував би на майбутньому IS). Seeded `seq_bootstrap`; `cpcv_validate_returns`; Optuna CLI завжди лишає сліпий holdout (20%, якщо `HOLDOUT_PCT=0`) і пише `holdout_sharpe` | ✅ код |
| M4 | `cmd_report` завжди має стрес: fees×2 + slippage×2 і вилучення топ-5 угод (`cost_concentration_stress`) | ✅ код |

### 5.4 — Дані під HFT

| ID | Що | Статус |
|---|---|---|
| D1 | `validate_trades` / `validate_funding` / `validate_depth` / `validate_bookticker` у `data/validate.py`. Fail-closed запис кешу (`save_trades`/`save_funding`). `load_research_data` пише `quality_*` звіти | ✅ код |
| D2 | `calibrate_queue_from_depth` + `estimate_spread_from_depth` — емпіричний спред/L1 qty для `QueuePositionModel`. **MM/OBI не відроджено**: потрібне `quality_ok` покриття depth5 (VPS/Tardis) і окремий OOS-аудит | ✅ калібрування; ⏳ історія L2 / revival |

### 5.5 — Live gate (без змін)

Paper Gate ≥8 тижнів на конфігурації LINK/BTC 1h maker + regime_scale(0.25)
+ vol-target sizing (`ENABLE_VOL_TARGET`). Live — тільки після Gate і явного запиту.

### 5.6 — Health plan: integrity, perf, architecture, ops ✅

| ID | Що | Статус |
|---|---|---|
| H-RI | `AuditMode` (`exploratory`/`final`) у `audit_cell`; final вимагає holdout+OOS burn+CSCV; `EXPLORATORY_PASS` не проходить live-гейт | ✅ код |
| H-RI2 | `parameter_sensitivity` на research-slice; `ensure_trades_coverage` fail-fast для `needs_trades` | ✅ код |
| H-PF | Job CPU budget → `resolve_sweep_workers`; preload cache symbol×TF у sweep ProcessPool | ✅ код |
| H-ARCH | `scalper_hft/application/use_cases.py`; CLI `overfit` через `RunCellAudit` | ✅ код |
| H-OPS | `live/metrics.py` (step_latency, ws_reconnects, recorder_queue_depth, JSON export) | ✅ код |

---

## Phase 6 — Closure & MFT maturity (2026-09 → 2026-11)

Після Wave 0 (код) пріоритет — **закрити Paper Gate**, не розширювати альфи.
Деталі критеріїв: [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md) §«Критерії виходу».
Чекліст ops: [reports/paper_v0.2.0_ops_checklist.md](reports/paper_v0.2.0_ops_checklist.md).

### 6.0 — Paper Gate closure [P0, ops]

| ID | Що | Статус |
|---|---|---|
| PG-1 | Тег `paper-v0.2.0` на VPS (`scripts/deploy_paper.sh`, новий sqlite) | ⏳ |
| PG-2 | Конфіг: LINK/BTC 1h maker, `lookback=120`, `regime_scale=0.25`, той самий `MAKER_FILL_SEED` | ⏳ |
| PG-3 | ≥8 тижнів без скидання sqlite / зміни z/lb/regime_scale | ⏳ |
| PG-4 | Щотижня: `paper-audit` + `is-report --days 7`; blended TCA < 3 bps | ⏳ |

**Готово коли:** 8 тижнів журналу на одному тегу. Це **не** дозвіл на live.

### 6.1 — Code hygiene [P0, код]

| ID | Що | Статус |
|---|---|---|
| H1 | `portfolio_var_limit`: `getattr` у `_portfolio_var_ok` (частковий runner у тестах) | ✅ |
| H2 | Актуалізація [analysis_hft_2026.md](analysis_hft_2026.md) (закриті GAP-2/3/5/6) | ✅ |
| H3 | Оновлення дати та Phase 6 у цьому ROADMAP | ✅ |

### 6.2 — Wave 1 research (паралельно з paper, не чіпати демон) [P1]

| ID | Що | Статус |
|---|---|---|
| W1-K | Kalman vs OLS bake-off → `docs/reports/kalman_ols_bakeoff_2026.md` | 📋 |
| W1-X | Cross-symbol sweep BTC/ETH/LINK для відхилених стратегій | 📋 |
| W1-VA | Value-added test (ΔSharpe портфеля vs LINK/BTC core) | 📋 |
| W9-1 | `ts_momentum` 1d: pre-registration + незалежне вікно (розширення універсуму до 30–50 перпів) — [iter9](reports/iter9_strategy_selection.md) | 📋 |
| W9-2 | Vol-targeting overlay для momentum-портфеля: pre-registered A/B (σ-вікно 20–90, t_NW 2.05–2.54 — пост-хок) | 📋 |
| W9-3 | Нативна 4h-історія (не лише 1d) для аудиту 4h-комірок поза 3-річним 1m-кешем | 📋 |

Дефолти (`use_kalman`, chase, ERC) **не** змінювати без PASS звіту.

### 6.3 — Regime portfolio (після Paper Gate) [P2]

| ID | Що | Статус |
|---|---|---|
| RS-1 | `RegimeStrategyMap` з OOS iter7 (selector +2.64, PBO=0.004 — exploratory) | 📋 |
| RS-2 | `funding_carry` у пулі дітей; taxonomy priors з емпіричної матриці | 📋 |
| RS-3 | `cell_audit --audit-mode final` на selector перед paper | 📋 |

### 6.4 — Live execution (лише після Gate + явний запит) [P2]

| ID | Що | Статус |
|---|---|---|
| L1 | `PairsLiveRunner` soak test (мін. ноціонал, reconcile, KillSwitch) | 📋 |
| L2 | Fill quality dashboard (IS bps, fill-rate, drift) | 📋 |
| L3 | Тег `live-v*`, IP-whitelist ключів | 📋 |

### 6.5 — HFT track (окремий продукт, не блокує pairs) [P3]

| ID | Що | Статус |
|---|---|---|
| HFT-1 | L2 archive `quality_ok` → revival `ob_imbalance` / `market_maker` OOS | ⏳ |
| HFT-2 | Nautilus benchmark vs `event_engine` fill parity | 📋 |
| HFT-3 | Tardis.dev історія для queue model (Phase 4) | 📋 |

### 6.6 — Honesty, live-parity, hygiene (аудит 2026-09-12) [P0–P1]

Свіжий код-аудит (live/risk/docs vs HFT/MFT практики). Не нові альфи.
Тег `paper-v0.2.0` **існує** в git; `main` попереду тега (research ок).
Gate = 8 тижнів **на VPS**, не факт створення тега.

| ID | Що | Пріоритет | Статус |
|---|---|---|---|
| DOC-1 | Одна книга: `VALIDATED_PAIRS` = LINK/BTC; прибрати 4-парний «стартовий портфель» | P0 | ✅ цей коміт (Phase 1) |
| DOC-2 | TODO / DESIGN / AGENTS / `pyproject` / CLI: продукт = MFT pairs, не HFT-скальп | P0 | ✅ 2026-09-12 |
| L0 | Live `PairsLiveRunner` hardcode `legging_mode="chase"` vs paper `strict_both`. Chase — свідомий захист від одноногої позиції, але **Paper Gate не міряє taker-вартість другої ноги**. Перед live-soak: або shadow-TCA chase у paper, або live теж `strict_both` до окремого PASS | P0 (до L1) | ✅ 2026-09-12 `chase_shadow` (paper лишається strict_both) |
| JOB-1 | `job succeeded` ≠ усі клітинки здорові: fail/warn при високій частці `error`/`degenerate` | P1 | 📋 |
| APP-1 | `run_cell_audit` ще качає trades/funding (R6 закрив лише `run_backtest`) | P1 | 📋 |
| REG-1 | 17 імен у `REGISTRY`; default sweep без відхилених (`--include-rejected`) | P1 | 📋 |
| HFT-0 | `nautilus_adapter` — stub (`NotImplementedError`); не продавати як інтегрований рушій | P1 | 📋 |
| MFT-1 | Перп-облік: isolated/cross, mark vs last, funding з mark, buffer до ліквідації — не first-class у live | P2 (після Gate) | 📋 |
| MFT-2 | `IMPACT_K=0`; `CostModel()` dataclass default 0.1 ≠ settings 0.0. Лише `from_settings` | P2 | 📋 |
| STRESS-G | Crash maxDD ≤ 2× BT додати в критерії Paper Gate | P2 | 📋 |

---

## Що категорично не робити

- Не вмикати live на `mean_reversion` / `cvd_momentum` / `funding_carry` / `basis_reversion` / 1m pairs — усі відхилені.
- Не оптимізувати z/lookback на повній вибірці без OOS/DSR контролю.
- Не збільшувати частоту pairs до 1m «щоб більше угод» — 1m знищується fee-drag.
- Не ставити taker на **вхід** pairs (лише post-only). Chase другої ноги — свідомий live-режим, не дефолт paper (Phase 6.6 L0).
- Не запускати реальний капітал до завершення 8-тижневого paper-трейдингу.

---

## Метрики прогресу

| Фаза | Головна метрика | Статус |
|---|---|---|
| 0 | Усі P0 закриті тестами; `cli pairs` і weekly-audit зелені | ✅ Закрито |
| 1 | Paper vs backtest tracking error; % unfilled post-only; 8 тижнів без розриву | ⏳ Після тегу з R1+R3, не на старій моделі філу |
| 1.5 | WS `/public`/`private` + keepalive; chase=taker; ADF-kill; Kalman OOS; IS→cost + halt; session/markout | код у циклі 2026-09-03; paper-gate ще відкритий |
| 1.6 | Persist+daemon; VPS paper з git-тегу; live-адаптер ніг лише після Gate | 📋 [DEPLOY_PLAN.md](DEPLOY_PLAN.md) |
| R | Fill-parity, API bind, .env-профілі, sync depth, тонкий application | 📋 [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md) |
| 2 | Hardening: reconcile в циклі; kill-switch; OLS/Johansen. Ladders/risk budget — модулі | ✅ Код + тести |
| 3 | ML meta-labeling, CFI, micro-price, sparse basket, stress/cohort валідація | ✅ Реалізовано в коді |
| 4 | L2 Tardis дані, черга лімітних ордерів, live під реальний капітал | 🔜 Наступний етап |
| 5.1 | AFML purge/embargo дефолти; DSR на OOS у report; haircut у sweep; pairs_arb v1.3 дефолти | ✅ Код цього циклу |
| 5.2 | vol-target у paper; max_leverage у рушії; partial-fill; capability basket/l2/multi_symbol | ✅ Код цього циклу |
| 5.3 | preferred_regimes з OOS-матриці; supervisor на haircut-рострі; ML PurgedKFold/CPCV; стрес у report | ✅ Код цього циклу |
| 5.4 | валідація trades/funding/depth; queue calibration з depth5 (MM/OBI ще ні) | ✅ Код цього циклу (revival ⏳) |
| 6.0 | Paper Gate closure (`paper-v0.2.0`, 8 тижнів VPS) | ⏳ Ops |
| 6.1 | Code hygiene (portfolio VaR regression, docs sync) | ✅ 2026-09-12 |
| 6.2 | Wave 1 research (Kalman, cross-symbol, value-added) | 📋 Паралельно з paper |
| 6.3 | Regime portfolio (selector OOS iter7) | 📋 Після Gate |
| 6.4 | Live execution (лише після Gate + явний запит) | 📋 |
| 6.5 | HFT track (L2/Tardis/Nautilus — окремий продукт) | 🔜 |
| 6.6 | Doc honesty; chase-cost vs paper; job integrity; registry quarantine | L0 ✅; решта 📋 |
