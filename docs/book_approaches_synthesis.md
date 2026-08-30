# Синтез: підходи з 5 книг → scalper-hft

Консолідований аналіз п'яти книг проти поточної кодової бази (стан: ітерація 9–10).
Мета — знайти підходи, які **ще не реалізовані**, з пріоритетами впровадження.
Детальні звіти по окремих книгах:

| Книга | Звіт |
|---|---|
| Advances in Financial ML (López de Prado) | (повний конспект у чаті; ключове — нижче) |
| Financial Signal Processing and ML (Akansu et al.) | `docs/reports/fspml_scalping_notes.md` |
| Inside the Black Box (Narang) | `docs/book_gap_analysis.md` |
| ML Algorithms with Applications in Finance (Eyal Gofer, PhD) | `docs/reports/book_online_learning_notes.md` |
| Predictive Marketing (Artun & Levin) | (повний конспект у чаті; ключове — нижче) |

## ✅ Реалізовано (Спринт 1, 2026-08-29)

| Підхід | Книга/розділ | Реалізація | Тести |
|---|---|---|---|
| Мета-лейблінг (повний) + sizing із імовірностей | AFML Ch.3.6–3.7, 10.3 | `ml/trainer.py::train_walk_forward_meta`, `ml/bet_sizing.py`, `strategies/ml_strategy.py` (meta_filter/prob_size/confidence_thr тепер працюють через proba) | `tests/test_sprint1.py` |
| Breakeven-поріг входу | Narang Ch.5 | `backtest/execution.py::apply_breakeven_gate`, прапорець `breakeven_gate` у `get_strategy`, `--breakeven-gate` у CLI backtest | `tests/test_sprint1.py::TestBreakevenGate` |
| Cohort decay | PM Ch.4, 8, 13 | `validation/cohort.py` (cohort_metrics/decay/report) + CLI `cohort` | `tests/test_sprint1.py::TestCohort` |
| Децильний lift-аналіз фіч | PM Ch.2, 9 | `validation/lift.py` (decile_lift/feature_lift_report/lift_summary) + CLI `lift` | `tests/test_sprint1.py::TestLift` |
| Hedge-блендінг (no-regret) | Gofer Ch.2 | `strategies/blend.py` (hedge_weights, HedgeBlend, hedge_blend_signals) + `ensemble mode='hedge'` | `tests/test_sprint1.py::TestHedge` |

## ✅ Реалізовано (Спринт 2, 2026-08-29)

| Підхід | Книга/розділ | Реалізація | Тести |
|---|---|---|---|
| VPIN / Kyle λ / Roll / Amihud / Corwin–Schultz / Parkinson / signed-flow AC | AFML Ch.19 | `features/microstructure.py` + `add_microstructure_features` | `tests/test_sprint2.py::TestMicrostructure` |
| HMM-режими (Gaussian HMM, Baum–Welch на numpy — hmmlearn недоступний офлайн) | FSPML Ch.4.5 | `features/hmm_regime.py` (GaussianHMM, `hmm_regime_features` → hmm_state + P(режим)) | `tests/test_sprint2.py::TestHmmRegime` |
| GARCH(1,1) σ_{t+1} + EWMA + ARCH-LM | FSPML Ch.7–8 | `features/volatility.py` (garch11_fit/vol/forecast, ewma_vol, arch_lm_test) | `tests/test_sprint2.py::TestGarch` |
| Емпіричний CostModel: vol-scaled slippage + Square-Root impact + калібровка | Narang Ch.5 | `CostModel.vol_aware_slippage/sqrt_law_impact/total_cost_per_side`, `estimate_impact_k_from_bars`, `estimate_spread_from_bookticker` | `tests/test_sprint2.py::TestCostModelSprint2` |
| ERC / risk-parity алокація + turnover tax | Narang Ch.6 | `portfolio/erc.py` (erc_weights, allocate_portfolio) + `run_pairs_portfolio(method='erc', turnover_rate=...)` + CLI `pairs-portfolio --method erc` | `tests/test_sprint2.py::TestErc` |
| MDI / MDA / SFI + PCA-Kendall перевірка | AFML Ch.8 | `ml/feature_importance.py` + CLI `featimp` | `tests/test_sprint2.py::TestFeatureImportance` |

## ✅ Реалізовано (Спринт 3, 2026-08-29)

| Підхід | Книга/розділ | Реалізація | Тести |
|---|---|---|---|
| Стрес-тест: crash / liquidity / vol_spike / funding_shock | Narang Ch.4, 10 | `validation/stress.py` (apply_stress/stress_test/stress_report) + CLI `stress` | `tests/test_sprint3.py::TestStress` |
| Портфельний risk budget: vol-targeting, VaR, розподіл ліміту збитків | Narang Ch.4, 10 | `portfolio/risk_budget.py` (vol_target_scale/portfolio_var/loss_budget_split) | `tests/test_sprint3.py::TestStress` |
| Динамічний розмір + лімітна ціна (сигмоїда) | AFML Ch.10.6 | `ml/bet_sizing.py::sigmoid_size/calibrate_omega/target_size/limit_price` | `tests/test_sprint3.py::TestSigmoidSizing` |
| Capacity-тест (share of wallet) | PM Ch.4 | `validation/capacity.py` (capacity_curve/saturation_scale/report) + CLI `capacity` | `tests/test_sprint3.py::TestCapacity` |
| Survival / Kaplan–Meier (динамічний holding) | PM Ch.7, 12, 13 | `validation/survival.py` (trade_durations/kaplan_meier/median_survival_time/survival_by_feature) + CLI `survival` | `tests/test_sprint3.py::TestSurvival` |
| Інтеграція micro/HMM/GARCH у ML-фічі | AFML Ch.19, FSPML Ch.4.5/7 | `ml/features.py::build_labeled_dataset(add_micro/add_hmm/add_garch)`; HMM — каузальна версія (`filtered_proba`, без lookahead) | `tests/test_sprint3.py::TestMlFeatureIntegration` |

## ✅ Реалізовано (Спринт 4, 2026-08-29)

| Підхід | Реалізація | Тести |
|---|---|---|
| MCP-сервер для трейдінгу | `scalper_hft/mcp_trading.py` (stdio, JSON-RPC, чистий stdlib): strategy_list / settings_summary / market_status / run_backtest / run_cohort / run_stress / run_capacity / paper_step; CLI `mcp`; конфіг `.mcp/mcp-config.md` (§4) | `tests/test_sprint4.py::TestMcpTrading` |
| Alpha-гіпотеза: HMM-гейтований mean reversion | `strategies/hmm_reversion.py` (каузальний HMM-гейт «спокійного» стану, без lookahead) + повний цикл валідації (walk-forward + DSR + sensitivity) | `tests/test_sprint4.py::TestHmmReversion` |
| Інтеграція фіч у live-шар | `live/trader.py`: `vol_scaled_size` (інверсне vol-масштабування розміру, кліп [0.25, 3]) + `hmm_blocked` (блок нових входів у «неспокійному» HMM-режимі; close ніколи не блокується) | `tests/test_sprint4.py::TestLiveSprint4` |
| Дашборд: cohort/stress/capacity | `dashboard.py` §5: cohort decay, stress_report, capacity-крива (plotly) | py_compile |

## ✅ Реалізовано (Спринт 5, 2026-08-30)

| Підхід | Книга/розділ | Реалізація | Тести |
|---|---|---|---|
| Micro-price розрахунок & fair maker котирування | Narang Ch.7, 15 | `scalper_hft/backtest/micro_price.py`, інтеграція в `backtest/execution.py::CostModel` | `tests/test_sprint5.py::test_micro_price` |
| Price ladder exits (драбина рівнів виходу) | Gofer Ch.7 | `scalper_hft/live/exit_ladders.py` (PriceLadderExit, каскадні рівні $2^i K$, partial fills) | `tests/test_sprint5.py::test_exit_ladders` |
| Clustered Feature Importance (CFI) | AFML Ch.8 | `scalper_hft/ml/clustered_importance.py` (кластеризація фіч + OOS scoring) + CLI `cfi` | `tests/test_sprint5.py::test_cfi` |
| Sparse Basket Arbitrage (мульти-активний кошик) | Narang Ch.3, FSPML Ch.2 | `scalper_hft/strategies/sparse_basket.py` (Lasso sparse weights + PCA-проекція) | `tests/test_sprint5.py::test_sparse_basket` |
| Multi-asset Bar Sampling (tick/volume/dollar/imbalance) | AFML Ch.2 | `scalper_hft/data/bars.py` (dollar_bars, imbalance_bars, tick_bars) | `tests/test_sprint5.py::test_bars` |
| Exp3 Multi-Armed Bandit для вибору активів/стратегій | Gofer Ch.2, 4 | `scalper_hft/strategies/bandit.py` (Exp3Bandit, exp3_select_signals) | `tests/test_sprint5.py::test_bandit` |

## Що вже є в проєкті (не дублювати)

- **AFML**: triple-barrier labeling (side-aware), sample weights (uniqueness + time-decay + sequential bootstrap), FFD + find_min_d, purged K-fold + embargo (label-aware), CPCV/PBO, Deflated Sharpe + estimate_n_trials, walk-forward, sensitivity, Optuna.
- **FSPML**: волатильність/тренд/сесія/funding regime-фільтри, CVD, OB imbalance (top + depth-weighted), spread, коінтеграція у pairs arb.
- **Narang**: повна архітектура Alpha → Risk → T-Cost → Portfolio → Execution; CostModel (fee + slippage + impact); подієвий maker-рушій з adverse selection; risk-check у live-трейдері.
- **Gofer**: періодичний walk-forward ретрейн (але **не** онлайн-алгоритми).
- **PM**: жодного перенесеного методу (книга ще не використовувалась).

---

## 🥇 Високий пріоритет (найбільший ефект на edge/ризик, S–M складність)

### 1. Повний мета-лейблінг + sizing із імовірностей — AFML Ch.3.6–3.7, 10.3
**Стан:** `MLStrategy._apply_meta_filter` — наближений контур, фактично не працює
(мета-таргет {0,1} vs прогноз {-1,+1} → фільтр ніколи не спрацьовує; немає асиметричних
бар'єрів, sizing, F1-скорингу).
**План:**
1. `get_events(..., side=Series)` → асиметричні `ptSl=[pt, sl]` відносно сторони; мета-мітка {0,1} = «перший дотик бар'єра дав прибуток у бік ставки».
2. Друга модель (LightGBM) → `P(meta=1)` з sample weights + purged CV, скоринг **F1** (не accuracy).
3. Розмір позиції `m = 2Φ(z) − 1`, `z = ln(p̂/(1−p̂))`; сторона — від primary; застосувати в `MLStrategy` та `live/trader.py`.
**Файли:** `ml/labeling.py`, `ml/trainer.py`, `strategies/ml_strategy.py`, `live/trader.py`.

### 2. Bet sizing + динамічна лімітна ціна (сигмоїда) — AFML Ch.10.3, 10.6
**Стан:** сигнал ±1 фіксованого розміру (`position_pct`).
**План:** `m[ω,x] = 2/(1+e^{−ωx}) − 1`, де `x = f_i − p_t` (прогноз vs ринок);
цільова позиція `q* = Int[Q·m]`; **лімітна ціна** `L = f_i − (1/ω)·ln((1+m)/(1−m))` —
готовий post-only рівень для maker-рушія. Коли `p_t → f_i` — позиція закривається сама.
**Файли:** `backtest/execution.py`, `backtest/event_engine.py`, `live/fills.py`.

### 3. MDI / MDA / SFI feature importance + PCA-перевірка — AFML Ch.8
**Стан:** лише сирий gain-importance у `MlResult.feature_importance`.
**План:** новий `ml/feature_importance.py`: MDI (IS, impurity, `max_features≈1`), MDA
(OOS, permutation, purged+embargoed CV, neg-log-loss/F1), SFI (пофічево), PCA +
weighted Kendall τ (MDI vs inverse-PCA-rank) > 0.8 як доказ, що патерн не оверфіт.
Інтегрувати у CLI-звіт перед висновком про edge («Backtesting is not a research tool»).
**Файли:** `ml/feature_importance.py` (новий), `cli.py`, `validation/cv.py`.

### 4. Мікроструктурні фічі: VPIN, Kyle λ, Roll, Amihud, Corwin–Schultz — AFML Ch.19
**Стан:** є CVD, OB imbalance, relative spread.
**План (розширити `features/indicators.py`):**
- `vpin(trades, V, n=50)` — flow toxicity, прямо калібрує adverse-selection для maker-рушія;
- `kyle_lambda` — OLS `Δp = λ·(b·V) + ε`; фіча — **t-value** λ;
- `roll_spread = 2√(−cov(Δp_t, Δp_{t−1}))` — ефективний спред з aggTrades;
- `amihud = |r| / dollar-vol`; `corwin_schultz(high, low)` — спред/vol з OHLC;
- `signed_flow_autocorr` — персистентність потоку (splitting/herding).
**Файли:** `features/indicators.py`, `ml/features.py`, `backtest/event_engine.py`.

### 5. Емпірична модель транзакційних витрат — Narang Ch.5
**Стан:** `CostModel` плоский: impact = 0.0, slippage = константа 2 bps.
**План:**
1. `slippage = f(σ_t, spread, τ)` — зростає з волатильністю;
2. impact за Square-Root Law `k_imp·σ_t·√(Q/ADV)` або Almgren–Chriss, калібровка `k_imp`
   на наявних L2/bookTicker даних;
3. `breakeven_move_pct` з property → **поріг входу** в генерації сигналів (не торгувати, якщо
   очікуваний рух < round-trip витрат);
4. piecewise-linear форма витрат за розміром ордера.
**Файли:** `backtest/execution.py`, `strategies/`, `live/trader.py`.

### 6. Portfolio Construction: ERC/risk-parity + turnover tax + кореляції — Narang Ch.6
**Стан:** рівні ваги у `pairs_portfolio.py`; `ensemble.py` — наївний mean/vote.
**План:**
1. Кореляційна матриця прибутковостей стратегій → **ERC** (equal risk contribution) або risk-parity;
2. **Turnover tax** в алокації (штраф за зміну ваг — для 1h pairs істотно);
3. Regime-gated ансамбль (ваги залежать від режиму `features/regimes.py`);
4. Grinold–Kahn факторний підхід: 12 стратегій = 12 факторів → лінійний оптимізатор з обмеженнями.
**Файли:** `backtest/pairs_portfolio.py`, `strategies/ensemble.py`, новий `portfolio/`.

### 7. Stress-testing + портфельний risk budget — Narang Ch.4, Ch.10
**Стан:** hard limits на рівні стратегії (`trader.risk_check`): позиція, денний збиток, серія.
**План:**
1. Модуль стрес-сценаріїв: краш −30%, ліквідність-криза (спред ×10, глибина ÷10),
   фандінг-шок, contagion пар — переграти історію зі зсунутими витратами;
2. Портфельний risk budget: денний ліміт збитків на **портфель**, не на ногу
   (вже заявлено у ROADMAP Phase 2.5 — зробити);
3. Vol-targeting / penalty-функції замість жорстких обмежень.
**Файли:** `live/trader.py`, `validation/stress.py` (новий), `config.py`.

### 8. HMM-режими (hmmlearn) → MS-TCM — FSPML Ch.4.5
**Стан:** rule-based regime-фільтри (vol/trend/session); hmmlearn лише у планах.
**План:**
1. `GaussianHMM(K=2..4)` на `[ret, |ret|, CVD, spread]` → Viterbi + posteriors
   → `P(regime)` як ML-фіча та gate стратегій (книга: Rand 0.94–0.97, прогноз 0.5% vs 26%);
2. rolling вікна 750 барів / крок 50; обережно з label switching;
3. пізніше — повний MS-TCM (switching VAR + group lasso, EM) як L-задача.
**Файли:** `features/regimes.py`, `ml/features.py`.

### 9. GARCH/EGARCH σ_{t+1} — FSPML Ch.7–8
**План:** `arch.arch_model(ret, vol='EGARCH')` → прогноз волатильності як фіча +
інверсне sizing (менша позиція у високій волі); ARCH-LM на залишках як діагностика.
**Складність:** S–M. **Файли:** `features/indicators.py`, `ml/features.py`.

### 10. Онлайн-блендінг Hedge (no-regret) — Gofer Ch.2 §2.1
**Стан:** `ensemble.py` — статичний mean/vote; концепт-дрейф не відстежується.
**План:** `w_{t+1} = w_t·e^{−η·l_t}`, ваги `p_t = w/Σw`; loss = `−log(1+ret_i)` після комісій;
адаптивне `η=√((8/q′)ln N)`. ~40 рядків numpy. Regret `O(√(T·ln N))` — гарантія
worst-case. Апгрейд `ensemble.py` і, потенційно, алокації між парами.
**Складність:** S. **Файли:** `strategies/ensemble.py`, новий `strategies/blend.py`.

### 11. Exp3 bandit — вибір символу/моделі — Gofer Ch.2, Ch.4 §4.7
**План:** bandit-вибір «який символ/модель/поріг зараз» з exploration–exploitation;
reward = прибуток за епізод після комісій. Онлайн-доповнення до Optuna для
мультисимвольного портфеля (у вас pairs-портфель).
**Складність:** S. **Файли:** новий `strategies/bandit.py`, `live/pairs_runner.py`.

### 12. Cohort analysis + моніторинг decay edge — PM Ch.4, 8, 13
**Стан:** моніторингу деградації edge у часі немає (є лише періодичний аудит).
**План:**
1. `trades.groupby(entry_month) → [sharpe, pnl_per_trade, win_rate]` — cohort-криві;
2. Тест тренду (Mann–Kendall або регресія cohort-Sharpe на час) → алерт «edge помер»;
3. **Silent attrition**: кількість угод та сама, а PnL/угода падає — EWMA-сигнал;
4. Блок у `dashboard.py` + розділ у `docs/STRATEGY_STATUS.md`.
**Складність:** S–M. **Файли:** `dashboard.py`, новий `validation/cohort.py`.

### 13. LTV-аналіз стратегії: capacity + value migration — PM Ch.4, 8
**План:**
1. Крива «LTV стратегії»: `cum_pnl` + `rolling_sharpe(EWMA)` — нахил = швидкість деградації;
2. **Capacity-тест**: бектест з масштабом позицій ×2, ×5, ×10 → точка насичення Sharpe(capital);
3. Матриця переходів стратегія×режим «працює/не працює» (аналог transition matrix).
**Складність:** M. **Файли:** `validation/capacity.py` (новий), `cli.py`.

### 14. Децильний lift-аналіз фіч — PM Ch.2, 9 (uplift-концепт)
**План:** для кожної фічі `Δ(x) = E[PnL | вхід, X∈bin] − E[PnL | вхід]` на OOS-трейдах;
`Δ≈0` → видаляти фічу; `Δ<0` → умовний фільтр входу. Контролювати confounding (режим)!
**Складність:** M. **Файли:** `ml/features.py`, новий `validation/lift.py`.

### 15. Калібрування ймовірностей → розмір позиції — PM Ch.2, 9 + AFML Ch.10.3
**План:** `CalibratedClassifierCV` (isotonic/Platt) на OOS-прогнозах; reliability diagram;
`size = base·(2·p_cal − 1)` з Kelly-капом (`f* = (b·p − q)/b` — у вас уже є `risk_of_ruin`);
децильний PnL-звіт як перевірка монотонності.
**Складність:** M. **Файли:** `ml/trainer.py`, `live/trader.py`.

### 16. Комбінування сигналів через constrained lasso — FSPML Ch.2.6
**План:** `LassoCV(positive=True)` forward-returns ~ затримані сигнали → sparse ваги;
benchmark 1/K; як альтернатива Hedge-блендінгу (п. 10) для стаціонарних періодів.
**Складність:** S. **Файли:** `strategies/ensemble.py`.

### 17. Rank-кореляції замість Pearson — FSPML Ch.8.2
**План:** Kendall τ / Spearman у фічах і валідації (книга: rank-міри робастні до важких
хвостів — Pearson не відхиляє H0 у 13–53% випадків, rank-міри — 0%).
**Складність:** S. **Файли:** `ml/features.py`, `validation/`.

### 18. Kalman fair price — FSPML (поза книгою, стандартний DSP)
**План:** `x̂⁺ = x̂ + K(y − x̂)`, `K = P/(P+R)` → fair price; фічі `x̂ − y`, `P`;
для maker-рушія — оцінка «справжньої» ціни між спредом. ~30 рядків numpy/filterpy.
**Складність:** S–M. **Файли:** `features/indicators.py`, `backtest/event_engine.py`.

---

## 🥈 Середній пріоритет

| Підхід | Книга/розділ | Що дає | Складність |
|---|---|---|---|
| CUSUM event-семплінг (`get_t_events`) | AFML Ch.2.5.2 | ML вчиться лише на інформативних зсувах, не на кожному барі | S |
| Dollar/imbalance-бари | AFML Ch.2.3 | IID-ближчі семпли з aggTrades; менше барів на день | M |
| Bagging + OOB + seq-bootstrap (K=3–7 LGBM) | AFML Ch.6 | Зменшення variance (у фінансах краще за boosting); std прогнозів = фіча впевненості | M |
| OTR: синтетичний бектест на O-U процесі | AFML Ch.13 | Пре-калібровка TP/SL без підгонки до одного шляху; heat-map 20×20 | M |
| HHI / DD / Time-under-Water / PSR / IS-метрики | AFML Ch.14 | Концентрація прибутків, 95-й перцентиль просідань, «return on execution costs» | S–M |
| Імовірність провалу стратегії (цільова p) | AFML Ch.15 | Орієнтир точності: для SR=2, n=5000 достатньо p≈0.514 | S |
| Sizing за конкуренцією бети | AFML Ch.10.2 | Обмеження одночасних лонгів/шортів у портфелі | M |
| Survival analysis → динамічний holding period | PM Ch.7, 12, 13 | Kaplan–Meier → Cox: час утримання за фічами замість фіксованого `holding_bars` | L |
| Кластеризація трейдів/днів (KMeans/GMM/HDBSCAN) | PM Ch.2, 6, 12 | Типи трейдів → умовні правила ввімкнення; кластер як ML-фіча | M |
| A/B + sequential testing (alpha-spending) | PM Ch.9, 13, 18 | Дисципліна live-експериментів: O'Brien–Fleming пороги | M |
| Attribution PnL за факторами | PM Ch.5 | Last-touch → Shapley лише при високих кореляціях | M |
| One-way trading: price-ladder exit | Gofer Ch.7 | Каскад часткових TP на рівнях `2^i·K` замість фіксованого TP | S |
| OGD/RFTL онлайн-baseline | Gofer Ch.2 | Дешевий онлайн-базлайн + детектор дрейфу (SGDClassifier partial_fit) | S |
| Variation-based η + anytime-regret моніторинг | Gofer Ch.2–3, 8 | Адаптивність + детектор аномалій (флеш-краш = перевищення regret-межі) | S–M |
| Quadratic variation як vol-фіча | Gofer Ch.5–6 | Заміна/доповнення vol-фільтрів | S |
| POET-коваріація / precision (TIGER) | FSPML Ch.6 | Стійка коваріація для sizing | M |
| Копули / tail dependence (λ_U, λ_L) | FSPML Ch.8.4 | Екстремальна залежність пар (поза Pearson) | M |
| CVaR-sizing / quantile-лейбли | FSPML Ch.10–11 | LightGBM `objective='quantile'` (0.05/0.95) → VaR-смуга; розмір з CVaR | S–M |
| Poisson/CSM order flow | FSPML Ch.9 | Overdispersion, спільні шоки BTC/ETH, backward-симуляція стресів | M |
| Execution: micro-price + IS-вимірювання | Narang Ch.7 | Краща ціна maker; фактичний slippage vs модель | S–M |
| Quintile/monotonicity + time-decay тест | Narang Ch.9 | Перевірка «сигнал → прибуток» по квінтилях; лаг входу (критично для 1m) | S |
| TWAP/VWAP/POV слайсинг | Narang Ch.7 | Лише для пар/дельта-нейтральних великих ноціоналів | M |
| Звіт про кількість trials у `cmd_report` | AFML Ch.11 | Дисципліна DSR: скільки всього спроб було | S |

---

## 🥉 Низький пріоритет / пізніше

- HRP-алокація (AFML Ch.16) — робастний замінник Markowitz для пар-портфеля, L.
- SADF/QADF/CADF/SMT структурні розриви (AFML Ch.17) — pump/dump-детектор для
  maker-рушія; **почати з дешевого CSW CUSUM (b₀.₀₅=4.6)**, SADF — O(T²), векторизувати, L.
- Ентропійні фічі Shannon/LZ (AFML Ch.18) — експериментальні, S–M.
- dropLabels (AFML Ch.3.9), class weights (Ch.4.8) — дрібниці, S.
- mpPandasObj розпаралелювання (AFML Ch.20) — практика для get_events/SADF, M.
- MS-TCM повний (FSPML Ch.4.5) — L, після hmmlearn.
- KLT/DCT деноізинг + декорреляція фіч (FSPML Ch.5) — умовно, S.
- Вейвлет-деноізинг (pywt, поза книгою) — лаг для 1s, низький.
- Періодограма/Welch (поза книгою) — діагностика, низький.
- HedC partial restarts / clustered experts / branching-cloning (Gofer Ch.4) — M, P3.
- Time-of-day сигнал, cross-sectional momentum, Open Interest/ліквідації (Narang Ch.3) — S–M.
- Negative churn / pool cycle рамка (PM Ch.13) — описова метрика, S.
- Granger-тести (FSPML Ch.4) — діагностика фіч, S.

---

## ⛔ Не впроваджувати (чесно)

- **Collaborative filtering для відбору активів** (PM Ch.2, 10) — кореляція ≠ edge;
  коінтеграція у pairs суворіша.
- **Повноцінний uplift-ML без рандомізації** (PM) — обсерваційні дані; лише децильний lift.
- **Survival-моделі як «істина» про вихід** (PM) — цензурування залежить від власних
  рішень; використовувати як описову аналітику.
- **Kelly прямо з книги** — AFML Ch.10 замінює його мапуванням імовірностей у розмір ставки.
- **«Реактивувати стратегію в 10× дешевше»** (PM Ch.4/13) — економіка каналів не переноситься.
- **Wavelets/Kalman як «з книги»** — у FSPML їх немає (лише вказівники); це стандартний DSP.

---

## Рекомендований порядок (Спринти 1–5 — ✅ виконано)

1. ~~**Спринт 1 (S):** мета-лейблінг + sizing, breakeven-поріг, cohort decay,
   децильний lift, Hedge-блендінг~~ → зроблено 2026-08-29.
2. ~~**Спринт 2 (M):** VPIN/Kyle λ/Roll фічі, HMM-режими, GARCH σ_{t+1}, емпіричний
   CostModel (Square-Root impact), ERC-алокація, MDI/MDA/SFI~~ → зроблено 2026-08-29.
3. ~~**Спринт 3:** stress-test + risk budget, сигмоїдний sizing/лімітна ціна,
   capacity-тест, survival/Kaplan–Meier, інтеграція micro/HMM/GARCH у ML-фічі~~ →
   зроблено 2026-08-29.
4. ~~**Спринт 4:** MCP-сервер для трейдінгу, alpha-гіпотеза hmm_reversion через
   повний цикл, live-інтеграція (vol-sizing + HMM-блок), дашборд cohort/stress/
   capacity~~ → зроблено 2026-08-29.
5. ~~**Спринт 5:** micro-price розрахунок, exit ladders (драбини виходу), CFI (кластеризована важливість фіч),
   sparse basket арбітраж, мульти-активні бари (dollar/tick/imbalance), Exp3 онлайн-бандит~~ →
   зроблено 2026-08-30.

Подальші кроки (за потребою): тривалий paper-прогін (≥8 тижнів) для валідації виконання без розходжень,
накопичення L2-даних глибини стакана та моніторинг фандінг-режимів.

Кожен підхід — через повний цикл: реалізація → тест → walk-forward + Deflated Sharpe →
sensitivity → paper (AGENTS.md, `skills/overfitting-audit.md`).
