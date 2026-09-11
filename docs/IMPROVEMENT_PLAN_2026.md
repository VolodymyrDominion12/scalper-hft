# План: Paper-Gate і MFT-hardening (2026-09-11)

> Статус: **Active — поточний цикл** | Замінює попередню версію цього файлу
> (quintile/time-decay/stress у `cmd_report` уже зроблені).
> Пов’язані: [CHANGE_PLAN_REVIEW.md](CHANGE_PLAN_REVIEW.md) (R1–R8) ·
> [DEPLOY_PLAN.md](DEPLOY_PLAN.md) · [STRATEGY_STATUS.md](STRATEGY_STATUS.md) ·
> [ROADMAP.md](ROADMAP.md)

Продукт — **MFT pairs_arb LINK/BTC 1h maker**, не субмілісекундний HFT.
Жодних нових стратегій і жодного live з реальними коштами, доки Wave 0 +
Ops (8 тижнів paper) не закриті.

```text
W0  код-блокери + чесний TCA     ← цей тиждень, 4–6 роб. днів
Ops paper-v0.2.0 на VPS          ← 8 тижнів календаря після тегу
W1  bake-off / cross-symbol      ← паралельно з paper, демон не чіпати
W2  друга пара + ERC/VaR         ← лише після Gate
W3  L2 / Nautilus / MM           ← окремий продукт, не цей цикл
```

---

## Принципи

1. Один PR = одна задача з таблиці нижче. Не змішувати TCA з systemd.
2. Кожна кодова задача: червоний тест → фікс → `uv run pytest tests/ -q` →
   `uv run ruff check --fix && uv run ruff format`.
3. `use_kalman`, `legging_mode=chase`, `USE_EXIT_LADDERS`, ERC у циклі —
   не дефолт, поки окремий bake-off / друга PASS-пара не скажуть інакше.
4. Без lookahead. Без зміни z / lookback / `regime_scale` на повній вибірці.
5. Зміна моделі філу або TCA під час живого paper — **скидання 8-тижневого
   годинника** і новий sqlite.

## Поза скоупом

- Нові 1m/5m taker-скальпи, VWAP/EMA/ORB.
- Revival `market_maker` / `ob_imbalance` у paper без `quality_ok` L2.
- Увімкнення відхилених: `mean_reversion`, `cvd_momentum`, `funding_*`,
  `basis_reversion`, `hmm_reversion`, `supertrend` як directional.
- Kalman / chase / ERC / ladders як дефолт.
- Повний rewrite `domain/` + Decimal у numpy-бектесті.
- Перейменування репо. Rust/Nautilus «для швидкості» 1h-пари.
- RL / funding-aware MM, multi-venue, `live-v*`.
- Повернення XRP/BTC у `VALIDATED_PAIRS` без нового pair-PASS + ADF у paper.

---

## Wave 0 — блокери чесного Paper-Gate

Мета: один CostModel, чесний IS, закриті R4–R7, тег `paper-v0.2.0`.
R1–R8 (крім ops-тегу) уже в коді.

### W0-R4 — Closeout exit_ladders [P2, ~0.5 д]

Код уже читає `Settings.use_exit_ladders` (дефолт false). Бракує тесту і
чесних документів.

**Файли:** `tests/test_trader_notional_cap.py` (або вузький новий тест) ·
`docs/DESIGN.md` §3.5 · `docs/TODO.md` · `docs/CHANGE_PLAN_REVIEW.md`

**Готово коли:** `trader.use_exit_ladders is False` за замовчуванням покрито
тестом; DESIGN каже «вимкнено прапорцем»; немає `getattr(..., False)` на
неіснуючому полі; чекбокс R4 у TODO закритий.

**Статус:** зроблено. `Settings.use_exit_ladders` (дефолт false); LiveTrader
читає поле напряму; DESIGN §3.5 — вимкнено прапорцем; тест у
`test_trader_notional_cap.py`.

### W0-R5 — `scripts/sync_depth.sh` [P1, ~1 д]

**Навіщо.** Depth5 пишеться на VPS; на research-машині архіву немає.
Paper-pairs не залежить; робимо паралельно.

**Файли:** `scripts/sync_depth.sh` (новий) · `docs/OB_RECORDER_RUNBOOK.md` ·
`docs/L2_DATA_PLAN.md` Фаза 2

**Поведінка:**

```text
sync_depth.sh [--dry-run] [user@host:path]
  rsync -avz --partial data/*depth5*.parquet *bookTicker*
  validate_depth / validate_bookticker → results/quality_depth.md
  exit ≠ 0 якщо quality_ok=false
```

Не комітити parquet. **Готово коли:** скрипт у репо; ранбук — одна команда;
`bash -n scripts/sync_depth.sh` проходить.

**Статус:** зроблено. `scripts/sync_depth.sh [--dry-run] [user@host:path]`;
CLI `depth-audit` → `results/quality_depth.md`; exit ≠ 0 якщо `quality_ok=false`.

### W0-R6 — Use-cases без завантаження ринку [P2, ~1 д]

**Навіщо.** `run_backtest()` качає klines/trades і будує **плоский**
`CostModel(...)`, тоді як `cmd_report` / `audit_cell` беруть
`CostModel.from_settings(settings, df=df)`. Два світи витрат.

**Файли:** `scalper_hft/application/use_cases.py` ·
`scalper_hft/cli/research_backtest.py` · `scalper_hft/research/job_handlers.py` ·
`tests/test_health_contracts.py`

**Поведінка:** `run_backtest(req, df, trades=, funding=)` не імпортує
`data.downloader` / `data.access`. Cost завжди `CostModel.from_settings`.
CLI як і раніше качає дані.

**Готово коли:** `use_cases.py` не імпортує downloader; pytest CLI/jobs зелений.

**Статус:** зроблено. `run_backtest(req, df, trades=, funding=)` + `CostModel.from_settings`;
I/O лишився в CLI/`job_handlers`.

### W0-R7 — Тести `trader_loop` [P1, ~1–2 д]

**Файли:** `tests/test_trader_loop.py` (новий) · точково `trader_loop.py` /
`sync_engine.py` / `pairs_live.py` / `pending_orders.py`

**Тести (мінімум):**

1. `pause` → немає submit.
2. DD-breaker flatten через `trader_loop` (не дублювати всю логіку, якщо вже є).
3. Мок fetch позицій кидає → не `except Exception: pass`.
4. Мережевий збій klines у paper **не** ставить KillSwitch (контракт H1).

**Готово коли:** є `tests/test_trader_loop.py`; немає голого `pass` на fetch
позицій; pytest зелений.

**Статус:** зроблено. `tests/test_trader_loop.py`; `_fill_symbols` більше не ковтає
fetch; submit/poll — мережа fail-closed, інше проброс.

### W0-TCA — Чесний Implementation Shortfall [P0, ~2 д]

**Навіщо.** `build_from_orders` ставить `mid_at_decision = fill_price = limit`
→ maker IS ≈ 0. Unfilled не мають opportunity cost. Калібровка CostModel з
такого звіту бреше.

**Файли:**

- `scalper_hft/live/is_log.py` — already has `markout_bps`; додати unfilled
- `scalper_hft/live/is_report.py` — mid з book/bar close рішення, не з limit
- `scalper_hft/live/pairs_engine.py` — логувати unfilled з mid і причиною
- `scalper_hft/validation/paper_audit.py` — секція IS + miss-cost
- `scalper_hft/cli/ops.py` → `cmd_is_report`
- `tests/test_is_report.py` (новий)

**Метрики в звіті:**

| Метрика | Формула |
|---|---|
| Fill IS (bps) | `(fill − mid_decision) / mid` зі знаком сторони |
| Next-bar markout | уже є `apply_next_bar_markout` |
| Opportunity cost | для unfilled: adverse move mid_decision → mid_cancel/timeout |
| Blended TCA | `fill_rate × IS + (1 − fill_rate) × miss_cost` |

**Не робити:** авто-запис `SLIPPAGE_BPS` у `.env` з IS. Лише друк рекомендації,
як зараз, і лише якщо `coverage_ok` (≥20 fills) **і** mid ≠ fill.

**Готово коли:** тест з limit=100, fill=100, mid=99.8 дає ненульовий buy IS;
unfilled збільшує blended TCA; `paper-audit` друкує IS; pytest зелений.

### W0-Q — Quintile / time-decay для пар на спред [P1, ~1 д]

**Навіщо.** `cmd_report` рахує `fwd = close.pct_change()`. `pairs_arb` торгує
−Δspread. Тест дивиться не ту альфу. CLI `quintile` для пар уже правильний
окремо — треба той самий forward у звіті пар.

**Файли:** `scalper_hft/cli/research_audit.py` · `scalper_hft/cli/pairs.py` ·
`scalper_hft/validation/audit_extensions.py` · `tests/test_improvements.py`

**Поведінка:** для `pairs_arb` / `cmd_pairs` з `--walkforward` або окремий
блок у pairs-звіті: `z` + `forward = −Δspread` (як `quintile_spread_study`
у `test_roadmap.py`). Time-decay — `pairs_time_decay`, не одноногий
`time_decay_test`.

**Готово коли:** тест на синтетичному монотонному спреді PASS; на
`close.pct_change` тієї ж ноги — не обов’язково PASS.

**Статус:** зроблено. `run_quintile_audit` / `run_time_decay_audit` на
`leg1`/`leg2` беруть `z` і `−Δspread`; `cmd_pairs` друкує блок; `report --leg1 --leg2`.

### W0-COST — Один CostModel на всіх шляхах [P1, частина R6]

Після R6: `cmd_pairs` теж `CostModel.from_settings(settings, df=df1)` (не
плоский конструктор). `IMPACT_K` лишається 0, доки Wave 1 не відкалібрує.

**Готово коли:** grep `CostModel(` у `cli/` і `application/` показує лише
`from_settings` або явні тестові фікстури.

**Статус:** зроблено. CLI (включно з `cmd_pairs`) і jobs — `from_settings`.

### W0-OPS — Тег і старт paper [операції, не код]

Після злиття **мінімум W0-TCA + R1+R3** (R5–R7 бажано в тому ж тегу).

1. Тег `paper-v0.2.0`. Не котити на живий `paper-v0.1.0` без скидання годинника.
2. VPS: `scripts/deploy_paper.sh paper-v0.2.0`, профіль
   `docs/env/vps-paper.env.example`, `DRY_RUN=true`, новий sqlite.
3. Конфіг: LINK/BTC 1h maker, `lookback=120`, `regime_scale=0.25`, той самий
   `MAKER_FILL_SEED`.
4. Щотижня `paper-audit` + `is-report --days 7`.
5. Клас C hotfix (z/lb/regime_scale) заборонений під час 8 тижнів.

Деталі: [DEPLOY_PLAN.md](DEPLOY_PLAN.md) фаза 2.
Чекліст: [reports/paper_v0.2.0_ops_checklist.md](reports/paper_v0.2.0_ops_checklist.md).

**Готово коли:** 8 тижнів журналу без обнулення sqlite. Це **не** дозвіл на live.

---

## Критерії виходу з Paper-Gate

```
LINK/BTC 1h maker, regime_scale=0.25, один git-тег
────────────────────────────────────────────────────
≥ 8 тижнів безперервного paper на VPS
Tracking error (paper vs BT equity) < 3% на тиждень
Fill rate post-only ≥ 70%
maxDD paper ≤ backtest maxDD × 1.5
MAE/MFE forensics: без аномалій
Rolling ADF p-value спреду < 0.05 (входи не вбиті kill >1 раз/міс)
Blended TCA maker (IS + miss) < 3 bps  ← після W0-TCA
```

Live — лише після Gate **і** явного запиту користувача. Немає `DRY_RUN=false`
у юнітах цього плану.

---

## Wave 1 — дослідження паралельно з paper

Демон на VPS не чіпати. Результати — звіти в `docs/reports/`, не зміна дефолтів.

### W1-K — Kalman vs OLS bake-off [M, 3–4 д]

```bash
uv run python -m scalper_hft.cli data-audit --days 1095
uv run python -m scalper_hft.cli pairs --strategy pairs_arb \
  --leg1 LINKUSDT --leg2 BTCUSDT --interval 1h --days 1095 --maker
uv run python -m scalper_hft.cli pairs --strategy pairs_arb \
  --leg1 LINKUSDT --leg2 BTCUSDT --interval 1h --days 1095 --maker --use-kalman
```

| Метрика | OLS (baseline) | Kalman (прийняти якщо) |
|---|---|---|
| WF avg OOS Sharpe | зафіксувати | ≥ baseline |
| WF pos_windows | ~65% | ≥ 60% |
| maxDD 3y | ~−27% з regime_scale | ≤ −30% |
| n_trades 3y | ~46 | ≥ 35 |

**Результат:** `docs/reports/kalman_ols_bakeoff_2026.md`. Дефолт `use_kalman=False`,
поки звіт не PASS.

### W1-X — Cross-symbol перевірка відхилених [S, запуск]

Закрити AAVE 15m bias. Не «шукати нову альфу».

```bash
uv run python -m scalper_hft.cli data-audit --days 1095
uv run python -m scalper_hft.cli backtest \
  --strategy cvd_momentum,ob_imbalance,hmm_reversion \
  --symbol BTCUSDT,ETHUSDT,LINKUSDT \
  --interval 1h --days 1095 --mode walkforward --enqueue
```

**Результат:** `docs/reports/cross_symbol_sweep_2026.md`. Увімкнення в paper —
лише якщо комірка проходить `cell_audit` **і** W1-VA.

### W1-VA — Value-added тест [S, 1–2 д]

**Навіщо.** Narang гл. 9: нова ідея в портфель, якщо ΔSharpe портфеля > 0,
не якщо її власний Sharpe > 0.

**Файли:** `scalper_hft/validation/benchmark.py` (або новий
`validation/value_added.py`) · CLI секція в `report` · тести

**Поведінка:** `Sharpe(LINK/BTC + candidate)` vs `Sharpe(LINK/BTC)`;
кореляція барних returns < 0.3.

**Готово коли:** є функція + тест на двох штучних рядах; CLI друкує ΔSharpe.

### W1-IMP — Калібровка `IMPACT_K` з depth5 [M, 2 д, після R5]

Не вмикати `0.1` «з стелі». Оцінити k на BTC/LINK depth5: taker walk по
5 рівнях vs mid. Для `PAIR_NOTIONAL_PCT=0.05–0.10` очікуваний impact
0.1–0.3 bps — можна лишити 0, якщо емпірика це підтвердить.

**Готово коли:** звіт з оцінкою k; `.env.example` коментар «коли вмикати».

---

## Wave 2 — портфель (після Paper-Gate)

Лише якщо Gate зелений.

1. Друга пара — свіжий pair-PASS (`pairs_gate`: WF pos ≥ 0.55, PBO < 0.5,
   n_trades ≥ 20) + rolling ADF у paper ≥ 4 тижні без щомісячного kill.
2. XRP/BTC: не в `VALIDATED_PAIRS` без цього. 3y −63% при +3.5% на 400д.
3. Тоді `PORTFOLIO_ALLOCATION_METHOD=erc` у `PairsPortfolioRunner`
   (`portfolio/sizing.py` уже є).
4. Portfolio VaR halt з `portfolio/risk_budget.py` (денний/тижневий уже
   частково в runner — перевірити, що VaR не мертвий дріт).
5. Turnover tax: ребаланс лише якщо Σ|Δw| > поріг.

Не вмикати `USE_EXIT_LADDERS` для pairs.

---

## Wave 3 — шлях HFT (не цей квартал)

Окреме рішення «робимо MM як продукт»:

1. Місяці `quality_ok` depth5 (R5 + VPS recorder уже пише).
2. Queue-position з L2, не з OHLC `event_engine`.
3. Nautilus як **незалежний** fill-engine, не обгортка `generate_signals(df)`.
4. Окремий OOS / Paper-Gate для MM. `scalper_core` Rust — лише якщо профілер
   покаже гарячий шлях; для 1h pairs не потрібно.

---

## Зведена таблиця

| ID | Завдання | Хвиля | Дні | Блокує paper-тег? | Статус |
|---|---|---|---|---|---|
| R1–R3, R8 | Fill-parity, API bind, .env профілі, документи | — | — | так (уже в коді) | ☑ |
| W0-R4 | Тест + docs `use_exit_ladders=false` | 0 | 0.5 | ні | ☑ |
| W0-R5 | `sync_depth.sh` | 0 | 1 | ні | ☑ |
| W0-R6 | I/O геть з use_cases + `from_settings` | 0 | 1 | бажано | ☑ |
| W0-R7 | Тести `trader_loop` | 0 | 1–2 | ні | ☑ |
| W0-TCA | IS mid ≠ fill + miss-cost + paper-audit | 0 | 2 | **так** | ☑ |
| W0-Q | Quintile/decay пар на −Δspread | 0 | 1 | ні | ☑ |
| W0-COST | `cmd_pairs` → `from_settings` | 0 | у R6 | бажано | ☑ |
| W0-OPS | Тег `paper-v0.2.0` + 8 тижнів | Ops | 8 тиж. | — | ☐ |
| W1-K | Kalman vs OLS | 1 | 3–4 | ні | ☐ |
| W1-X | Cross-symbol BTC/ETH/LINK | 1 | запуск | ні | ☐ |
| W1-VA | Value-added тест | 1 | 1–2 | ні | ☐ |
| W1-IMP | Калібровка IMPACT_K | 1 | 2 | ні | ☐ |
| W2 | Друга пара + ERC/VaR | 2 | після Gate | — | ☐ |
| W3 | L2 queue + Nautilus MM | 3 | пізніше | — | ☐ |

**Порядок коду Wave 0:** W0-TCA → W0-R6/COST → W0-Q → W0-R7 → W0-R5 → W0-R4 → тег.

---

## Що не відкривати, поки Wave 0 відкрита

- Sweep нових стратегій / iter8 альф.
- Paper regime-selector з iter7 (OOS вікна спалені, DSR 0.666 при n_trials=270).
- Зміна `VALIDATED_PAIRS`.
- `DRY_RUN=false`.

*Наступний перегляд: після тегу `paper-v0.2.0` або закриття всіх ☐ у Wave 0.*
