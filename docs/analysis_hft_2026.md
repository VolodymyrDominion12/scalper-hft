# Аналіз scalper-hft vs кращі практики HFT/MFT 2026

> Дата аналізу: 11 вересня 2026 | Актуалізація: **12 вересня 2026**
>
> Виконуваний план: [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md) ·
> Роадмап Phase 6: [ROADMAP.md](ROADMAP.md#phase-6--closure--mft-maturity-2026-09--2026-11)

---

## 🏆 Загальна оцінка: **7.5/10 — Зріла MFT квант-платформа**

Проєкт — між **research-інструментом** і **production MFT-системою** (хвилини–години).
Архітектура правильна (Narang, AFML, López de Prado). Це **не** субмілісекундний HFT:
Python/ccxt, бари 1h, sync loop — прийнятно для `pairs_arb`, недостатньо для tick-MM.

| Категорія | Оцінка |
|---|---|
| Квант-дослідження / анти-перенавчання | **9/10** |
| MFT (retail, 1h pairs) | **7.5/10** |
| Справжній HFT (субмс–мс) | **2/10** |
| Production readiness | **6/10** (Paper Gate не закритий) |

---

## 1. СИЛЬНІ СТОРОНИ

### ✅ Методологія дослідження (9/10)
- Walk-forward + purge/embargo (AFML Ch.7/11)
- Deflated Sharpe на OOS; CSCV PBO = 0.000 для `pairs_arb LINK/BTC`
- Triple-barrier + sample weights; OOS burn registry (`oos_usage.md`)
- Capability contract (`MissingDataError`) — fail-fast
- **Quintile / time-decay / stress** у `audit_extensions.py` → `cmd_report` / `cell_audit` (Wave 0-Q) ✅

### ✅ Модель витрат (7.5/10)
- Maker 0.02% / Taker 0.05%; `CostModel.from_settings` — єдиний шлях (W0-COST) ✅
- Vol-aware slippage + Square-Root impact (`impact_k`, дефолт 0 до калібровки)
- `QueuePositionModel` / micro-price для MM-бектесту
- `breakeven_gate` — property; не вбудований у всі стратегії (залишковий gap)

### ✅ Live-шар безпеки (8/10)
- `DRY_RUN=true` за замовчуванням; DrawdownBreaker; cooldown; macro calendar
- Reconciliation + KillSwitch; Paper Gate ≥8 тижнів
- **IS benchmark** (W0-TCA): `is_log.py`, `is_report.py`, секція в `paper-audit` ✅
- Portfolio hist-VaR halt (`PORTFOLIO_VAR_LIMIT`) у `PairsPortfolioRunner` ✅

### ✅ Мікроструктура та режими (8/10)
- VPIN, Kyle λ, Roll, Amihud, Corwin–Schultz, OBI
- HMM + rule-based режими; `RegimeSupervisor`; `preferred_regimes_from_matrix`

### ✅ Anti-overfitting культура (9/10)
- Заборона testnet-даних для research; `AuditMode` exploratory/final
- `haircut_roster`; реєстр пасток у AGENTS.md

---

## 2. GAPS vs практики HFT/MFT 2026 (актуальний стан)

### 🟡 GAP-1: Cost Model — частково емпіричний

**Закрито (Wave 0):** vol-aware slippage, `from_settings`, IS-звіт з paper fills.

**Відкрито:**
- `IMPACT_K=0` за замовчуванням — sqrt-law не калібрований на власних fills
- Авто-запис `SLIPPAGE_BPS` з IS у `.env` **свідомо не робиться** (лише рекомендація)
- Per-symbol calibration (BTC vs ALT) — немає

**Пріоритет:** Wave 1 після ≥20 fills у paper (критерій `coverage_ok` у W0-TCA).

---

### ✅ GAP-2: Quintile / Monotonicity — ЗАКРИТО (2026-09-11)

`validation/audit_extensions.py` → `run_quintile_audit`; для пар — `−Δspread`
(не `close.pct_change`). У `cmd_report`, `cell_audit`, CLI `report --leg1 --leg2`.

---

### ✅ GAP-3: Time-Decay — ЗАКРИТО (2026-09-11)

`pairs_time_decay` для пар; `run_time_decay_audit` у стандартному аудиті.

---

### 🟡 GAP-4: Portfolio-Level Risk

**Є:** `ENABLE_VOL_TARGET`, `portfolio_var_limit` (hist-VaR 95%), CLI `--method erc`.

**Немає в live-циклі:** `portfolio/risk_budget.py`; ERC не дефолт у `PairsPortfolioRunner`.

**Пріоритет:** Phase 6.3 / Wave 2 (після Paper Gate).

---

### ✅ GAP-5: Implementation Shortfall — ЗАКРИТО (W0-TCA)

`is_report.py`: fill IS, opportunity cost unfilled, blended TCA; CLI `is-report`.

---

### ✅ GAP-6: Stress Testing у пайплайні — ЗАКРИТО (Phase 5.3 M4)

`stress_report` у `cmd_report` / `audit_extensions`; `stress_pass` у `cell_audit`.

**Відкрито:** stress не в критеріях Paper Gate (рекомендація: maxDD crash ≤ 2× BT).

---

### 🔴 GAP-7: Paper Gate не закритий (критичний для production)

Wave 0 код ✅; **W0-OPS** ⏳ — тег `paper-v0.2.0`, 8 тижнів VPS, щотижневий `paper-audit`.

---

### 🟡 GAP-8: HFT-інфраструктура (окремий track)

L2 queue у live, async OMS, colocation — Phase 6.5 / Phase 4. Не блокує pairs.

---

## 3. ПОРІВНЯННЯ З ПРАКТИКАМИ 2026

### 3.1 Тестування стратегій

| Компонент | Стан | Gap |
|---|---|---|
| Walk-forward + purge/embargo | ✅ | — |
| Deflated Sharpe (OOS) | ✅ | — |
| CSCV/PBO | ✅ | — |
| Quintile test | ✅ в `cell_audit` | — |
| Time-decay | ✅ в `cell_audit` | — |
| Stress scenarios | ✅ в `cmd_report` | 🟡 не в Paper Gate |
| Sensitivity | ✅ | — |
| IS benchmark | ✅ W0-TCA | 🟡 feedback у CostModel вручну |
| OOS burn registry | ✅ | — |
| Value-added test | 📋 W1-VA | 🟡 |

### 3.2 Cost Model

| Компонент | Стан |
|---|---|
| Fees | ✅ |
| Vol-aware slippage | ✅ |
| Square-root impact | ⚠️ код є, `IMPACT_K=0` |
| IS → CostModel auto | ❌ (навмисно) |
| Per-symbol calibration | ❌ |

### 3.3 Live-шар

| Компонент | Стан |
|---|---|
| Circuit breaker / reconciliation | ✅ |
| Vol-target sizing | ✅ |
| Portfolio VaR halt | ✅ (новий) |
| IS metrics / `is-report` | ✅ |
| WebSocket keepalive | ✅ Phase 1.5 |
| Async OMS | ❌ (не потрібно для 1h MFT) |
| Paper Gate 8 тижнів | ⏳ |

---

## 4. РЕКОМЕНДАЦІЇ ДЛЯ ДОСЛІДЖЕНЬ

### P0 — зараз (без нових альф)
1. **W0-OPS:** `paper-v0.2.0` + 8 тижнів paper ([IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md))
2. Щотижня: `paper-audit` + `is-report --days 7`

### P1 — паралельно з paper (Wave 1)
3. **Kalman vs OLS** bake-off (W1-K)
4. **Cross-symbol** BTC/ETH/LINK для відхилених (W1-X) — закрити AAVE-bias
5. **Value-added test** перед другою парою (W1-VA)

### P2 — після Paper Gate
6. **Regime selector** (iter7: +2.64 OOS, PBO=0.004 — exploratory, DSR portfolio 0.666)
7. **ERC + risk_budget** у live portfolio runner
8. **Друга пара** лише з новим pair-PASS + ADF у paper

### P3 — HFT track (окремий продукт)
9. L2 `quality_ok` → `ob_imbalance` / `market_maker` OOS
10. Nautilus / Tardis fill parity

### Протокол нової стратегії (2026)

```
Крок 1: Spec YAML                           ✅
Крок 2: IS backtest                         ✅
Крок 3: Quintile (монотонність)             ✅ audit_extensions
Крок 4: Time-decay (лаг 0→3)                  ✅ audit_extensions
Крок 5: Walk-forward 3y                     ✅
Крок 6: DSR OOS                             ✅
Крок 7: Stress                              ✅ cmd_report
Крок 8: Sensitivity                         ✅
Крок 9: CSCV/PBO                            ✅
Крок 10: Paper Gate 8 тижнів                ⏳ W0-OPS
```

---

## 5. ПЛАН ПОКРАЩЕНЬ (пріоритизований, 2026-09-12)

### 🔴 КРИТИЧНО

| # | Завдання | Статус |
|---|---|---|
| 1 | Paper Gate W0-OPS (`paper-v0.2.0`, 8 тижнів) | ⏳ ops |
| 2 | Kalman vs OLS bake-off (W1-K) | 📋 |
| 3 | Cross-symbol sweep BTC/ETH/LINK (W1-X) | 📋 |
| 4 | Rolling ADF kill у paper | ✅ Phase 1.5 B2 |

### 🟡 ВАЖЛИВО (після Gate)

| # | Завдання | Статус |
|---|---|---|
| 5 | ERC у `PairsPortfolioRunner` | 📋 Wave 2 |
| 6 | `risk_budget` у live-циклі | 📋 |
| 7 | Value-added test (W1-VA) | 📋 |
| 8 | Stress у критеріях Paper Gate | 📋 |
| 9 | `IMPACT_K` калібровка з IS (≥20 fills) | 📋 |
| 10 | Regime selector final audit | 📋 Phase 6.3 |

### 🟢 ДОВГОСТРОКОВІ

| # | Завдання |
|---|---|
| 11 | L2 Tardis / Nautilus (Phase 6.5) |
| 12 | `sparse_basket` OOS (5+ монет) |
| 13 | Kelly/half-Kelly sizing |
| 14 | Domain layer / Decimal у live paths |

---

## 6. ВИСНОВКИ

### Що реально добре
**`pairs_arb LINK/BTC 1h maker`** — єдиний validated кандидат (PBO=0, `regime_scale=0.25`).
Правильна тактика: **Paper Gate першим**, нові альфи — Wave 1 паралельно, без зміни дефолтів.

### Що турбує
1. Paper Gate не закритий — найбільший gap до production.
2. Meta-стратегії (`ensemble`, `regime_supervisor`) слабкі OOS; iter7 selector — promising, не validated.
3. AAVEUSDT sweep bias — перевірити відхилені на BTC/ETH/LINK перед остаточним вердиктом.

### Головна рекомендація

> **80% дисципліна (paper + TCA + audit), 20% нові ідеї.**
> Не перейменовувати проєкт у «HFT» — це чесна **MFT pairs-arb платформа**.

---

## 7. ТЕХНІЧНІ БОРГИ (актуальний)

| Проблема | Місце | Пріоритет | Статус |
|---|---|---|---|
| Paper Gate не закритий | ops / VPS | 🔴 | ⏳ |
| `portfolio_var_limit` у частковому runner | `pairs_runner.py` | 🔴 | ✅ 2026-09-12 |
| Worker code version | `research/code_version.py` | 🟡 | ✅ hash у worker |
| `IMPACT_K=0` | `CostModel` | 🟡 | відкрито (`from_settings` = 0.0; bare `CostModel()` = 0.1) |
| Live chase vs paper `strict_both` | `pairs_runner.PairsLiveRunner` | 🔴 до live | відкрито (Phase 6.6 L0) |
| `risk_budget` не в live | `portfolio/` | 🟡 | відкрито |
| AAVEUSDT sweep bias | research | 🟡 | W1-X |
| `mypy strict` частково | `pyproject.toml` | 🟢 | |
| Job succeeded при degenerate-клітинках | `research/job_worker.py` | 🟡 | Phase 6.6 JOB-1 |

---

## 8. Аудит 2026-09-12 (код vs практики)

Повторна перевірка live/risk/docs. Вердикт **не змінився**: зріла **MFT**-платформа, не HFT.
Нове, чого не було в §2:

1. **Paper vs live execution gap.** `PairsEngine` дефолт `strict_both`; `PairsLiveAdapter` /
   `PairsLiveRunner` — `legging_mode="chase"` (taker IOC другої ноги). Це правильний захист
   від одноногої книги на біржі, але 8-тижневий paper **не** накопичує цей taker-cost.
   Не вмикати live, поки chase не в shadow-TCA або не вирівняний з paper.
2. **Тег ≠ Gate.** `paper-v0.2.0` є в git; `main` на ~7 комітів попереду — так і має бути.
   Відкритий пункт — безперервний прогін на VPS, не створення тега.
3. **Narang pipeline частково бібліотека:** ERC / `risk_budget.py` не в live-циклі (одна пара —
   ок). Isolated/cross, mark-price funding, MMR-ліквідація — не first-class.
4. **Назва.** `AGENTS.md` / `DESIGN.md` / `pyproject` вирівняні на MFT (2026-09-12), щоб агенти
   не пропонували 1m taker-скальп «бо репо називається HFT».

Деталі в роадмапі: [ROADMAP.md](ROADMAP.md#66--honesty-live-parity-hygiene-аудит-2026-09-12-p0p1).
