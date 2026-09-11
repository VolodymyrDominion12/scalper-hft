# Аналіз scalper-hft vs кращі практики HFT/MFT 2026

> Дата аналізу: 11 вересня 2026 | Модель: Claude Sonnet 4.6 (Thinking)
>
> **Актуалізація того ж дня:** quintile/time-decay/stress уже в `cmd_report`.
> Виконуваний план: [IMPROVEMENT_PLAN_2026.md](IMPROVEMENT_PLAN_2026.md).
> Частина GAP-2/3 нижче — застаріла як «відсутні в пайплайні».

---

## 🏆 Загальна оцінка: **7.2/10 — Зріла квант-платформа**

Проект знаходиться між **research-інструментом** і **production-системою**.
Архітектура правильна, академічна база глибока (Narang, AFML, López de Prado),
але реалізація кожного шару — «перша версія», характерна для проектів 2022–2024.
У 2026 стандарти виросли.

---

## 1. СИЛЬНІ СТОРОНИ — що зроблено правильно

### ✅ Методологія дослідження (9/10)
- **Walk-forward + purge/embargo** (AFML Ch.7/11) — правильно реалізовано
- **Deflated Sharpe Ratio** — на OOS (а не IS) — прогрес після аудиту
- **CSCV/PBO** — `PBO = 0.000` для `pairs_arb LINK/BTC` — найкращий результат
- **Triple-barrier labeling + sample weights** — відповідає стандарту López de Prado
- **OOS реєстр** (`oos_usage.md`) — є; ключова практика 2026 (burning data transparency)
- **Capability contract** (`MissingDataError`) — fail-fast замість тихої деградації ✅

### ✅ Модель витрат (7/10)
- Maker 0.02% / Taker 0.05% + slippage 2 bps обов'язково — правильно
- `breakeven_gate` є — але лише property, не вхід у генерацію сигналів (Gap!)
- `QueuePositionModel` (micro-price) — реалізований

### ✅ Live-шар безпеки (8/10)
- `DRY_RUN=true` за замовчуванням ✅
- `DrawdownBreaker` (peak-to-trough circuit breaker) ✅
- Cooldown + consecutive losses halt ✅
- Macro calendar gate (новинний фільтр) ✅
- Reconciliation (`fetch_positions` + kill-switch) ✅
- Paper Gate 8 тижнів перед live ✅ — індустріальний стандарт

### ✅ Мікроструктурні фічі (8/10)
- VPIN, Kyle λ (з t-value), Roll spread, Amihud, Corwin-Schultz, Parkinson vol ✅
- `signed_flow_autocorr` (splitting/herding) ✅
- `kyle_lambda_series` — векторизована через `sliding_window_view` (10–50x швидше) ✅
- L2 OBI (order book imbalance) ✅

### ✅ Режимний детектор (8/10)
- HMM + rule-based (EMA/vol) — каузальний, без lookahead
- `RegimeSupervisor` з `contextual_hedge` (Hedge no-regret, Exp3Bandit) ✅
- `preferred_regimes_from_matrix` — OOS-валідовані режими ✅
- `min_dwell_bars` — гістерезис (зменшує churn ваг) ✅

### ✅ Anti-overfitting культура (9/10)
- Заборона testnet даних (фальшивий edge!) — hard-coded guard ✅
- Реєстр "пасток" в AGENTS.md — задокументовані помилки ✅
- `haircut_roster` — лише стратегії що пройшли DSR ✅
- `AuditMode` (exploratory/final) ✅

---

## 2. КРИТИЧНІ GAPS vs практики HFT/MFT 2026

### 🔴 GAP-1: Cost Model — flat, не емпіричний (Критичний)

**Стандарт 2026:** Market impact моделюється square-root law або Almgren-Chriss,
калібрується на власних fills. Slippage — функція vol×spread, не константа.

**Поточний стан:** `CostModel` — flat 2 bps слипейдж, `impact_frac = 0.0` зашито.
`book_gap_analysis.md` вже ідентифікував це. Проблема: **fills.py** пише
логи, але вони **не** зворотньо зв'язані з `CostModel`.

**Наслідок:** Пари з 30–60% ноціоналу мають ненульовий impact. Tracking error
між paper і backtest частково звідси.

```python
# Мінімальна зміна: vol-aware slippage
slippage_bps = base_slippage * (1 + k_vol * realized_vol / baseline_vol)
# k_vol ≈ 0.5–1.5 (калібрується на fills)
```

---

### 🔴 GAP-2: Quintile/Monotonicity Study (Критичний для нових альф)

**Стандарт 2026:** Обов'язковий тест перед ухваленням будь-якої стратегії.
Сигнал → 5 квінтилів → mid forward-returns мають бути монотонні (Spearman ρ > 0.8).

**Поточний стан:** `quintile.py` існує, але **не вбудований у `cmd_report`**.

**Наслідок:** Немає автоматичної перевірки "чи не випадковий" edge нових стратегій.

---

### 🔴 GAP-3: Time-Decay Test (Критичний для 1h+ стратегій)

**Стандарт 2026:** Вхід з лагом 0,1,2,3 бари. Якщо Sharpe падає за 1 бар →
стратегія залежить від виконання (векторизований backtest переоцінює edge).

**Поточний стан:** `time_decay.py` існує але **не в стандартному пайплайні звіту**.

**Особливо важливо для `pairs_arb`:** 1h-пари — чи виживає edge при затримці 1 бар?

---

### 🟡 GAP-4: Portfolio-Level Risk (Середній)

**Стандарт 2026:** ERC між парами + агрегований VaR + daily vol-targeting
на рівні портфеля, не тільки per-pair.

**Поточний стан:** `ENABLE_VOL_TARGET` є для pairs, `portfolio/risk_budget.py` — модуль,
але **не в live-циклі**. `PairsPortfolioRunner` — тільки equal-weight.

**Рекомендація:** Замінити equal-weight на online ERC (`w_i ∝ 1/realized_vol_7d`).

---

### 🟡 GAP-5: Implementation Shortfall Benchmark (Середній)

**Стандарт 2026:** Кожен fill = `(price - mid_at_decision) / mid_at_decision`.
Щоденний звіт: realized slippage maker vs taker vs `CostModel`. Зворотній зв'язок у модель.

**Поточний стан:** `fills.py` пише логи, але немає автоматичного агрегування IS-метрики.

---

### 🟡 GAP-6: Stress Testing — немає у стандартному пайплайні (Середній)

**Стандарт 2026:** Кожен кандидат проходить: crash (-30% за 2 дні),
liquidity crisis (spread ×10), funding shock, contagion.

**Поточний стан:** `validation/stress.py` реалізований але **не включений у `cmd_report`
і не в критеріях Paper Gate**.

---

## 3. ПОРІВНЯННЯ З ПРАКТИКАМИ 2026

### 3.1 Тестування стратегій

| Компонент | Ваш проект | Стандарт 2026 | Gap |
|---|---|---|---|
| Walk-forward | ✅ purge/embargo | ✅ AFML Ch.7/11 | — |
| Deflated Sharpe | ✅ на OOS | ✅ на OOS | — |
| CSCV/PBO | ✅ PBO=0.000 | ✅ PBO < 0.5 | — |
| Quintile test | ⚠️ є, не в пайплайні | ✅ обов'язковий | 🔴 |
| Time-decay | ⚠️ є, не автоматичний | ✅ обов'язковий | 🔴 |
| Stress scenarios | ⚠️ є, не у пайплайні | ✅ crash/liquidity/contagion | 🟡 |
| Sensitivity | ✅ sensitivity.py | ✅ плато, не пік | — |
| IS benchmark | ❌ відсутній | ✅ fill vs mid | 🟡 |
| OOS burn registry | ✅ oos_usage.md | ✅ | — |
| Value-added test | ❌ відсутній | ✅ ΔSharpe vs portfolio | 🟡 |

### 3.2 Cost Model

| Компонент | Ваш проект | Стандарт 2026 |
|---|---|---|
| Fees | ✅ 0.02%/0.05% | ✅ |
| Slippage | ⚠️ 2 bps константа | Vol-aware: f(σ, spread) |
| Market Impact | ❌ impact=0.0 | √-law / Almgren-Chriss |
| Timing cost | ❌ немає | σ×√T_hold |
| Per-symbol calibration | ❌ однакові для всіх | Окремо BTC/ALT |
| IS feedback loop | ❌ fills.py → /dev/null | fills → CostModel |

### 3.3 Live-шар

| Компонент | Ваш проект | Стандарт 2026 |
|---|---|---|
| Circuit breaker | ✅ DrawdownBreaker | ✅ |
| News filter | ✅ economic_calendar | ✅ |
| Vol-target sizing | ✅ ENABLE_VOL_TARGET | ✅ |
| Reconciliation | ✅ | ✅ |
| WebSocket keepalive | ⏳ Phase 1.5 | ✅ обов'язковий |
| Fill quality monitoring | ❌ немає IS metrics | ✅ real-time |
| Portfolio-level halt | ⚠️ per-pair, не портфель | ✅ portfolio VaR |
| Async order management | ⚠️ sync engine | Async event-driven |

---

## 4. РЕКОМЕНДАЦІЇ ДЛЯ ДОСЛІДЖЕННЯ СТРАТЕГІЙ

### 4.1 Нові перспективні напрями (2026)

#### 🌟 Kalman Filter Hedge Ratio (Висока пріоритетність)
Ваша `CHANGE_PLAN.md` планує це (Phase 1.5 C). Kalman динамічно
адаптує hedge ratio — особливо важливо для BTC-пар де кореляція змінюється з режимом.

```
Дослідження: OOS bake-off OLS vs Kalman на LINK/BTC 1h (3y)
Метрика: WF avg OOS Sharpe, n_trades, maxDD
Гіпотеза: Kalman знизить |entry signal noise| на 10-20%
```

#### 🌟 Cointegration Decay Detection (Висока пріоритетність)
Пара XRP/BTC: **−63%** на 3y але **+3.49%** на 400д — класичний симптом
структурного зламу коінтеграції.

```
Rolling Engle-Granger або Johansen (window=90d) з alert при
P-value > 0.05 → автоматичний блок входів (вже в Phase 1.5 B2 — пріоритизувати!).
```

#### 🌟 Funding-Aware Entry Timing (Середня пріоритетність)
Фандінг структурно низький (2025-26), але є window-ефект: за 1 годину до розрахунку
фандінгу відбувається rush позиціонування — мікро-альфа для pairs арбітражу.

```
Дослідження: mean entry return for [0, 30, 60, 90, 120, 180] minutes
before funding settlement vs average
```

#### 🌟 Sparse Basket Optimization (Середня пріоритетність)
`sparse_basket.py` реалізований але **не пройшов OOS аудит**.

```
Plan: sweep на 10 символів × 3 TF (1h/4h/1d)
Перевірити: чи знижує Lasso-кошик correlation з pairs_arb LINK/BTC
```

### 4.2 Системні проблеми дослідження

#### ⚠️ Проблема 1: AAVEUSDT sweep bias
`oos_usage.md` показує: **всі** стратегії тестувалися на AAVEUSDT (15m).
AAVE — малоліквідний алт, його поведінка ≠ BTC/ETH/LINK.

```
Рекомендація: Обов'язковий тест на BTCUSDT + ETHUSDT + LINKUSDT перед
будь-яким висновком про "edge" або відхиленням стратегії.
```

#### ⚠️ Проблема 2: Worker cache invalidation
З AGENTS.md: "довгоживучий job worker тримає старий код" → 1259 клітинок з
AttributeError, записались як `succeeded`. Критичний технічний борг.

```python
# Рекомендація: hash коду стратегії у cell_key
cell_key = f"{strategy}:{symbol}:{interval}:{params_hash}:{code_version}"
# При code_version зміні → примусовий restart worker
```

#### ⚠️ Проблема 3: hmm_reversion відхилена з неточної причини
1 угода/30д — мала вибірка, але не причина злому. Реальна причина:
HMM-гейт надто агресивний. Варто дослідити `hmm_fit_bars` sweep перед
остаточним відхиленням на BTCUSDT 1h.

### 4.3 Протокол нової стратегії (2026 стандарт)

```
Крок 1: Формулювання гіпотези (spec YAML)            ✅ у вас є
Крок 2: IS backtest на 60% даних (перевірка механіки) ✅
Крок 3: Quintile study (монотонність сигналу)         🔴 ВІДСУТНІЙ
Крок 4: Time-decay test (лаг 0→3 бари)               🔴 ВІДСУТНІЙ
Крок 5: Walk-forward на повних 3y                     ✅
Крок 6: DSR на OOS WF returns                        ✅
Крок 7: Stress scenarios (crash, liquidity)           🟡 не в пайплайні
Крок 8: Sensitivity (плато, не пік)                   ✅
Крок 9: CSCV/PBO                                     ✅
Крок 10: Paper Gate (8 тижнів)                        ✅
```

Кроки 3, 4 відсутні — саме вони фільтрують "уявні" edge.

```python
# Мінімальна реалізація quintile_study для cmd_report:
def quintile_study(signals: pd.Series, fwd_returns: pd.Series, n: int = 5):
    q = pd.qcut(signals, n, labels=False, duplicates='drop')
    means = fwd_returns.groupby(q).mean()
    rho = means.corr(pd.Series(range(len(means))), method='spearman')
    return means, rho  # rho > 0.8 → монотонний сигнал → pass
```

---

## 5. ПЛАН ПОКРАЩЕНЬ (пріоритизований)

### 🔴 КРИТИЧНО (зараз):

| # | Завдання | Складність | Вплив |
|---|---|---|---|
| 1 | Вбудувати **Quintile Study** у `cmd_report` | S (1-2 дні) | Фільтрує фейкові edge |
| 2 | Вбудувати **Time-Decay Test** у `cmd_report` | S (1 день) | Виявляє execution-залежні стратегії |
| 3 | **Rolling ADF kill** для пар (Phase 1.5 B2) | M (2-3 дні) | Захист від XRP/BTC-ситуації |
| 4 | **Kalman vs OLS bake-off** (Phase 1.5 C) | M (3-4 дні) | Покращення hedge ratio |
| 5 | OOS sweep на **BTC + ETH + LINK** (не тільки AAVE) | S (запуск) | Крос-символна валідація |

### 🟡 ВАЖЛИВО (наступний спринт):

| # | Завдання | Складність | Вплив |
|---|---|---|---|
| 6 | **IS-benchmark** fills → daily_report | M (3 дні) | Зворотній зв'язок у cost model |
| 7 | **Vol-aware slippage** у `CostModel` | S (1 день) | Точніший backtest |
| 8 | **Stress scenarios** у `cmd_report` + Paper Gate | S (1-2 дні) | Реальна оцінка tail risk |
| 9 | **ERC** для portfolio runner | M (2 дні) | Диверсифікація ризику |
| 10 | **sparse_basket OOS аудит** (5+ монет) | M (4 дні) | Новий кандидат для portfolio |

### 🟢 ДОВГОСТРОКОВІ:

| # | Завдання | Складність | Вплив |
|---|---|---|---|
| 11 | L2 Tardis.dev integration (MM revival) | L | Новий тип стратегії |
| 12 | WebSocket async private stream keepalive | M | Production readiness |
| 13 | Market impact (√-law) у CostModel | M | Точніший великий ноціонал |
| 14 | Kelly/half-Kelly для position sizing | M | Оптимальний розмір |
| 15 | Value-added test нової стратегії vs portfolio | S | Кращий відбір |

---

## 6. МОЯ ДУМКА ПРО СТАН ДОСЛІДЖЕНЬ

### Що реально добре:

**`pairs_arb LINK/BTC`** — єдиний правильно валідований кандидат.
PBO=0.000, maxDD знижено вдвічі через `regime_scale`. Заслуговує довіри.
**Правильна тактика: не шукати нові стратегії, а пройти Paper Gate.**

**Відхилення `mean_reversion`** — правильне. OOS Sharpe < 0 = немає edge.
Тиск "продовжити оптимізацію" — класична пастка, ви правильно зупинились.

### Що турбує:

1. **Фокус на нових стратегіях при незавершеному Paper Gate** — типова помилка.
   Платформа знайшла кандидата. **Пріоритет: VPS + paper + 8 тижнів, потім нові дослідження.**

2. **AAVEUSDT як "представницький" символ** для sweep — проблема.
   Всі відхилені стратегії тестувались тільки на AAVE 15m.
   Можливо, `cvd_momentum` або `ob_imbalance` мають edge на BTC 1h.
   Варто перевірити top-2 rejected на BTCUSDT перед остаточним відхиленням.

3. **`ml_strategy`** без чіткого OOS Sharpe на реальних даних.
   Meta-labeling правильний підхід, але потребує ≥2 роки тіків для надійності.

4. **Paper Gate критерії** не включають stress testing —
   рекомендую додати як go/no-go criterion (наприклад: maxDD при crash-30% ≤ 2×backtest maxDD).

### Головна рекомендація:

> **Зупиніть розширення (нові стратегії) і поглибте існуюче:
> LINK/BTC paper + quintile/time-decay у стандартний пайплайн + IS benchmark.
> Квант-торгівля = 80% дисципліна, 20% нові ідеї.**

---

## 7. ТЕХНІЧНІ БОРГИ

| Проблема | Місце | Пріоритет |
|---|---|---|
| Worker cache invalidation при зміні коду | `validation/sweep.py` | 🔴 Критичний |
| `quintile.py` не в `cmd_report` | `validation/` | 🔴 Критичний |
| `time_decay.py` не в `cmd_report` | `validation/` | 🔴 Критичний |
| `stress.py` не в `cmd_report` | `validation/` | 🟡 Важливий |
| `impact_frac = 0.0` зашито | `backtest/execution.py` | 🟡 Середній |
| `fills.py` без зворотного зв'язку | `live/fills.py` | 🟡 Середній |
| AAVEUSDT sweep bias | `validation/sweep.py` | 🟡 Середній |
| `mypy strict=false` у більшості модулів | `pyproject.toml` | 🟢 Низький |

---

*Аналіз на основі детального вивчення кодової бази та порівняння з:
Narang 2026, López de Prado AFML, Kyle (1985), Roll (1984), Amihud (2002),
Avellaneda-Stoikov (2008), Bailey-López de Prado (2014) DSR.*
