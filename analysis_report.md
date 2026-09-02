# Аналіз проекту scalper-hft

> Станом на 2026-09-02. **339 пройшло, 3 пропущено, 0 помилок. Ruff — чистий.**

---

## ✅ Загальний стан: ДУЖЕ ДОБРИЙ

Проект — зріла, добре структурована квант-платформа з гарним покриттям тестами та документацією. Архітектура відповідає книзі Narang і є консистентною. Нижче — конкретні знахідки, від критичних до дрібних.

---

## 🔴 Баги / некоректності

### 1. `risk_of_ruin` — математично некоректна формула
**Файл:** [`scalper_hft/backtest/metrics.py:136-144`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/backtest/metrics.py#L136-L144)

```python
# Поточний код — НЕКОРЕКТНО:
edge = avg_trade_return       # частка, напр. 0.001 (0.1%)
f = 0.01                      # фіксована частка ризику
b = edge / f                  # b = 0.1 → не є odds ratio!
risk_of_ruin = ((1-b)/(1+b)) ** (1.0/f)
```

**Проблема:** `b` трактується як "перевага" (advantage), але формула Kelly вимагає **odds ratio** (середній виграш / середній програш), а не `edge/f`. При `edge=0.01` → `b=1.0` → `P(ruin)=0`, що не відповідає реальності.

**Правильна формула** (Vince, "Mathematics of Money Management"):
```python
# P(ruin) = ((1-p)/p)^(R/W), де p=win_rate, R=ruin_threshold, W=bet_size
# Або спрощена Kelly: P(ruin) = exp(-2*edge*N), де N=capital/stake
```

**Рекомендація:** або виправити на стандартну формулу, або позначити метрику як `experimental=True` і додати попередження у summary.

---

### 2. `compute_metrics` — масштаб ануалізації не залежить від таймфрейму
**Файл:** [`scalper_hft/backtest/metrics.py:83-84`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/backtest/metrics.py#L83-L84)

```python
# Завжди використовує 365*24*60 (хвилинний масштаб):
ann_vol = ret.std(ddof=0) * math.sqrt(365 * 24 * 60)  # ← 525 600
sharpe  = ret.mean() / ret.std(ddof=0) * math.sqrt(365 * 24 * 60)
```

**Проблема:** при подачі 1h даних Sharpe завищується у `sqrt(60) ≈ 7.7x`. Для `pairs_arb` на 1h це **критично**: реальний річний Sharpe ~0.8 виглядає як ~6.2.

**Рекомендація:**
```python
# Автовизначення таймфрейму:
median_bars_per_day = pd.Timedelta("1D") / equity.index.to_series().diff().median()
ann_factor = math.sqrt(max(median_bars_per_day, 1) * 365)
```

> [!WARNING]
> Це може впливати на порогові перевірки DSR/PBO, якщо вони спираються на `compute_metrics` для 1h-даних.

---

### 3. `estimate_spread_from_bookticker` — неправильний fallback на `bid_qty`/`ask_qty`
**Файл:** [`scalper_hft/backtest/execution.py:142-144`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/backtest/execution.py#L142-L144)

```python
bid_col = "bid" if "bid" in bt.columns else "bid_qty"   # ← bid_qty — це КІЛЬКІСТЬ, не ціна!
ask_col = "ask" if "ask" in bt.columns else "ask_qty"
```

**Проблема:** `bid_qty`/`ask_qty` — кількість контрактів, а не ціна. Spread із кількостей буде безглуздим числом.

---

## 🟡 Покращення / важливі зауваження

### 4. `Exp3Bandit` і `HedgeBlend` — не є `Strategy`, не в REGISTRY
**Файл:** [`scalper_hft/strategies/__init__.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/strategies/__init__.py)

`Exp3Bandit` (base=`object`) і `HedgeBlend` — корисні утилітарні класи, але:
- вони **не є** підкласами `Strategy` і не можуть бути напряму використані через `get_strategy()` / CLI
- `DESIGN.md` списує `blend` і `bandit` у таблиці стратегій разом із `ensemble`, що вводить в оману — `blend` це **алгоритм агрегації**, а не стратегія

**Рекомендація:** додати примітку в `DESIGN.md` або зробити обгортку `BlendStrategy(Strategy)`.

---

### 5. `pairs_arb.py` — `pair_legs()` в `risk_gate.py` не відповідає реальному pair_id
**Файл:** [`scalper_hft/live/risk_gate.py:62-63`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/live/risk_gate.py#L62-L63)

```python
def pair_legs(pid: str) -> frozenset[str]:
    return frozenset(part for part in pid.split("/") if part)
```

Формат пар: `XRPUSDT/BTCUSDT`. `pair_legs("XRPUSDT/BTCUSDT")` → `{"XRPUSDT", "BTCUSDT"}`.
Формат позиційного ключа: `XRPUSDT/BTCUSDT:XRPUSDT`.
Функція `pos_pair_id` правильно стрипає `:XRPUSDT`, але `pair_legs` отримує вже `pid` без суфікса — **ОК**. Але якщо хтось передасть повний ключ позиції — результат невірний. Варто додати assert або doc.

---

### 6. `walk_forward.py` — Sharpe розраховується `_sharpe_from_equity`, а не з `compute_metrics`
**Файл:** [`scalper_hft/validation/walk_forward.py:64-68`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/validation/walk_forward.py#L64-L68)

```python
def _sharpe_from_equity(equity: pd.Series) -> float:
    ret = equity.pct_change().dropna()
    return float(ret.mean() / ret.std(ddof=0) * np.sqrt(len(ret)))   # ← per-period, не річний!
```

**Проблема:** це не ануалізований Sharpe — це `Sharpe * sqrt(N_bars)`. Для OOS вікна в 480 барів: `sqrt(480) ≈ 22`. Результати walk-forward несумісні зі звітом `compute_metrics`. Проте для **порівняння IS↔OOS** це консистентно і має сенс (WFA порівнює однакові вікна). Важливо задокументувати цю відмінність.

---

### 7. `HedgeBlend.step()` — η обчислюється і одразу зберігається, але `eta=None` тоді не перерахується
**Файл:** [`scalper_hft/strategies/blend.py:95-98`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/strategies/blend.py#L95-L98)

```python
if self.eta is None:
    self._q += ...
    self.eta = float(np.sqrt(...))   # ← після першого step eta стає float і більше не оновлюється!
```

**Проблема:** `eta` фіксується після першого кроку і більше не оновлюється з новими даними. Функція `hedge_weights()` обчислює `eta` один раз на весь батч — це правильно. Але `HedgeBlend` задуманий як **інкрементальний** для live — тоді адаптивне η повинно оновлюватись щоразу. Потрібно або документувати як known limitation, або не присвоювати `self.eta` в `step()`.

---

### 8. `estimate_spread_from_bookticker` не враховується в `run_backtest`
**Файл:** [`scalper_hft/backtest/execution.py:134-150`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/backtest/execution.py#L134-L150)

Функція написана, але **ніде не викликається в backtest pipeline** (тільки в `mcp_trading.py`). Якщо є реальні bookTicker дані — вони не впливають на CostModel у бектесті автоматично.

---

### 9. `pyproject.toml` — Python 3.11+ але `_TRADING_DAYS` і mypy target — 3.12
**Файл:** [`pyproject.toml:5`](file:///home/volodymyr/PycharmProjects/scalper-hft/pyproject.toml#L5)

```toml
requires-python = ">=3.11"   # але mypy і ruff таргетують 3.12
```

У `.venv` — `python3.13`. Невідповідність не критична, але краще синхронізувати до `>=3.12`.

---

## ✅ Документація vs Код: відповідність

| Аспект | Статус | Коментар |
|---|---|---|
| `DESIGN.md` → структура модулів | ✅ | Всі модулі існують і відповідають опису |
| `DESIGN.md` → список стратегій | ⚠ | `blend`/`bandit` описані як стратегії, але не є `Strategy` підкласами |
| `DESIGN.md` → lookahead модель | ✅ | `signals.shift(1)` правильно в `engine.py:157` |
| `AGENTS.md` → комісії в бектестах | ✅ | `CostModel` maker/taker правильно реалізовані |
| `AGENTS.md` → DRY_RUN default | ✅ | `config.py:44` — `dry_run=True` за замовчуванням |
| `STRATEGY_STATUS.md` → відхилені | ✅ | `mean_reversion`, `cvd_momentum` є в REGISTRY але задокументовані як "відхилені" |
| `ROADMAP.md` → Phase 0 P0-баги | ✅ | Тести `test_p0.py` зелені |
| Walk-forward документація | ✅ | Відповідає реалізації |
| Funding cash-flow (no-lookahead) | ✅ | `engine.py:223` — `searchsorted(side='right')-1` — правильно |
| `VALID_PAIRS` в `pairs_runner.py` | ✅ | Відповідає `STRATEGY_STATUS.md` (XRP/BTC, BTC/ETH, LINK/BTC) |

---

## 💡 Рекомендації з покращень

### Пріоритет 1 — виправити
1. **`risk_of_ruin`** — виправити формулу або замінити на стандартну Kelly-based
2. **`estimate_spread_from_bookticker`** — виправити fallback на `bid_qty`
3. **Sharpe annualization** — автовизначення таймфрейму в `compute_metrics`

### Пріоритет 2 — покращення якості
4. Додати параметр `interval` до `compute_metrics` для правильного масштабу  
5. Задокументувати відмінність між Sharpe у WFA і в `summary()`
6. Виправити `HedgeBlend.step()`: не фіксувати `self.eta` назавжди
7. Уточнити в `DESIGN.md`: `blend`/`bandit` — утиліти, не стратегії

### Пріоритет 3 — технічний борг
8. Синхронізувати `requires-python = ">=3.12"` (або `>=3.13`)
9. Підключити `estimate_spread_from_bookticker` до pipeline у `CostModel` при наявності даних
10. Додати `assert ":" not in pid` в `pair_legs()` для захисту від неправильного виклику

---

## 📊 Метрики якості коду

| Метрика | Значення |
|---|---|
| Тести | **339 passed, 3 skipped, 0 failed** |
| Ruff лінтер | ✅ `All checks passed` |
| Mypy | `strict=false` (розумно для проекту такого розміру) |
| Тест lookahead | ✅ `test_engine_no_lookahead` присутній і зелений |
| Секрети в коді | ✅ Жодного — лише `.env` читання |
| DRY_RUN default | ✅ `True` |
