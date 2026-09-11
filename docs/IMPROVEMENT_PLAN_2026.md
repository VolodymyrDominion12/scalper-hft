# План покращень scalper-hft: Research Pipeline & Production Hardening (2026)

> Статус: **Active** | Створено: 2026-09-11 | На основі аналізу vs кращих практик HFT/MFT 2026
> Пов'язані документи: [ROADMAP.md](ROADMAP.md) · [STRATEGY_STATUS.md](STRATEGY_STATUS.md) · [CHANGE_PLAN.md](CHANGE_PLAN.md)

---

## Контекст та мотивація

Аналіз кодової бази (вересень 2026) підтвердив: платформа архітектурно правильна, але
реалізація кожного шару — «перша версія». Три системних проблеми вимагають вирішення
до будь-якого розширення списку стратегій:

1. **Research pipeline має сліпі зони**: `quintile.py` і `time_decay.py` існують, але
   не вбудовані у стандартний `cmd_report`. Стратегії проходять WF/DSR/CSCV, але
   не перевіряються на монотонність сигналу та залежність від швидкості виконання.

2. **CostModel не має зворотного зв'язку**: `fills.py` логує Implementation Shortfall,
   але ці дані не агрегуються і не повертаються у модель. `vol_aware_slippage` і
   `sqrt_law_impact` реалізовані у `CostModel`, але **не активовані за замовчуванням**.

3. **Символьна упередженість sweep**: всі стратегії тестувались переважно на AAVEUSDT
   (15m). Відхилені стратегії потребують верифікації на BTCUSDT/ETHUSDT/LINKUSDT.

**Головне правило цього плану**: жодних нових стратегій до завершення Phase A.
`pairs_arb LINK/BTC` — валідований кандидат. Пріоритет: Paper Gate + pipeline hardening.

---

## Phase A — Research Pipeline Hardening (2 тижні)

> **Мета**: Автоматизувати критичні тести (quintile + time-decay + stress) у `cmd_report`
> так, щоб будь-яка нова стратегія проходила повний 10-кроковий протокол без ручного
> виклику окремих команд.

### A1 — Quintile Study у `cmd_report` [🔴 КРИТИЧНО]

**Що робити**: Вбудувати `quintile_spread_study()` з `validation/quintile.py` у
`cmd_report` як обов'язковий розділ звіту.

**Файли**:
- Читати: `scalper_hft/validation/quintile.py` L22–45
- Редагувати: `scalper_hft/cli/research_audit.py` → функція `cmd_report`

**Детальна специфікація**:

```python
# У cmd_report, після секції [3] SENSITIVITY — додати:

print("\n[4] QUINTILE STUDY (Narang гл. 9 — monotonicity)")
print("─" * 50)

from scalper_hft.validation.quintile import quintile_spread_study

full_signals = strategy.generate_signals(df, ...)
fwd_returns = df["close"].pct_change().shift(-1).fillna(0.0)

try:
    q_result = quintile_spread_study(z=full_signals, forward=fwd_returns, n_bins=5)
    print(q_result.summary())

    QUINTILE_SPEARMAN_MIN = 0.7
    if not q_result.monotonic or abs(q_result.spearman) < QUINTILE_SPEARMAN_MIN:
        print(f"⚠ QUINTILE FAIL: Spearman ρ={q_result.spearman:+.3f} — сигнал не монотонний")
    else:
        print(f"✅ QUINTILE PASS: Spearman ρ={q_result.spearman:+.3f}")
except ValueError as e:
    print(f"  quintile: {e}")
```

**Критерій PASS**: `|Spearman ρ| ≥ 0.7` та `monotonic = True`.

**Важливо для pairs**: Для `pairs_arb` сигнал = z-score спреду; forward = `-Δspread`.
Адаптувати виклик у `cmd_pairs_report` аналогічно.

---

### A2 — Time-Decay Test у `cmd_report` [🔴 КРИТИЧНО]

**Що робити**: Вбудувати `time_decay_test()` / `pairs_time_decay()` у `cmd_report`
як розділ після quintile study.

**Файли**:
- Читати: `scalper_hft/validation/time_decay.py` L76–115
- Редагувати: `scalper_hft/cli/research_audit.py`

**Детальна специфікація**:

```python
# Після [4] QUINTILE STUDY — додати:

print("\n[5] TIME-DECAY TEST (Narang гл. 9 — execution dependency)")
print("─" * 50)

from scalper_hft.validation.time_decay import time_decay_test, pairs_time_decay

if not is_pairs_strategy:
    td = time_decay_test(df, strategy, max_lag=3, cost=cost, trades=trades,
                         funding=funding, position_pct=position_pct)
else:
    td = pairs_time_decay(signals=signals, spread=spread, max_lag=3)

print(td.summary())

lag0_sr, lag1_sr = td.sharpes[0], td.sharpes[1]
if lag0_sr > 0 and lag1_sr < 0:
    print("⚠ TIME-DECAY FAIL: Sharpe вмирає за 1 бар — execution-dependent!")
elif lag0_sr > 0 and lag1_sr / lag0_sr < 0.5:
    print(f"⚠ TIME-DECAY WARN: Деградація {1 - lag1_sr/lag0_sr:.0%} за 1 бар")
else:
    print(f"✅ TIME-DECAY PASS: Sharpe зберігається при lag=1 ({lag1_sr:+.3f})")
```

**Критерій PASS для pairs 1h**: Sharpe при lag=1 ≥ 50% від lag=0.

---

### A3 — Stress Scenarios у `cmd_report` та Paper Gate [🟡 ВАЖЛИВО]

**Що робити**: Вбудувати 4 стандартних стрес-сценарії у `cmd_report` і додати
stress-поріг до критеріїв Paper Gate у `validation/pairs_gate.py`.

**Файли**:
- Читати: `scalper_hft/validation/stress.py` L24–29
- Редагувати: `scalper_hft/cli/research_audit.py`
- Редагувати: `scalper_hft/validation/pairs_gate.py`

```python
# Розширити вже наявну stress-секцію [M4]:

from scalper_hft.validation.stress import run_stress_suite, SCENARIOS

stress_results = run_stress_suite(
    returns=result.bar_returns,
    scenarios=SCENARIOS,
    cost_frac=cost.round_trip_maker() / 2,
)

for name, metrics in stress_results.items():
    ok = metrics["max_dd"] <= result.metrics.max_dd * 2.5
    print(f"  {'✅' if ok else '⚠'} {name:15s}: maxDD={metrics['max_dd']:.1%}")
```

**Paper Gate критерії** (додати у `pairs_gate.py`):
- `stress_crash_max_dd ≤ backtest_max_dd × 2.5`
- `stress_liquidity_max_dd ≤ backtest_max_dd × 4.0`

---

### A4 — Cross-Symbol Validation Sweep [🟡 ВАЖЛИВО]

**Що робити**: Запустити sweep на BTCUSDT, ETHUSDT, LINKUSDT для стратегій що
раніше тестувались лише на AAVEUSDT.

```bash
# 0) Аудит кешу (обов'язково)
uv run python -m scalper_hft.cli data-audit --days 1095

# 1) Cross-symbol sweep
uv run python -m scalper_hft.cli backtest \
  --strategy cvd_momentum,ob_imbalance,cross_momentum \
  --symbol BTCUSDT,ETHUSDT,LINKUSDT \
  --interval 1h --days 1095 \
  --mode walkforward --workers 4 --enqueue

# 2) Перегляд результатів
uv run python -m scalper_hft.cli sweep-report \
  --filter "strategy in ['cvd_momentum','ob_imbalance','cross_momentum']"
```

**Мета**: Верифікація чи відхилені на AAVE стратегії мають edge на BTC/ETH/LINK.
**Результат**: `docs/reports/cross_symbol_sweep_2026.md`

---

### A5 — Worker Cache Invalidation Guard [🔴 КРИТИЧНО]

**Проблема**: Задокументована пастка в AGENTS.md — 1259 клітинок з AttributeError
записались як `succeeded` через застарілий код у довгоживучому worker.

**Файли**:
- Редагувати: `scalper_hft/validation/sweep.py`
- Редагувати: `scalper_hft/research/sweep_store.py`

```python
# sweep.py — додати хеш коду стратегії:
import hashlib, inspect

def _strategy_code_hash(strategy_name: str) -> str:
    from scalper_hft.strategies import REGISTRY
    cls = REGISTRY.get(strategy_name)
    if cls is None:
        return "unknown"
    return hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()[:8]

# SweepStore.row_exists() — перевіряти code_hash:
def row_exists(self, key: str, code_hash: str | None = None) -> bool:
    """Клітинка вважається виконаною тільки якщо code_hash збігається."""
    ...
```

---

## Phase B — Cost Model Evolution (1 тиждень)

> **Мета**: Активувати вже реалізовані `vol_aware_slippage` і `sqrt_law_impact`,
> підключити IS feedback loop з `fills.py`.

### B1 — Vol-Aware Slippage активація [🟡 ВАЖЛИВО]

**Контекст**: `execution.py` L72–81 вже містить `vol_aware_slippage(vol_frac)`.
Але `vol_ref = 0.0` за замовчуванням → завжди повертає базовий `slippage_frac`.

**Файли**: `scalper_hft/backtest/execution.py`, `scalper_hft/config.py`

**Конфігурація** (додати у `.env.example`):
```bash
# Vol-aware slippage: 0 = вимкнено (flat), > 0 = калібровочна vol
VOL_AWARE_SLIPPAGE_REF=0.0
VOL_AWARE_SLIPPAGE_EXP=1.0
```

**Фабричний метод**:
```python
@classmethod
def from_settings(cls, settings=None, df: pd.DataFrame | None = None) -> "CostModel":
    """Авто-калібровка vol_ref з Parkinson-vol якщо не задано."""
    s = settings or get_settings()
    vol_ref = _env_float("VOL_AWARE_SLIPPAGE_REF", 0.0)
    if vol_ref <= 0 and df is not None:
        from scalper_hft.features.microstructure import parkinson_vol
        pv = parkinson_vol(df["high"], df["low"], window=20)
        vol_ref = float(pv.dropna().median()) if not pv.dropna().empty else 0.0
    return cls(maker_fee=s.maker_fee, taker_fee=s.taker_fee,
               slippage_frac=s.slippage_bps / 10_000.0, vol_ref=vol_ref,
               vol_exp=_env_float("VOL_AWARE_SLIPPAGE_EXP", 1.0))
```

---

### B2 — Implementation Shortfall Feedback Loop [🟡 ВАЖЛИВО]

**Контекст**: `calibrate_from_is()` і `with_is_slippage()` вже реалізовані у
`execution.py`. Потрібно парсити IS з fills і агрегувати щоденно.

**Новий файл**: `scalper_hft/live/is_report.py`

```python
"""Implementation Shortfall агрегатор (Phase B2)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ISReport:
    n_fills: int
    median_is_bps: float
    maker_is_bps: float
    taker_is_bps: float
    model_slippage_bps: float
    coverage_ok: bool   # ≥ 20 fills для калібровки

    def summary(self) -> str:
        delta = self.median_is_bps - self.model_slippage_bps
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
        return (
            f"IS Report: {self.n_fills} fills | "
            f"Realized={self.median_is_bps:.1f} bps {arrow} Model={self.model_slippage_bps:.1f} bps\n"
            f"  Maker={self.maker_is_bps:.1f} bps | Taker={self.taker_is_bps:.1f} bps"
        )
```

**CLI команда** (додати у `cli/ops.py`):
```bash
uv run python -m scalper_hft.cli is-report --days 7
```

---

### B3 — Market Impact для pairs [🟢 ДОВГОСТРОКОВЕ]

**Контекст**: `sqrt_law_impact()` реалізована (L83–98). Активувати для pairs де
ноціонал > 5% ADV.

```bash
# .env.example:
IMPACT_K=0.0         # 0 = вимкнено; > 0 = увімкнено (починати з 0.05)
```

**Замітка**: При `PAIR_NOTIONAL_PCT=0.05–0.10` impact ≈ 0.1–0.3 bps (незначний).
Активувати коли портфель зросте до 5+ пар.

---

## Phase C — Portfolio & Live Upgrades (2 тижні)

> **Мета**: Підключити вже реалізовані модулі у live-цикл, запустити Kalman bake-off.

### C1 — Kalman vs OLS Bake-off [🔴 КРИТИЧНО]

**Дослідницький протокол**:

```bash
# Step 1: OLS baseline (зафіксувати метрики)
uv run python -m scalper_hft.cli pairs \
  --strategy pairs_arb --leg1 LINKUSDT --leg2 BTCUSDT \
  --interval 1h --days 1095 --maker

# Step 2: Kalman run
uv run python -m scalper_hft.cli pairs \
  --strategy pairs_arb --leg1 LINKUSDT --leg2 BTCUSDT \
  --interval 1h --days 1095 --maker --use-kalman

# Step 3: Kalman sensitivity sweep (delta і Ve параметри)
uv run python -m scalper_hft.cli backtest \
  --strategy pairs_arb --leg1 LINKUSDT --leg2 BTCUSDT \
  --interval 1h --days 1095 \
  --params "kalman_delta=0.0001,0.001,0.01;kalman_ve=0.001,0.01,0.1" \
  --use-kalman --enqueue
```

**Критерії вибору Kalman**:

| Метрика | OLS (baseline) | Kalman (target) |
|---|---|---|
| WF avg OOS Sharpe | +0.0064 | ≥ +0.0064 |
| WF pos_windows | 65% | ≥ 60% |
| maxDD (3y) | −27% | ≤ −30% |
| n_trades (3y) | ~46 | ≥ 35 |
| Sensitivity plateau | ✅ | ✅ |

**Результат**: `docs/reports/kalman_ols_bakeoff_2026.md`

---

### C2 — Rolling ADF Kill [🔴 КРИТИЧНО]

**Проблема**: XRP/BTC: −63% на 3y але +3.49% на 400д = симптом втрати коінтеграції.

**Файли**: `scalper_hft/live/pairs_engine.py`, `scalper_hft/live/pair_health.py`

```python
# pair_health.py — додати:
from statsmodels.tsa.stattools import adfuller

def check_cointegration_health(
    spread: pd.Series,
    window: int = 90 * 24,   # 90 днів для 1h
    pvalue_threshold: float = 0.05,
) -> tuple[bool, float]:
    """Rolling ADF на spread[-window:].
    Returns: (is_healthy, p_value)
    Блокує входи (не flatten) при p_value > threshold.
    """
    if len(spread) < window // 2:
        return True, 0.0   # warmup — дозволяємо
    sample = spread.iloc[-window:].dropna()
    if len(sample) < 50:
        return True, 0.0
    try:
        result = adfuller(sample, maxlag=12, autolag="AIC")
        p = float(result[1])
        return p <= pvalue_threshold, p
    except Exception:
        return True, 0.0   # fail-open: не блокуємо при помилці ADF
```

**Конфігурація**:
```bash
ADF_PVALUE_THRESHOLD=0.05
ADF_WINDOW_DAYS=90
```

---

### C3 — ERC для Portfolio Runner [🟡 ВАЖЛИВО]

**Контекст**: `portfolio/risk_budget.py` (ERC) вже реалізований, але не у live-циклі.

**Файли**: `scalper_hft/live/pairs_runner.py`

```python
# Замість equal-weight:
if self.allocation_method == "erc":
    from scalper_hft.portfolio.risk_budget import erc_vol_target_sizes
    realized_vols = self._compute_pair_vols(window_days=7)
    sizes = erc_vol_target_sizes(
        pair_ids=list(self._open_pairs.keys()),
        realized_vols=realized_vols,
        total_notional=self._equity * self._portfolio_notional_pct,
        min_size=self._min_notional,
    )
else:
    sizes = {pid: equal_size for pid in self._open_pairs}
```

**Конфігурація**:
```bash
PORTFOLIO_ALLOCATION_METHOD=equal   # equal | erc
ERC_VOL_WINDOW_DAYS=7
```

---

### C4 — Sparse Basket OOS Аудит [🟡 ВАЖЛИВО]

```bash
# Sweep на кошику (10 символів)
uv run python -m scalper_hft.cli backtest \
  --strategy sparse_basket \
  --symbol BTCUSDT,ETHUSDT,SOLUSDT,LINKUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT \
  --interval 1h,4h --days 1095 --mode walkforward --workers 4 --enqueue

# Overfit аудит кращого кандидата
uv run python -m scalper_hft.cli overfit \
  --strategy sparse_basket --symbol BTCUSDT --interval 1h --days 1095
```

**Критерій прийняття**: WF pos_windows ≥ 0.55, PBO < 0.5, n_trades ≥ 20.
**Перевірити**: correlation(`sparse_basket`, `pairs_arb LINK/BTC`) < 0.3.

---

## Протокол дослідження нових стратегій (v2.0, вересень 2026)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ПРОТОКОЛ НОВОЇ СТРАТЕГІЇ scalper-hft (v2.0)                            │
├────┬──────────────────────────────────────────────┬────────────┬────────┤
│  # │ Крок                                         │ Команда    │ Статус │
├────┼──────────────────────────────────────────────┼────────────┼────────┤
│  1 │ Формулювання гіпотези (spec YAML)            │ manual     │ ✅     │
│  2 │ IS backtest (60% даних, перевірка механіки)  │ backtest   │ ✅     │
│  3 │ Quintile Study (монотонність сигналу) [NEW]  │ report     │ 🆕 A1  │
│  4 │ Time-Decay Test (лаг 0→3 бари) [NEW]        │ report     │ 🆕 A2  │
│  5 │ Walk-forward (3y + purge/embargo)            │ overfit    │ ✅     │
│  6 │ DSR на конкатенованих OOS WF returns         │ overfit    │ ✅     │
│  7 │ Stress Scenarios (4 сценарії) [NEW]          │ report     │ 🆕 A3  │
│  8 │ Sensitivity (параметри — плато, не пік)      │ overfit    │ ✅     │
│  9 │ CSCV/PBO (< 0.5)                            │ cscv       │ ✅     │
│ 10 │ Paper Gate (≥ 8 тижнів on VPS)              │ paper-run  │ ✅     │
└────┴──────────────────────────────────────────────┴────────────┴────────┘
```

**Після Phase A** — кроки 3,4,7 виконуються автоматично у `cmd_report`:
```bash
uv run python -m scalper_hft.cli overfit --strategy <name> \
  --symbol <SYM>USDT --interval 1h --days 1095
# Звіт автоматично: WF + DSR + sensitivity + quintile + time-decay + stress
```

---

## Оновлені критерії Paper Gate

```
Критерії виходу з Paper Gate (pairs_arb LINK/BTC, 1h, maker):
─────────────────────────────────────────────────────────────
✅ ≥ 8 тижнів безперервного paper-трейдингу на VPS
✅ Tracking error (paper vs backtest equity) < 3% на тиждень
✅ Fill rate post-only ≥ 70%
✅ maxDD paper ≤ backtest maxDD × 1.5
✅ MAE/MFE forensics: без аномалій
✅ Rolling ADF p-value LINK/BTC spread < 0.05        [NEW — Phase C2]
✅ IS report: realized_slippage_maker < 3 bps         [NEW — Phase B2]
✅ Stress crash maxDD ≤ backtest crash maxDD × 2      [NEW — Phase A3]
```

---

## Відповіді на типові питання

**Q: Чи варто запускати sweep на нових стратегіях зараз?**
A: Ні. Пріоритет — Phase A (2 тижні) + C1/C2 (1 тиждень) + Paper Gate.

**Q: Чи потрібно переробити `CostModel` повністю?**
A: Ні. `vol_aware_slippage` і `sqrt_law_impact` вже є — потрібно лише активувати
через `VOL_AWARE_SLIPPAGE_REF > 0` і `IMPACT_K > 0`.

**Q: XRP/BTC — відновлювати чи відхиляти?**
A: Умовно відновити після C2. Запустити paper з rolling ADF kill. Якщо ADF спрацьовує
частіше ніж раз на місяць → остаточно відхилити.

**Q: Коли активувати `ml_strategy`?**
A: Тільки після Phase A. Потребує ≥ 2 роки aggTrades для надійності. Не вмикати
до завершення Paper Gate.

---

## Зведена таблиця завдань

| ID | Завдання | Phase | Складність | Вплив | Статус |
|---|---|---|---|---|---|
| A1 | Quintile Study у `cmd_report` | A | S (1-2д) | 🔴 | ☐ |
| A2 | Time-Decay Test у `cmd_report` | A | S (1д) | 🔴 | ☐ |
| A3 | Stress scenarios у `cmd_report` + Paper Gate | A | S (1-2д) | 🟡 | ☑ |
| A4 | Cross-symbol sweep (BTC+ETH+LINK) | A | S (запуск) | 🟡 | ☐ |
| A5 | Worker cache invalidation guard | A | M (2-3д) | 🔴 | ☑ |
| B1 | Vol-aware slippage активація | B | S (1д) | 🟡 | ☑ |
| B2 | IS feedback loop (`is_report.py`) | B | M (3д) | 🟡 | ☑ |
| B3 | Market impact для pairs | B | M (2д) | 🟢 | ☑ |
| C1 | Kalman vs OLS bake-off | C | M (3-4д) | 🔴 | ☐ |
| C2 | Rolling ADF kill | C | M (2-3д) | 🔴 | ☑ |
| C3 | ERC для portfolio runner | C | M (2д) | 🟡 | ☑ |
| C4 | Sparse basket OOS аудит | C | M (4д) | 🟡 | ☐ |

**Складність**: S = 1–2 дні | M = 3–5 днів
**Вплив**: 🔴 критичний | 🟡 важливий | 🟢 довгострокове

---

*Документ створено: 2026-09-11. Наступний перегляд: після завершення Phase A.*
*Аналіз: Claude Sonnet 4.6 (Thinking) | Версія плану: 1.0*
