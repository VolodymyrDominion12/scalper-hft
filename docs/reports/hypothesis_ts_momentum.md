# Pre-registration: ts_momentum 1d long-only портфель

**Дата реєстрації:** 2026-09-12  
**Статус:** зареєстровано (до наступного OOS-прогону)  
**Автор циклу:** iter10 (`experiments/iter10_research_cycle.py`)

---

## 1. Гіпотеза (фальсифікована)

На **денних** USDT-M перпах Binance time-series momentum у режимі **long-only**
(`allow_short=False`, `lookback=20`) на **рівноважному портфелі з 15 ліквідних
символів** дає позитивний risk-adjusted return після maker-комісій (2 bps/бік),
який:

1. перевищує baseline short+long на тому ж універсумі;
2. має **t-статистику Newey–West(20) ≥ 2.0** на OOS-дохідностях портфеля;
3. не зникає при maker-виконанні (round-trip ~4 bps).

**Механізм (чому long-only):** на горизонті 2019–2026 крипто-перпи переважно в
bull-фазі; short-ноги momentum несуть додатковий funding-drag і зворотний тренд.
Зменшення turnover (без шортів) знижує fee-drag.

---

## 2. Фіксовані параметри (не підганяти post-hoc)

| Параметр | Значення | Обґрунтування |
|---|---|---|
| `strategy` | `ts_momentum` | spec v1.0, iter9 виділення з cross_momentum |
| `interval` | `1d` | єдиний ТФ з невід'ємним edge після витрат |
| `lookback` | 20 | плато iter9 (lb 20–60); lb=20 — дефолт spec |
| `top_pct` | 0.2 | дефолт spec |
| `quantile_window` | 3 | дефолт spec |
| `signal_smooth` | 1 | дефолт spec (варіант smooth=3 — окрема гіпотеза) |
| `allow_short` | **false** | iter10: port Sharpe +1.37 vs +0.76 baseline |
| `execution` | maker (post-only) | round-trip ~4 bps |
| `універсум` | 15 перпів (див. нижче) | той самий, що iter9 |

**Символи (фіксований список):**
`BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, LINKUSDT, DOGEUSDT, ADAUSDT,
AVAXUSDT, UNIUSDT, NEARUSDT, DOTUSDT, ATOMUSDT, LTCUSDT, AAVEUSDT`

---

## 3. Метод перевірки

### 3.1 In-sample (вже спалено — лише для контексту)

3-річний кеш 1m (2023-09 → 2026-09): **OOS спалено** (`docs/reports/oos_usage.md`).
Повторна підгонка на цьому вікні **заборонена**.

### 3.2 Out-of-sample (дозволений прогін)

| Тест | Метод | Поріг |
|---|---|---|
| Довга історія | нативні 1d klines, WF train=250/test=125 | mean OOS SR > 0 |
| Портфель | рівноважний mean OOS-return по символах | Sharpe > 0.5, t_NW ≥ 2.0 |
| Paper | `paper-run-ts-momentum --daemon` ≥ 8 тижнів | tracking error vs BT, maxDD ≤ BT×1.5 |
| Robustness | sensitivity lookback ∈ {10,20,30,40,60} | плато, не пік на одній точці |

### 3.3 Критерії PASS (monitoring → validated)

| Критерій | Поріг |
|---|---|
| Портфель Sharpe (maker, OOS) | ≥ 0.5 |
| t_Newey–West(20) | ≥ 2.0 |
| Bootstrap 95% CI нижня межа | > 0 |
| Paper tracking error | fill-rate ≥ 70%, maxDD ≤ BT×1.5 |
| DSR | інформативний, не гейт (n_trials ≈ 300) |

**Не є гейтом:** single-symbol OOS SR > 0.3 (портфельна стратегія).

---

## 4. Baseline (iter10, до paper)

| Метрика | baseline (short+long) | **long-only** |
|---|---:|---:|
| Port Sharpe | +0.758 | **+1.365** |
| t_Newey–West | +1.87 | **+2.61** |
| Символів OOS > 0 | 12/15 | 11/15 |
| OOS угод | 2644 | 1315 |

Vol-target overlay (20d, post-hoc): +1.086 / t_NW +2.54 — **не входить** в основну
гіпотезу; окремий risk-overlay тест після paper PASS.

---

## 5. Ризики та обмеження

1. **Період-специфічність:** long-only може деградувати у ведмежому ринку.
2. **Кореляція:** середня парна кореляція OOS ≈ 0.36 — диверсифікація обмежена.
3. **Buy&hold:** baseline портфель B&H Sharpe +0.86 — momentum не домінує B&H на
   2020–2026, лише покращує risk-adjusted profile.
4. **Множинність:** iter10 перебрав 6 варіантів; long-only обрано post-hoc → ця
   pre-registration фіксує параметри **до** paper.

---

## 6. Paper-протокол

```bash
# Перевірка даних
uv run python -m scalper_hft.cli data-audit --days 1095

# Paper-моніторинг (1d, long-only, 15 символів)
DRY_RUN=true MAKER_EXECUTION=true \
  uv run python -m scalper_hft.cli paper-run-ts-momentum \
  --daemon --sleep 3600 --control results/control.json

# Щотижневий аудит
uv run python -m scalper_hft.cli paper-audit \
  --db results/paper_ts_momentum.sqlite
```

**Разом із:** `pairs_arb` LINK/BTC 1h (`paper-run-pairs --portfolio`) — не замінює,
а доповнює validated ядро.

---

## 7. Вердикт (заповнюється після paper)

| Етап | Дата | Результат |
|---|---|---|
| Pre-registration | 2026-09-12 | ✅ зареєстровано |
| Paper старт | — | очікує |
| Paper 8 тижнів | — | очікує |
| Validated | — | очікує |
