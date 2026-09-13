# Iteration 15 — Attribution & Research Cycle Report (2026-09-13)

Pre-registration: [hypothesis_iter15.md](hypothesis_iter15.md)

## H15-A: Kalman vs OLS — FAIL ❌

Детальний звіт: [kalman_ols_bakeoff_2026.md](kalman_ols_bakeoff_2026.md)

| | OLS | Kalman |
|---|---|---|
| WF pos (49 вікон) | **55%** ✅ | 22% ❌ |
| WF OOS SRh avg | **+0.010** | −0.028 |
| Угоди | 72 | 525 (fee-drag) |

**Висновок: `use_kalman=False` підтверджено.** Kalman 7× більше угод → fee-drag.

---

## H15-B: ts_momentum Attribution CORE_15 — INCONCLUSIVE ⚠️

### WF pos per symbol (1d, 1095d, 8 вікон кожен)

| Символ | WF pos | Оцінка |
|---|---|---|
| NEARUSDT | **75%** | 🟢 Top |
| DOTUSDT | **75%** | 🟢 Top (несподіваний) |
| ATOMUSDT | 62% | 🟡 Добрий |
| SOLUSDT | 50% | 🟡 Нейтральний |
| ADAUSDT | 50% | 🟡 Нейтральний |
| AVAXUSDT | 50% | 🟡 Нейтральний |
| LINKUSDT | 50% | 🟡 Нейтральний |
| DOGEUSDT | 38% | 🔴 Слабкий |
| XRPUSDT | 38% | 🔴 Слабкий |
| BNBUSDT | 38% | 🔴 Слабкий |
| ETHUSDT | 38% | 🔴 Слабкий |
| BTCUSDT | 38% | 🔴 Слабкий |
| AAVEUSDT | 38% | 🔴 Слабкий |
| LTCUSDT | 38% | 🔴 Слабкий |
| UNIUSDT | **25%** | 🔴 Найгірший |

### Чому INCONCLUSIVE, а не FAIL або PASS

**Лише 8 вікон** на символ при 1d/1095d — надто мало для статистичних висновків:
- Різниця між 75% і 38% = лише 3 вікна з 8 (p-value > 0.1)
- Iter12 attribution (contribution до портфелю) показував DOGE/NEAR/SOL найкращими,
  а LTC/DOT/LINK найгіршими — тут DOTUSDT на 75%, що суперечить
- Per-symbol OOS pos ≠ portfolio contribution (різні метрики)
- Pre-registered критерій H15-B: Port Sharpe (CORE_12) ≥ +1.365 — **не перевірений**
  (потребує повного портфельного бектесту з --symbol list через спеціалізований CLI)

### Рекомендація

Зберегти CORE_15 без змін. UNIUSDT — кандидат на виключення, але 8 вікон
недостатньо. Потрібен повноцінний портфельний бектест через `paper-run-ts-momentum`
або sweep на 15 символах з агрегацією портфельних метрик.

---

## H15-D: Funding carry + filter — FAIL ❌

- BTCUSDT 1h, 1095d: 128 вікон, OOS>0 = **42%** < поріг 55%
- Підтверджує iter9 вердикт: funding carry структурно збитковий 2025–26

---

## H15-E: Sparse basket — НЕ ЗАПУЩЕНО ⚠️

`walkforward --strategy sparse_basket` вимагає `basket_df`
(multi-symbol DataFrame), що недоступний через стандартний CLI.
Потрібен окремий дослідницький скрипт або `regime-backtest`.

---

## Зведений вердикт Iteration 15

| Гіпотеза | Результат | Дія |
|---|---|---|
| H15-A Kalman | **FAIL** | `use_kalman=False` підтверджено |
| H15-B Attribution | **INCONCLUSIVE** | CORE_15 без змін; UNIUSDT — watchlist |
| H15-C AVAX/NEAR + Kalman | **BLOCKED** (H15-A FAIL) | Не запускаємо |
| H15-D Funding carry | **FAIL** | Підтверджує iter9 відхилення |
| H15-E Sparse basket | **BLOCKED** (CLI ліміт) | Потребує спец скрипту |

### Головний пріоритет залишається: **Paper Gate** (8 тижнів VPS)

Поточний стан paper (аудит 2026-09-13):
- bars=2160, return=+1.90%, win rate=60%, fill-rate=100%
- 10 closed trades, expectancy=+0.0102, max lose streak=2

### Що далі (Wave 2 — після Gate)

Прості вісі вичерпані (iter9→15). Наступні кроки вимагають або:
1. **Нових даних** (L2 depth5, накопичення ob_imbalance)
2. **Нових інфраструктурних можливостей** (basket_df pipeline для sparse_basket)
3. **Часу** (8 тижнів paper → Gate → друга пара за Wave 2)

*Дата: 2026-09-13 | Iter: 15 | Agent: Antigravity*
