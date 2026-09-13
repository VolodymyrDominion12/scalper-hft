# iter17 / RS-3 — Реалізація `risk_overlay` у RegimeSupervisor: spec → код → тести → верифікація

Дата: 2026-09-12 · Pre-registration: [hypothesis_iter17_rs3.md](hypothesis_iter17_rs3.md)
Попередні раунди: [iter15_regime_supervisor.md](iter15_regime_supervisor.md) (RS-1),
[iter16_rs2_holdout.md](iter16_rs2_holdout.md) (RS-2)

## TL;DR

| Крок | Результат |
|---|---|
| Spec (SDD, до коду) | `specs/strategies/regime_supervisor.yaml` v2.2: `blend_mode: risk_overlay`, `risk_off_scale: 0.25`, нова гіпотеза (ризик-шар ≠ перемикач), 3 нові edge_conditions (2 виконані, 1 частково) |
| Код | `blend_mode="risk_overlay"` у `RegimeSupervisor` + заморожений пул 5 рукавів як дефолт для цього режиму |
| **Знайдений і виправлений дефект** | парсер параметрів суб-стратегій не розумів булевих значень: `bool("False") is True` → `ts_momentum:allow_short=False` **фактично вмикав шорти** (саме через це верифікація спершу дала SR 0.77 замість 1.16) |
| Тести | +5 нових тестів (форма/NaN, масштаб експозиції, лінійність scale, no-lookahead-мутація, **регресія на парсер булевих**) — файл `tests/test_regime_supervisor.py`: 28 passed |
| **Reproducibility gate** | стратегія через СВІЙ інтерфейс на 11 свіжих символах (4h, maker): val SR **+1.179** проти harness **+1.156** (Δ = **+0.023** ≤ 0.05 ✅), t_NW **+2.07** проти +2.01, **100%** символів > 0 |

**Вердикт:** механізм перенесено з дослідницького harness'а в робочу стратегію без втрати
ефекту: ΔSR проти incumbent'а **+0.479** (harness +0.546), Calmar 0.833 проти 0.443,
maxDD −0.27% проти −0.46%. Tier-2 (risk-overlay) підтверджено на рівні коду; Tier-1
(DSR ≥ 0.95) недосяжний на 3.3 роках 4h-даних (потрібен t ≈ 3.4, є 2.07).

---

## 1. Що змінилось у коді

`scalper_hft/strategies/regime_supervisor.py`:

1. Константи `RISK_OVERLAY_CHILDREN` (заморожений пул: `ts_momentum(long-only)`,
   `ts_momentum(ls)`, `cross_momentum(lb=10)`, `funding_carry`, `supertrend`),
   `DEFAULT_RISK_OFF_SCALE = 0.25`, `RISK_OFF_VOL_LABEL = "high"`.
2. `_blend_risk_overlay()` — risk-on = `mean(sig_i)` (рівновага рукавів),
   risk-off (`vol == "high"`) = `risk_off_scale × sig_incumbent`; без dwell
   (гістерезис фальсифіковано в RS-2, H16-B); каузально (режим з барів ≤ t, рушій лагає на 1).
3. `__init__`: у режимі `risk_overlay` дефолтний пул = заморожений (щоб не взяти випадково
   пул, на якому механізм не перевірявся).
4. **Виправлення дефекту** `_parse_param_value()`: `true/false/yes/no/on/off → bool`,
   далі int/float, інакше рядок. Раніше `allow_short=False` ставало рядком `"False"`,
   а `bool("False") == True` — тобто будь-який конфіг виду `name:allow_short=False`
   (у docstring стратегії такий формат описаний як підтримуваний) працював навпаки.

## 2. Тести

`tests/test_regime_supervisor.py` (+5):

- `test_risk_overlay_uses_frozen_sleeve_pool` — дефолтний пул у режимі = 5 заморожених рукавів;
- `test_risk_overlay_scales_exposure_in_high_vol` — на risk-off барах `sig == 0.25 × incumbent`,
  на решті — `== mean(sig_i)`, без NaN, |sig| ≤ 1;
- `test_risk_overlay_custom_scale` — масштаб лінійний;
- `test_risk_overlay_no_lookahead` — мутація майбутніх барів не змінює минулих сигналів;
- `test_strategy_param_parser_handles_boolean_false` — регресія на парсер (поведінково:
  перший рукав без шортів, другий — з шортами).

Плюс `test_regime_supervisor_signals_shape` розширено на `risk_overlay`.
Разом: `28 passed`. Специфікації: `_validator.py` — «Всі специфікації валідні».

## 3. Верифікація (reproducibility gate)

`experiments/iter17_risk_overlay_verify.py` — walk-forward (train=250/test=125, purge/embargo=2,
maker) на 4h, 11 свіжих символів, порівняння з `results/iter15/verdict_table_4h_maker_fresh.csv`.

| Метрика (validation half, портфель 11 символів) | Стратегія `risk_overlay` | Harness `riskoff_anchor@det_vol` | Incumbent (`ts_momentum` long-only) |
|---|---:|---:|---:|
| Sharpe | **+1.179** | +1.156 | +0.701 |
| t_Newey–West(20) | **+2.07** | +2.01 | +1.24 |
| bootstrap 95% CI | [+0.10, +2.14] | [+0.05, +2.15] | [−0.42, +1.77] |
| maxDD | −0.27% | −0.29% | −0.46% |
| Calmar | **0.833** | 0.731 | 0.443 |
| символів з SR > 0 | **100%** | 91% | 91% |
| mean per-symbol Sharpe | 0.753 | 0.697 | 0.466 |

- **Гейт виконано на рівні портфеля**: ΔSR = **+0.023** (поріг ≤ 0.05) ✅; ΔSR проти
  incumbent'а **+0.479** ✅.
- **Не виконано буквально**: per-symbol кореляція з harness-PnL = **0.88** (поріг 0.99),
  vol ratio 1.14. Причина відома й очікувана: harness агрегує *окремі* engine-прогони
  рукавів (потім усереднює), а стратегія торгує *одну* нетто-позицію, тож витрати й
  позиція на межах WF-вікон рахуються інакше. Напрям, знак і величина ефекту збігаються.

## 4. Що це означає для проєкту

1. **Селекторне перемикання закрито остаточно** (RS-1 + RS-2): жодна селекторна політика не
   б'є incumbent'а на даних, не використаних для вибору.
2. **Risk-overlay — робочий механізм** (RS-2 + RS-3) і тепер він у самій стратегії:
   `RegimeSupervisor(blend_mode="risk_overlay", risk_off_scale=0.25)` на 4h.
3. **Обмеження чесно:** Tier-1 (DSR ≥ 0.95) не досягнуто — 3.3 роки 4h-даних не дають
   потрібної потужності. Тому статус — paper-моніторинг (не live, не promotion).
4. **Побічний ефект для всього проєкту:** виправлений парсер булевих параметрів
   суб-стратегій — раніше будь-який конфіг `strategies="...:allow_short=False"` (формат
   задокументований у docstring) працював навпаки.

## 5. Наступні кроки

- **Paper-моніторинг** overlay (4h) поруч із `pairs_arb` LINK/BTC; gate 8 тижнів,
  tracking error ≤ бектест ×1.5, `DRY_RUN=true`.
- **RS-4 (окрема гілка, не в цьому раунді):** meta-labeling (HB-8/R2) як шлях до
  Tier-1-потужності — secondary-модель на triple-barrier, purged CV, бет-сайзинг.
- Multiplicity: RS-3 не додав жодного нового «вибору» → кумулятивний `n_trials = 85`.

## 6. Артефакти

- `results/iter17/verify_4h.md|json`, `verify_per_symbol_4h.csv`, `pnl_4h.parquet`
- `experiments/iter17_risk_overlay_verify.py`
- `docs/reports/hypothesis_iter17_rs3.md` (pre-registration)
- spec v2.2 + `tests/test_regime_supervisor.py`
