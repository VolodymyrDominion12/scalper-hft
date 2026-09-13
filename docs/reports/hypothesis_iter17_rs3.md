# Pre-registration · Раунд RS-3 (iter17) — реалізація `risk_overlay` у RegimeSupervisor

Дата фіксації: 2026-09-12 (ДО змін у коді стратегії)
Попередні раунди: [iter15_regime_supervisor.md](iter15_regime_supervisor.md) (RS-1, reject селектора),
[iter16_rs2_holdout.md](iter16_rs2_holdout.md) (RS-2, Tier-2 для risk-scaling)
Протокол SDD: `.agents/skills/spec-driven-strategy/SKILL.md` — spec ПЕРЕД кодом.

> RS-1+RS-2 дали єдиний механізм, який вижив на справжньому holdout (11 свіжих символів):
> рівноважний портфель 5 рукавів + експозиція ×0.25 у режимі високої **власної** волатильності
> на 4h (детектор `det_vol` — перцентиль rv(60)). RS-3 переносить це з дослідницького
> harness'а в саму стратегію `RegimeSupervisor` і перевіряє відтворюваність.

## 1. Що саме реалізується (заморожені рішення з RS-1/RS-2)

| Елемент | Значення (заморожено) | Джерело |
|---|---|---|
| `blend_mode` | **`risk_overlay`** (новий режим) | RS-2: `riskoff_anchor@det_vol` |
| risk-on leg | рівновага всіх суб-стратегій пулу (вага 1/n) | RS-1 P1/RS-2 |
| risk-off leg | `risk_off_scale × incumbent` (incumbent = перша суб-стратегія пулу) | RS-1 P6 |
| `risk_off_scale` | **0.25** | RS-1/RS-2 (не підбирати!) |
| детектор | `det_vol` = власний vol-перцентиль (low/normal/high); risk-off ⟺ `high` | RS-2: переносний |
| підтвердження | **немає** (off_dwell=1) — dwell фальсифіковано (H16-B ❌) | RS-2 |
| цільовий ТФ | **4h** (1d — не працює, 1h — збитковий) | RS-2 §3, RS-1 §2 |
| пул рукавів | `ts_momentum(allow_short=False)`, `ts_momentum(allow_short=True)`, `cross_momentum(lb=10)`, `funding_carry`, `supertrend` | RS-1 SLEEVES |

**Заборонено в цьому раунді:** додавати нові параметри, підбирати `risk_off_scale`, змінювати
детектор або ТФ, додавати dwell. Будь-яка така зміна = новий раунд із новою pre-registration.

## 2. Зміни в коді (spec → код → тести)

1. `specs/strategies/regime_supervisor.yaml`: `version: 2.2`, додати `blend_mode: risk_overlay`
   у `params_doc` + `risk_off_scale`, оновити `hypothesis` (перемикання → масштабування).
2. `scalper_hft/strategies/regime_supervisor.py`: режим `risk_overlay` у `generate_signals`
   (каузально: режим на барі t рахується з даних ≤ t; рушій лагає на 1 бар, як і для решти
   режимів) + `DEFAULT_BLEND_MODE` НЕ змінювати (дефолт лишається `regime_soft`).
3. Тести: (а) сигнал у `{-1, 0, 1}` × scale, без NaN; (б) no-lookahead mutation-тест
   (мутація майбутніх барів не змінює сигнал t); (в) risk-off бар дає |signal| ≤ scale×|incumbent|;
   (г) spec-тести `tests/test_strategy_specs.py -k regime_supervisor`.

## 3. Критерій прийняття раунду (reproducibility gate)

RS-3 вважається виконаним, якщо на тих самих збережених даних стратегія через СВІЙ інтерфейс
(`RegimeSupervisor(blend_mode="risk_overlay", strategies=...)`) відтворює PnL політики
`riskoff_anchor@det_vol` з `results/iter15/sleeves/4h_maker_*.parquet`:

- кореляція денних PnL ≥ **0.99** і різниця Sharpe ≤ **0.05** на кожному з 11 свіжих символів;
- якщо розбіжність більша — зафіксувати причину (різниця в лагах/сематиці детектора) і
  виправити код, а не «підганяти» числа.

## 4. Після RS-3

- Звіт `docs/reports/iter17_rs3_implementation.md` + оновлення `STRATEGY_STATUS.md`.
- Пропозиція: **paper-моніторинг** risk-overlay на 4h поруч із `pairs_arb` LINK/BTC
  (не live; `DRY_RUN=true`), 8 тижнів, tracking error ≤ бектест ×1.5.
- Окрема гілка (не в цьому раунді): meta-labeling (HB-8/R2) як шлях до Tier-1-потужності.

## 5. Multiplicity

RS-3 не додає жодного нового «вибору» (реалізація замороженого механізму) → `n_trials`
лишається **85** (63 RS-1 + 22 RS-2). Якщо під час реалізації виникне спокуса щось підібрати —
це нова гіпотеза, і її треба реєструвати окремо.
