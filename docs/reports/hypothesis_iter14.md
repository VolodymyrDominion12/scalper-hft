# iter14 — pre-registration: CS-momentum, 12-1 TSMOM, друга пара

Дата: 2026-09-12 · Цикл: improvement-loop (після iter13).

## Мотивація

iter9→13 вичерпали «дешеві» осі навколо `ts_momentum` CORE_15 (1d/4h/1w,
vol-target, нові імена, weekly). Monitoring-набір лишається:

- `pairs_arb` LINK/BTC 1h maker — **validated** (можна стартувати paper);
- `ts_momentum` 1d/4h long-only CORE_15 — **monitoring**.

Щоб не підганяти ту саму альфу, цей цикл тестує **три інші механізми**,
зафіксовані до прогону. Усі три — або інша сім'я, або інша специфікація,
або інший об'єкт (нова пара). Сітки малі; гейт жорсткий; після метрик
нових гіпотез не додаю.

## Дані

- H14-A / H14-B: нативні **1d** klines `binanceusdm` (LIVE), CORE_15,
  ~2500 днів (2019-11 → 2026-09). Той самий універсум, що iter11, але
  **інша альфа** (крос-секція / skip-month), не `ts_momentum` rolling-quantile.
- H14-C: скан коінтеграції на нативних **1d** closes selection half 2019–2022
  (у кеші 1m/1h більшості символів є лише з 2023-09 — 1h-скан на 2019–22
  фізично неможливий без нового даунлоаду; 1d є з лістингу). Validation
  торгівлі — **1h** maker на доступній 1h-історії (~1095d, 2023-09→2026-09)
  + funding. Відбір пар не бачить 1h-валідацію.
- Перед прогоном: `data-audit --days 1095` (exit 0).

## Гіпотези (зафіксовано ДО прогону)

### H14-A: true cross-sectional momentum (портфель)

Ранжування 15 символів за lookback-return; лонг top-`top_pct`.
Це **не** `ts_momentum` (власний квантиль історії) і не 1D-сигнал
`cross_momentum` для першої колонки — рівноважний CS-портфель.

- `top_pct=0.2` заморожено.
- Сітка lookback (дні): `{10, 20, 60}`. Три значення, не підганяю.
- Первинний варіант: **long-only** (рівновага серед топу, поза топом = кеш).
- Контроль: long-short (top +0.5 / bottom −0.5, dollar-neutral) на
  обраному lookback — очікування з iter11: шорти слабші.
- Виконання: ваги на закритті t → дохідність бару t+1; maker 2 bps на
  `|Δw|` (оборот).
- Two-stage: selection 2019-01-01 → 2022-12-31 обирає lookback з макс.
  Sharpe long-only; validation 2023-01-01 → 2026-12-31 невидима.
- **Рішення:** → monitoring IFF на validation
  `Sharpe > 0` AND `t_NW ≥ 2.0` AND bootstrap 95% CI не містить 0.
  Інакше — відхилено. Жодного іншого порогу після.

### H14-B: 12-1 skip-month TSMOM (академічна специфікація)

Moskowitz–Ooi–Pedersen: знак дохідності за 252 дні **без останніх 21**.
Параметри **заморожені** (немає сітки) — lookback=252, skip=21,
long-only, рівновага серед імен з додатним skip-return.

- Немає selection по параметрах: 2019–2022 = IS-підтвердження,
  2023–2026 = єдиний validation.
- Контроль: той самий lookback **без skip** (252d, skip=0).
- Витрати / лаг — як у H14-A.
- **Рішення:** той самий гейт на validation 2023–2026.

### H14-C: друга пара з IS-скану коінтеграції

`scan_pairs` **лише** на 1d selection half (2019–2022). З tradable
виключаю вже досліджені пари (не «перевідкриваю» відхилені):

`LINK/BTC`, `XRP/BTC`, `LINK/ETH`, `BTC/ETH`, `ETH/SOL`, `BTC/SOL`
(обидва порядки ніг).

Беру **топ-3** за ADF p-value серед tradable, що лишились.
Рецепт `pairs_arb` **заморожений** (як VALIDATED_PAIRS, без підбору z/lb):

`entry_z=2.0`, `exit_z=0.3`, `lookback=120`, `regime_scale=True`,
`regime_scale_factor=0.25`, maker, `position_pct=0.1`.

Validation: WF на повній 1h-історії (train=1500 / test=500) + PnL на
вікні 2023-01-01 → 2026-12-31.

- **Рішення:** пара → candidate (НЕ validated, НЕ в `VALIDATED_PAIRS`) IFF
  `WF pos ≥ 0.55` AND `n_windows ≥ 3` AND `n_trades ≥ 20` на validation
  half AND `validation total_return > 0`. CSCV/PBO — окремий крок перед
  будь-якою зміною `VALIDATED_PAIRS`.

### H14-D: value-added (діагностика, не гейт)

Якщо H14-A або H14-B має validation Sharpe > 0: 50/50 з наявним
`ts_momentum` 1d CORE_15 (`results/iter12/ts_oos_matrix_1d.parquet`).
Combined Sharpe > max(standalone) AND combined maxDD кращий → фіксую
диверсифікацію (construction evidence, не paper-gate).

## Що НЕ роблю

- Не підбираю top_pct / skip / z / lookback / імена після метрик.
- Не додаю H14-E+ після прогону («поки не вийде»).
- Не чіпаю спалені 1w-вікна і не реоптимізую `ts_momentum` CORE_15.
- Не запускаю live. Paper pairs — вже дозволений validated-кандидат;
  цей цикл лише шукає **додатковий** рукав.

## Зупинка циклу

Три гіпотези = запланований цикл. Після звіту: або промоція комірок,
що пройшли гейт, або **стоп** і старт paper на поточному наборі
(pairs LINK/BTC ± monitoring ts_momentum). Новий цикл — лише з новою
pre-registration і новими даними/механізмом.
