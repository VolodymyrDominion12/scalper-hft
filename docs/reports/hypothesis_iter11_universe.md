# Pre-registration: iter11 — розширений універсум і нативний 4h

**Дата реєстрації:** 2026-09-12 (до прогону `experiments/iter11_expansion_cycle.py`)  
**Статус:** зареєстровано  
**База:** iter10 long-only, `docs/reports/hypothesis_ts_momentum.md`

---

## Навіщо цей цикл (не підгонка)

iter10 обрав long-only **post-hoc** на тих самих 15 символах, що й iter9.
Єдиний чесний наступний крок — **нові імена**, які не входили в скрин,
і **нативний 4h** (не ресемпл з 1m 3-річного кешу).

Параметри альфи **заморожені**. Не підбираємо lookback / top_pct / smooth.

| Параметр | Значення |
|---|---|
| strategy | `ts_momentum` |
| lookback | 20 |
| top_pct | 0.2 |
| quantile_window | 3 |
| signal_smooth | 1 |
| allow_short | **false** (основна гіпотеза) / true (контроль) |
| execution | maker |

---

## Гіпотези (фальсифіковані)

**H1 — незалежний універсум.** Рівноважний long-only портфель на **лише нових**
символах (список нижче, не з CANONICAL_15) має t_Newey–West(20) ≥ 2.0 і
портфельний Sharpe ≥ 0.5 після maker-комісій на нативних 1d klines.

Якщо H1 FAIL — long-only на 15 іменах був відбором, не edge. Не paper.

**H2 — об'єднаний універсум.** CORE_15 ∪ NEW дає t_NW ≥ 2.0 (більше імен
має підняти t-стат при збереженні кореляції < 0.5).

**H3 — нативний 4h.** Той самий long-only на CORE_15, нативні 4h klines,
календарні вікна WF ≈ 1d (train=1500 / test=750 барів 4h). Поріг той самий:
Sharpe ≥ 0.5 і t_NW ≥ 2.0. Контроль: short+long на 4h.

**H4 — період.** На NEW-універсумі OOS Sharpe > 0 в обох підперіодах
2019–2022 і 2023–2026. Якщо плюс лише в бичачому 2020–21 — період-специфічність.

Vol-target overlay рахуємо як діагностику, **не** як критерій відбору
(пост-хок iter9).

---

## Фіксовані списки (не відкидати імена після прогону)

**CORE_15** (pre-registered iter10, реплікація):
`BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, LINKUSDT, DOGEUSDT, ADAUSDT,
AVAXUSDT, UNIUSDT, NEARUSDT, DOTUSDT, ATOMUSDT, LTCUSDT, AAVEUSDT`

**NEW_30** (незалежний тест; пропуск лише якщо немає лістингу / замало барів):
`BCHUSDT, ETCUSDT, TRXUSDT, XLMUSDT, FILUSDT, EOSUSDT, VETUSDT, ALGOUSDT,
FTMUSDT, MKRUSDT, SNXUSDT, CRVUSDT, COMPUSDT, GRTUSDT, SANDUSDT, MANAUSDT,
AXSUSDT, EGLDUSDT, ICPUSDT, THETAUSDT, APTUSDT, OPUSDT, ARBUSDT, INJUSDT,
SUIUSDT, NEOUSDT, ZECUSDT, DASHUSDT, IOTAUSDT, YFIUSDT`

Символ, якого немає на біржі або `< train+test+10` барів — skip, не заміна
іншим «кращим» іменем.

Пропуск **до** повторного прогону (не після метрик): `EOSUSDT` (немає USDT-M
лістингу), `FTMUSDT` і `MKRUSDT` (валідатор 1d відхилив 3 спайки / ~0.14%).
NEW = 27 імен.

---

## Вердикт → paper

| Результат | Дія |
|---|---|
| H1 PASS | paper-моніторинг long-only на CORE∪NEW (або CORE, якщо H2 гірший) |
| H1 FAIL, H2 PASS | лишаємо CORE_15 monitoring; NEW не додавати |
| H3 PASS | окремий 4h-рукав у monitoring, не заміна 1d |
| Усі FAIL | не paper; причина в звіті iter11 |

pairs_arb LINK/BTC 1h **не чіпаємо** — вже validated.

## Вердикт після прогону (2026-09-12)

| Гіпотеза | Результат |
|---|---|
| H1 NEW_27 | **FAIL** (t_NW +1.70, CI містить 0) — нові імена не в paper |
| H2 FULL_42 | PASS* лише бо CORE тягне t-стат — не розширювати книгу |
| H3 CORE 4h long-only | **PASS** — 4h-рукав у monitoring |
| H0 CORE 1d long-only | **PASS** — реплікація, CI [+0.49, +2.25] |
| Paper | pairs_arb LINK/BTC 1h + ts_momentum CORE_15 1d + CORE_15 4h |
