# Роадмап scalper-hft

Стан на 2026-08-28 (після ітерацій 8–10). Цей документ — план розвитку після
аудиту коду, стратегій і live-шару. Детальний статус стратегій:
[STRATEGY_STATUS.md](STRATEGY_STATUS.md), пари — [pairs_audit.md](pairs_audit.md).

## Чесний вердикт

Проєкт — зріла **research-платформа** (бектест, walk-forward, DSR, CSCV, облік
комісій/фандінгу) з одним валідованим кандидатом: **pairs_arb на 1h, maker**.
Це **не** готовий HFT-скальпер. Taker-скальпінг на 1m відхилено fee-drag;
funding/basis сплять у низькому режимі 2025–26. Phase 0 (облік, закритий бар,
post-only, REGISTRY) закрито. Live все ще не готовий до грошей: немає 2-ніг
paper runner і моделі пропущених maker-філлів.

**Правило розгортання:** жоден live, доки paper pairs не пройде ≥8 тижнів без
розходження з бектестом.

---

## Phase 0 — P0 баги (закрито 2026-08-28)

`pairs_arb` у REGISTRY; ф'ючерсний облік `PaperAccount` (cash + UPNL, без
подвоєння PnL); сигнал/філл з закритого бару; close-before-flip + limit/post-only;
paper-replay реверсує і рахує daily-loss на mark-to-market. Тести:
`tests/test_live.py`.

| ID | Проблема | Статус |
|---|---|---|
| P0-1 | `pairs_arb` немає в REGISTRY | ✅ |
| P0-2 | `PaperAccount.equity` подвоює PnL | ✅ |
| P0-3 | Сигнал з формуючого бару | ✅ |
| P0-4 | Market-ордери + flip без close | ✅ |
| P0-5 | Replay не реверсує; risk на зламаному equity | ✅ |

---

## Phase 1 — Paper pairs (2–4 тижні)

Єдиний шлях до грошей. Бектест pairs уже є; live-циклу для 2 ніг **немає**.

1. **PairsPaperRunner**: дві ноги, z-score на закритому 1h барі, maker post-only
   на обох, рівні ноціонали (як у бектесті), funding обох ніг.
2. **Реалістичний філл**: post-only може не виконатись — таймаут, реквот, лог
   unfilled. Бектест зараз припускає 100% філл на close — це оптимістично.
3. **CLI** `paper-run-pairs --leg1 XRPUSDT --leg2 BTCUSDT --interval 1h --maker`.
4. **Ризик портфеля**: ≤30% ноціоналу на пару, ≤60% сумарно; стоп після 2
   місяців поспіль збитку (вже в STRATEGY_STATUS).
5. **Персистенція**: SQLite (угоди, ордери, equity, fill/reject) замість CSV.
6. **Дашборд**: XRP/LINK + секція pairs (зараз лише BTC/ETH/SOL і REGISTRY без pairs).
7. **Weekly-audit**: прибрати акцент з відхиленого `funding_carry`; ганяти
   валідовані пари + портфель; Telegram — OOS/DD, не «funding низький».
8. **Ключі**: валідні testnet, потім paper на mainnet (`DRY_RUN=true`).
9. **systemd linger**: `loginctl enable-linger` для depth-рекордера.

Критерій виходу: ≥8 тижнів paper, кореляція paper vs backtest PnL, розходження
філлів < порогу (зафіксувати в звіті), maxDD у межах бектесту × 1.5.

Стартовий портфель (з [pairs_audit.md](pairs_audit.md)):
**XRP/BTC + BTC/ETH + LINK/BTC**, рівні ваги, 1h, maker.

---

## Phase 2 — Production hardening (4–6 тижнів, після зеленого paper)

1. Звірка позицій з біржею кожен цикл (`fetch_positions`); kill-switch при
   розходженні.
2. Reduce-only на закритті; GTX/post-only на вході; ніколи market на pairs.
3. asyncio + ccxt.pro (або наявний WS) замість REST-полінгу — для 1h не
   критично, для OB/MM — обов'язково.
4. Telegram: fill, reject, daily PnL, risk-block, kill-switch.
5. Денний/тижневий ліміт збитків на **портфель**, не на одну ногу.
6. Hedge-ratio: зараз 1:1 через log-ratio. Додати rolling OLS/Johansen і
   порівняти OOS з поточною моделлю (не міняти live, доки OOS не кращий).
7. Модель пропущених філлів у бектесті (ймовірність філла від distance-to-mid)
   — щоб paper не був сюрпризом.

Live з реальним капіталом — лише явний запит і після Phase 1 gate.
Стартовий розмір: малий (напр. 5–10% цільового), scale-up за правилом.

---

## Phase 3 — Наступні альфи (паралельно з paper, не замість)

Не воскрешати відхилені 1m-стратегії. Нові ідеї лише після того ж циклу:
walk-forward → DSR → sensitivity → paper-replay.

| Тема | Умова старту | Навіщо |
|---|---|---|
| Depth-weighted `ob_imbalance` | ≥2–4 тижні depth5 | Єдиний шлях до справжнього скальпу; top-of-book біполярний |
| Market maker на L2 | Tardis або власний L2 + nautilus | Поточна OHLC-модель філлів нечесна |
| Funding/basis wake-up | weekly-audit: >5% точок >36% річних | Режимно-сплячі; код тримати, не крутити на 1m |
| Коінтеграційний універсум | скрипт скану пар (не ручний список) | XRP/LINK знайдені ітеративно; наступні пари — систематично |
| ML | triple-barrier + DSR на OOS, не accuracy | Поточний LightGBM — напрямок бару без витрат; для pairs майже не потрібен |

---

## Phase 4 — HFT-інфра (лише якщо Phase 1–2 живуть)

Retail-латентність 50–200 мс. Тіковий HFT з дому не виграється. Ця фаза —
якщо pairs працює в paper/live і з’являється мікроструктурний edge:

1. 1s/5s свічки + Tardis L2.
2. nautilus_trader як альтернативний L2-рушій (аудит філлів vs власний event engine).
3. Colocation / VPS ближче до Binance — лише під MM/OB, не під 1h pairs.

---

## Що не робити

- Не вмикати live на `mean_reversion` / `cvd_momentum` / `funding_carry` /
  `basis_reversion` / 1m pairs — усі відхилені.
- Не оптимізувати z/lookback на повній вибірці і не оголошувати «новий edge».
- Не збільшувати частоту pairs «щоб більше угод» — 1m уже вбитий fee-drag.
- Не ставити taker на pairs (аудит: taker гірший за maker).
- Не масштабувати ноціонал, доки paper-філли не збігаються з моделлю.

---

## Метрики прогресу

| Фаза | Головна метрика |
|---|---|
| 0 | Усі P0 закриті тестами; `cli pairs` і weekly-audit зелені |
| 1 | Paper vs backtest tracking error; % unfilled post-only; тижні без розриву обліку |
| 2 | Розходження позицій з біржею = 0; час до kill-switch |
| 3 | Новий кандидат: OOS>0, DSR, ≥100 угод або чесне «мало угод — лише paper» |
| 4 | Не починати без живого PnL з Phase 2 |
