# Конспект: "Inside the Black Box" — Rishi K. Narang (2-ге видання)

Практичні висновки для scalper-hft. Повний текст книги: `Inside the Black Box - Rishi K. Narang.pdf`.

## Архітектура квант-системи (гл. 2)

```
        ┌─────────────┐  ┌───────────┐  ┌───────────────────────┐
        │ Alpha Model │  │ Risk Model│  │ Transaction Cost Model│
        └──────┬──────┘  └─────┬─────┘  └──────────┬────────────┘
               └───────────────┼───────────────────┘
                               ▼
                   ┌────────────────────────┐
                   │ Portfolio Construction │
                   └───────────┬────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Execution Model    │
                    └──────────────────────┘
        Підмурок: Data (гл. 8) та Research (гл. 9)
```

У scalper-hft ця модель реалізована так:
- **Alpha Model** → `strategies/` (mean_reversion, cvd_momentum, ob_imbalance, market_maker);
- **Risk Model** → `live/trader.py` (ліміт позиції, денний ліміт збитків, серія збитків);
- **Transaction Cost Model** → `backtest/execution.py` (комісії + slippage + impact);
- **Portfolio Construction** → `position_pct` / `max_open_positions`;
- **Execution** → векторизований рушій (філ на відкритті наступного бару) та подієвий (maker);
- **Data** → `data/` (klines, aggTrades, funding, parquet-кеш);
- **Research** → `validation/` (walk-forward, purged CV, deflated Sharpe, sensitivity) + `ml/`.

## Гл. 3 — Alpha Models

- Теоретичні (trend, mean reversion, arbitrage) vs дата-драйвові (ML).
- **Mean reversion** купує перепроданість, продає перекупленість — **найменш чутливий до slippage**
  (книга прямо каже: покупка "проти руху" дає кращі філи). → наша стратегія `mean_reversion`.
- **Momentum/trend** страждає від slippage найбільше (купуєш те, що вже рухається). → наш
  `cvd_momentum` використовує потік заявок для входу на ранній стадії руху.

## Гл. 4 — Risk Models

- Два завдання: обмежити **обсяг** ризику (скільки) і **типи** ризику (чого уникати).
- Практика: фільтри режиму (`features/regimes.py`), денні ліміти збитків, ліміт позицій.

## Гл. 5 — Transaction Cost Models (критично для скальпінгу!)

Три компоненти:
1. **Комісії** — фіксовані, легко моделюються (Binance: maker 0.02%, taker 0.05%);
2. **Slippage** — зміна ціни між рішенням і виконанням; залежить від латентності та волатильності;
   mean-reversion стратегії страждають найменше, trend — найбільше;
3. **Market impact** — для retail-розмірів на глибоких стаканах Binance нехтовний.

> Найважливіше: якщо очікуваний прибуток менший за витрати — стратегія мертва незалежно від сигналу.
> У нашому `CostModel.breakeven_move_pct` це рахується явно.

## Гл. 7 — Execution

- **Aggressive vs passive**: aggressive (market) — гарантія філа, платиш спред; passive (limit) —
  економиш спред, але ризикуєш adverse selection і невиконанням.
- **Adverse selection**: твій лімітний ордер частіше заповнюється, коли це погано (ринок йде проти тебе).
- Бенчмарки: mid-market, VWAP.
- Для скальпінгу: якщо стратегія taker — потрібен великий рух; якщо maker — потрібен контроль
  інвентаря та швидкі скасування. → `backtest/event_engine.py` (maker) моделює це спрощено.

## Гл. 8 — Data

- Якість > кількість: clean data, консистентність, без lookahead у фічах.
- У нас: aggTrades (buy/sell флаг) → CVD; klines → індикатори; funding → фільтри. Все в parquet.

## Гл. 9 — Research: науковий метод (серце анти-перенавчання)

1. **Спостереження** → **теорія** → **наслідки** → **тестування (фальсифікація, Popper)**.
2. **In-sample (train)**: підбір параметрів; ризик — "відповіді в кінці підручника".
3. **Out-of-sample (test)**: єдиний чесний доказ генералізації.
4. Широта і довжина вибірки: більше даних = більше сценаріїв, але вищий ризик підгонки.
5. **Метрики "хорошої" моделі**: графік кумулятивного прибутку (якість кривої!), Sharpe,
   просідання, win rate, profit factor. → `backtest/metrics.py`.

## Гл. 10–11 — Ризики та критика квантів

- Model risk, regime change risk, exogenous shock, contagion.
- **"Quants Are Guilty of Data Mining"** (гл. 11): головна критика — data mining дає
  ілюзорні edge. Відповідь — сувора OOS-валідація та коригування на множинне тестування
  (deflated Sharpe у нас).

## Гл. 13–16 — High-Speed / High-Frequency Trading

- **Структура HFT**: CMM (contractual market making), **NCMM** (noncontractual — постиш обидві
  сторони, ризик adverse selection), арбітраж, fast alpha.
- **Чому швидкість має значення**: 3 операції — поставити пасивний ордер, зняти ліквідність,
  скасувати пасивний. Усі три чутливі до латентності.
- **Реалізм**: retail не може конкурувати на тіках; наші стратегії працюють на горизонті
  секунди–хвилини з мікроструктурними сигналами (imbalance, CVD), що є "fast alpha" у термінах книги.
- Гл. 16 — контроверсії HFT (фронт-раннінг, волатильність): важливо для розуміння ризиків
  конкуренції з професійними маркет-мейкерами.
