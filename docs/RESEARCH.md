# Research: HFT / Скальпінг на Binance USDT-M (2023–2025)

Зведення web-дослідження (3 напрямки: практики HFT, Python-фреймворки, біржі).

---

## 1. Що реально працює для retail

- **Maker-based market making** — єдиний систематичний retail-edge, і лише за низьких
  комісій/ребейтів. Binance USDT-M: 0.02% maker / 0.05% taker; BNB-дисконт ~10%
  (→ 0.018%/0.045%); VIP-рівні до негативних maker-комісій (rebates).
  Джерела: [Binance VIP Portal](https://www.binance.com/en-GB/vip-portal/liquidity-hub),
  [Paradex: scalping fees](https://paradex.trade/blog/scalping-fees-shouldnt-kill-your-edge).
- **Round-trip math**: taker–taker ≈ 0.10% ноціоналу (з 10× плечем = ~1% капіталу за цикл);
  maker–maker ≈ 0.04%. При ~0.03% типовому скальп-цілі BTC — taker-скальпінг майже
  неможливий без сильного направленого руху.
- **Funding-rate arbitrage** (довгий спот / короткий перп або дельта-нейтральні пари) —
  найдоступніший retail-edge, бо збирає фандінг, а не тіки.
  Джерело: [Elsevier: RL market making with funding dynamics](https://www.sciencedirect.com/science/article/pii/S240591882600022X).
- **Cross-exchange арбітраж** закритий для retail (latency-bound).
  Джерело: [Volity: Why Retail Cannot Compete](https://volity.io/crypto/hft-crypto-trading/).

## 2. Ключові мікроструктурні сигнали

- **Order book imbalance** — дисбаланс глибини bid/ask прогнозує напрямок на секундах.
  [KuCoin explainer](https://www.kucoin.com/knowledge-base/Analysis/what-is-order-book-imbalance-in-crypto).
- **Trade flow / CVD** — кумулятивний об'ємний дельта (агресивні покупці/продавці);
  комбінується з VWAP та детекцією великих трейдів.
  [CVD guide](https://incrypted.com/en/how-use-cumulative-volume-delta/),
  [Elite-Metrics-Trade-Bybit](https://github.com/nssanta/Elite-Metrics-Trade-Bybit).
- **Spread/depth dynamics**, **funding rates** (перегріті лонги), **liquidation cascades**.
  [BitMEX: State of Crypto Perps 2025](https://mainnet.fxrbb.com/blog/site-announcement/state-of-crypto-perps-2025).

## 3. Архітектура

- **Event-driven WebSockets, не REST-полінг** — кілька символів за одне з'єднання, менша латентність.
  [CoinAPI](https://www.coinapi.io/blog/why-websocket-multiple-updates-beat-rest-apis-for-real-time-crypto-trading).
- **Тікові/depth-стріми, а не 1s/5s klines** — свічки вбивають мікроструктуру.
- **Latency з дому: ~50–200 ms** — retail не виграє тікові перегони; стратегії мають жити на горизонті
  секунди–хвилини. [Kraken colocation (2025)](https://www.coindesk.com/business/2025/03/14/kraken-to-offer-superfast-trading-with-planned-launch-of-colocation-service).
- **Python-стек**: asyncio + `websockets`/`ccxt.pro`, heartbeat/reconnect, монотонні годинники, Polars/numpy.

## 4. Критичні пастки

- **Fee drag** — більшість бектестів занижує комісії та ігнорує гіршу ціну філла.
  [dev.to: fee assumption](https://dev.to/jacktrader/the-fee-assumption-quietly-wrong-in-most-backtests-1jpa).
- **Slippage** — market-ордери перетинають спред; на волатильних віках філл систематично гірший.
- **Adverse selection** — HFT-фірми котирують всередині застарілого погляду retail; лімітки
  retail "з'їдаються" у несприятливий бік.
- **Wash trading / фейкові об'єми** — ставитись до volume скептично.
  [DeFiLlama halts data](https://investor.wedbush.com/wedbush/article/breakingcrypto-2025-10-7-defillama-halts-astar-data-amid-wash-trading-scandal-rocking-defi-transparency).
- **Overfitting** — багатопараметричні тікові стратегії підганяються під шум.

## 5. Реалістичні очікування

- Прибутковість crypto fat-tailed: швидкі трейдери втрачають **220× більше** за повільних через куртозис.
  [xMoney](https://www.xmoney.com/blog/cryptokurtosis-why-fast-crypto-traders-lose-220x-more).
- Досяжний retail-edge: **кілька bps на угоду після комісій**, win rate 40–55%,
  прибуток — через асиметричний R:R і малий ризик на угоду.
  [BloFin scalping](https://blofin.com/academy/education/crypto-scalping),
  [arXiv market-neutral](https://ar5iv.labs.arxiv.org/html/2405.15461).
- **Ризик розорення**: малі частки позиції, жорсткий стоп на серію збитків; високий леверидж
  у скальпінгу = негативна лотерея для більшості.

---

## 6. Python-фреймворки (оцінка для HFT/скальпінгу)

| Фреймворк | HFT-придатність | Сильні сторони | Слабкі сторони |
|---|---|---|---|
| **nautilus_trader** | 9/10 | Rust-ядро, наносекундний event-driven бектест, L2/order-book, **Binance futures adapter** для live | Крута крива навчання, все збираєш сам |
| **ccxt / ccxt.pro** | 8/10 (шар) | Стандарт абстракції бірж, websocket-стріми, живий проект | Це транспорт, не стратегії |
| **vectorbt** | 6/10 | Найшвидші параметричні свати на OHLCV | Немає tick/order-book, немає live |
| **hummingbot** | 6/10 | Бойовий market-making з Binance | Слабкий бектестер |
| **freqtrade + FreqAI** | 5/10 | Кращі docs/community, live+futures, rolling ML | Тільки OHLCV-філи, не субсекундні |
| **backtrader / backtesting.py / jesse / OctoBot** | 3–5/10 | Простота | Бар-моделі, немає HFT-фіч |

Джерела: [vectorbt](https://github.com/polakowo/vectorbt), [nautilus Binance](https://nautilustrader.io/binance/),
[ccxt PR #28091](https://github.com/ccxt/ccxt/pull/28091), [freqtrade](https://pypi.org/project/freqtrade/2024.8/).

## 7. Анти-перенавчання (бібліотеки)

- **purgedcv** — purging/embargo, combinatorial purged CV, deflated Sharpe.
  [PyPI](https://pypi.org/project/purgedcv/), [GitHub](https://github.com/eslazarev/purged-cross-validation).
- **oos-lab** — PSR, Deflated Sharpe, PBO/CSCV, walk-forward.
  [GitHub](https://github.com/OutOfSampleLab/oos-lab).
- **mlfinlab/mlfinlib** — CPCV, deflated Sharpe, triple-barrier labeling (López de Prado).
  [combinatorial.py](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py).
- **Optuna** — TPE-пошук з walk-forward цільовою функцією.
- **sklearn TimeSeriesSplit** — базовий; без purging/embargo (в нашій системі — `validation/cv.py`).

## 8. Вибір біржі (для Python-скальпера)

| Біржа | Maker/Taker | Коментар |
|---|---|---|
| **Binance USDT-M** ✅ | 0.02%/0.05% (BNB → 0.018/0.045) | Найнижчі ефективні комісії, найглибші стакани, найкращі API-доки, безкоштовний testnet і historical dumps, працює в Україні |
| Bybit | 0.02%/0.05% | Запасний варіант для EEA (MiCA), відмінний v5 API |
| OKX | 0.02%/0.05% | Демо-торгівля, 240 WS-каналів |
| Hyperliquid | 0.045%/0.10%, maker rebates до −0.0125% | No-KYC, повний стакан on-chain, погодинний фандінг, тонші стакани |
| Kraken Futures | 0.02%/0.05% | Регульована EU-альтернатива |
| Coinbase / Kraken spot | 0.40–1.20% RT | Скальпінг неможливий (комісії) |

**Висновок:** Binance USDT-M futures + ccxt (або binance-connector). Для maker-ребейтів без KYC —
Hyperliquid. Деталі: [Binance FAQ](https://www.binance.com/en/support/faq/detail/98488a516eb84e3eb34605683dffd554),
[Hyperliquid fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees).

## 9. Дані

- **Безкоштовно**: [Binance public dumps](https://github.com/ltftf/binance-historical-data)
  (data.binance.vision — klines, trades, aggTrades, depth snapshots),
  [CryptoDataDownload](https://www.cryptodatadownload.com/about/).
- **Платно (tick-level)**: [Tardis.dev](https://docs.tardis.dev/python-client/quickstart.md) — стандарт
  для серйозних tick-бектестів; Databento, Kaiko, Amberdata.
- **Testnet**: Binance/Bybit testnet — лише для інтеграційного тестування; стакани розріджені,
  філи нереалістичні → **не** використовувати для валідації edge
  ([Binance dev discussion](https://dev.binance.vision/t/suggestion-for-testnet-liquidity-improvement/2187/5)).

## 10. Фінальний стек scalper-hft

- **Дані**: ccxt REST (klines/aggTrades/funding) → parquet-кеш; live — ccxt.pro/WS (roadmap).
- **Бектест**: власний векторизований + подієвий (цей проект); для серйозного L2 — nautilus_trader.
- **Валідація**: власні walk-forward / purged CV / deflated Sharpe / sensitivity (цей проект)
  + purgedcv/oos-lab для перехресної перевірки.
- **ML**: LightGBM walk-forward + hmmlearn (roadmap).
- **Live**: ccxt, paper/testnet за замовчуванням.
