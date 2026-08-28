# TODO / Backlog (стан після ітерації 2)

## ✅ Зроблено в ітерації 2
- [x] **funding_carry стратегія** — збір фандінгу на перекосах; DSR=1.0, PBO=0.000,
      позитивна на BTC/ETH/SOL (90 днів 1m) — перша стратегія, що пройшла аудит
- [x] **data.binance.vision** завантажувач історичних aggTrades (5.4M трейдів за тиждень)
- [x] **Виправлена інфляція Sharpe**: додано годинний Sharpe/Sortino (без ануалізації)
- [x] **Regime-фільтри** підключені до mean_reversion (trend_strength, volatility_regime)
- [x] **Крос-символьний аудит** funding_carry: BTC/ETH/SOL
- [x] **CSCV/PBO** (Combinatorial Purged CV, López de Prado) + CLI `cscv` + тести
- [x] **Рекордер bookTicker** (WebSocket) + CLI `record-bookticker` — 7k снапшотів/хв

## Дані (data)
- [ ] 1s/5s свічки для реалістичних maker-філів
- [ ] Tardis.dev tick-дані (опційно, для аудиту філів nautilus_trader)
- [ ] Прогресивне збереження aggTrades під час завантаження (зараз — одним файлом)

## Стратегії
- [ ] funding_carry → paper-торгівля (LiveTrader вже підтримує funding)
- [ ] Funding + price filter (не входити в шорт під час сильного аптренду — тренд-фільтр)
- [ ] OB imbalance: перевірити на накопичених bookTicker снапшотах
- [ ] Market maker: калібрування adverse_sel_haircut на L2-даних
- [ ] Delta-neutral funding arb (перп + спот) — потребує спотового API

## Валідація
- [ ] Triple-barrier labeling для ML (mlfinlab/mlfinlib)
- [ ] Статистична значущість між стратегіями (множинне порівняння)

## Live
- [ ] asyncio + ccxt.pro WebSocket-цикл (klines + bookTicker + user stream)
- [ ] Maker-ордери post-only з контролем інвентаря
- [ ] Telegram-сповіщення (ключі є у trade-bots/.env)
- [ ] Persistence угод у SQLite

## Інфраструктура
- [ ] Streamlit-дашборд (моніторинг стратегій, equity, параметрів)
- [ ] GitHub Actions: щотижневий аудит стратегій на свіжих даних
- [ ] .env з робочими ключами (поточні ключі невалідні — оновити на Binance)

## Відомі проблеми
- [ ] AggTrades download через REST повільний (rate limit) — перевагу віддавати
      data.binance.vision
- [ ] market_maker використовує барові high/low для філів — потрібні справжні L2
