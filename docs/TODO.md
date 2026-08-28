# TODO / Backlog (наступні ітерації циклу інвестігейт→реалізація→тест→аудит)

## Дані (data)
- [ ] Історичні aggTrades > 2 діб через data.binance.vision (безкоштовні dumps)
- [ ] Запис bookTicker (best bid/ask) у реальному часі → parquet, для OB-стратегій
- [ ] 1s/5s свічки для реалістичних maker-філів
- [ ] Tardis.dev tick-дані (опційно, для аудиту філів nautilus_trader)

## Стратегії
- [ ] Налаштувати mean_reversion під 1m/5m (поточні дефолти надто консервативні)
- [ ] CVD-стратегія: додати фільтр режиму (тренд/флет) через features/regimes.py
- [ ] OB imbalance: перевірити на реальних снапшотах стакана (після запису)
- [ ] Market maker: калібрування adverse_sel_haircut на L2-даних
- [ ] Funding-rate стратегія (delta-neutral perp pair) — доступний retail-edge

## Валідація
- [ ] Combinatorial Purged CV (повна CSCV-оцінка PBO, не тільки бутстреп)
- [ ] Triple-barrier labeling для ML (mlfinlab/mlfinlib)
- [ ] Статистична значущість між стратегіями (множинне порівняння)

## Live
- [ ] asyncio + ccxt.pro WebSocket-цикл (klines + bookTicker + user stream)
- [ ] Maker-ордери post-only з контролем інвентаря (дослідження: maker edge)
- [ ] Telegram-сповіщення (ключі є у trade-bots/.env)
- [ ] Persistence угод у SQLite

## Інфраструктура
- [ ] Streamlit-дашборд (моніторинг стратегій, equity, параметрів)
- [ ] GitHub Actions: щотижневий аудит стратегій на свіжих даних
- [ ] .env з робочими ключами (поточні ключі невалідні — оновити на Binance)

## Відомі проблеми
- [ ] Sharpe на 1m-даних сильно ануалізований (sqrt(525600)) — додати
      "період-незалежний" варіант (напр. mean/std на годину)
- [ ] AggTrades download повільний (rate limit) — прогресивне збереження батчів
