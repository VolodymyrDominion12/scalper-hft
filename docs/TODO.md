# TODO / Backlog (стан після ітерації 5)

## ✅ Зроблено в ітерації 5
- [x] **Delta-neutral funding arb** (перп+спот): спот-дані (BinanceClient market_type),
      стратегія `funding_arb`, рушій `backtest/delta_neutral.py` (basis PnL +
      funding 1 раз/8h + 2-leg комісії, maker_execution flag), CLI `arb`
- [x] **Аудит arb (30 днів, 3 символи)**: taker — ret −1.6…−2.2%; maker (post-only) —
      −0.38…−0.50%; funding внесок +0.12…+0.17% при 50% ноціоналу → **відхилено**:
      фандінг ~5%/рік не покриває тертя; перспектива — високий фандінг-режим/ребейти
- [x] **Накопичення depth5**: 10k+ снапшотів (74+ хв) → WeightedDepthImbalance
      (|extreme| 46% vs 83% top-of-book); OB-бектест на 63 барах з реальним
      imbalance — плагінг валідовано, потрібні години даних
- [x] **Тести**: +1 delta-neutral облік — всього **16**

## Дані (data)
- [ ] Запустити systemd scalper-record (накопичення bookTicker/depth5, дні/тижні) —
      інструкція: deploy/scalper-record.service
- [ ] Спот-кеш: перевірити розширення понад 30 днів (для довших арб-бектестів)

## Стратегії (чесний статус після аудитів)
- [ ] funding_arb: сценарій високого фандінгу (історичні періоди >10% річних)
      та/або VIP-ребейти; maker-виконання обов'язкове
- [ ] OB depth-weighted: після накопичення годин depth5
- [ ] Maker post-only live-виконання (потрібні ключі)
- [ ] Ідея: basis-торгівля (перп премія до споту як окремий сигнал)

## Live
- [ ] Live-режим з валідними Binance-ключами (поточні ключі невалідні)
- [ ] Persistence угод у SQLite
