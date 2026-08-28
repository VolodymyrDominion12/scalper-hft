# TODO / Backlog (стан після ітерації 7)

## ✅ Зроблено в ітерації 7
- [x] **Статистичний арбітраж пар**: стратегія `pairs_arb` (z-score log-ratio),
      рушій `backtest/pairs.py` (2 перп-ноги, funding обох ніг з правильними
      знаками, 2-leg комісії, maker flag), CLI `pairs`
- [x] **Аудит пар (чесно)**:
      - BTC/ETH 1h 90д: +0.35% (maker) / −0.19% (taker), 9 угод — найстабільніша,
        але статистично слабка;
      - BTC/SOL, ETH/SOL: −3…−7% (відносний тренд SOL зламав реверсію);
      - BTC/ETH 1m: овертрейдинг −5…−31% (fee-drag) → **низькочастотний лише**
  - [x] **systemd user-юніт** scalper-record ВСТАНОВЛЕНО і ПРАЦЮЄ
        (~/.config/systemd/user/scalper-record.service, enabled) — depth5
        накопичується безперервно (Restart=always)
- [x] **Моніторинг фандінг-режиму** у weekly-audit workflow: авто-алерт у Telegram
      при вході у високий режим (>36% річних у >5% точок)
- [x] **Тести**: +1 pairs accounting — всього **19**

## Дані (data)
- [ ] Лінгер для systemd: `loginctl enable-linger volodymyr` (щоб сервіс працював
      після logout); зараз працює лише у сесії
- [ ] Спот-кеш 90+ днів для довших тестів

## Стратегії (чесний статус)
- [ ] pairs_arb: тільки BTC/ETH на 1h+, з maker-комісіями; потрібна більша
      статистика (рік даних) для висновку
- [ ] OB depth-weighted: після накопичення днів depth5 (сервіс працює!)
- [ ] funding/basis: регім-гейтинг (низький режим 2026 — сплять)
- [ ] Maker post-only live (потрібні валідні ключі)

## Live
- [ ] Live-режим з валідними Binance-ключами (поточні ключі невалідні)
- [ ] Persistence угод у SQLite
