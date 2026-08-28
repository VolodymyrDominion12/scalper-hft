# TODO / Backlog (стан після ітерації 3)

## ✅ Зроблено в ітерації 3
- [x] **funding_carry + trend-фільтр** (`trend_block`): аудит показав, що фільтр
      ПОГІРШУЄ carry на цьому вікні (шорт під час ралі збирає більше фандінгу,
      ніж втрачає на ціні) → параметр за замовчуванням вимкнено (1.0)
- [x] **Paper-прогін** (`paper-run`): циклічний трейдер зі свіжими даними,
      збереженням equity/угод у `results/`, опцією `--notify`
- [x] **Telegram-сповіщення** (`live/telegram.py`): ключі з trade-bots/.env,
      валідність підтверджена (getMe OK, тестове повідомлення надіслано)
- [x] **Валідація OB imbalance на реальних даних** (docs/ob_validation.md):
      top-of-book imbalance вкрай біполярний (83% |imb|>0.5); depth5 стабільніший;
      потрібне накопичення годин запису
- [x] **Рекордер depth5** (5 рівнів стакана, `--depth`) з автоперепідключенням
- [x] **Streamlit-дашборд** (`scalper_hft/dashboard.py`): кеш, бектест, equity,
      Deflated Sharpe, paper-run результати
- [x] **Виправлення таймзон**: рекордер пише UTC; `_utc_now()`; `.value` замість
      `.timestamp()` (була помилка 3 год у epoch-конверсії); `_interval_ms` (був зламаний);
      свіжість кешу klines залежить від інтервалу (2 бари)

## Дані (data)
- [ ] Безперервне накопичення bookTicker/depth5 (systemd/cron) — дні/тижні
- [ ] 1s/5s свічки для реалістичних maker-філів
- [ ] Tardis.dev tick-дані (опційно, для аудиту філів nautilus_trader)

## Стратегії
- [ ] `WeightedDepthImbalance` фіча (зважена глибина 5+ рівнів + EMA) — після накопичення depth5
- [ ] OB-аудит на накопичених даних за стандартною процедурою
- [ ] funding_carry: paper-прогін на тиждень + порівняння з бектестом
- [ ] Market maker: калібрування adverse_sel_haircut на depth5 даних
- [ ] Delta-neutral funding arb (перп + спот) — потребує спотового API

## Live
- [ ] Live-режим з валідними Binance-ключами (поточні ключі невалідні)
- [ ] Maker-ордери post-only з контролем інвентаря
- [ ] Persistence угод у SQLite
- [ ] asyncio-цикл з ccxt.pro (klines + bookTicker + user stream)

## Інфраструктура
- [ ] GitHub Actions: щотижневий аудит стратегій на свіжих даних
- [ ] systemd-юніт для рекордера bookTicker/depth5

## Відомі проблеми
- [ ] BookTicker стрім іноді закривається сервером (реконект додано; слідкувати)
- [ ] streamlit dashboard.py виконує код при імпорті (стандартний патерн streamlit,
      але ускладнює юніт-тестування)
