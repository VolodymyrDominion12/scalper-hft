# План: отримання та використання даних стакана (L2 / depth5) для HFT-досліджень

Дата: 2026-09-07 · Статус: ПЛАН (очікує рішень користувача + доступу до VPS)
Контекст: `market_maker` та depth-weighted `ob_imbalance` — єдиний шлях до
HFT-класу alpha, але не валідуються без історичної глибини стакана
(див. `docs/STRATEGY_STATUS.md` — «Очікують даних/інфраструктури»).

## Що вже є в коді (перевірено)

| Компонент | Файл | Стан |
|---|---|---|
| WS-рекордер bookTicker | `scalper_hft/live/bookticker_recorder.py` | ✅ reconnect-цикл, flush буфера у `finally` |
| WS-рекордер **depth5** | той самий, `_record_depth()` | ✅ 5 рівнів, розгорнуті колонки `bid1..5, ask1..5 (+_qty)`, ts = час події |
| CLI | `scalper_hft.cli record-bookticker --symbol S --minutes M --depth` | ✅ |
| Безперервний цикл | `scripts/record_loop.sh SYMBOL MIN` | ✅ (одна сесія на запуск) |
| systemd на VPS | `deploy/scalper-record.service` | ⚠ записує **лише BTCUSDT**, `User=volodymyr`, WorkingDirectory = локальний шлях (на VPS шлях інший!) |
| Дані | `data/{SYMBOL}_depth5.parquet`, `{SYMBOL}_bookTicker.parquet` | ⚠ на цій машині НЕМАЄ (перевірено) |

**Головна невідомість (поза моїм доступом):** стан на VPS — чи працює сервіс,
скільки даних накопичено, чи вірний шлях у юніті. Це треба перевірити вам.

## Мета даних

- **Ядро досліджень**: BTCUSDT (якір) + LINKUSDT (ядро пар) + XRPUSDT/ETHUSDT
  (об'єм для OB-досліджень) — depth5, ~100 мс.
- **Використання**: (a) depth-weighted imbalance → напрямковий сигнал на 1s–1m;
  (b) реальний спред/глибина → чесний event-бектест `market_maker`;
  (c) spread dynamics → покращення CostModel/філ-моделей.

## Фаза 0 — Аудит поточного стану на VPS (робить користувач або дає доступ)

```bash
# на VPS (у каталозі scalper-hft)
systemctl --user status scalper-record   # або sudo systemctl status scalper-record
du -sh data/*depth5.parquet 2>/dev/null; ls -la data/*depth5* data/*bookTicker* 2>/dev/null
# скільки днів/записів:
.venv/bin/python -c "import pandas as pd,glob;
for f in glob.glob('data/*depth5*.parquet'):
    d=pd.read_parquet(f); print(f, len(d), d.ts.min(), d.ts.max())"
df -h   # диск
```
Критерій виходу: знаємо символи, покриття у часі, обсяг, швидкість росту.

## Фаза 1 — Розширення запису (код, деплой)

1. **Кілька символів паралельно** — depth5 для 4 символів:
   - Варіант A (рекомендовано): systemd **template-юніт** `scalper-record@.service`
     (`ExecStart=.../scripts/record_loop.sh %i 1440`), включити для
     `BTCUSDT`, `LINKUSDT`, `XRPUSDT`, `ETHUSDT` — по процесу на символ.
   - Варіант B: розширити `_record_depth` на мультиплекс кількох символів в
     одному процесі (менше процесів, більше коду).
2. **Виправити шлях юніта під VPS** (у `deploy/scalper-record.service`
   WorkingDirectory/ExecStart зараз локальний шлях машини розробника).
3. **Розмір файлів/ротація**: parquet дозаписується; на ~1–2 тижні перевірити
   розмір і за потреби додати daily-ротацію (`{SYM}_depth5_YYYYMMDD.parquet`)
   — зараз один файл на символ росте вічно.
4. Тести: `pytest tests/ -q` зелений після змін; локально 2-хвилинний
   smoke-запис depth5 для одного символу (потрібен мережевий доступ до Binance).

## Фаза 2 — Синхронізація даних VPS → research-машина

- `scripts/sync_depth.sh` (rsync по ssh): `rsync -avz vps:…/data/*depth5*.parquet ./data/`
- daily cron/CI-крок; перевірка цілісності (ts монотонний, дедуп за ts).
- Після цього depth5 з'являється локально → можна будувати датасети.

## Фаза 3 — Патлайн снапшот → бари/фічі (код)

- `scalper_hft/data/` або `features/microstructure.py` (розширення):
  - валідація: пропуски/дублікати ts, викиди (qty≤0, px≤0);
  - **depth-weighted imbalance** на кожному снапшоті:
    `imb_dw = (Σ w_i·bid_qty_i − Σ w_i·ask_qty_i)/(Σ w_i·(bid+ask)_qty_i)` (w — ваги рівнів);
    + spread_bps, book pressure тощо;
  - агрегація у бари (1s/5s/1m): last/first/mean imbalance, мін/макс спред.
- Тест без lookahead (фічі лише ≤ t).

## Фаза 4 — Дослідження (цикл самовдосконалення)

1. `ob_imbalance` (depth-weighted) на 1s–1m барах з реальним imbalance:
   бектест → WF → DSR (порівняти з top-of-book версією).
2. `market_maker` event-бектест з виміряним спредом/глибиною замість наближень
   з діапазону бару; sensitivity до філ-моделі.
3. Spread dynamics → калібровка `CostModel`/adverse selection.
4. Валідовані клітинки — у ядро «сутності» за правилами
   `docs/reports/entity_recommendation.md`.

## Фаза 5 (альтернатива/прискорення) — купити історію

- **Tardis.dev** (futures depth5/крокові дані, є безкоштовні семпли),
  Databento тощо — дає РЕТРОСПЕКТИВНУ глибину за місяці/роки одразу,
  замість очікування накопичення тижнями. Ціна ~$ за символи/періоди.
- Рішення: якщо мета — швидкий HFT-пілот, купити історію BTC/LINK depth5
  за ~3–12 місяців і почати Фазу 3–4 негайно; рекордер продовжує писати
  live-хвіст.

## Рішення, які потрібні від користувача

1. Доступ/підтвердження стану VPS (Фаза 0) — я сам VPS не бачу.
2. Набір символів для запису (рекомендую: BTCUSDT, LINKUSDT, XRPUSDT, ETHUSDT).
3. Терміновість: чекати накопичення (тижні) чи купити Tardis (Фаза 5).
4. Чи можна вносити зміни в `deploy/scalper-record.service`/`scripts/record_loop.sh`
   (деплой на VPS робиться тегом, за DEPLOY_PLAN).

## Ризики та примітки

- Binance futures `depth5@100ms` — публічний потік, ключів не треба.
- Обсяг: орієнтовно 10⁵–10⁶ снапшотів/день/символ → parquet ~50–200 МБ/день/символ;
  рахувати по Фазі 0 і стежити за диском VPS.
- Формат futures depthUpdate (`b`/`a` — тільки змінені рівні) вже оброблений у
  рекордері, але це **дельти**, не повні снапшоти: для досліджень потрібен
  реконструйований стан стакана (початковий снапшот REST + застосування дельт)
  — перевірити, чи потік `depth5@100ms` дає повні топ-5 (Binance futures
  надсилає повні топ-5 у кожному повідомленні цього стріму) — уточнити на етапі
  валідації даних.
- Годинники: ts = час події (UTC, naive) — вже виправлено в рекордері.
