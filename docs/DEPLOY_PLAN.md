# План: research локально, робот на VPS

Стан на 2026-09-03. Мета — розділити два рантайми: локальна машина лишається
дослідницькою, VPS крутить лише заморожений git-тег. Валідований кандидат
лишається **pairs_arb 1h maker**. Live з реальними коштами — лише після
Paper-Gate (≥8 тижнів) і явного запиту.

Контекст дослідження: попередній розбір практик (один репо, два середовища,
hotfix від running tag, drain між барами). Цей документ — робочий план
імплементації, не нові альфи.

Паралельний план коду live-шару: [CHANGE_PLAN.md](CHANGE_PLAN.md).
Роадмап: [ROADMAP.md](ROADMAP.md).

## Принципи

1. Один репозиторій, два рантайми. Локально ніколи `DRY_RUN=false`. VPS ніколи
   не ганяє Optuna / walk-forward / нові альфи.
2. На VPS — checkout **тегу**, не `git pull origin main` і не rsync робочого дерева.
3. `.env` на VPS не в git і не перезаписується деплоєм. Ключі Binance: IP
   whitelist лише VPS, futures trade, без withdraw. Локальні ключі — інші або відсутні.
4. Для 1h maker рестарт між барами нормальний. Hot-reload (`importlib`) і
   Kubernetes — поза скоупом.
5. Один PR = одна фаза. Кожна фаза: тести → `uv run pytest tests/ -q` →
   `uv run ruff check --fix && uv run ruff format`.
6. Класи змін на вже запущеному боті:
   - **A інфра** (логи, Telegram) — рестарт, позиції тримати;
   - **B execution/risk** — drain pending, рестарт, гідратація, reconcile;
   - **C альфа/параметри** — не hotfix; новий paper-тег і повторний gate.

## Поза скоупом (свідомо не чіпати)

- Live з реальними коштами всередині фаз 0–1 (спочатку paper-демон на VPS).
- Docker/K8s для бота (Docker лишається для Postgres research-кешу).
- Colo / субмілісекундна латентність (1h maker цього не потребує).
- Відкритий Streamlit / pgAdmin в інтернет.
- Нові альфи, зміна z/lookback, увімкнення Kalman/chase як дефолту.
- Flatten позицій на кожен деплой (дорого і ламає paper-статистику).

---

## Фаза 0 — Persist стану + daemon + control plane (P0, код)

**Навіщо.** Зараз `PaperAccount` лише в RAM, `paper-run-pairs` закінчується після
`--iterations`, `PairsPaperRunner._last_ts` губиться. `systemd Restart=always`
обнулить книгу. Без цього VPS бреше після кожного рестарту.

**Файли**

- `scalper_hft/live/account.py` — серіалізація `Position` / cash / realized_pnl /
  consecutive_losses / day_start_equity
- `scalper_hft/live/store.py` — таблиця `account_snapshot` (JSON або колонки) +
  `last_bar_ts` у `meta`
- `scalper_hft/live/control.py` — **новий**: читання `control.json` на початку
  `step` (`pause` / `no_new_entries` / `flatten` — flatten лише явний)
- `scalper_hft/live/pairs_runner.py` — restore на `__init__`, save після успішного
  `step`, SIGTERM → save + вихід 0, daemon-цикл до наступного bar close
- `scalper_hft/cli/paper.py` — `paper-run-pairs --daemon` (ігнорує скінченні iterations)
- `tests/test_store.py` / `tests/test_live.py` — restore після «рестарту» в тесті;
  control flag блокує entry, не exit

**Поведінка**

| Подія | Дія |
|---|---|
| Старт процесу | `PaperStore.load_snapshot()` → гідратація рахунку; якщо знімка немає — новий рахунок (як зараз) |
| Успішний `step` | атомарний знімок (позиції, cash, pending engine, `_last_ts`) |
| Той самий бар | `hold:same_bar` як зараз, знімок не обов'язковий |
| `control.json` → `no_new_entries` | виходи дозволені, нові входи заборонені |
| `control.json` → `pause` | `step` no-op, процес живий (heartbeat) |
| SIGTERM / SIGINT | знімок, лог, exit 0 (щоб systemd не вважав крешем) |

Сон демона: будити ~60–90 с до очікуваного close 1h-бара, не крутити `sleep 300`
наосліп. Публічні klines, без live-ключів.

**Готово коли:** тест «step → destroy runner → новий runner з того ж sqlite»
відновлює позиції, cash і не дублює бар; `pause` не ставить входи; pytest зелений.

---

## Фаза 1 — systemd-юніт paper + реліз тегом (P0, інфра)

**Навіщо.** Рекордер уже має `deploy/scalper-record.service`. Бота немає.
Без юніта + linger paper-gate на VPS не пережиє logout/reboot.

**Файли**

- `deploy/scalper-paper-pairs.service` — user-юніт, шаблон з плейсхолдерами
  `WorkingDirectory=` / `EnvironmentFile=`
- `deploy/README.md` — встановлення linger, checkout тегу, **не** чіпати `.env`
- `scripts/deploy_paper.sh` — на VPS: `git fetch --tags && git checkout "$TAG"`
  + `systemctl --user restart scalper-paper-pairs` (не pull main)
- `scalper_hft/live/telegram.py` виклик при старті: tag/SHA, `DRY_RUN`, mode=paper
- `.env.example` — коментар: VPS `.env` окремий, не копіювати з ноута

**Юніт (контракт)**

```text
Type=simple
Restart=always
RestartSec=10
EnvironmentFile=%h/scalper-hft/.env
WorkingDirectory=…  # лише checkout тегу
ExecStart=uv run python -m scalper_hft.cli paper-run-pairs --portfolio --daemon --notify
```

`KillSignal=SIGTERM`, `TimeoutStopSec=30`. `.env` на сервері: `DRY_RUN=true`.

**Git-контракт релізу**

```text
main              — кандидат після тестів; не автодеплой
experiment/…      — лише локально
hotfix/…          — гілка ВІД running tag, потім paper-vX.Y.Z+1
paper-vX.Y.Z      — іммутабельний зріз для VPS paper
live-vX.Y.Z       — той самий SHA після Paper-Gate (фаза 3)
```

Перший реліз: після зеленої фази 0 — тег `paper-v0.1.0`.

**Готово коли:** на чистій машині (або локально як dry-run юніта) процес
переживає `systemctl --user restart` без обнулення sqlite-книги; Telegram
показує SHA; `deploy_paper.sh` не чіпає `.env`.

---

## Фаза 2 — VPS paper 8 тижнів (операції, не код альфи)

**Навіщо.** Це і є Paper-Gate з [ROADMAP.md](ROADMAP.md). Код фази 0–1 має
крутитись безрозривно. Локально паралельно йде research на інших гілках.

**Чекліст машини**

- [ ] Окремий користувач (не root), SSH лише ключі
- [ ] `loginctl enable-linger` (як у рекордера)
- [ ] chrony/NTP; диск під `results/paper_pairs.sqlite`
- [ ] Бекап sqlite (cron/rclone), не в git
- [ ] Binance ключі для paper/testnet **або** порожні ключі (симуляція на публічних klines)
- [ ] Telegram bot окремий або той самий chat, але старт-повідомлення з hostname
- [ ] Streamlit/pgAdmin не слухають `0.0.0.0` (або firewall drop)

**Операційний ритм**

| Коли | Що |
|---|---|
| Щотижня | `paper-audit` vs бектест; maxDD ≤ BT × 1.5 |
| Після crash/reboot | перевірити snapshot + `hold:same_bar` на першому кроці |
| Hotfix класу A/B | протокол фази 0 control + checkout нового **мінорного** тегу |
| Hotfix класу C | заборонено на цьому прогоні; новий paper з нуля або після flatten+документації |

**Вікно деплою 1h:** після `step` на закритому барі t і не в останні 2–3 хв години
(pending post-only).

**Готово коли:** ≥8 тижнів без розриву журналу; tracking error прийнятний;
немає «рестарт обнулив equity». Це **не** дозвіл на live — лише вихід з Phase 1 Gate.

---

## Фаза 3 — Live-адаптер ніг pairs (код, без увімкнення грошей)

**Навіщо.** `PairsPaperRunner` свідомо кидає, якщо `DRY_RUN=false` (M3). Paper
симулює філи локально; live має ставити реальні GTX/reduce-only на обидві ноги
і звірятись з біржею.

**Файли (орієнтир)**

- `scalper_hft/live/pairs_live.py` — **новий** адаптер: place/cancel/poll двох ніг
- `scalper_hft/live/orders.py` / `trader.py` — пере використати idempotent client
  order id, post-only вхід, reduce-only вихід; ніколи market на pairs
- `scalper_hft/live/reconcile.py` — на старті live: `fetch_positions` +
  `fetch_open_orders` → гідратація; порожній локальний рахунок заборонений, якщо
  на біржі є ноги
- `tests/test_pairs_live.py` — мок біржі, без мережі

**Boot live (коли колись увімкнуть)**

1. Exchange = source of truth.
2. Якщо drift — `KillSwitch`, не торгувати.
3. Pending GTX підхопити або скасувати за правилами engine.
4. Telegram: tag, DRY_RUN=false, список позицій.

Дефолт CLI і `.env.example` лишаються `DRY_RUN=true`. Немає тихого шляху
ввімкнути live.

**Готово коли:** тести з моком проходять all-or-none / chase=taker / unwind;
`DRY_RUN=false` без адаптера як і раніше падає; адаптер не викликається з paper.

---

## Фаза 4 — Live-тег (лише після Gate + явного запиту)

**Навіщо.** Окремий іммутабельний зріз і окремі ключі. Не «перемкнути .env на
тому ж paper-процесі».

**Умови старту (усі обов'язкові)**

1. Фаза 2 (8 тижнів paper) закрита аудитом.
2. Фаза 3 в main, тести зелені.
3. Користувач явно просить live (правило AGENTS.md / algo-trading-safety).
4. Нові ключі: IP = VPS, без withdraw, окремі від ноута.
5. Тег `live-v0.1.0` = SHA перевіреного paper + адаптера.
6. Малий ноціонал (окремі `PAIR_NOTIONAL_PCT` у VPS `.env`, не в коді).

Окремий systemd-юніт `scalper-live-pairs.service` (не той самий, що paper),
щоб paper можна було лишити як тінь або вимкнути свідомо.

**Готово коли:** юніт стартує, reconcile ok, Telegram з `live-v*` і DRY_RUN=false.
Ця фаза **не планується в календар**, доки фаза 2 не закрита.

---

## Порядок

```text
0   persist + daemon + control.json     ← блокер будь-якого VPS
1   systemd paper + tag paper-v0.1.0    ← блокер 8 тижнів
2   VPS paper-gate (операції)           ← паралельно з локальним research
3   live adapter ніг (код, DRY_RUN)     ← можна кодити під час фази 2
4   live-v* + ключі                     ← лише після 2 і явного запиту
```

Локальне дослідження **не чекає** фаз 0–4: гілки `experiment/…` як зараз.
На VPS вони не потрапляють, бо деплой іде тегом.

## Протокол hotfix (після фази 1)

```text
1. git checkout -b hotfix/… paper-vX.Y.Z     # НЕ від latest main
2. фікс + тести
3. git tag paper-vX.Y.Z+1 && push --tags
4. control.json → no_new_entries
5. дочекатися idle (немає in-flight двох ніг); не :59 години
6. на VPS: scripts/deploy_paper.sh paper-vX.Y.Z+1
7. reconcile / snapshot ok → зняти no_new_entries
8. cherry-pick у main
```

Відкат = checkout попереднього тегу, той самий drain.

## Критерій «план виконано»

- [x] Фаза 0: snapshot/restore; `--daemon`; control flags; SIGTERM save
- [x] Фаза 1 (код): `deploy/scalper-paper-pairs.service`; SHA в Telegram на старті; `.env` не в релізі
- [ ] Фаза 1 (реліз): тег `paper-v0.1.0` і checkout на VPS
- [ ] Фаза 2: 8 тижнів paper на VPS; `paper-audit` зелений
- [ ] Фаза 3: live-адаптер ніг під моком; paper як і раніше без реальних ордерів
- [ ] Фаза 4: не стартує без явного запиту і закритої фази 2
- [ ] `uv run pytest tests/ -q` зелений після кожної кодової фази
- [ ] На VPS немає `git pull origin main`; `.env` не в релізі

## Стан коду (2026-09-03)

Фази 0–1 у репозиторії: persist `PaperAccount`, `--daemon`, `control.json`,
systemd-юніт і `scripts/deploy_paper.sh`. Наступне — тег `paper-v0.1.0` і
paper-gate на VPS (фаза 2). Не вмикати live.
