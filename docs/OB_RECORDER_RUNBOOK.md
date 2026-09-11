# Рунбук: запуск збору даних стакана (depth5) — крок за кроком

Дата: 2026-09-07 · Статус: ПЕРЕВІРЕНО ЛОКАЛЬНО (live-запис працює)
Мета документа: що саме треба зробити, щоб запустити безперервний запис
bookTicker/depth5 (L1/L2) і як це має відбуватися. Це виконавчий план до
`docs/L2_DATA_PLAN.md` (там — фази досліджень і рішення).

---

## Крок 0. Передумови (перевірено на цій машині)

| Перевірка | Команда | Очікувано |
|---|---|---|
| Python + venv | `ls .venv/bin/python` | існує (3.12/3.13) |
| websockets | `.venv/bin/python -c "import websockets; print(websockets.__version__)"` | версія (тут 16.1.1) |
| Доступ до Binance WS | див. Крок 2 (smoke) | повідомлення depthUpdate |
| Ключі | **НЕ потрібні** — публічний потік | `.env` не чіпати |

Якщо `websockets` немає: `.venv/bin/pip install websockets` (або `uv sync --extra live`).

## Крок 1. Короткий тест запису (2 хвилини)

```bash
cd /home/volodymyr/PycharmProjects/scalper-hft
.venv/bin/python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 2 --depth
# Очікуємо: "BTCUSDT: записано ~1000+ depth5 снапшотів"
```

Перевірка даних:
```bash
.venv/bin/python - <<'EOF'
import pandas as pd
d = pd.read_parquet("data/BTCUSDT_depth5.parquet")
print("rows:", len(d)); print(d.head(1).to_string())
EOF
```
Очікувано: рядки з колонками `bid1..bid5, ask1..ask5 (+_qty)`, індекс `ts` (UTC),
ціни/обсяги реалістичні (BTC ~ $79k). **Результат на цій машині: 1057 снапшотів за 2 хв.**

## Крок 2. Безперервний запис на довгий термін

### Варіант A — локально (для початку; машина має бути ввімкнена 24/7)
```bash
# одна сесія на добу в циклі (перезапуск щодня), лог у файл:
nohup bash scripts/record_loop.sh BTCUSDT 1440 > logs_record_depth.log 2>&1 &
# або як systemd user-сервіс (автостарт):
mkdir -p ~/.config/systemd/user
# (адаптуйте WorkingDirectory/ExecStart у deploy/scalper-record.service під цю машину)
cp deploy/scalper-record.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now scalper-record
journalctl --user -u scalper-record -f   # логи
```

### Варіант B — VPS (рекомендовано для 24/7; робиться тегом за DEPLOY_PLAN)

**Куди класти юніт і що правити (відповідь на «куди перенести»):**

1. **Файл юніта кладеться в системний каталог systemd:**
   ```bash
   sudo cp deploy/scalper-record.service /etc/systemd/system/scalper-record.service
   ```
   (або на рівні користувача: `~/.config/systemd/user/scalper-record.service` —
   тоді без `sudo`, але сервіс працює лише поки користувач залогінений, якщо не
   ввімкнено `loginctl enable-linger $USER`).

2. **ОБОВ'ЯЗКОВО виправити шляхи у файлі ДО копіювання** — зараз там шлях
   машини розробника (`User=volodymyr`,
   `WorkingDirectory=/home/volodymyr/PycharmProjects/scalper-hft`,
   `ExecStart=…/scripts/record_loop.sh BTCUSDT 1440`). На VPS мають бути ваш
   користувач і каталог репо, напр.:
   ```ini
   User=<VPS_USER>
   WorkingDirectory=/home/<VPS_USER>/scalper-hft
   ExecStart=/home/<VPS_USER>/scalper-hft/scripts/record_loop.sh BTCUSDT 1440
   ```
   Заміна без ручного редагування (на VPS, у каталозі репо):
   ```bash
   sed -i "s|User=volodymyr|User=$(whoami)|; \
           s|/home/volodymyr/PycharmProjects/scalper-hft|$PWD|g" deploy/scalper-record.service
   grep -E "User=|WorkingDirectory|ExecStart" deploy/scalper-record.service   # перевірка
   ```

3. **Вимоги до каталогу на VPS** (щоб юніт стартував):
   - репо на `$PWD` (той самий, що в `WorkingDirectory`);
   - venv із websockets: `cd $PWD && uv sync --extra live` (або
     `.venv/bin/pip install websockets`);
   - `scripts/record_loop.sh` виконуваний: `chmod +x scripts/record_loop.sh`;
   - ключі/`.env` НЕ потрібні (публічний стрім).

4. **Запуск і контроль:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now scalper-record
   sudo systemctl status scalper-record          # active (running)
   journalctl -u scalper-record -f               # логи
   du -sh data/*depth5*.parquet                  # файл росте
   ```

5. **Кілька символів** — юніт зараз записує лише `BTCUSDT`. Для кожного
   символу — окремий інстанс (systemd template):
   ```ini
   # /etc/systemd/system/scalper-record@.service
   [Service]
   Type=simple
   User=<VPS_USER>
   WorkingDirectory=/home/<VPS_USER>/scalper-hft
   ExecStart=/home/<VPS_USER>/scalper-hft/scripts/record_loop.sh %i 1440
   Restart=always
   RestartSec=10
   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable --now scalper-record@BTCUSDT.service \
                          scalper-record@LINKUSDT.service \
                          scalper-record@XRPUSDT.service \
                          scalper-record@ETHUSDT.service
   ```

> ⚠ Примітка: юніт у репо (`deploy/scalper-record.service`) залишається шаблоном
> зі шляхом машини розробника; на VPS завжди правте копію (п.2) перед `cp`.

## Крок 3. Скільки даних і чи росте файл

```bash
du -sh data/*depth5*.parquet
.venv/bin/python -c "import pandas as pd,glob;
for f in glob.glob('data/*depth5*.parquet'):
    d=pd.read_parquet(f); print(f, len(d), d.index.min(), d.index.max())"
```
Орієнтир: ~0.5–1 млн снапшотів/день/символ (на BTCUSDT тут ~8–9 снапшотів/с)
→ parquet ~50–200 МБ/день/символ. Через ~1 тиждень оцінити фактичний розмір.

## Крок 4. Перенесення даних на research-машину (коли є VPS)

Одна команда (W0-R5). Parquet не комітити. Рекордер — публічний WS, без API-ключів.

```bash
# на локальній машині (ssh-ключ налаштовано):
bash scripts/sync_depth.sh [--dry-run] [user@host:path]
# дефолт джерела: ${VPS_USER:-tradebot}@${VPS_HOST}:${REMOTE_DIR:-~/scalper-hft/data}
# після копії: validate_depth / validate_bookticker → results/quality_depth.md
# exit ≠ 0, якщо quality_ok=false
```
Після цього дані доступні для Фази 3–4 (`docs/L2_DATA_PLAN.md`):
depth-weighted imbalance → бари → `ob_imbalance`/`market_maker` дослідження.

## Як це має відбуватися (порядок і критерії)

1. **Сьогодні (локально)**: Крок 1 — smoke ✅ (вже зроблено); запустити Крок 2A,
   залишити на ніч → завтра перевірити Крок 3 (файл росте).
2. **Цього тижня**: рішення про символи (рекомендую BTCUSDT+LINKUSDT+XRPUSDT+
   ETHUSDT) і про VPS (є сервер? ОС? ssh?) → деплой Крок 2B з виправленим шляхом.
3. **Після 1–2 тижнів запису**: rsync (Крок 4) → патлайн Фази 3 → перші
   дослідження depth-weighted imbalance (Фаза 4 L2_DATA_PLAN).
4. **Альтернатива**: якщо чекати не можна — Tardis.dev історія (Фаза 5) одразу.

## Відомі проблеми та як їх ловити

- **Немає рядків у логах** — нормально: рекордер пише підсумок лише наприкінці
  (`logger.info`); перевіряйте файл parquet, а не stdout.
- **Файл не росте** — обрив WS: рекордер має reconnect-цикл; див. логи на
  `WARN` («перепідключення») та час `ts` останнього рядка.
- **Дублікати/розриви ts** — дедуп за ts на етапі патлайну (Фаза 3).
- **Диск** — `du` щодня; за потреби ротація за датами (пункт Фази 1 L2_PLAN).
- **Запис при Ctrl+C** — буфер скидається у `finally` (вже реалізовано).
