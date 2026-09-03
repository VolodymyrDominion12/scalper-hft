# Перенесення PostgreSQL-кешу між комп’ютерами

Кеш ринкових даних при `DATA_BACKEND=postgres` живе **не** в `data/*.parquet`,
а в Docker-volume `postgres_data` контейнера `scalper_postgres`.
Повторно качати історію з Binance не треба: знімаємо дамп, переносимо файл,
відновлюємо на другому компі.

Контейнер і доступи за замовчуванням (див. `docker-compose.yml` і `.env.example`):

| | |
|---|---|
| контейнер | `scalper_postgres` |
| образ | `postgres:16-alpine` |
| БД / юзер / пароль | `scalper` / `scalper` / `scalper` |
| зовнішній порт | **5436** (всередині контейнера 5432) |
| volume | `postgres_data` |

Таблиці: `klines`, `agg_trades`, `funding`, `meta`.
Гігабайти майже завжди в `agg_trades`; `klines` і `funding` легкі.

**Не копіюй** каталог Docker volume руками (`/var/lib/docker/volumes/...`) —
між машинами це крихко. Не коміть дамп у git.

## 1. Перевірити, що Postgres піднятий і скільки важить

На машині, де вже є історія:

```bash
docker compose ps
docker exec scalper_postgres psql -U scalper -d scalper -c \
  "SELECT pg_size_pretty(pg_database_size('scalper'));"

docker exec scalper_postgres psql -U scalper -d scalper -c \
  "SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
   FROM pg_catalog.pg_statio_user_tables
   ORDER BY pg_total_relation_size(relid) DESC;"
```

## 2. Зняти дамп (машина-джерело)

Кастомний формат `-Fc` уже стиснутий — зручно для кількох ГБ.

```bash
docker exec scalper_postgres pg_dump -U scalper -d scalper -Fc -Z 6 -f /tmp/scalper.dump
docker cp scalper_postgres:/tmp/scalper.dump ./scalper.dump
docker exec scalper_postgres rm /tmp/scalper.dump
ls -lh scalper.dump
```

Якщо на другому компі потрібні лише свічки (без CVD / тіків):

```bash
docker exec scalper_postgres pg_dump -U scalper -d scalper -Fc -Z 6 \
  -t klines -t funding -t meta -f /tmp/scalper.dump
docker cp scalper_postgres:/tmp/scalper.dump ./scalper.dump
docker exec scalper_postgres rm /tmp/scalper.dump
```

## 3. Перенести файл

Той самий `scalper.dump` — флешка, зовнішній SSD або LAN:

```bash
# LAN (підстав юзера і IP/hostname другого компа)
rsync -avh --progress scalper.dump USER@OTHER_PC:~/PycharmProjects/scalper-hft/

# або SSD
rsync -avh --progress scalper.dump /media/volodymyr/SSD/
```

Дамп не комітити. Після успішного restore файл можна видалити.

## 4. Відновити (машина-призначення)

Той самий репозиторій. У `.env` мають збігатися бекенд і креденшали:

```bash
DATA_BACKEND=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5436
POSTGRES_DB=scalper
POSTGRES_USER=scalper
POSTGRES_PASSWORD=scalper
```

```bash
docker compose up -d postgres
docker compose ps   # дочекайся healthy

docker cp ./scalper.dump scalper_postgres:/tmp/scalper.dump
docker exec scalper_postgres pg_restore -U scalper -d scalper \
  --clean --if-exists --no-owner --jobs=4 /tmp/scalper.dump
docker exec scalper_postgres rm /tmp/scalper.dump
```

`--clean --if-exists` замінює порожні таблиці (схема створюється сама при
першому підключенні через `PostgresStore.ensure_schema`).

Попередження на кшталт «role scalper does not exist» / «extension …» можна
ігнорувати, якщо `--no-owner` стоїть і таблиці на місці.

## 5. Перевірка

```bash
docker exec scalper_postgres psql -U scalper -d scalper -c \
  "SELECT symbol, interval, COUNT(*) FROM klines GROUP BY 1,2 ORDER BY 1,2;"

docker exec scalper_postgres psql -U scalper -d scalper -c \
  "SELECT symbol, COUNT(*), MIN(ts), MAX(ts) FROM agg_trades GROUP BY 1 ORDER BY 1;"
```

Далі `download` лише докачує хвіст, а не всю історію:

```bash
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 90
```

Для `aggTrades` це критично: REST Binance віддає лише ~останні 2 доби.
Старіша історія живе **лише** в цій БД (або в parquet-кеші). Якщо дамп не
перенести — з біржі її вже не відновити.

## Що не робити

- Копіювати raw Docker volume між компами.
- Відкривати Postgres у LAN «наживу» замість одного файлу дампу.
- Залишати `scalper.dump` у репозиторії або в хмарі з публічним лінком.
- Очікувати, що два Postgres самі синхронізуються. Разовий дамп — нормальний
  варіант для двох машин. Постійна реплікація тут не налаштована.

Якщо часто працюєш на двох компах і не потрібен Postgres — простіше
`DATA_BACKEND=parquet` і Syncthing на каталог `data/`.
