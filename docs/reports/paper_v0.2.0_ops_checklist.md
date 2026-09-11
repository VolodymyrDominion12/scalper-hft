# Чекліст: тег `paper-v0.2.0` і 8 тижнів Paper-Gate

Дата: 2026-09-11 · Ops, не код альфи.
Конфіг: **pairs_arb LINK/BTC 1h maker**, `lookback=120`, `regime_scale=True`,
`regime_scale_factor=0.25`, `MAKER_FILL_SEED=42`, `ENABLE_VOL_TARGET=true`,
`USE_EXIT_LADDERS=false`, `DRY_RUN=true`.

Джерела: [IMPROVEMENT_PLAN_2026.md](../IMPROVEMENT_PLAN_2026.md) W0-OPS ·
[DEPLOY_PLAN.md](../DEPLOY_PLAN.md) фаза 2 ·
[CHANGE_PLAN_REVIEW.md](../CHANGE_PLAN_REVIEW.md) Ops ·
[deploy/README.md](../../deploy/README.md).
Попередній прогін: [paper_v0.1.0_vps_runbook.md](paper_v0.1.0_vps_runbook.md)
(годинник **скидається** — R1 змінив модель філу).

Це **не** дозвіл на live. Немає `DRY_RUN=false` у юнітах і немає тегу `live-v*`.

---

## 0. Перед тегом (локально)

Wave 0 код (TCA, R4–R7) має бути **в git**, інакше тег зріже незакомічені зміни.

- [ ] `git status` чистий (або свідомий коміт Wave 0). Не комітити `.env`, ключі, parquet, `results/`.
- [ ] `uv run pytest tests/ -q` зелений (останній прогін: 1428 passed, 9 skipped).
- [ ] `uv run ruff check scalper_hft tests` без нових помилок.
- [ ] У `deploy/` немає `DRY_RUN=false`, `use_kalman`, `legging_mode=chase`, `USE_EXIT_LADDERS=true`.
- [ ] `VALIDATED_PAIRS` = лише LINK/BTC, lb=120, regime_scale 0.25
      (`scalper_hft/live/pairs_runner.py`).
- [ ] Юніт лишається `paper-run-pairs --portfolio --daemon --notify`.

```bash
git log -1 --oneline
git status
uv run pytest tests/ -q
```

---

## 1. Тег (локально, після коміту)

Не котити цей SHA на живий `paper-v0.1.0` без нового sqlite і скидання 8 тижнів.

- [ ] Annotated tag на потрібному SHA (після merge/коміту Wave 0):

```bash
git tag -a paper-v0.2.0 -m "Paper-Gate v0.2.0: fill-parity + TCA + Wave 0 (R4–R7)"
git push origin paper-v0.2.0
# якщо main ще не запушений з Wave 0:
# git push origin HEAD
```

- [ ] `git rev-parse paper-v0.2.0` записати (SHA в Telegram-банері).
- [ ] Тег **не** рухати (`git tag -f` заборонено, якщо VPS уже на ньому).

---

## 2. Зупинка старого прогону `paper-v0.1.0` (VPS)

Модель філу + vol-target + схема IS (mid) змінюють облік → **новий sqlite**.

- [ ] Не в останні 2–3 хв години (pending post-only).
- [ ] Drain входів:

```bash
# results/control.json
{"pause": false, "no_new_entries": true, "flatten": false}
```

- [ ] Дочекатись idle (немає in-flight двох ніг). **Не** ставити `flatten`, якщо не хочеш закрити книгу v0.1.0 окремим рішенням.
- [ ] Архів старої книги (не в git):

```bash
systemctl --user stop scalper-paper-pairs
mkdir -p ~/scalper-archive/paper-v0.1.0
mv results/paper_pairs.sqlite ~/scalper-archive/paper-v0.1.0/
# якщо є control.json / is logs — туди ж
```

---

## 3. `.env` на VPS (не з ноута)

Деплой **не** перезаписує `.env`. Шаблон: `docs/env/vps-paper.env.example`.

- [ ] Скопіювати профіль і заповнити ключі **на сервері**:

```bash
cp docs/env/vps-paper.env.example .env
# nano .env  — ключі лише тут
```

Обов’язкові значення:

| Змінна | Значення |
|---|---|
| `DRY_RUN` | `true` |
| `EXCHANGE` | `binance-testnet` (paper; ордери симульовані) |
| `DATA_EXCHANGE` | `binanceusdm` (live-історія, не testnet) |
| `MAKER_EXECUTION` | `true` |
| `MAKER_FILL_SEED` | `42` (той самий, що BT / paper-audit) |
| `ENABLE_VOL_TARGET` | `true` |
| `VOL_TARGET_ANN` | `0.10` |
| `PAIR_NOTIONAL_PCT` | `0.30` |
| `USE_EXIT_LADDERS` | `false` |
| `REQUIRE_AUDIT_PASS` | `true` |
| `API_HOST` | `127.0.0.1` |

- [ ] `echo $DRY_RUN $EXCHANGE $DATA_EXCHANGE` у сесії шелу **не** має перекривати `.env` (load_dotenv не б’є вже експортовані змінні).
- [ ] Ключі Binance: IP whitelist лише VPS, futures trade, **без withdraw**. Для симуляції на публічних klines ключі можуть бути порожні.
- [ ] Streamlit / API / pgAdmin не слухають `0.0.0.0`.

---

## 4. Деплой тегу

- [ ] Шляхи в `~/.config/systemd/user/scalper-paper-pairs.service` збігаються з клоном (`WorkingDirectory`, `EnvironmentFile`).
- [ ] `loginctl show-user "$USER" | grep Linger` → `yes`.
- [ ] Checkout **тегу**, не `git pull origin main`:

```bash
./scripts/deploy_paper.sh paper-v0.2.0
# = git fetch --tags && git checkout paper-v0.2.0 && uv sync --frozen && restart
```

- [ ] `systemctl --user --no-pager --full status scalper-paper-pairs` — active.
- [ ] Банер у журналі / Telegram: тег або SHA, `DRY_RUN=true`, mode=paper.

```bash
journalctl --user -u scalper-paper-pairs -n 80 --no-pager
git describe --tags --exact-match
```

- [ ] Новий `results/paper_pairs.sqlite` з’явився після першого успішного `step`.
- [ ] Зняти drain після гідратації (порожня книга — ок):

```json
{"pause": false, "no_new_entries": false, "flatten": false}
```

- [ ] Перший крок після рестарту: `hold:same_bar` або нормальний quote — **не** обнулений equity.

---

## 5. Щотижневий ритм (8 календарних тижнів)

Старт годинника = перший успішний `step` на `paper-v0.2.0`, не дата тегу.

| Коли | Що |
|---|---|
| Щотижня (напр. пн) | `paper-audit` + `is-report --days 7` |
| Після crash/reboot | snapshot restore, `hold:same_bar`, sqlite не порожній |
| Hotfix A (логи/Telegram) | мінорний тег `paper-v0.2.1`, позиції тримати |
| Hotfix B (execution/risk) | `no_new_entries` → idle → новий мінорний тег |
| Hotfix C (z / lookback / regime_scale / Kalman / chase) | **заборонено**; новий major-тег і годинник з нуля |

```bash
# на VPS, той самий MAKER_FILL_SEED=42
uv run python -m scalper_hft.cli paper-audit \
  --db results/paper_pairs.sqlite --dd-mult 1.5
uv run python -m scalper_hft.cli is-report --days 7
```

Tracking error потребує кривої BT за **той самий** період (`--bt-equity ts,equity.csv`).
Без неї audit все одно друкує fill-rate і maxDD paper.

Бекап (не в git):

```bash
# приклад cron 03:00 UTC
0 3 * * * rclone copy ~/PycharmProjects/scalper-hft/results/paper_pairs.sqlite remote:scalper-backup/paper-v0.2.0/
```

Паралельно (не блокує Gate): `bash scripts/sync_depth.sh` з research-машини.

---

## 6. Критерії виходу з Paper-Gate

Усі пункти — на **одному** тезі `paper-v0.2.0` (або мінорних A/B без зміни альфи).

- [ ] ≥ 8 тижнів безперервного paper на VPS (журнал без gap > 1h-бара).
- [ ] Tracking error (paper vs BT equity) **< 3% на тиждень**.
- [ ] Fill rate post-only **≥ 70%**.
- [ ] maxDD paper **≤** backtest maxDD × 1.5 (`--dd-mult 1.5`).
- [ ] MAE/MFE forensics: без аномалій.
- [ ] Rolling ADF p-value спреду < 0.05 (входи не вбиті kill > 1 раз/міс).
- [ ] Blended TCA maker (IS + miss) **< 3 bps** (`is-report`; mid ≠ fill).
- [ ] Немає «рестарт обнулив equity».
- [ ] `USE_EXIT_LADDERS` лишився `false`; Kalman/chase не дефолт.

**Пройдений Gate ≠ live.** Live — окремий тег `live-v*`, окремі ключі, окремий юніт,
і лише після **явного** запиту.

---

## 7. Заборонено під час 8 тижнів

- Котити Wave 0 / R1 на `paper-v0.1.0` «щоб не скидати годинник».
- `git pull origin main` на VPS.
- Копіювати `.env` з ноута.
- Міняти z / lookback / `regime_scale` / `MAKER_FILL_SEED`.
- `USE_EXIT_LADDERS=true`, Kalman, chase, ERC у циклі, друга пара в `VALIDATED_PAIRS`.
- `DRY_RUN=false`, тег `live-v*`.
- Авто-запис `SLIPPAGE_BPS` з IS у `.env`.
- Flatten на кожен деплой.

---

## 8. Відкат

```bash
# control.json → no_new_entries, дочекатись idle
./scripts/deploy_paper.sh paper-v0.1.0   # лише якщо свідомо вертаєтесь на стару модель філу
# або paper-v0.2.0, якщо відкочуєте зіпсований 0.2.1
```

Відкат на `paper-v0.1.0` **не** продовжує годинник v0.2.0.
