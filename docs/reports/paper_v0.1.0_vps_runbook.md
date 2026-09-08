# Ранбук: paper-v0.1.0 на VPS (Фаза 2 — Paper-Gate)

Дата: 2026-09-08 · Тег: `paper-v0.1.0` · SHA: `ca0a97e` (main) ·
Конфіг: `LINK/BTC lb=120, regime_scale=True, regime_scale_factor=0.25`.

Цей ранбук — покрокова інструкція розгортання paper-бота на VPS та
8-тижневого Paper-Gate з [DEPLOY_PLAN.md](../DEPLOY_PLAN.md), Фаза 2.

## 0. Попередня умова (на локальній машині)

- [x] Тег `paper-v0.1.0` створено та запушено на origin.
- [x] `uv run pytest tests/ -q` зелений (1000 passed).
- [x] CSCV/PBO = 0.000 (PASS); regime_scale validated.
- [x] `VALIDATED_PAIRS` = `LINK/BTC lb=120, regime_scale=True, factor=0.25`.

## 1. Підготовка VPS (одноразово)

### 1.1 Користувач та SSH

```bash
# На VPS, окремий користувач (не root)
sudo adduser scalper
sudo usermod -aG sudo scalper  # або через sudoers, на твій вибір
sudo su - scalper

# SSH лише ключі (заборони password auth у /etc/ssh/sshd_config):
#   PasswordAuthentication no
#   PubkeyAuthentication yes
# Перевір: ssh scalper@<vps-ip> заходить без пароля
```

### 1.2 Час (NTP) та linger

```bash
sudo apt install -y chrony && sudo systemctl enable --now chrony
timedatectl status  # має бути NTP synchronized

# linger — щоб user-юніти пережили logout/reboot
loginctl enable-linger scalper
loginctl show-user scalper | grep Linger  # має бути yes
```

### 1.3 Залежності

```bash
# Python 3.12 + uv
sudo apt install -y python3.12 python3.12-venv git
curl -LsSf https://astral.sh/uv/install.sh | sh
# .local/bin має бути в PATH (додай у ~/.bashrc: export PATH="$HOME/.local/bin:$PATH")
```

### 1.4 Firewall (не слухати 0.0.0.0 без потреби)

```bash
# Streamlit/pgAdmin/API не відкривати; paper-бот лише вихідні з'єднання
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw enable
```

## 2. Перше встановлення бота

### 2.1 Clone на тег (НЕ на main)

```bash
cd ~
git clone --branch paper-v0.1.0 git@github.com:VolodymyrDominion12/scalper-hft.git PycharmProjects/scalper-hft
cd PycharmProjects/scalper-hft
uv sync --frozen
```

### 2.2 `.env` (окремий, не з ноута)

```bash
cp .env.example .env
nano .env
```

Мінімум для paper:
```
DRY_RUN=true
EXCHANGE=binance-testnet
MAKER_EXECUTION=true
PAIR_NOTIONAL_PCT=0.30
PORTFOLIO_NOTIONAL_PCT=0.60
DAILY_LOSS_LIMIT=0.03
WEEKLY_LOSS_LIMIT=0.06
TELEGRAM_BOT_TOKEN=<окремий токен або порожньо>
TELEGRAM_CHAT_ID=<твій chat_id>
```

**Ключі Binance:** для paper/testnet — опційні (симуляція на публічних klines).
Якщо є — IP whitelist лише VPS, futures trade, **без withdraw**.

### 2.3 systemd user-юніт

```bash
mkdir -p ~/.config/systemd/user
cp deploy/scalper-paper-pairs.service ~/.config/systemd/user/
# Перевір шляхи у юніті (WorkingDirectory, EnvironmentFile) — за замовчуванням
#   %h/PycharmProjects/scalper-hft
systemctl --user daemon-reload
systemctl --user enable --now scalper-paper-pairs
systemctl --user --no-pager --full status scalper-paper-pairs
```

### 2.4 Перевірка старту

```bash
journalctl --user -u scalper-paper-pairs -f --no-pager
```

Очікуваний банер:
```
scalper-hft paper-pairs | paper-v0.1.0 | DRY_RUN=true | daemon portfolio
```

Якщо Telegram налаштований — повідомлення прийде в чат з тим самим рядком.
Рівень equity через кілька барів: `hold:same_bar` → `quoted want=...` → `filled`.

## 3. Контроль без SSH-kill

`results/control.json` (створюється автоматично):

```json
{"pause": false, "no_new_entries": true, "flatten": false}
```

- `no_new_entries: true` — зупинити нові входи (виходи дозволені). Без рестарту.
- `pause: true` — `step` no-op (процес живий, heartbeat).
- `flatten: true` — лише якщо явно хочеш закрити все (НЕ на деплой).

Невалідний JSON → pause (fail-closed).

## 4. Операційний ритм (8 тижнів)

| Коли | Що | Команда |
|---|---|---|
| Щотижня (пн) | `paper-audit` vs бектест; maxDD ≤ BT × 1.5 | `uv run python -m scalper_hft.cli paper-audit --portfolio --days 7` |
| Після crash/reboot | перевірити snapshot + `hold:same_bar` на першому кроці | `journalctl --user -u scalper-paper-pairs -n 50` |
| Hotfix класу A (логі/Telegram) | рестарт, позиції тримати | `./scripts/deploy_paper.sh paper-v0.1.1` |
| Hotfix класу B (execution/risk) | drain pending, рестарт, reconcile | спершу `no_new_entries`, потім тег |
| Hotfix класу C (альфа/параметри) | ЗАБОРОНЕНО на цьому прогоні | новий paper-тег + повторний gate з нуля |

### Вікно деплою 1h

Деплой лише після `step` на закритому барі t і **не** в останні 2–3 хв години
(pending post-only ордери).

## 5. Критерії проходження Paper-Gate (8 тижнів)

- [ ] ≥8 тижнів безрозривного журналу (`journalctl` без gap > 1 бару)
- [ ] `paper-audit` зелений щотижня: tracking error прийнятна, maxDD ≤ BT × 1.5
- [ ] Немає «рестарт обнулив equity» (snapshot restore працює)
- [ ] Fill-rate maker стабільний (не деградує)
- [ ] Жодного `KillSwitch` від `SyncEngine` (drift позицій)

**Це НЕ дозвіл на live** — лише вихід з Фази 2. Live (Фаза 3–4) — окремий
адаптер, окремі ключі, окремий тег `live-v*`, і лише після явного запиту.

## 6. Відкат

```bash
# Відкат до попереднього тегу (напр. при проблемі):
./scripts/deploy_paper.sh paper-v0.1.0  # або попередній стабільний тег
```

Перед відкатом: `control.json → no_new_entries`, дочекатись idle (немає in-flight ніг).

## 7. Бекап `results/paper_pairs.sqlite`

```bash
# cron щодня (rclone на S3/Backblaze):
0 3 * * * rclone copy ~/PycharmProjects/scalper-hft/results/paper_pairs.sqlite remote:scalper-backup/
```

SQLite у git НЕ комітити.

---

## Поточний статус

- Фаза 0: ✅ (persist, daemon, control)
- Фаза 1 код: ✅ (systemd unit, deploy script, SHA в Telegram)
- Фаза 1 реліз: ✅ (тег `paper-v0.1.0` запушено 2026-09-08)
- **Фаза 2: цей ранбук → виконати на VPS**
- Фаза 3: код live-адаптера (можна кодити під час Фази 2)
- Фаза 4: лише після Gate + явного запиту
