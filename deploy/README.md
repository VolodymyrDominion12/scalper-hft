# Деплой paper-бота на VPS

Paper pairs крутиться як systemd **user**-юніт з git-тегу. Локальна машина
лишається для research. Повний план: [docs/DEPLOY_PLAN.md](../docs/DEPLOY_PLAN.md).

## Принципи

- На сервері `git checkout <tag>`, ніколи `git pull origin main`.
- `.env` на VPS **не** в git і скрипт деплою його не перезаписує.
- `DRY_RUN=true`. Live — окремий тег і явний запит після Paper-Gate.
- Ключі Binance (якщо є) — IP whitelist лише цієї машини, без withdraw.

## Перше встановлення

```bash
# clone на тег, не на випадковий tip
git clone --branch paper-v0.1.0 <repo-url> ~/PycharmProjects/scalper-hft
cd ~/PycharmProjects/scalper-hft
uv sync --frozen
cp .env.example .env   # заповнити локально на сервері; DRY_RUN=true
mkdir -p ~/.config/systemd/user
cp deploy/scalper-paper-pairs.service ~/.config/systemd/user/
# за потреби поправ WorkingDirectory / EnvironmentFile у юніті
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now scalper-paper-pairs
systemctl --user status scalper-paper-pairs
```

Контроль без SSH-kill процесу — `results/control.json`:

```json
{"pause": false, "no_new_entries": true, "flatten": false}
```

`flatten` лише якщо явно `true`. Невалідний JSON → pause (fail-closed).

## Оновлення (hotfix)

```bash
# на VPS, від running tag:
./scripts/deploy_paper.sh paper-v0.1.1
```

Перед цим: `no_new_entries`, не в останні 2–3 хв години. Відкат — попередній тег.
