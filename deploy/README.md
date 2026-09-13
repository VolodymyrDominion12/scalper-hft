# Деплой paper-бота на VPS

Paper-боти крутяться як systemd **user**-юніти з git-тегу. Локальна машина
лишається для research. Повний план: [docs/DEPLOY_PLAN.md](../docs/DEPLOY_PLAN.md).

## Принципи

- На сервері `git checkout <tag>`, ніколи `git pull origin main`.
- `.env` на VPS **не** в git і скрипт деплою його не перезаписує.
- `DRY_RUN=true`. Live — окремий тег і явний запит після Paper-Gate.
- Ключі Binance (якщо є) — IP whitelist лише цієї машини, без withdraw.

## Набір paper-ботів (paper-v0.3.0)

| Юніт | Команда | Кандидат | Статус |
|---|---|---|---|
| `scalper-paper-pairs` | `paper-run-pairs --portfolio --daemon` | pairs_arb LINK/BTC 1h maker (regime_scale 0.25) | validated |
| `scalper-paper-tsmom@1d` | `paper-run-ts-momentum --interval 1d --daemon` | ts_momentum 1d long-only CORE_15 | monitoring |
| `scalper-paper-tsmom@4h` | `paper-run-ts-momentum --interval 4h --daemon` | ts_momentum 4h long-only CORE_15 | monitoring |

Журнали: `results/paper_pairs.sqlite`, `results/paper_ts_momentum_1d.sqlite`,
`results/paper_ts_momentum_4h.sqlite`. Спільний стоп-кран — `results/control.json`.

`scalper-paper-pairs` fail-closed: без свіжого **PASS** для комірки
`LINKUSDT/BTCUSDT 1h` у `results/audit_verdicts.jsonl` юніт не стартує
(`live/audit_gate.py`, вік вердикта ≤ `AUDIT_MAX_AGE_DAYS`, дефолт 30 діб).
`results/` не в git — файл вердиктів засівається окремо (див. нижче).

## Перше встановлення

```bash
# clone на тег, не на випадковий tip
git clone --branch paper-v0.3.0 <repo-url> ~/scalper-hft
cd ~/scalper-hft
~/.local/bin/uv sync --frozen
cp docs/env/vps-paper.env.example .env   # заповнити ключі локально; DRY_RUN=true
mkdir -p results ~/.config/systemd/user
cp deploy/scalper-paper-pairs.service deploy/scalper-paper-tsmom@.service ~/.config/systemd/user/
# шляхи в юнітах — під VPS-layout %h/scalper-hft; на іншому каталозі поправ
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now scalper-paper-pairs scalper-paper-tsmom@1d scalper-paper-tsmom@4h
systemctl --user status scalper-paper-pairs scalper-paper-tsmom@1d scalper-paper-tsmom@4h
```

### Засів гейта (pairs)

`results/audit_verdicts.jsonl` — артефакт research-машини (walk-forward на VPS
свідомо не ганяємо). Копіюємо лише свіжий PASS:

```bash
# локально (research-машина)
rsync -av results/audit_verdicts.jsonl tradebot@<vps>:~/scalper-hft/results/
```

Перевірка віку: `uv run python -m scalper_hft.cli paper-audit` і логи юніта —
`audit_gate` пише причину відмови, якщо вердикт протух.

## Контроль без SSH-kill процесу

`results/control.json`:

```json
{"pause": false, "no_new_entries": true, "flatten": false}
```

`flatten` лише якщо явно `true`. Невалідний JSON → pause (fail-closed).
Один файл керує всіма трьома юнітами (у них спільний `--control`).

## FastAPI (опційно)

`scalper-api.service` біндиться на **localhost** (`API_HOST=127.0.0.1` у `.env`).
Не відкривайте `0.0.0.0` без `API_SECRET_KEY` ≥32 символів — lifespan
`require_safe_api_bind` зупинить процес. Доступ ззовні — лише SSH-тунель:

```bash
ssh -L 8000:127.0.0.1:8000 user@vps
```

## Оновлення (hotfix)

```bash
# на VPS, від running tag:
./scripts/deploy_paper.sh paper-v0.3.1
```

Перед цим: `no_new_entries`, не в останні 2–3 хв години. Відкат — попередній тег.
`deploy_paper.sh` рестартить лише `scalper-paper-pairs`; після нього вручну
`systemctl --user restart scalper-paper-tsmom@1d scalper-paper-tsmom@4h`.

## Щоденні/щотижневі перевірки

```bash
systemctl --user list-units 'scalper-paper*' --no-pager
journalctl --user -u scalper-paper-pairs -n 50 --no-pager
# pairs: tracking error vs бектест + MAE/MFE forensics (є ордери/філи)
uv run python -m scalper_hft.cli paper-audit --db results/paper_pairs.sqlite
# ts_momentum: у журналі лише equity/trades (без maker-ордерів) —
# дивимось equity_portfolio, а не paper-audit
sqlite3 results/paper_ts_momentum_1d.sqlite \
  "select ts, equity from equity where symbol='TSMOM_PORTFOLIO' order by ts desc limit 10;"
```
