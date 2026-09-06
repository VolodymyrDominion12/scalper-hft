# Концепція Моніторингу та Торгової Платформи

> Версія: 2026-09-06 | Статус: Архітектурна концепція  
> Контекст: розширення поверх вже реалізованих Phase 0–3 (pairs_arb, RegimeSupervisor, ML-стек, SQLite-store, Telegram-повідомлення).

---

## Огляд архітектури

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRADING PLATFORM                                │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Binance  │   │  Bybit   │   │   OKX    │   │  Gate.io │  ...       │
│  │ USDT-M   │   │ Linear   │   │  Swap    │   │  Perps   │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       └──────────────┴──────────────┴──────────────┘                   │
│                              │                                          │
│                    ┌─────────▼──────────┐                              │
│                    │   ExchangeAdapter  │  (уніфікований інтерфейс)    │
│                    │   ccxt-based       │                              │
│                    └─────────┬──────────┘                              │
│          ┌───────────────────┼───────────────────┐                     │
│          │                   │                   │                     │
│   ┌──────▼──────┐   ┌────────▼──────┐   ┌───────▼──────┐             │
│   │  DataLayer  │   │   BotEngine   │   │  TradeStore  │             │
│   │  (Parquet)  │   │ (Supervisor)  │   │  (SQLite/PG) │             │
│   └──────┬──────┘   └────────┬──────┘   └───────┬──────┘             │
│          └───────────────────┼───────────────────┘                     │
│              ┌───────────────┼───────────────┐                         │
│      ┌───────▼──────┐ ┌──────▼──────┐ ┌────▼──────────┐              │
│      │  Streamlit   │ │  Telegram   │ │  REST/WebSocket│              │
│      │  Dashboard   │ │    Bot      │ │  API (FastAPI) │              │
│      └──────────────┘ └─────────────┘ └───────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Багатобіржева архітектура даних

### 1.1 Структура файлової системи (кеш/дані)

```
data/
├── binance/
│   ├── klines/
│   │   ├── BTCUSDT/1h/2025-01.parquet
│   │   ├── BTCUSDT/4h/2025-01.parquet
│   │   └── XRPUSDT/1h/2025-01.parquet
│   ├── funding/
│   │   └── BTCUSDT/2025-01.parquet
│   ├── trades/           ← aggTrades для CVD
│   │   └── BTCUSDT/2025-01.parquet
│   └── orderbook/
│       └── BTCUSDT/depth5-snapshots/2025-01.parquet
├── bybit/
│   ├── klines/
│   └── funding/
├── okx/
│   └── ...
└── _meta/
    ├── exchanges.json    ← реєстр бірж і символів
    └── cache_index.json  ← які дані є локально
```

**Принцип**: кожна біржа — окремий namespace. Дослідження (`research/`)
автоматично шукає дані в `data/{exchange}/klines/{symbol}/{interval}/`.

### 1.2 ExchangeRegistry

```python
# scalper_hft/data/exchange_registry.py

SUPPORTED_EXCHANGES = {
    "binance": {
        "ccxt_id": "binanceusdm",
        "type": "futures_usdt",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "has_funding": True,
        "has_agg_trades": True,
    },
    "bybit": {
        "ccxt_id": "bybit",
        "type": "linear",
        "maker_fee": 0.0001,
        "taker_fee": 0.0006,
        "has_funding": True,
        "has_agg_trades": False,
    },
    "okx": {
        "ccxt_id": "okx",
        "type": "swap",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "has_funding": True,
        "has_agg_trades": False,
    },
}
```

### 1.3 Розширення DataClient

```python
# scalper_hft/data/client.py (додати підтримку exchange=)


def fetch_klines(
    symbol: str,
    interval: str,
    exchange: str = "binance",  # ← новий параметр
    days: int = 90,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_path = Path(f"data/{exchange}/klines/{symbol}/{interval}")
    ...
```

**CLI-приклад:**
```bash
uv run python -m scalper_hft.cli download --symbol BTCUSDT --interval 1h --days 90 --exchange bybit
uv run python -m scalper_hft.cli backtest --strategy pairs_arb --exchange bybit --symbol BTCUSDT --interval 1h
```

---

## 2. Стратегії: гнучкість і конфігурація

### 2.1 YAML-схема конфігурації стратегії

```yaml
# configs/strategies/btcusdt_1h_binance.yaml
name: regime_supervisor
exchange: binance
symbol: BTCUSDT
interval: 1h
mode: paper  # paper | live

strategies: "mean_reversion,supertrend,hmm_reversion"
blend_mode: contextual_hedge
n_hmm_states: 3

risk:
  notional_pct: 0.05
  max_dd: 0.10
  daily_loss_limit: 0.02

regime_overrides:
  trend_up:
    preferred: "supertrend,cross_momentum"
    weight_boost: 1.5
  trend_down:
    preferred: "mean_reversion,hmm_reversion"
    weight_boost: 1.5
  range:
    preferred: "mean_reversion,ob_imbalance"
    weight_boost: 1.0
```

### 2.2 Один бот — багато стратегій для ринкових режимів

```
RegimeSupervisor (вже реалізований)
    │
    ├── Режим: trend_up   → supertrend + cross_momentum (вищі ваги)
    ├── Режим: trend_down → mean_reversion + hmm_reversion (вищі ваги)
    ├── Режим: range      → ob_imbalance + market_maker (вищі ваги)
    ├── Режим: high_vol   → зменшити всі позиції × 0.5
    └── Режим: crisis     → flatten + halt
```

**Ключова ідея:** один процес містить N суб-стратегій. `RegimeSupervisor`
(вже є в `strategies/regime_supervisor.py`) детектує режим і зважує сигнали.
Потрібно лише `SupervisorConfig.yaml` + завантаження з `control.json`.

---

## 3. TradeStore: єдине сховище угод

### 3.1 Розширена схема (від поточного `store.py`)

```sql
-- Додати поля exchange і mode в існуючі таблиці
ALTER TABLE trades ADD COLUMN exchange TEXT NOT NULL DEFAULT 'binance';
ALTER TABLE trades ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper';

-- Новий: стан рахунків по біржах
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    exchange TEXT NOT NULL,
    mode TEXT NOT NULL,       -- paper | live
    balance REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    margin_used REAL NOT NULL,
    available REAL NOT NULL
);

-- Новий: відкриті позиції (синхронізація з біржею)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,       -- long | short
    size REAL NOT NULL,
    entry_price REAL NOT NULL,
    mark_price REAL,
    unrealized_pnl REAL,
    mode TEXT NOT NULL
);

-- Новий: активні боти
CREATE TABLE IF NOT EXISTS bots (
    bot_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    strategy TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,    -- running | paused | stopped | error
    started_at TEXT NOT NULL,
    last_heartbeat TEXT,
    config_json TEXT NOT NULL
);
```

### 3.2 Стратегія синхронізації (SyncEngine)

| Подія | Метод | Частота |
|---|---|---|
| Новий ордер | WebSocket user-stream | Real-time |
| Баланс рахунку | REST `/account` | Кожні 60 с |
| Позиції | REST + WS | Кожні 30 с + real-time |
| PnL/equity | Обчислення з позицій | Кожні 30 с |
| Заповнені ордери | WS `/tradeUpdate` | Real-time |

```python
# scalper_hft/live/sync_engine.py  ← НОВИЙ


class SyncEngine:
    """Синхронізує локальний TradeStore з реальною біржею."""

    def sync_account(self, exchange: str) -> None:
        """Оновити баланс, маржу і PnL."""
        ...

    def sync_positions(self, exchange: str) -> None:
        """Зрівняти відкриті позиції. KillSwitch якщо drift."""
        ...

    def sync_fills(self, exchange: str) -> None:
        """Завантажити нові заповнення через REST або WS."""
        ...
```

---

## 4. Telegram Bot: мобільний моніторинг

### 4.1 Команди бота

```
/status          → поточний баланс, PnL, відкриті позиції
/trades [N]      → останні N угод (default 10)
/bots            → список запущених ботів і їх стан
/pause [bot_id]  → призупинити бот (control.json no_new_entries)
/resume [bot_id] → відновити
/stop [bot_id]   → зупинити бот (SIGTERM)
/risk            → поточне використання ризик-лімітів
/equity          → equity curve (PNG графік)
/regime          → поточний ринковий режим по символах
/alert_on        → увімкнути push-сповіщення
/alert_off       → вимкнути
/help            → список команд
```

### 4.2 Push-сповіщення (автоматичні)

```
📈 OPEN  XRP/BTC  z=2.34  long XRP / short BTC
   Ціна: 0.000028 | Ноціонал: $50 | Режим: range

📉 CLOSE XRP/BTC  PnL: +$1.23 (+2.46%)
   Тривалість: 3h 42m | MAE: -$0.15 | MFE: +$1.87

⚠️ RISK  Binance | MaxDD досягнуто 8.3%
   Нові входи заблоковано. /resume для повторного старту.

💛 REGIME_CHANGE  BTCUSDT → trend_up
   Ваги: supertrend 0.45, cross_momentum 0.35, mean_reversion 0.20

❤️ HEARTBEAT  Бот живий | Paper | Binance
   Equity: $1,023.45 (+2.3%) | Позицій: 3 | Uptime: 42h
```

### 4.3 Структура (python-telegram-bot v20+ async)

```python
# scalper_hft/live/telegram_bot.py  ← НОВИЙ

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


class TelegramBotServer:
    """Async Telegram Bot для управління та моніторингу."""

    def __init__(self, store: UnifiedTradeStore, token: str, allowed_chat_ids: list[int]):
        self.store = store
        self.allowed_chat_ids = allowed_chat_ids  # whitelist security
        self.app = Application.builder().token(token).build()
        self._register_handlers()

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        account = self.store.latest_account()
        positions = self.store.open_positions()
        # ... форматування та відправка
```

**Безпека:**
- `TELEGRAM_ALLOWED_CHAT_IDS` — whitelist у `.env`
- Команди модифікації (pause/stop) — додатковий PIN: `/confirm <pin>`
- Rate limiting: ≤30 команд/хв з одного chat_id

---

## 5. Streamlit Dashboard: локальний центр аналітики

### 5.1 Структура сторінок

```
dashboard.py (точка входу, захищена паролем)
├── 🏠 Overview          → зведена панель по всіх біржах і ботах
├── 📊 Live Trading      → real-time equity, позиції, P&L
│   ├── Paper Trading
│   └── Live Trading
├── 📈 Research Hub      → бектести, walk-forward, порівняння стратегій
│   ├── Backtest Results
│   ├── Strategy Comparison (Sharpe heatmap по біржах)
│   └── Overfitting Audit (DSR, CSCV summary)
├── 🤖 Bots Manager      → список ботів, статус, управління
├── 💱 Multi-Exchange    → дані з різних бірж на одному графіку
├── 📋 Trade History     → журнал угод із фільтрами
├── ⚠️ Risk Monitor      → ліміти, drawdown, режими ринку
└── ⚙️ Settings          → конфігурація (API ключі маскуються)
```

### 5.2 Аутентифікація (session-based)

```python
# scalper_hft/dashboard_auth.py

import hashlib, secrets
import streamlit as st


def check_password() -> bool:
    """Захист паролем з session tokens."""
    if "auth_token" in st.session_state:
        return _validate_token(st.session_state["auth_token"])

    with st.form("login"):
        pwd = st.text_input("Пароль", type="password")
        if st.form_submit_button("Увійти"):
            if _check_password(pwd):
                st.session_state["auth_token"] = secrets.token_hex(32)
                st.rerun()
            else:
                st.error("Невірний пароль")
    return False
```

**Змінні оточення:**
```bash
DASHBOARD_PASSWORD_HASH=bcrypt_hash  # у .env
```

### 5.3 Multi-Exchange сторінка

```python
# scalper_hft/app_pages/multi_exchange.py


def render_multi_exchange() -> None:
    exchanges = st.multiselect("Біржі", ["binance", "bybit", "okx"])
    symbol = st.text_input("Символ", "BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h", "4h"])

    # Порівняльний графік цін (Plotly subplots)
    dfs = {ex: load_klines(ex, symbol, interval) for ex in exchanges}
    fig = plot_price_comparison(dfs)
    st.plotly_chart(fig)

    # Funding rate comparison
    fig2 = plot_funding_rates(exchanges, symbol)
    st.plotly_chart(fig2)

    # Correlation matrix між біржами
    render_correlation_heatmap(dfs)
```

---

## 6. REST API (FastAPI): програматичний доступ

```python
# scalper_hft/api/server.py


@app.get("/api/v1/account/{exchange}")
async def get_account(exchange: str, token: str = Depends(verify_token)): ...


@app.get("/api/v1/positions")
async def get_positions(exchange: str | None = None): ...


@app.get("/api/v1/trades")
async def get_trades(limit: int = 50, exchange: str | None = None): ...


@app.post("/api/v1/bots/{bot_id}/pause")
async def pause_bot(bot_id: str, token: str = Depends(verify_token)): ...


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time оновлення позицій і угод."""
    ...
```

**Дашборд і Telegram Bot використовують цей API** — єдина точка правди.

---

## 7. Кешування: стратегія і рівні

| Рівень | Технологія | Що кешує | TTL |
|---|---|---|---|
| L1 In-Memory | dict + TTL | ціни, баланси | 5–30 с |
| L2 SQLite (WAL) | unified_store.sqlite | угоди, equity, snapshot | постійно |
| L3 Parquet | `data/{exchange}/...` | klines, funding, trades | незмінні файли |
| L4 PostgreSQL | (optional, USE_POSTGRES=true) | aggTrades CVD (млн рядків) | постійно |

```python
# scalper_hft/data/cache_manager.py


class CacheManager:
    def get_klines(self, exchange: str, symbol: str, interval: str) -> pd.DataFrame:
        """
        1. Перевіряє cache_index.json чи є свіжий parquet.
        2. Якщо застарів → fetch REST, дописує parquet.
        3. Повертає DataFrame.
        """
        ...

    def invalidate_if_stale(self, max_age_hours: int = 2) -> None: ...
```

---

## 8. Фазовий план реалізації

### Пріоритет 1 — Telegram Bot (команди, не тільки push)

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/live/telegram_bot.py` | **НОВИЙ**: CommandHandler, async, whitelist | 1–2 дні |
| `scalper_hft/live/telegram.py` | Додати `register_handlers()` | 0.5 дні |
| `scalper_hft/cli.py` | `telegram-bot start` команда | 0.5 дні |
| `.env.example` | `TELEGRAM_ALLOWED_CHAT_IDS` | 1 год |
| `tests/test_telegram_bot.py` | Мок Update, тест команд | 1 день |

**Залежності:** `python-telegram-bot>=20.0`

---

### Пріоритет 2 — UnifiedTradeStore

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/live/store.py` | Додати `exchange`, `mode` поля | 1 день |
| `scalper_hft/live/store.py` | Нові таблиці: `accounts`, `positions`, `bots` | 1 день |
| `scalper_hft/live/sync_engine.py` | **НОВИЙ**: REST sync циклу | 2 дні |
| `tests/test_store_v2.py` | Міграційні тести | 1 день |

---

### Пріоритет 3 — Dashboard аутентифікація + Overview

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/dashboard_auth.py` | **НОВИЙ**: login form, session token | 1 день |
| `scalper_hft/app_pages/overview.py` | Оновити: multi-exchange KPI | 1 день |
| `scalper_hft/dashboard.py` | Підключити auth до всіх сторінок | 0.5 дні |

---

### Пріоритет 4 — Багатобіржева підтримка (DataLayer)

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/data/exchange_registry.py` | **НОВИЙ**: реєстр бірж | 0.5 дні |
| `scalper_hft/data/client.py` | `exchange=` параметр скрізь | 1 день |
| `scalper_hft/cli.py` | `--exchange` flag у download/backtest | 1 день |
| `scalper_hft/app_pages/multi_exchange.py` | **НОВИЙ**: порівняльні графіки | 2 дні |

---

### Пріоритет 5 — REST API (FastAPI)

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/api/server.py` | **НОВИЙ**: FastAPI endpoints | 2 дні |
| `scalper_hft/api/auth.py` | Bearer token auth | 1 день |
| `scalper_hft/cli.py` | `api start --port 8080` | 0.5 дні |
| `deploy/scalper-api.service` | systemd unit | 0.5 дні |

---

### Пріоритет 6 — SupervisorConfig + YAML-конфігурація

| Файл | Дія | Оцінка |
|---|---|---|
| `scalper_hft/live/supervisor_config.py` | **НОВИЙ**: dataclasses | 1 день |
| `configs/strategies/*.yaml` | Приклади конфігурацій | 1 день |
| `scalper_hft/cli.py` | `run --config configs/btcusdt_1h.yaml` | 1 день |

---

## 9. Операційна модель: де що виконується

```
┌────────────────────────────────────────────────────────────────┐
│                    ЛОКАЛЬНА МАШИНА                             │
│  • Research / Optuna / Walk-forward / DSR / CSCV              │
│  • Streamlit Dashboard (localhost:8501, паролем)              │
│  • Бектести на даних з різних бірж                            │
│  • DRY_RUN=true ЗАВЖДИ                                         │
└────────────────────────────────────────────────────────────────┘
              ↕ SSH тунель (rclone SQLite sync)
┌────────────────────────────────────────────────────────────────┐
│                    VPS (PAPER/LIVE)                            │
│  • systemd: scalper-paper-pairs (git-тег)                      │
│  • Telegram Bot Server (відповідає на команди)                │
│  • SyncEngine (REST/WS синхронізація з біржею)                │
│  • SQLite store → rclone → backup storage                      │
│  • FastAPI (localhost only, без публічного доступу)           │
│  • DRY_RUN=true (paper) або =false (live, після Gate)          │
└────────────────────────────────────────────────────────────────┘
              ↕ Telegram API (push/pull команди)
┌────────────────────────────────────────────────────────────────┐
│                    ТЕЛЕФОН (Telegram)                          │
│  /status, /trades, /risk, /pause, /resume, /equity            │
│  Push: угоди, режими, ризик-тригери, heartbeat                │
└────────────────────────────────────────────────────────────────┘
```

### SSH тунель для дашборду VPS

```bash
# Прокинути VPS Streamlit локально через SSH
ssh -L 8502:localhost:8501 user@vps-ip -N

# localhost:8502 → VPS Streamlit (захищений паролем, не публічний)
```

---

## 10. Нові залежності

```toml
[project.dependencies]
# Existing: ccxt, pandas, streamlit, etc. +

python-telegram-bot = ">=20.8"      # async Telegram Bot

fastapi = ">=0.110"                 # REST API
uvicorn = {extras = ["standard"], version = ">=0.28"}

pyyaml = ">=6.0"                   # YAML конфіги стратегій
bcrypt = ">=4.0"                    # паролі dashboard
aiohttp = ">=3.9"                   # async HTTP для SyncEngine
```

---

## 11. Чого не робити

- ❌ Не відкривати Streamlit/API в інтернет без reverse proxy + TLS
- ❌ Не зберігати API ключі в SQLite (тільки в `.env`)
- ❌ Не давати Telegram боту право виходу без PIN-підтвердження
- ❌ Не запускати research на VPS (лише paper/live daemon)
- ❌ Не синхронізувати між біржами автоматично (ризик подвоєння позицій)
- ❌ Не додавати нові біржі без тестів (мок для кожного ExchangeAdapter)

---

## 12. Наступні кроки (рекомендований порядок)

1. **Telegram Bot команди** (П1) — найбільша практична цінність прямо зараз
2. **UnifiedTradeStore** (П2) — основа для multi-exchange dashboard
3. **Dashboard auth** (П3) — безпека перед відкриттям
4. **Exchange registry + data layer** (П4) — bybit/okx дані
5. **FastAPI** (П5) — програматичний доступ і WebSocket live updates
6. **YAML конфіги стратегій** (П6) — гнучкість без перекомпіляції

> Перевірка після кожної фази: `uv run pytest tests/ -q` має бути зеленим.
