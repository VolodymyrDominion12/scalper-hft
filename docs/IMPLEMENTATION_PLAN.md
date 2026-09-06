# Покроковий план змін: Monitoring & Trading Platform

> Статус: очікує підтвердження  
> Основа: MONITORING_CONCEPT.md | Правила: AGENTS.md  
> Перевірка після кожного спринту: `uv run pytest tests/ -q` 🟢

---

## Порядок спринтів

```
Sprint 1  →  Telegram Bot (команди + push)            ~5 днів
Sprint 2  →  UnifiedTradeStore + SyncEngine           ~5 днів
Sprint 3  →  Dashboard Auth + Multi-Exchange UI        ~5 днів
Sprint 4  →  Exchange Registry + DataLayer            ~4 дні
Sprint 5  →  FastAPI REST + WebSocket                 ~5 днів
Sprint 6  →  YAML Strategy Configs + SupervisorConfig ~4 дні
```

---

## Sprint 1 — Telegram Bot (інтерактивний)

> **Мета**: з телефону отримати `/status`, `/trades`, `/pause`, керувати ботами.  
> Поточний стан: є лише `send_telegram()` (односторонній push).

### Крок 1.1 — Нова залежність

**Файл:** [`pyproject.toml`](file:///home/volodymyr/PycharmProjects/scalper-hft/pyproject.toml)

```diff
 [project.optional-dependencies]
 live = [
     "websockets>=12",
+    "python-telegram-bot>=20.8",
 ]
```

```bash
uv add "python-telegram-bot>=20.8" --optional live
```

### Крок 1.2 — Нові поля Settings

**Файл:** [`scalper_hft/config.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/config.py)

```diff
 # Telegram (опційно; без ключів send — no-op)
 telegram_bot_token: str = ...
 telegram_chat_id: str = ...
+
+# Telegram Bot — список дозволених chat_id через кому (whitelist)
+telegram_allowed_chat_ids: str = field(
+    default_factory=lambda: os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
+)
+# PIN для деструктивних команд (pause/stop/resume)
+telegram_bot_pin: str = field(
+    default_factory=lambda: os.getenv("TELEGRAM_BOT_PIN", "")
+)
```

### Крок 1.3 — Telegram Bot сервер [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/live/telegram_bot.py`

```python
"""Інтерактивний Telegram Bot (python-telegram-bot v20, async).

Команди:
    /status   — баланс, PnL, позиції
    /trades   — останні N угод
    /bots     — список активних ботів
    /pause    — призупинити бот (no_new_entries)
    /resume   — відновити бот
    /equity   — equity curve PNG
    /risk     — ризик-ліміти
    /regime   — поточний режим ринку
    /help     — список команд

Безпека:
    - TELEGRAM_ALLOWED_CHAT_IDS — whitelist (через кому)
    - /pause /stop /resume — потребують TELEGRAM_BOT_PIN
    - Rate limit: ≤30 команд/хв з одного chat_id
"""
```

**Методи:**
- `_is_allowed(update)` — перевірка whitelist
- `_check_pin(text, update)` — перевірка PIN перед деструктивними командами
- `cmd_status(update, ctx)` — баланс + позиції з `PaperStore`
- `cmd_trades(update, ctx)` — `store.all_trades().tail(N)`
- `cmd_bots(update, ctx)` — читати таблицю `bots` (Sprint 2)
- `cmd_pause(update, ctx)` — писати `control.json` (вже є `control.py`)
- `cmd_resume(update, ctx)` — знімати `no_new_entries` з `control.json`
- `cmd_equity(update, ctx)` — генерувати PNG через matplotlib, відправляти фото
- `cmd_regime(update, ctx)` — читати останній snapshot стану
- `run_polling()` — запуск бота в окремому thread

### Крок 1.4 — Розширення `telegram.py`

**Файл:** [`scalper_hft/live/telegram.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/live/telegram.py)

Додати функції push-сповіщень:
```diff
+def notify_regime_change(symbol: str, old: str, new: str, weights: dict) -> bool:
+    """💛 REGIME_CHANGE: symbol → new_regime. Ваги: ..."""
+
+def notify_heartbeat(exchange: str, mode: str, equity: float,
+                     pct: float, n_positions: int, uptime_h: float) -> bool:
+    """❤️ HEARTBEAT: бот живий + equity."""
+
+def notify_risk_block(exchange: str, reason: str, dd_pct: float) -> bool:
+    """⚠️ RISK: MaxDD/DailyLoss досягнуто, нові входи заблоковано."""
```

### Крок 1.5 — CLI команда

**Файл:** [`scalper_hft/cli.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/cli.py)

```bash
# Нова команда:
uv run python -m scalper_hft.cli telegram-bot start [--store results/paper_pairs.sqlite]
```

### Крок 1.6 — `.env.example`

```diff
+# Telegram Bot (інтерактивний)
+TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321   # whitelist через кому
+TELEGRAM_BOT_PIN=1234                            # PIN для pause/stop/resume
```

### Крок 1.7 — Тести

**Файл:** `tests/test_telegram_bot.py` [НОВИЙ]

```python
# Мок telegram.Update + telegram.ext.ContextTypes
# Тест: незнайомий chat_id → rejected
# Тест: /status → форматований текст з equity
# Тест: /pause без PIN → відхилено
# Тест: /pause <pin> → control.json оновлено
# Тест: /trades 5 → 5 рядків
# Тест: rate limit → 31-й запит блокується
```

**Критерій виходу Sprint 1:**
- [ ] `uv run pytest tests/test_telegram_bot.py -q` — зелений
- [ ] `uv run python -m scalper_hft.cli telegram-bot start` — стартує без помилок
- [ ] З телефону: `/status` повертає equity і позиції

---

## Sprint 2 — UnifiedTradeStore + SyncEngine

> **Мета**: єдине SQLite сховище з підтримкою `exchange` та `mode`; автоматична синхронізація балансу і позицій.

### Крок 2.1 — Міграція схеми

**Файл:** [`scalper_hft/live/store.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/live/store.py)

```diff
 class PaperStore:
+    """Розширено до UnifiedTradeStore: exchange + mode у всіх таблицях."""
+
+    SCHEMA_VERSION = 2
```

Зміни схеми (backward-compatible через `ALTER TABLE IF NOT EXISTS`):

```sql
-- Міграція v1 → v2
ALTER TABLE equity   ADD COLUMN exchange TEXT NOT NULL DEFAULT 'binance';
ALTER TABLE equity   ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper';
ALTER TABLE orders   ADD COLUMN exchange TEXT NOT NULL DEFAULT 'binance';
ALTER TABLE orders   ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper';
ALTER TABLE trades   ADD COLUMN exchange TEXT NOT NULL DEFAULT 'binance';
ALTER TABLE trades   ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper';

-- Нові таблиці
CREATE TABLE IF NOT EXISTS accounts ( ... );    -- баланс по біржах
CREATE TABLE IF NOT EXISTS positions ( ... );   -- відкриті позиції
CREATE TABLE IF NOT EXISTS bots ( ... );        -- реєстр ботів
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);
```

**Нові методи `PaperStore`:**
```python
def log_account(
    self, ts, exchange: str, mode: str, balance: float, unrealized_pnl: float, margin_used: float, available: float
) -> None: ...


def log_position(
    self,
    ts,
    exchange: str,
    symbol: str,
    side: str,
    size: float,
    entry_price: float,
    mark_price: float | None,
    unrealized_pnl: float | None,
    mode: str,
) -> None: ...


def upsert_bot(
    self, bot_id: str, exchange: str, symbol: str, interval: str, strategy: str, mode: str, status: str, config: dict
) -> None: ...


def update_bot_heartbeat(self, bot_id: str) -> None: ...


def open_positions(self, exchange: str | None = None) -> pd.DataFrame: ...
def latest_account(self, exchange: str = "binance") -> dict | None: ...
```

### Крок 2.2 — SyncEngine [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/live/sync_engine.py`

```python
"""
SyncEngine: синхронізує локальний TradeStore з біржею через REST.

Запускається в окремому daemon-thread кожні SYNC_INTERVAL секунд.
Paper mode: sync_account і sync_positions — no-op (симуляція).
Live mode: реальні запити через ccxt.
"""


class SyncEngine:
    def __init__(self, store: PaperStore, exchange_id: str, mode: str = "paper", interval_sec: int = 30): ...

    def sync_account(self) -> None:
        """REST /account → store.log_account()"""

    def sync_positions(self) -> None:
        """REST /positions → store.log_position() + KillSwitch якщо drift"""

    def sync_fills(self) -> None:
        """REST /userTrades (last 500) → порівняти з store.trades"""

    def start(self) -> None:
        """Запуск daemon thread з SYNC_INTERVAL циклом."""

    def stop(self) -> None:
        """Graceful shutdown."""
```

### Крок 2.3 — Інтеграція в `pairs_runner.py`

**Файл:** [`scalper_hft/live/pairs_runner.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/live/pairs_runner.py)

```diff
+from scalper_hft.live.sync_engine import SyncEngine
+
 class PairsPaperRunner:
     def __init__(self, ...):
         ...
+        self._sync = SyncEngine(self._store, exchange_id="binance",
+                                mode="paper")
+        self._sync.start()
+
+    def __del__(self):
+        self._sync.stop()
```

### Крок 2.4 — Тести

**Файл:** `tests/test_store_v2.py` [НОВИЙ]

```python
# Тест: міграція з v1 SQLite → v2 (зберігає існуючі дані)
# Тест: log_account, open_positions, upsert_bot, update_heartbeat
# Тест: SyncEngine paper mode → no-op (не викликає ccxt)
# Тест: SyncEngine live mode → мок ccxt, перевірити log_account викликано
```

Розширити `tests/test_store.py` (існуючий):
- Перевірити backward-compatibility після `ALTER TABLE`

**Критерій виходу Sprint 2:**
- [ ] `uv run pytest tests/test_store.py tests/test_store_v2.py -q` — зелений
- [ ] Існуючий SQLite файл успішно мігрує без втрати даних
- [ ] Telegram `/status` показує дані з `latest_account()`

---

## Sprint 3 — Dashboard Auth + Multi-Exchange UI

> **Мета**: захист паролем, нові сторінки Multi-Exchange і Research Hub.

### Крок 3.1 — Нові залежності

```diff
 [project.optional-dependencies]
 dashboard = [
     "streamlit>=1.30",
+    "bcrypt>=4.0",
 ]
```

### Крок 3.2 — Dashboard Auth [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/dashboard_auth.py`

```python
"""
Захист Streamlit dashboard паролем.

Зберігає bcrypt-хеш у .env як DASHBOARD_PASSWORD_HASH.
Після успішного входу — session token у st.session_state (8 год TTL).

Генерація хешу:
    uv run python -m scalper_hft.cli dashboard-hash <пароль>
"""


def check_password() -> bool:
    """Повертає True якщо сесія авторизована."""
    ...


def render_login_form() -> None:
    """Форма входу: пароль + кнопка Увійти."""
    ...
```

```diff
 # .env.example
+DASHBOARD_PASSWORD_HASH=<bcrypt_hash>  # генерується через cli dashboard-hash
```

### Крок 3.3 — Підключення auth до dashboard

**Файл:** [`scalper_hft/dashboard.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/dashboard.py)

```diff
+from scalper_hft.dashboard_auth import check_password
+
+if not check_password():
+    st.stop()
```

### Крок 3.4 — Нова сторінка: Multi-Exchange [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/app_pages/multi_exchange.py`

Секції сторінки:
- **Фільтри**: мультиселект бірж, символ, таймфрейм, діапазон дат
- **Price Comparison**: Plotly subplots (normalized returns по біржах)
- **Spread / Basis**: різниця цін між біржами
- **Funding Rate Comparison**: bar chart по біржах
- **Correlation Heatmap**: кореляція між біржами для символу
- **Data Availability**: таблиця — які дані є локально по кожній біржі

### Крок 3.5 — Оновлення Overview

**Файл:** [`scalper_hft/app_pages/overview.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/app_pages/overview.py)

```diff
+# Верхній рядок: KPI по всіх біржах і ботах
+col1, col2, col3, col4 = st.columns(4)
+col1.metric("Total Equity", ...)     # сума по всіх mode/exchange
+col2.metric("Open Positions", ...)   # кількість + по скільки бірж
+col3.metric("Active Bots", ...)      # running/total з таблиці bots
+col4.metric("Today PnL", ...)
+
+# Додати: таблиця активних ботів зі статусом
+render_bot_status_cards(store)
```

### Крок 3.6 — Нова сторінка: Research Hub [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/app_pages/research_hub.py`

Вкладки:
- **Backtests** — сканувати `results/{exchange}/{strategy}/` → таблиця з Sharpe/MaxDD
- **Strategy Comparison** — Sharpe heatmap: рядки = стратегії, колонки = біржі
- **Overfitting Audit** — summary DSR, CSCV PBO з останніх `report` запусків

### Крок 3.7 — CLI: генерація хешу пароля

**Файл:** [`scalper_hft/cli.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/cli.py)

```bash
# Нова утиліта:
uv run python -m scalper_hft.cli dashboard-hash <пароль>
# → виводить DASHBOARD_PASSWORD_HASH=<hash> для .env
```

### Крок 3.8 — Тести

Розширити `tests/test_dashboard.py` (існуючий):
```python
# Тест: check_password() з неправильним хешем → False
# Тест: check_password() з правильним → True + session token
# Тест: multi_exchange page imports → немає ImportError
# Тест: research_hub page imports → немає ImportError
```

**Критерій виходу Sprint 3:**
- [ ] Dashboard вимагає пароль при відкритті
- [ ] Multi-Exchange сторінка завантажується з локальними parquet даними
- [ ] `uv run pytest tests/test_dashboard.py -q` — зелений

---

## Sprint 4 — Exchange Registry + DataLayer

> **Мета**: `--exchange` параметр у всіх CLI командах; bybit/okx дані в окремих папках.

### Крок 4.1 — ExchangeRegistry [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/data/exchange_registry.py`

```python
"""Реєстр підтримуваних бірж і їх параметрів."""


@dataclass(frozen=True)
class ExchangeSpec:
    ccxt_id: str
    exchange_type: str  # futures_usdt | linear | swap
    maker_fee: float
    taker_fee: float
    has_funding: bool
    has_agg_trades: bool
    kline_limit: int = 1000  # макс свічок за запит
    funding_interval_h: int = 8


REGISTRY: dict[str, ExchangeSpec] = {
    "binance": ExchangeSpec(...),
    "bybit": ExchangeSpec(...),
    "okx": ExchangeSpec(...),
    "gateio": ExchangeSpec(...),
}


def get_exchange(name: str) -> ExchangeSpec: ...
def list_exchanges() -> list[str]: ...
```

### Крок 4.2 — DataClient: `exchange=` параметр

**Файл:** [`scalper_hft/data/client.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/data/)

Зміни:
```diff
-def fetch_klines(symbol, interval, days=90, use_cache=True):
+def fetch_klines(symbol, interval, exchange="binance", days=90, use_cache=True):
+    spec = get_exchange(exchange)
+    cache_path = settings.data_dir_abs / exchange / "klines" / symbol / interval
     ...
```

Аналогічно для `fetch_funding_rates()`, `fetch_agg_trades()`.

### Крок 4.3 — CacheManager [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/data/cache_manager.py`

```python
"""
Управління кешем Parquet по біржах.

data/_meta/cache_index.json:
{
  "binance": {
    "BTCUSDT/1h": {"last_ts": "2026-09-05T23:00:00Z", "rows": 8760}
  },
  "bybit": { ... }
}
"""


class CacheManager:
    def is_fresh(self, exchange, symbol, interval, max_age_hours=2) -> bool: ...

    def mark_updated(self, exchange, symbol, interval, last_ts: str, rows: int) -> None: ...

    def list_available(self, exchange: str | None = None) -> dict: ...
```

### Крок 4.4 — CLI: `--exchange` у всіх командах

**Файл:** [`scalper_hft/cli.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/cli.py)

```diff
+@click.option("--exchange", default="binance",
+              type=click.Choice(["binance","bybit","okx","gateio"]),
+              help="Біржа (default: binance)")
```

Додати до команд: `download`, `backtest`, `pairs`, `overfit`, `cscv`, `report`.

### Крок 4.5 — Оновлення Research (завантаження даних)

**Файл:** [`scalper_hft/app_pages/research.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/app_pages/research.py)

```diff
+exchange = st.selectbox("Біржа", list_exchanges(), index=0)
-df = load_klines(symbol, interval)
+df = load_klines(symbol, interval, exchange=exchange)
```

### Крок 4.6 — Тести

**Файл:** `tests/test_exchange_registry.py` [НОВИЙ]

```python
# Тест: get_exchange("binance") → правильні fees
# Тест: get_exchange("unknown") → KeyError
# Тест: fetch_klines з exchange="bybit" → кеш в data/bybit/...
# Тест: CacheManager.is_fresh() — TTL логіка
# Тест: CacheManager.list_available() — сканує реальну структуру
```

**Критерій виходу Sprint 4:**
- [ ] `uv run python -m scalper_hft.cli download --exchange bybit --symbol BTCUSDT --interval 1h --days 30` — завантажує в `data/bybit/`
- [ ] `uv run pytest tests/test_exchange_registry.py -q` — зелений

---

## Sprint 5 — FastAPI REST + WebSocket

> **Мета**: programmatic доступ до даних; WebSocket live-updates для dashboard.

### Крок 5.1 — Нові залежності

```diff
 [project.optional-dependencies]
+api = [
+    "fastapi>=0.110",
+    "uvicorn[standard]>=0.28",
+    "pyjwt>=2.8",
+]
```

### Крок 5.2 — API сервер [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/api/server.py`

```python
"""FastAPI REST + WebSocket API.

Аутентифікація: Bearer JWT (DASHBOARD_API_SECRET у .env).

Endpoints:
    GET  /api/v1/account/{exchange}    — баланс
    GET  /api/v1/positions             — відкриті позиції
    GET  /api/v1/trades                — угоди
    GET  /api/v1/equity                — equity curve (JSON)
    GET  /api/v1/bots                  — статус ботів
    POST /api/v1/bots/{bot_id}/pause   — призупинити
    POST /api/v1/bots/{bot_id}/resume  — відновити
    GET  /api/v1/exchanges             — список бірж і доступних даних
    WS   /ws/live                      — real-time позиції та угоди
"""
```

**Файл:** `scalper_hft/api/auth.py`

```python
"""JWT Bearer token аутентифікація для API."""


def create_token(secret: str, expires_hours: int = 24) -> str: ...
def verify_token(token: str = Depends(security)) -> dict: ...
```

### Крок 5.3 — WebSocket live-broadcast

**Файл:** `scalper_hft/api/broadcaster.py`

```python
"""
ConnectionManager: зберігає активні WS-з'єднання.
SyncEngine та PairsPaperRunner надсилають події через broadcast().
"""


class ConnectionManager:
    async def connect(self, ws: WebSocket) -> None: ...
    def disconnect(self, ws: WebSocket) -> None: ...
    async def broadcast(self, event: dict) -> None: ...


manager = ConnectionManager()  # singleton
```

### Крок 5.4 — CLI команди

**Файл:** [`scalper_hft/cli.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/cli.py)

```bash
uv run python -m scalper_hft.cli api start [--port 8080] [--host 127.0.0.1]
uv run python -m scalper_hft.cli api token   # генерує JWT для .env
```

### Крок 5.5 — systemd unit [НОВИЙ ФАЙЛ]

**Файл:** `deploy/scalper-api.service`

```ini
[Unit]
Description=Scalper HFT API Server
After=network.target

[Service]
Type=simple
Restart=on-failure
EnvironmentFile=%h/scalper-hft/.env
WorkingDirectory=...
ExecStart=uv run python -m scalper_hft.cli api start --host 127.0.0.1 --port 8080

[Install]
WantedBy=default.target
```

### Крок 5.6 — Streamlit: використовує API

**Файл:** [`scalper_hft/app_pages/overview.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/app_pages/overview.py)

```diff
+# Якщо API_BASE_URL налаштовано → через HTTP; інакше → пряме читання SQLite
+if settings.api_base_url:
+    data = requests.get(f"{settings.api_base_url}/api/v1/positions").json()
+else:
+    data = store.open_positions().to_dict("records")
```

### Крок 5.7 — Тести

**Файл:** `tests/test_api.py` [НОВИЙ]

```python
from fastapi.testclient import TestClient

# Тест: GET /api/v1/account/binance без токена → 401
# Тест: GET /api/v1/account/binance з токеном → 200, поля balance
# Тест: POST /api/v1/bots/test_bot/pause → control.json оновлено
# Тест: WS /ws/live → підключення + отримання broadcast події
```

**Критерій виходу Sprint 5:**
- [ ] `uv run python -m scalper_hft.cli api start` — стартує на 127.0.0.1:8080
- [ ] `curl -H "Authorization: Bearer <token>" localhost:8080/api/v1/positions`
- [ ] `uv run pytest tests/test_api.py -q` — зелений

---

## Sprint 6 — YAML Strategy Configs + SupervisorConfig

> **Мета**: гнучке управління стратегіями без зміни коду; один бот — N стратегій для N режимів.

### Крок 6.1 — Нові залежності

```diff
+pyyaml = ">=6.0"
```

### Крок 6.2 — SupervisorConfig [НОВИЙ ФАЙЛ]

**Файл:** `scalper_hft/live/supervisor_config.py`

```python
"""
Конфігурація запуску RegimeSupervisor з YAML-файлу.

YAML-схема:
    name: regime_supervisor
    exchange: binance
    symbol: BTCUSDT
    interval: 1h
    mode: paper
    strategies: "mean_reversion,supertrend,hmm_reversion"
    blend_mode: contextual_hedge
    risk:
      notional_pct: 0.05
      max_dd: 0.10
    regime_overrides:
      trend_up:
        preferred: "supertrend,cross_momentum"
        weight_boost: 1.5
"""


@dataclass
class RiskConfig:
    notional_pct: float = 0.05
    max_dd: float = 0.10
    daily_loss_limit: float = 0.02


@dataclass
class RegimeOverride:
    preferred: list[str]
    weight_boost: float = 1.0


@dataclass
class SupervisorConfig:
    name: str
    exchange: str
    symbol: str
    interval: str
    mode: str
    strategies: str
    blend_mode: str = "contextual_hedge"
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime_overrides: dict[str, RegimeOverride] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SupervisorConfig": ...

    def to_strategy_params(self) -> dict: ...
```

### Крок 6.3 — Приклади конфігів [НОВІ ФАЙЛИ]

```
configs/
└── strategies/
    ├── btcusdt_1h_binance.yaml
    ├── xrpbtc_pairs_1h.yaml
    └── multi_regime_bybit.yaml
```

**Файл:** `configs/strategies/btcusdt_1h_binance.yaml`

```yaml
name: regime_supervisor
exchange: binance
symbol: BTCUSDT
interval: 1h
mode: paper

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

### Крок 6.4 — CLI: запуск з YAML

**Файл:** [`scalper_hft/cli.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/cli.py)

```bash
# Нова команда:
uv run python -m scalper_hft.cli run --config configs/strategies/btcusdt_1h_binance.yaml

# Або залишити існуючий стиль + --exchange:
uv run python -m scalper_hft.cli paper-run-pairs --config configs/strategies/xrpbtc_pairs_1h.yaml
```

### Крок 6.5 — Інтеграція в RegimeSupervisor

**Файл:** [`scalper_hft/strategies/regime_supervisor.py`](file:///home/volodymyr/PycharmProjects/scalper-hft/scalper_hft/strategies/regime_supervisor.py)

```diff
+    @classmethod
+    def from_config(cls, cfg: SupervisorConfig) -> "RegimeSupervisor":
+        """Створити з SupervisorConfig (завантаженого з YAML)."""
+        params = cfg.to_strategy_params()
+        return cls(**params)
```

### Крок 6.6 — Тести

**Файл:** `tests/test_supervisor_config.py` [НОВИЙ]

```python
# Тест: SupervisorConfig.from_yaml(valid_yaml) → правильні поля
# Тест: from_yaml(invalid_yaml) → ValueError з описом
# Тест: RegimeSupervisor.from_config(cfg) → стратегія стартує без помилок
# Тест: to_strategy_params() → містить усі необхідні поля для __init__
```

**Критерій виходу Sprint 6:**
- [ ] `uv run python -m scalper_hft.cli run --config configs/strategies/btcusdt_1h_binance.yaml` — запускає paper
- [ ] `uv run pytest tests/test_supervisor_config.py -q` — зелений

---

## Загальні правила для всіх спринтів

> [!IMPORTANT]
> Після кожного спринту обов'язково:
> ```bash
> uv run pytest tests/ -q
> uv run ruff check --fix && uv run ruff format
> ```
> Обидві команди мають завершуватись без помилок.

> [!WARNING]
> `DRY_RUN=true` на локальній машині завжди.  
> SyncEngine у live mode — лише після Paper-Gate (Phase 2 з DEPLOY_PLAN.md).  
> API endpoint `/api/v1/bots/{id}/pause` — лише через localhost, не публічно.

> [!NOTE]
> Git workflow: один PR на спринт.  
> Назви гілок: `feature/sprint1-telegram-bot`, `feature/sprint2-unified-store`, etc.

---

## Залежності між спринтами

```
Sprint 1 (Telegram Bot)
    │
    ├──→ Sprint 2 (Store) → Sprint 3 (Dashboard) → Sprint 5 (API)
    │                              │
    └──────────────────────────────┘
    
Sprint 4 (Exchange Registry) → Sprint 3 (Dashboard multi-exchange)
Sprint 6 (YAML Configs) — незалежний, можна паралельно зі Sprint 4-5
```

**Можна починати Sprint 4 і Sprint 6 паралельно зі Sprint 2-3.**

---

## Підсумок: нові файли і файли-зміни

| Файл | Тип | Sprint |
|---|---|---|
| `scalper_hft/live/telegram_bot.py` | **НОВИЙ** | 1 |
| `scalper_hft/live/telegram.py` | **ЗМІНА** | 1 |
| `scalper_hft/config.py` | **ЗМІНА** | 1 |
| `scalper_hft/cli.py` | **ЗМІНА** | 1, 2, 3, 4, 5, 6 |
| `tests/test_telegram_bot.py` | **НОВИЙ** | 1 |
| `scalper_hft/live/store.py` | **ЗМІНА** | 2 |
| `scalper_hft/live/sync_engine.py` | **НОВИЙ** | 2 |
| `scalper_hft/live/pairs_runner.py` | **ЗМІНА** | 2 |
| `tests/test_store_v2.py` | **НОВИЙ** | 2 |
| `scalper_hft/dashboard_auth.py` | **НОВИЙ** | 3 |
| `scalper_hft/dashboard.py` | **ЗМІНА** | 3 |
| `scalper_hft/app_pages/overview.py` | **ЗМІНА** | 3 |
| `scalper_hft/app_pages/multi_exchange.py` | **НОВИЙ** | 3 |
| `scalper_hft/app_pages/research_hub.py` | **НОВИЙ** | 3 |
| `scalper_hft/data/exchange_registry.py` | **НОВИЙ** | 4 |
| `scalper_hft/data/client.py` | **ЗМІНА** | 4 |
| `scalper_hft/data/cache_manager.py` | **НОВИЙ** | 4 |
| `scalper_hft/app_pages/research.py` | **ЗМІНА** | 4 |
| `tests/test_exchange_registry.py` | **НОВИЙ** | 4 |
| `scalper_hft/api/server.py` | **НОВИЙ** | 5 |
| `scalper_hft/api/auth.py` | **НОВИЙ** | 5 |
| `scalper_hft/api/broadcaster.py` | **НОВИЙ** | 5 |
| `deploy/scalper-api.service` | **НОВИЙ** | 5 |
| `tests/test_api.py` | **НОВИЙ** | 5 |
| `scalper_hft/live/supervisor_config.py` | **НОВИЙ** | 6 |
| `configs/strategies/*.yaml` | **НОВІ** | 6 |
| `scalper_hft/strategies/regime_supervisor.py` | **ЗМІНА** | 6 |
| `tests/test_supervisor_config.py` | **НОВИЙ** | 6 |
| `pyproject.toml` | **ЗМІНА** | 1, 3, 5, 6 |
| `.env.example` | **ЗМІНА** | 1, 3, 5 |

**Разом**: 12 нових файлів, 16 змін у існуючих файлів, ~25–30 робочих днів.
