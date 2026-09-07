"""Тести для PaperStore v2 (UnifiedTradeStore): міграція схеми, нові таблиці, SyncEngine."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from scalper_hft.live.store import SCHEMA_VERSION, PaperStore
from scalper_hft.live.sync_engine import SyncEngine

# ─── Фікстури ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    """Свіжий PaperStore у тимчасовій директорії."""
    return PaperStore(tmp_path / "test_v2.sqlite")


@pytest.fixture
def v1_store_path(tmp_path: Path) -> Path:
    """SQLite файл із v1 схемою (без exchange/mode колонок)."""
    db_path = tmp_path / "v1_store.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE equity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, pair TEXT NOT NULL,
            equity REAL NOT NULL, cash REAL NOT NULL, realized_pnl REAL NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, pair TEXT NOT NULL, symbol TEXT NOT NULL,
            side TEXT NOT NULL, size REAL NOT NULL, price REAL NOT NULL,
            status TEXT NOT NULL, reason TEXT NOT NULL
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, pair TEXT NOT NULL, symbol TEXT NOT NULL,
            side TEXT NOT NULL, size REAL NOT NULL, entry_price REAL,
            exit_price REAL, pnl REAL, kind TEXT NOT NULL
        );
        CREATE TABLE months (pair TEXT NOT NULL, month TEXT NOT NULL, pnl REAL NOT NULL, PRIMARY KEY (pair, month));
        CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
        CREATE TABLE snapshots (id TEXT PRIMARY KEY, payload TEXT NOT NULL, saved_at TEXT NOT NULL);
    """)
    # Заповнити тестовими даними v1
    conn.execute(
        "INSERT INTO equity (ts, pair, equity, cash, realized_pnl) VALUES ('2026-01-01', 'XRP/BTC', 1000, 900, 100)"
    )
    conn.execute(
        "INSERT INTO trades (ts, pair, symbol, side, size, entry_price, kind) VALUES ('2026-01-01', 'XRP/BTC', 'XRPUSDT', 'buy', 100, 0.5, 'trade')"
    )
    conn.execute(
        "INSERT INTO orders (ts, pair, symbol, side, size, price, status, reason) VALUES ('2026-01-01', 'XRP/BTC', 'XRPUSDT', 'buy', 100, 0.5, 'filled', 'entry')"
    )
    conn.commit()
    conn.close()
    return db_path


# ─── Тести схеми і міграції ───────────────────────────────────────────────────


def test_new_store_schema_version(store: PaperStore) -> None:
    """Новий store має правильну версію схеми."""
    assert store.schema_version() == SCHEMA_VERSION


def test_migration_v1_to_v2_preserves_data(v1_store_path: Path) -> None:
    """Міграція v1 → v2: існуючі дані збережено."""
    store = PaperStore(v1_store_path)

    # v1 дані доступні
    equity = store.all_equity()
    assert len(equity) == 1
    assert equity.iloc[0]["pair"] == "XRP/BTC"
    assert equity.iloc[0]["equity"] == 1000.0

    trades = store.all_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["symbol"] == "XRPUSDT"

    orders = store.all_orders()
    assert len(orders) == 1


def test_migration_adds_exchange_column(v1_store_path: Path) -> None:
    """Після міграції в таблиці equity є колонка exchange."""
    PaperStore(v1_store_path).close()
    conn = sqlite3.connect(v1_store_path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(equity)").fetchall()]
    conn.close()
    assert "exchange" in cols
    assert "mode" in cols


def test_migration_adds_orders_columns(v1_store_path: Path) -> None:
    """Після міграції в orders є exchange і mode."""
    PaperStore(v1_store_path)
    conn = sqlite3.connect(v1_store_path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(orders)").fetchall()]
    conn.close()
    assert "exchange" in cols
    assert "mode" in cols


def test_migration_adds_trades_columns(v1_store_path: Path) -> None:
    """Після міграції в trades є exchange і mode."""
    PaperStore(v1_store_path)
    conn = sqlite3.connect(v1_store_path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()]
    conn.close()
    assert "exchange" in cols
    assert "mode" in cols


def test_migration_idempotent(v1_store_path: Path) -> None:
    """Подвійна міграція не ламає базу."""
    PaperStore(v1_store_path)
    store2 = PaperStore(v1_store_path)  # друге відкриття
    assert store2.schema_version() == SCHEMA_VERSION
    assert len(store2.all_equity()) == 1


def test_new_tables_created(store: PaperStore) -> None:
    """Нові таблиці v2 створені в свіжій базі."""
    conn = store._conn
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "accounts" in tables
    assert "positions" in tables
    assert "bots" in tables


# ─── Тести log_equity з exchange/mode ─────────────────────────────────────────


def test_log_equity_with_exchange(store: PaperStore) -> None:
    """log_equity зберігає exchange і mode."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_equity(ts, "XRP/BTC", equity=1050.0, cash=950.0, realized_pnl=50.0, exchange="bybit", mode="live")
    row = store._conn.execute("SELECT exchange, mode FROM equity WHERE pair='XRP/BTC'").fetchone()
    assert row["exchange"] == "bybit"
    assert row["mode"] == "live"


def test_log_equity_default_exchange(store: PaperStore) -> None:
    """log_equity без explicit exchange → default 'binance'/'paper'."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_equity(ts, "XRP/BTC", 1000.0, 900.0, 100.0)
    row = store._conn.execute("SELECT exchange, mode FROM equity").fetchone()
    assert row["exchange"] == "binance"
    assert row["mode"] == "paper"


# ─── Тести accounts ───────────────────────────────────────────────────────────


def test_log_and_latest_account(store: PaperStore) -> None:
    """log_account → latest_account повертає останній запис."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_account(ts, "binance", "paper", 10000.0, 250.0, 1000.0, 9000.0)
    acc = store.latest_account("binance", "paper")
    assert acc is not None
    assert acc["balance"] == 10000.0
    assert acc["unrealized_pnl"] == 250.0
    assert acc["available"] == 9000.0


def test_latest_account_empty(store: PaperStore) -> None:
    """latest_account без даних → None."""
    assert store.latest_account("okx", "paper") is None


def test_latest_account_exchange_filtered(store: PaperStore) -> None:
    """latest_account фільтрує по exchange."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_account(ts, "binance", "paper", 10000.0, 0.0, 0.0, 10000.0)
    store.log_account(ts, "bybit", "paper", 5000.0, 0.0, 0.0, 5000.0)

    binance_acc = store.latest_account("binance", "paper")
    bybit_acc = store.latest_account("bybit", "paper")

    assert binance_acc is not None
    assert binance_acc["balance"] == 10000.0
    assert bybit_acc is not None
    assert bybit_acc["balance"] == 5000.0


def test_recent_accounts_returns_df(store: PaperStore) -> None:
    """recent_accounts повертає DataFrame з правильними колонками."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_account(ts, "binance", "paper", 10000.0, 100.0, 500.0, 9500.0)
    df = store.recent_accounts("binance", "paper", limit=10)
    assert not df.empty
    assert "balance" in df.columns
    assert "unrealized_pnl" in df.columns


def test_all_accounts_empty(store: PaperStore) -> None:
    """all_accounts без даних → порожній DataFrame."""
    df = store.all_accounts()
    assert df.empty
    assert "exchange" in df.columns
    assert "balance" in df.columns


def test_all_accounts_includes_all_exchanges(store: PaperStore) -> None:
    """all_accounts повертає знімки з усіх бірж, не лише однієї."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_account(ts, "binance", "paper", 10000.0, 0.0, 0.0, 10000.0)
    store.log_account(ts, "bybit", "live", 5000.0, 1.0, 0.0, 5000.0)
    df = store.all_accounts()
    assert set(df["exchange"]) == {"binance", "bybit"}
    assert set(df["mode"]) == {"paper", "live"}


def test_paper_store_context_manager_closes_connection(tmp_path: Path) -> None:
    """with PaperStore закриває SQLite-з'єднання на виході."""
    path = tmp_path / "ctx.sqlite"
    with PaperStore(path) as ctx_store:
        assert ctx_store.schema_version() == SCHEMA_VERSION
    with pytest.raises(sqlite3.ProgrammingError):
        ctx_store.schema_version()


def test_dashboard_kpi_read_pattern(tmp_path: Path) -> None:
    """Overview KPI читає accounts/bots через публічний API в with-блоці."""
    path = tmp_path / "kpi.sqlite"
    seed = PaperStore(path)
    ts = pd.Timestamp.now(tz="UTC")
    seed.log_account(ts, "binance", "paper", 10000.0, 25.0, 0.0, 10000.0)
    seed.upsert_bot("bot1", "binance", "BTCUSDT", "1h", "pairs_arb", "paper", "running")
    seed.close()

    with PaperStore(path) as p_store:
        accounts = p_store.all_accounts()
        bots = p_store.all_bots()

    assert not accounts.empty
    assert accounts.iloc[0]["balance"] == 10000.0
    assert list(bots["bot_id"]) == ["bot1"]
    assert "last_heartbeat" in bots.columns
    assert "pid" not in bots.columns
    assert "last_ping" not in bots.columns


# ─── Тести positions ──────────────────────────────────────────────────────────


def test_log_and_open_positions(store: PaperStore) -> None:
    """log_position → open_positions повертає відкриті позиції."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_position(ts, "binance", "BTCUSDT", "long", 0.1, 60000.0, mark_price=61000.0, unrealized_pnl=100.0)
    store.log_position(ts, "binance", "ETHUSDT", "short", 1.0, 3000.0, mark_price=2950.0, unrealized_pnl=50.0)

    df = store.open_positions(exchange="binance")
    assert len(df) == 2
    symbols = set(df["symbol"])
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


def test_open_positions_zero_size_excluded(store: PaperStore) -> None:
    """Позиції з size=0 не відображаються як відкриті."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_position(ts, "binance", "BTCUSDT", "long", 0.1, 60000.0)
    store.log_position(ts, "binance", "BTCUSDT", "long", 0.0, 60000.0)  # закрита

    df = store.open_positions(exchange="binance")
    assert len(df) == 0  # остання позиція має size=0


def test_open_positions_exchange_filter(store: PaperStore) -> None:
    """open_positions фільтрує по exchange."""
    ts = pd.Timestamp.now(tz="UTC")
    store.log_position(ts, "binance", "BTCUSDT", "long", 0.1, 60000.0)
    store.log_position(ts, "bybit", "ETHUSDT", "short", 1.0, 3000.0)

    binance_pos = store.open_positions(exchange="binance")
    bybit_pos = store.open_positions(exchange="bybit")

    assert len(binance_pos) == 1
    assert binance_pos.iloc[0]["symbol"] == "BTCUSDT"
    assert len(bybit_pos) == 1
    assert bybit_pos.iloc[0]["symbol"] == "ETHUSDT"


# ─── Тести bots ───────────────────────────────────────────────────────────────


def test_upsert_bot_new(store: PaperStore) -> None:
    """upsert_bot реєструє нового бота."""
    store.upsert_bot(
        "bot_xrpbtc",
        "binance",
        "XRPUSDT",
        "1h",
        "regime_supervisor",
        "paper",
        "running",
        config={"blend_mode": "contextual_hedge"},
    )
    df = store.all_bots()
    assert len(df) == 1
    assert df.iloc[0]["bot_id"] == "bot_xrpbtc"
    assert df.iloc[0]["status"] == "running"


def test_upsert_bot_update(store: PaperStore) -> None:
    """upsert_bot оновлює існуючого бота."""
    store.upsert_bot("bot1", "binance", "BTCUSDT", "1h", "pairs_arb", "paper", "running")
    store.upsert_bot("bot1", "binance", "BTCUSDT", "1h", "pairs_arb", "paper", "paused")
    df = store.all_bots()
    assert len(df) == 1  # не дублює
    assert df.iloc[0]["status"] == "paused"


def test_update_bot_heartbeat(store: PaperStore) -> None:
    """update_bot_heartbeat оновлює last_heartbeat."""
    store.upsert_bot("bot1", "binance", "BTCUSDT", "1h", "pairs_arb", "paper", "running")
    store.update_bot_heartbeat("bot1")
    df = store.all_bots()
    assert df.iloc[0]["last_heartbeat"] is not None


def test_set_bot_status(store: PaperStore) -> None:
    """set_bot_status змінює статус бота."""
    store.upsert_bot("bot1", "binance", "BTCUSDT", "1h", "pairs_arb", "paper", "running")
    store.set_bot_status("bot1", "stopped")
    df = store.all_bots()
    assert df.iloc[0]["status"] == "stopped"


def test_all_bots_empty(store: PaperStore) -> None:
    """all_bots без ботів → порожній DataFrame."""
    df = store.all_bots()
    assert df.empty


# ─── Тести SyncEngine ─────────────────────────────────────────────────────────


def test_sync_engine_paper_account_noop(tmp_path: Path) -> None:
    """SyncEngine paper mode sync_account — no-op, не викликає ccxt."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, exchange_id="binance", mode="paper")

    with patch("scalper_hft.live.sync_engine.SyncEngine._get_exchange") as mock_ex:
        engine.sync_account()
        mock_ex.assert_not_called()


def test_sync_engine_paper_positions_noop(tmp_path: Path) -> None:
    """SyncEngine paper mode sync_positions — no-op."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, exchange_id="binance", mode="paper")

    with patch("scalper_hft.live.sync_engine.SyncEngine._get_exchange") as mock_ex:
        engine.sync_positions()
        mock_ex.assert_not_called()


def test_sync_engine_paper_fills_noop(tmp_path: Path) -> None:
    """SyncEngine paper mode sync_fills — no-op."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, exchange_id="binance", mode="paper")
    with patch("scalper_hft.live.sync_engine.SyncEngine._get_exchange") as mock_ex:
        engine.sync_fills()
        mock_ex.assert_not_called()


def test_sync_engine_live_account_calls_ccxt(tmp_path: Path) -> None:
    """SyncEngine live mode sync_account → викликає ccxt і пише до store."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, exchange_id="binance", mode="live")

    # Мок ccxt exchange
    mock_exchange = MagicMock()
    mock_exchange.fetch_balance.return_value = {"USDT": {"total": 10000.0, "free": 9000.0, "used": 1000.0}}
    mock_exchange.fetch_positions.return_value = []

    with patch.object(engine, "_get_exchange", return_value=mock_exchange):
        engine.sync_account()

    mock_exchange.fetch_balance.assert_called_once()

    acc = store.latest_account("binance", "live")
    assert acc is not None
    assert acc["balance"] == 10000.0
    assert acc["available"] == 9000.0


def test_sync_engine_live_positions_calls_ccxt(tmp_path: Path) -> None:
    """SyncEngine live mode sync_positions → викликає ccxt і пише до store."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, exchange_id="binance", mode="live")

    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": 0.1,
            "entryPrice": 60000.0,
            "markPrice": 61000.0,
            "unrealizedPnl": 100.0,
        }
    ]

    with patch.object(engine, "_get_exchange", return_value=mock_exchange):
        engine.sync_positions()

    mock_exchange.fetch_positions.assert_called_once()
    df = store.open_positions(exchange="binance", mode="live")
    assert len(df) == 1
    assert df.iloc[0]["entry_price"] == 60000.0


def test_sync_engine_start_stop(tmp_path: Path) -> None:
    """SyncEngine start() запускає thread, stop() зупиняє."""
    import time

    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(
        store,
        exchange_id="binance",
        mode="paper",
        account_interval_sec=10,
        position_interval_sec=10,
    )
    engine.start()
    assert engine.is_running
    time.sleep(0.1)
    engine.stop()
    assert not engine.is_running


def test_sync_engine_double_start_noop(tmp_path: Path) -> None:
    """Подвійний start() не ламає engine."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, mode="paper")
    engine.start()
    engine.start()  # другий виклик — без помилок
    assert engine.is_running
    engine.stop()


def test_sync_engine_live_positions_kill_switch(tmp_path: Path) -> None:
    """Live sync_positions з account → halt_if_drift при розходженні."""
    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.reconcile import KillSwitch

    store = PaperStore(tmp_path / "s.sqlite")
    account = PaperAccount(10_000.0)
    account.open_position("BTCUSDT", "long", 0.1, 60_000.0, pd.Timestamp("2024-01-01"))
    engine = SyncEngine(
        store,
        mode="live",
        account=account,
        scope={"BTCUSDT"},
        dry_run=False,
    )
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = []

    with patch.object(engine, "_get_exchange", return_value=mock_exchange):
        with pytest.raises(KillSwitch):
            engine.sync_positions()


def test_sync_engine_live_fills_logs_missing(tmp_path: Path) -> None:
    """sync_fills записує біржові fills, яких немає локально."""
    store = PaperStore(tmp_path / "s.sqlite")
    engine = SyncEngine(store, mode="live", scope={"BTCUSDT"}, dry_run=False)
    mock_exchange = MagicMock()
    mock_exchange.fetch_my_trades.return_value = [
        {
            "id": "999001",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 50000.0,
            "timestamp": 1_700_000_000_000,
        }
    ]
    mock_exchange.load_markets.return_value = {"BTC/USDT:USDT": {}}

    with patch.object(engine, "_get_exchange", return_value=mock_exchange):
        engine.sync_fills()

    orders = store.all_orders()
    assert len(orders) == 1
    assert orders.iloc[0]["status"] == "exchange_sync"
    assert "999001" in str(orders.iloc[0]["reason"])


def test_sync_engine_kill_switch_callback(tmp_path: Path) -> None:
    """on_kill_switch отримує reason при drift (як у _run_loop)."""
    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.reconcile import KillSwitch

    store = PaperStore(tmp_path / "s.sqlite")
    account = PaperAccount(10_000.0)
    account.open_position("ETHUSDT", "long", 1.0, 3000.0, pd.Timestamp("2024-01-01"))
    fired: list[str] = []
    engine = SyncEngine(
        store,
        mode="live",
        account=account,
        scope={"ETHUSDT"},
        dry_run=False,
        on_kill_switch=fired.append,
    )
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = []

    with patch.object(engine, "_get_exchange", return_value=mock_exchange):
        try:
            engine.sync_positions()
        except KillSwitch as exc:
            if engine._on_kill_switch is not None:
                engine._on_kill_switch(str(exc))

    assert fired and "ETHUSDT" in fired[0]


def test_store_thread_safe_writes(tmp_path: Path) -> None:
    """Паралельні записи в PaperStore не падають (WAL + lock)."""
    import threading

    store = PaperStore(tmp_path / "s.sqlite")

    def writer(i: int) -> None:
        for j in range(20):
            store.log_order(
                pd.Timestamp("2024-01-01"),
                f"p{i}",
                "BTCUSDT",
                "buy",
                0.01,
                50000.0,
                "test",
                f"{i}-{j}",
            )

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(store.all_orders()) == 80
