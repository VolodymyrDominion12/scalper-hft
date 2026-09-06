"""SQLite-журнал paper pairs (v2): equity, ордери, угоди, місячний PnL,
runtime snapshot, рахунки по біржах, відкриті позиції, реєстр ботів.

v2 backward-compatible: існуючі SQLite файли мігрують автоматично
через ALTER TABLE IF NOT EXISTS (SQLite ≥ 3.37 / Python 3.12).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = 2


class PaperStore:
    """Персистенція paper/live прогонів.

    v2 — UnifiedTradeStore:
        - equity, orders, trades, months, meta, snapshots (v1)
        - accounts, positions, bots (нові в v2)
        - поля exchange і mode в equity/orders/trades (додаються міграцією)

    Path за замовчуванням results/paper_pairs.sqlite.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path("results") / "paper_pairs.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=15.0)
        self._conn.row_factory = sqlite3.Row
        self._init()
        self._migrate()

    def _init(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                pair TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                realized_pnl REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                kind TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS months (
                pair TEXT NOT NULL,
                month TEXT NOT NULL,
                pnl REAL NOT NULL,
                PRIMARY KEY (pair, month)
            );
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                saved_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
            -- v2: рахунки по біржах
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance',
                mode TEXT NOT NULL DEFAULT 'paper',
                balance REAL NOT NULL DEFAULT 0.0,
                unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                margin_used REAL NOT NULL DEFAULT 0.0,
                available REAL NOT NULL DEFAULT 0.0
            );
            -- v2: відкриті позиції (синхронізація з біржею)
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance',
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                mark_price REAL,
                unrealized_pnl REAL,
                mode TEXT NOT NULL DEFAULT 'paper'
            );
            -- v2: реєстр активних ботів
            CREATE TABLE IF NOT EXISTS bots (
                bot_id TEXT PRIMARY KEY,
                exchange TEXT NOT NULL DEFAULT 'binance',
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL DEFAULT '1h',
                strategy TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paper',
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                last_heartbeat TEXT,
                config_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Backward-compatible міграція v1 → v2.

        Додає нові колонки через ALTER TABLE якщо їх немає.
        SQLite не підтримує ADD COLUMN IF NOT EXISTS до 3.37;
        використовуємо спробу і ігноруємо OperationalError.
        """
        migrations = [
            ("equity", "exchange TEXT NOT NULL DEFAULT 'binance'"),
            ("equity", "mode TEXT NOT NULL DEFAULT 'paper'"),
            ("orders", "exchange TEXT NOT NULL DEFAULT 'binance'"),
            ("orders", "mode TEXT NOT NULL DEFAULT 'paper'"),
            ("trades", "exchange TEXT NOT NULL DEFAULT 'binance'"),
            ("trades", "mode TEXT NOT NULL DEFAULT 'paper'"),
        ]
        cur = self._conn.cursor()
        for table, col_def in migrations:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # колонка вже існує — ок

        # Записати версію схеми (таблиця вже є після _init())
        tbl_exists = cur.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()[0]
        if not tbl_exists:
            # Ця гілка не мала б спрацювати після _init(), але на всяк випадок
            cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")

        existing = cur.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        if existing == 0:
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            cur.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PaperStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ─── v1: equity / orders / trades / months ───────────────────────────────

    def log_equity(
        self,
        ts: pd.Timestamp,
        pair: str,
        equity: float,
        cash: float,
        realized_pnl: float,
        exchange: str = "binance",
        mode: str = "paper",
    ) -> None:
        self._conn.execute(
            "INSERT INTO equity (ts, pair, equity, cash, realized_pnl, exchange, mode) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(ts), pair, equity, cash, realized_pnl, exchange, mode),
        )
        self._conn.commit()

    def log_order(
        self,
        ts: pd.Timestamp,
        pair: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        status: str,
        reason: str,
        exchange: str = "binance",
        mode: str = "paper",
    ) -> None:
        self._conn.execute(
            "INSERT INTO orders (ts, pair, symbol, side, size, price, status, reason, exchange, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts), pair, symbol, side, size, price, status, reason, exchange, mode),
        )
        self._conn.commit()

    def log_trade(
        self,
        ts: pd.Timestamp,
        pair: str,
        trade: dict,
        exchange: str = "binance",
        mode: str = "paper",
    ) -> None:
        kind = str(trade.get("type", "trade"))
        self._conn.execute(
            "INSERT INTO trades (ts, pair, symbol, side, size, entry_price, exit_price, pnl, kind, exchange, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(ts),
                pair,
                trade.get("symbol", ""),
                trade.get("side", ""),
                float(trade.get("size") or 0.0),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("pnl"),
                kind,
                exchange,
                mode,
            ),
        )
        self._conn.commit()

    def log_month(self, pair: str, month: str, pnl: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO months (pair, month, pnl) VALUES (?, ?, ?)",
            (pair, month, pnl),
        )
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
        return str(row["v"]) if row else default

    def recent_equity(self, limit: int = 500) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, pair, equity, cash, realized_pnl FROM equity ORDER BY id DESC LIMIT ?",
            self._conn,
            params=(limit,),
        )

    def recent_orders(self, limit: int = 200) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, pair, symbol, side, size, price, status, reason FROM orders ORDER BY id DESC LIMIT ?",
            self._conn,
            params=(limit,),
        )

    def all_equity(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, pair, equity, cash, realized_pnl FROM equity ORDER BY id",
            self._conn,
        )

    def all_orders(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, pair, symbol, side, size, price, status, reason FROM orders ORDER BY id",
            self._conn,
        )

    def all_trades(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, pair, symbol, side, size, entry_price, exit_price, pnl, kind FROM trades ORDER BY id",
            self._conn,
        )

    def all_months(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT pair, month, pnl FROM months ORDER BY month, pair", self._conn)

    def fill_stats(self, pair: str | None = None) -> dict[str, int]:
        if pair:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM orders WHERE pair = ? GROUP BY status",
                (pair,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM orders GROUP BY status").fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def save_runtime(self, payload: dict[str, Any], snapshot_id: str = "runtime") -> None:
        """Атомарно замінити runtime-знімок (рахунок + engine)."""
        body = json.dumps(payload, default=str)
        saved_at = str(pd.Timestamp.now(tz="UTC").tz_convert(None))
        self._conn.execute(
            "INSERT OR REPLACE INTO snapshots (id, payload, saved_at) VALUES (?, ?, ?)",
            (snapshot_id, body, saved_at),
        )
        self._conn.commit()

    def load_runtime(self, snapshot_id: str = "runtime") -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload FROM snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(str(row["payload"]))
        return data if isinstance(data, dict) else None

    # ─── v2: accounts ─────────────────────────────────────────────────────────

    def log_account(
        self,
        ts: pd.Timestamp,
        exchange: str,
        mode: str,
        balance: float,
        unrealized_pnl: float,
        margin_used: float,
        available: float,
    ) -> None:
        """Зберегти знімок балансу рахунку."""
        self._conn.execute(
            "INSERT INTO accounts (ts, exchange, mode, balance, unrealized_pnl, margin_used, available) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(ts), exchange, mode, balance, unrealized_pnl, margin_used, available),
        )
        self._conn.commit()

    def latest_account(self, exchange: str = "binance", mode: str = "paper") -> dict[str, Any] | None:
        """Останній знімок балансу для біржі/режиму."""
        row = self._conn.execute(
            "SELECT * FROM accounts WHERE exchange = ? AND mode = ? ORDER BY id DESC LIMIT 1",
            (exchange, mode),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def recent_accounts(self, exchange: str = "binance", mode: str = "paper", limit: int = 200) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, exchange, mode, balance, unrealized_pnl, margin_used, available FROM accounts WHERE exchange = ? AND mode = ? ORDER BY id DESC LIMIT ?",
            self._conn,
            params=(exchange, mode, limit),
        )

    def all_accounts(self) -> pd.DataFrame:
        """Усі знімки балансів по біржах і режимах."""
        return pd.read_sql_query(
            "SELECT ts, exchange, mode, balance, unrealized_pnl, margin_used, available FROM accounts ORDER BY id",
            self._conn,
        )

    # ─── v2: positions ────────────────────────────────────────────────────────

    def log_position(
        self,
        ts: pd.Timestamp | str | None = None,
        exchange: str = "binance",
        symbol: str = "",
        side: str = "flat",
        size: float = 0.0,
        entry_price: float = 0.0,
        mark_price: float | None = None,
        unrealized_pnl: float | None = None,
        mode: str = "paper",
    ) -> None:
        """Зберегти знімок позиції (append-only; open positions — з останніх записів).

        Підтримує як явні keyword arguments, так і виклики з/без ts (захист від зсуву аргументів).
        """
        # Захист: якщо перший аргумент — біржа (наприклад, "binance", а ts пропущено),
        # а side отримав числове значення (розмір), нормалізуємо позиційні параметри
        if isinstance(side, (int, float)) and isinstance(symbol, str) and symbol in ("long", "short", "flat"):
            mode = str(unrealized_pnl) if isinstance(unrealized_pnl, str) else mode
            unrealized_pnl = float(mark_price) if mark_price is not None else None
            mark_price = float(entry_price) if entry_price != 0.0 else None
            entry_price = float(size)
            size = float(side)
            side = str(symbol)
            symbol = str(exchange)
            exchange = str(ts) if ts is not None else "binance"
            ts_val = pd.Timestamp.now(tz="UTC")
        else:
            ts_val = ts if ts is not None else pd.Timestamp.now(tz="UTC")

        self._conn.execute(
            "INSERT INTO positions (ts, exchange, symbol, side, size, entry_price, mark_price, unrealized_pnl, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts_val), str(exchange), str(symbol), str(side), float(size), float(entry_price), mark_price, unrealized_pnl, str(mode)),
        )
        self._conn.commit()

    def open_positions(self, exchange: str | None = None, mode: str | None = None) -> pd.DataFrame:
        """Останній знімок позицій (по одному рядку на symbol, з найбільшим id)."""
        where_parts = []
        params: list[Any] = []
        if exchange:
            where_parts.append("exchange = ?")
            params.append(exchange)
        if mode:
            where_parts.append("mode = ?")
            params.append(mode)
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        # Останній запис для кожного (exchange, symbol, mode)
        sql = f"""
            SELECT p.*
            FROM positions p
            INNER JOIN (
                SELECT exchange, symbol, mode, MAX(id) AS max_id
                FROM positions {where}
                GROUP BY exchange, symbol, mode
            ) latest ON p.id = latest.max_id AND p.exchange = latest.exchange
                     AND p.symbol = latest.symbol AND p.mode = latest.mode
            WHERE p.size > 0
        """
        return pd.read_sql_query(sql, self._conn, params=params)

    # ─── v2: bots ─────────────────────────────────────────────────────────────

    def upsert_bot(
        self,
        bot_id: str,
        exchange: str,
        symbol: str,
        interval: str,
        strategy: str,
        mode: str,
        status: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Зареєструвати або оновити бота в реєстрі."""
        now = str(pd.Timestamp.now(tz="UTC").tz_convert(None))
        config_json = json.dumps(config or {}, default=str)
        existing = self._conn.execute("SELECT bot_id FROM bots WHERE bot_id = ?", (bot_id,)).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE bots SET status = ?, last_heartbeat = ?, config_json = ? WHERE bot_id = ?",
                (status, now, config_json, bot_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO bots (bot_id, exchange, symbol, interval, strategy, mode, status, started_at, last_heartbeat, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (bot_id, exchange, symbol, interval, strategy, mode, status, now, now, config_json),
            )
        self._conn.commit()

    def update_bot_heartbeat(self, bot_id: str, status: str = "running") -> None:
        """Оновити timestamp останнього heartbeat бота."""
        now = str(pd.Timestamp.now(tz="UTC").tz_convert(None))
        self._conn.execute(
            "UPDATE bots SET last_heartbeat = ?, status = ? WHERE bot_id = ?",
            (now, status, bot_id),
        )
        self._conn.commit()

    def set_bot_status(self, bot_id: str, status: str) -> None:
        """Встановити статус бота (running/paused/stopped/error)."""
        self._conn.execute("UPDATE bots SET status = ? WHERE bot_id = ?", (status, bot_id))
        self._conn.commit()

    def all_bots(self) -> pd.DataFrame:
        """Всі зареєстровані боти."""
        return pd.read_sql_query(
            "SELECT bot_id, exchange, symbol, interval, strategy, mode, status, started_at, last_heartbeat FROM bots ORDER BY started_at DESC",
            self._conn,
        )

    def schema_version(self) -> int:
        """Повернути поточну версію схеми."""
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        return int(row[0]) if row else 1
