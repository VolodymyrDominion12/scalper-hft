"""SQLite-журнал paper pairs: equity, ордери, угоди, місячний PnL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class PaperStore:
    """Персистенція paper-прогону. Path за замовчуванням results/paper_pairs.sqlite."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path("results") / "paper_pairs.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        cur = self._conn.cursor()
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
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log_equity(self, ts: pd.Timestamp, pair: str, equity: float, cash: float, realized_pnl: float) -> None:
        self._conn.execute(
            "INSERT INTO equity (ts, pair, equity, cash, realized_pnl) VALUES (?, ?, ?, ?, ?)",
            (str(ts), pair, equity, cash, realized_pnl),
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
    ) -> None:
        self._conn.execute(
            "INSERT INTO orders (ts, pair, symbol, side, size, price, status, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts), pair, symbol, side, size, price, status, reason),
        )
        self._conn.commit()

    def log_trade(self, ts: pd.Timestamp, pair: str, trade: dict) -> None:
        kind = str(trade.get("type", "trade"))
        self._conn.execute(
            "INSERT INTO trades (ts, pair, symbol, side, size, entry_price, exit_price, pnl, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def fill_stats(self, pair: str | None = None) -> dict[str, int]:
        if pair:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM orders WHERE pair = ? GROUP BY status",
                (pair,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM orders GROUP BY status").fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}
