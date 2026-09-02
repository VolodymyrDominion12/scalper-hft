"""Sweep Store — збереження результатів масового прогону у SQLite.

Мета: при масових sweep-прогонах (всі стратегії × символи × таймфрейми)
результати накопичуються поступово і не губляться при збоях. Підтримує
`--resume` режим: пропускає вже виконані комбінації.

Використання:
    store = SweepStore("results/sweep.db")
    store.upsert(row)                         # зберегти результат клітинки
    done = store.already_done("mean_rev", "BTCUSDT", "5m", days=90, mode="backtest")
    df = store.load()                         # завантажити все у DataFrame
    store.close()
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ── Розширений SweepRow ───────────────────────────────────────────────────────


@dataclass
class SweepRow:
    """Результат одного бектесту (стратегія × символ × таймфрейм)."""

    strategy: str
    symbol: str
    interval: str
    days: int = 0
    mode: str = "backtest"  # backtest | walkforward
    n_bars: int = 0
    n_trades: int = 0
    total_return: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    calmar: float = float("nan")
    max_dd: float = float("nan")
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    avg_trade: float = float("nan")
    exposure: float = float("nan")
    trades_per_day: float = float("nan")
    # Walk-forward OOS метрики
    avg_is_sharpe: float = float("nan")
    avg_oos_sharpe: float = float("nan")
    oos_positive_frac: float = float("nan")
    # Filter tracing
    n_raw_signals: int = 0  # всього raw сигналів до фільтрів
    n_filtered: int = 0  # заблоковано фільтрами
    filter_attribution: str = ""  # JSON: {filter_name: n_blocked}
    # Статус
    status: str = "ok"
    error: str = ""
    run_ts: str = ""  # ISO timestamp прогону

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def filter_attr_dict(self) -> dict[str, int]:
        """Розпарсити JSON filter_attribution у dict."""
        if not self.filter_attribution:
            return {}
        try:
            return json.loads(self.filter_attribution)
        except (json.JSONDecodeError, TypeError):
            return {}


# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sweep_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    interval            TEXT NOT NULL,
    days                INTEGER DEFAULT 0,
    mode                TEXT DEFAULT 'backtest',
    n_bars              INTEGER DEFAULT 0,
    n_trades            INTEGER DEFAULT 0,
    total_return        REAL,
    sharpe              REAL,
    sortino             REAL,
    calmar              REAL,
    max_dd              REAL,
    win_rate            REAL,
    profit_factor       REAL,
    avg_trade           REAL,
    exposure            REAL,
    trades_per_day      REAL,
    avg_is_sharpe       REAL,
    avg_oos_sharpe      REAL,
    oos_positive_frac   REAL,
    n_raw_signals       INTEGER DEFAULT 0,
    n_filtered          INTEGER DEFAULT 0,
    filter_attribution  TEXT DEFAULT '',
    status              TEXT DEFAULT 'ok',
    error               TEXT DEFAULT '',
    run_ts              TEXT DEFAULT '',
    UNIQUE(strategy, symbol, interval, days, mode)
);
"""

_UPSERT_SQL = """
INSERT INTO sweep_results
    (strategy, symbol, interval, days, mode, n_bars, n_trades,
     total_return, sharpe, sortino, calmar, max_dd, win_rate, profit_factor,
     avg_trade, exposure, trades_per_day, avg_is_sharpe, avg_oos_sharpe,
     oos_positive_frac, n_raw_signals, n_filtered, filter_attribution,
     status, error, run_ts)
VALUES
    (:strategy, :symbol, :interval, :days, :mode, :n_bars, :n_trades,
     :total_return, :sharpe, :sortino, :calmar, :max_dd, :win_rate, :profit_factor,
     :avg_trade, :exposure, :trades_per_day, :avg_is_sharpe, :avg_oos_sharpe,
     :oos_positive_frac, :n_raw_signals, :n_filtered, :filter_attribution,
     :status, :error, :run_ts)
ON CONFLICT(strategy, symbol, interval, days, mode)
DO UPDATE SET
    n_bars=excluded.n_bars, n_trades=excluded.n_trades,
    total_return=excluded.total_return, sharpe=excluded.sharpe,
    sortino=excluded.sortino, calmar=excluded.calmar,
    max_dd=excluded.max_dd, win_rate=excluded.win_rate,
    profit_factor=excluded.profit_factor, avg_trade=excluded.avg_trade,
    exposure=excluded.exposure, trades_per_day=excluded.trades_per_day,
    avg_is_sharpe=excluded.avg_is_sharpe, avg_oos_sharpe=excluded.avg_oos_sharpe,
    oos_positive_frac=excluded.oos_positive_frac,
    n_raw_signals=excluded.n_raw_signals, n_filtered=excluded.n_filtered,
    filter_attribution=excluded.filter_attribution,
    status=excluded.status, error=excluded.error, run_ts=excluded.run_ts;
"""


# ── Store ─────────────────────────────────────────────────────────────────────


class SweepStore:
    """SQLite-сховище для результатів sweep-прогону.

    Thread-safe (check_same_thread=False): sweep.py запускає клітинки
    у ThreadPoolExecutor, кожна клітинка може викликати upsert().
    """

    def __init__(self, path: str | Path = "results/sweep.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def upsert(self, row: SweepRow) -> None:
        """Зберегти або оновити результат клітинки."""
        import datetime

        d = row.as_dict()
        if not d.get("run_ts"):
            d["run_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
        self._conn.execute(_UPSERT_SQL, d)
        self._conn.commit()

    def already_done(
        self,
        strategy: str,
        symbol: str,
        interval: str,
        days: int,
        mode: str = "backtest",
    ) -> bool:
        """Чи вже є результат для цієї комбінації зі статусом 'ok'?"""
        cur = self._conn.execute(
            "SELECT status FROM sweep_results WHERE strategy=? AND symbol=? AND interval=? AND days=? AND mode=?",
            (strategy, symbol, interval, days, mode),
        )
        row = cur.fetchone()
        return row is not None and row[0] == "ok"

    def load(self, **filters: Any) -> pd.DataFrame:
        """Завантажити всі результати у DataFrame.

        filters: keyword args для фільтрації (strategy=..., symbol=..., mode=...).
        """
        query = "SELECT * FROM sweep_results WHERE 1=1"
        params: list[Any] = []
        for col, val in filters.items():
            query += f" AND {col}=?"
            params.append(val)
        query += " ORDER BY sharpe DESC NULLS LAST"
        return pd.read_sql_query(query, self._conn, params=params)

    def load_filter_attribution(self) -> pd.DataFrame:
        """Зведена таблиця: filter_name × стратегія × total blocked."""
        df = self.load()
        if df.empty or "filter_attribution" not in df.columns:
            return pd.DataFrame()
        rows = []
        for _, r in df.iterrows():
            attr = r.get("filter_attribution", "")
            if not attr:
                continue
            try:
                d = json.loads(attr)
            except (json.JSONDecodeError, TypeError):
                continue
            for fname, cnt in d.items():
                rows.append(
                    {
                        "strategy": r["strategy"],
                        "symbol": r["symbol"],
                        "interval": r["interval"],
                        "filter_name": fname,
                        "n_blocked": cnt,
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def summary(self) -> pd.DataFrame:
        """Топ стратегій за середнім Sharpe по всіх символах/TF."""
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        return (
            df[df["status"] == "ok"]
            .groupby("strategy")
            .agg(
                mean_sharpe=("sharpe", "mean"),
                max_sharpe=("sharpe", "max"),
                n_cells=("sharpe", "count"),
                mean_n_trades=("n_trades", "mean"),
            )
            .sort_values("mean_sharpe", ascending=False)
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SweepStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
