"""Абстракція кешу ринкових даних: parquet (за замовчуванням) або PostgreSQL.

Єдиний інтерфейс для завантаження/збереження klines, aggTrades і funding,
щоб не переписувати downloader/бектест під кожен бекенд:

    store = get_store()                       # за Settings.data_backend
    df = store.load_klines("BTCUSDT", "1m")   # -> DataFrame | None
    store.save_klines("BTCUSDT", "1m", df)

PostgreSQL (DATA_BACKEND=postgres) — для майбутнього дослідження багатьох
інструментів: дані завантажуються один раз і живуть у Docker-volume,
схема створюється автоматично (ensure_schema).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataStore(Protocol):
    """Контракт кешу. Усі методи повертають DataFrame з UTC-naive DatetimeIndex."""

    def ensure_schema(self) -> None: ...

    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None: ...
    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None: ...

    def load_spot_klines(self, symbol: str, interval: str) -> pd.DataFrame | None: ...
    def save_spot_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None: ...

    def load_trades(self, symbol: str) -> pd.DataFrame | None: ...
    def save_trades(self, symbol: str, df: pd.DataFrame) -> None: ...

    def load_funding(self, symbol: str) -> pd.DataFrame | None: ...
    def save_funding(self, symbol: str, df: pd.DataFrame) -> None: ...

    def list_klines(self) -> list[tuple[str, str]]:
        """(symbol, interval) пари, наявні в кеші (для планування sweep)."""
        ...


class ParquetStore:
    """Старий добрий parquet-кеш у data/ (повна зворотна сумісність)."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        from scalper_hft.config import get_settings

        self.data_dir = Path(data_dir) if data_dir else get_settings().data_dir_abs

    def ensure_schema(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── klines ────────────────────────────────────────────────────────────────
    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        from scalper_hft.data.storage import klines_path, load_klines

        return load_klines(klines_path(self.data_dir, symbol, interval))

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        from scalper_hft.data.storage import klines_path, save_klines

        save_klines(klines_path(self.data_dir, symbol, interval), df)

    def load_spot_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        from scalper_hft.data.storage import load_klines, spot_klines_path

        return load_klines(spot_klines_path(self.data_dir, symbol, interval))

    def save_spot_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        from scalper_hft.data.storage import save_klines, spot_klines_path

        save_klines(spot_klines_path(self.data_dir, symbol, interval), df)

    # ── trades / funding ──────────────────────────────────────────────────────
    def load_trades(self, symbol: str) -> pd.DataFrame | None:
        from scalper_hft.data.storage import load_trades, trades_path

        return load_trades(trades_path(self.data_dir, symbol))

    def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
        from scalper_hft.data.storage import save_trades, trades_path

        save_trades(trades_path(self.data_dir, symbol), df)

    def load_funding(self, symbol: str) -> pd.DataFrame | None:
        from scalper_hft.data.storage import funding_path, load_funding

        return load_funding(funding_path(self.data_dir, symbol))

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        from scalper_hft.data.storage import funding_path, save_funding

        save_funding(funding_path(self.data_dir, symbol), df)

    def list_klines(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not self.data_dir.exists():
            return out
        for path in sorted(self.data_dir.glob("*_klines.parquet")):
            parts = path.stem.split("_")
            if len(parts) >= 3 and parts[-1] == "klines":
                interval = parts[-2]
                symbol = "_".join(parts[:-2])
                out.append((symbol, interval))
        return out


class PostgresStore:
    """PostgreSQL-кеш. Схема створюється автоматично; записи — upsert.

    Таблиці:
        klines(symbol, interval, ts, open, high, low, close, volume)
        agg_trades(symbol, trade_id, ts, price, amount, side)
        funding(symbol, ts, funding_rate)
        meta(key, value)   — службова (напр. останній trade_id)
    """

    _KLINES_COLS = ["open", "high", "low", "close", "volume"]
    _TRADES_COLS = ["trade_id", "price", "amount", "side"]
    _FUNDING_COLS = ["fundingRate"]  # колонка pandas (у БД — funding_rate)

    def __init__(self, conninfo: str | None = None) -> None:
        from scalper_hft.config import get_settings

        self.conninfo = conninfo or get_settings().postgres_conninfo
        self._schema_ready = False

    # ── підключення ───────────────────────────────────────────────────────────
    def _connect(self):
        import psycopg

        return psycopg.connect(self.conninfo, connect_timeout=10)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        ddl = """
        CREATE TABLE IF NOT EXISTS klines (
            symbol   TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts       TIMESTAMPTZ NOT NULL,
            open     DOUBLE PRECISION,
            high     DOUBLE PRECISION,
            low      DOUBLE PRECISION,
            close    DOUBLE PRECISION,
            volume   DOUBLE PRECISION,
            PRIMARY KEY (symbol, interval, ts)
        );
        CREATE TABLE IF NOT EXISTS agg_trades (
            symbol   TEXT NOT NULL,
            trade_id BIGINT NOT NULL,
            ts       TIMESTAMPTZ NOT NULL,
            price    DOUBLE PRECISION,
            amount   DOUBLE PRECISION,
            side     TEXT,
            PRIMARY KEY (symbol, trade_id)
        );
        CREATE INDEX IF NOT EXISTS idx_agg_trades_ts ON agg_trades (symbol, ts);
        CREATE TABLE IF NOT EXISTS funding (
            symbol       TEXT NOT NULL,
            ts           TIMESTAMPTZ NOT NULL,
            funding_rate DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        self._schema_ready = True

    def _query(self, sql: str, params: tuple) -> pd.DataFrame | None:
        import psycopg

        try:
            self.ensure_schema()
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d.name for d in cur.description] if cur.description else []
                rows = cur.fetchall()
        except psycopg.OperationalError as exc:
            logger.warning("Postgres недоступний (%s) — повертаю порожній кеш", exc)
            return None
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=cols)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        return df.set_index("ts").sort_index()

    @staticmethod
    def _copy(conn, table: str, cols: list[str], rows: list[tuple]) -> None:
        """Bulk-insert через COPY (швидко для мільйонів рядків). NaN → NULL."""
        import math

        clean = []
        for row in rows:
            clean.append(tuple(None if isinstance(v, float) and math.isnan(v) else v for v in row))
        with conn.cursor() as cur:
            with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as copy:
                for row in clean:
                    copy.write_row(row)

    @staticmethod
    def _lock_replace(conn, *keys: str) -> None:
        """Серіалізувати replace (symbol, interval) через pg_advisory_xact_lock.

        DELETE+COPY «атомарний» лише в межах одного з'єднання: два паралельні
        воркери на тому самому ключі (sweep workers>1, кілька процесів) ловили
        unique violation на klines_pkey / втрачали чужі щойно записані рядки.
        Xact-лок тримається до commit/rollback і працює між тредами й процесами.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))", keys)

    def _replace_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Атомарна заміна рядків klines для (symbol, interval), race-safe."""
        self.ensure_schema()
        cols = ["symbol", "interval", "ts", *self._KLINES_COLS]
        rows = [
            (symbol, interval, row[0].to_pydatetime(), *row[1:])
            for row in df[self._KLINES_COLS].itertuples(index=True, name=None)
        ]
        with self._connect() as conn:
            self._lock_replace(conn, symbol, interval)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM klines WHERE symbol = %s AND interval = %s", (symbol, interval))
            if rows:
                self._copy(conn, "klines", cols, rows)
            conn.commit()

    def _replace(self, table: str, symbol: str, df: pd.DataFrame) -> None:
        """Атомарна заміна рядків agg_trades / funding для символу, race-safe."""
        self.ensure_schema()
        with self._connect() as conn:
            self._lock_replace(conn, table, symbol)
            with conn.cursor() as cur:
                rows: list[tuple[Any, ...]]
                if table == "agg_trades":
                    cur.execute("DELETE FROM agg_trades WHERE symbol = %s", (symbol,))
                    cols = ["symbol", "trade_id", "ts", "price", "amount", "side"]
                    rows = [
                        (symbol, int(tid), ts.to_pydatetime(), float(p), float(a), s)
                        for ts, tid, p, a, s in zip(df.index, df["trade_id"], df["price"], df["amount"], df["side"])
                    ]
                elif table == "funding":
                    cur.execute("DELETE FROM funding WHERE symbol = %s", (symbol,))
                    cols = ["symbol", "ts", "funding_rate"]
                    rows = [(symbol, ts.to_pydatetime(), float(r)) for ts, r in df["fundingRate"].items()]
                else:  # pragma: no cover
                    raise ValueError(f"невідома таблиця {table}")

                if rows:
                    self._copy(conn, table, cols, rows)
            conn.commit()

    # ── klines ────────────────────────────────────────────────────────────────
    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        return self._query(
            "SELECT ts, open, high, low, close, volume FROM klines WHERE symbol = %s AND interval = %s ORDER BY ts",
            (symbol, interval),
        )

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._replace_klines(symbol, interval, df)
        logger.info("Postgres: збережено klines %s %s (%d рядків)", symbol, interval, len(df))

    def load_spot_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        return self._query(
            "SELECT ts, open, high, low, close, volume FROM klines WHERE symbol = %s AND interval = %s ORDER BY ts",
            (symbol, f"spot_{interval}"),
        )

    def save_spot_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._replace_klines(symbol, f"spot_{interval}", df)
        logger.info("Postgres: збережено spot klines %s %s (%d рядків)", symbol, interval, len(df))

    # ── trades / funding ──────────────────────────────────────────────────────
    def load_trades(self, symbol: str) -> pd.DataFrame | None:
        df = self._query(
            "SELECT ts, trade_id, price, amount, side FROM agg_trades WHERE symbol = %s ORDER BY ts",
            (symbol,),
        )
        return df

    def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._replace("agg_trades", symbol, df)
        logger.info("Postgres: збережено aggTrades %s (%d рядків)", symbol, len(df))

    def load_funding(self, symbol: str) -> pd.DataFrame | None:
        return self._query(
            'SELECT ts, funding_rate AS "fundingRate" FROM funding WHERE symbol = %s ORDER BY ts',
            (symbol,),
        )

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._replace("funding", symbol, df)
        logger.info("Postgres: збережено funding %s (%d рядків)", symbol, len(df))

    def list_klines(self) -> list[tuple[str, str]]:
        import psycopg

        try:
            self.ensure_schema()
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol, interval FROM klines ORDER BY symbol, interval")
                return [tuple(r) for r in cur.fetchall()]
        except psycopg.OperationalError as exc:
            logger.warning("Postgres недоступний (%s)", exc)
            return []


def get_store() -> MarketDataStore:
    """Фабрика бекенду кешу за Settings.data_backend ('parquet' | 'postgres')."""
    from scalper_hft.config import get_settings

    backend = get_settings().data_backend
    if backend == "postgres":
        store = PostgresStore()
        store.ensure_schema()
        return store
    return ParquetStore()


__all__ = ["MarketDataStore", "ParquetStore", "PostgresStore", "get_store"]
