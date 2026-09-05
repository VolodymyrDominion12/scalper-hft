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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CachePeek:
    """Кількість рядків і часовий span без завантаження OHLCV."""

    n_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class SymbolCacheStats:
    """Зріз кешу одного символу для дашборду свіжості."""

    klines_1m: CachePeek
    intervals: tuple[str, ...]
    funding: CachePeek
    trades: CachePeek


_EMPTY_PEEK = CachePeek(0, None, None)
_TF_CANDIDATES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")


def empty_symbol_stats() -> SymbolCacheStats:
    """Порожній зріз: символу немає ні в parquet, ні в Postgres."""
    return SymbolCacheStats(klines_1m=_EMPTY_PEEK, intervals=(), funding=_EMPTY_PEEK, trades=_EMPTY_PEEK)


def peek_cache_file(path: Path) -> CachePeek:
    """Row count і min/max timestamps parquet без завантаження OHLCV-колонок."""
    if not path.exists():
        return _EMPTY_PEEK
    try:
        pf = pq.ParquetFile(path)
    except Exception:  # noqa: BLE001 — битий файл = порожній зріз
        return _EMPTY_PEEK
    n = int(pf.metadata.num_rows)
    if n <= 0:
        return _EMPTY_PEEK
    names = list(pf.schema_arrow.names)
    ts_name: str | None = None
    for candidate in ("__index_level_0__", "timestamp", "ts", "time"):
        if candidate in names:
            ts_name = candidate
            break
    start, end = _span_from_statistics(pf, names, ts_name)
    if start is None and ts_name is not None and n <= 2_000_000:
        start, end = _span_from_column(path, ts_name)
    return CachePeek(n, start, end)


def _span_from_statistics(
    pf: pq.ParquetFile,
    names: list[str],
    ts_name: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if ts_name is None:
        return None, None
    col_idx = names.index(ts_name)
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(col_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            continue
        try:
            mins.append(pd.Timestamp(stats.min))
            maxs.append(pd.Timestamp(stats.max))
        except (ValueError, TypeError, OverflowError):
            return None, None
    if not mins:
        return None, None
    return min(mins), max(maxs)


def _span_from_column(path: Path, ts_name: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    try:
        series = pd.read_parquet(path, columns=[ts_name]).iloc[:, 0]
    except Exception:  # noqa: BLE001
        try:
            idx = pd.read_parquet(path, columns=[]).index
        except Exception:  # noqa: BLE001
            return None, None
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
            return None, None
        return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if ts.empty:
        return None, None
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def _naive_utc(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _estimate_bars(start: pd.Timestamp | None, end: pd.Timestamp | None, interval: str) -> int:
    """Кількість барів з span (без COUNT по мільйонах рядків)."""
    if start is None or end is None:
        return 0
    from scalper_hft.data.resample import INTERVAL_MINUTES

    step = float(INTERVAL_MINUTES.get(interval, 1.0))
    if step <= 0:
        return 0
    return int((end - start).total_seconds() / (60.0 * step)) + 1


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

    def symbol_stats(self, symbols: Sequence[str]) -> dict[str, SymbolCacheStats]:
        """COUNT/MIN/MAX по символах без завантаження рядків (дашборд свіжості)."""
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

    def symbol_stats(self, symbols: Sequence[str]) -> dict[str, SymbolCacheStats]:
        intervals_by_sym: dict[str, list[str]] = {}
        for sym, interval in self.list_klines():
            if "spot" in interval:
                continue
            intervals_by_sym.setdefault(sym, []).append(interval)
        out: dict[str, SymbolCacheStats] = {}
        for sym in symbols:
            out[sym] = SymbolCacheStats(
                klines_1m=peek_cache_file(self.data_dir / f"{sym}_1m_klines.parquet"),
                intervals=tuple(intervals_by_sym.get(sym, ())),
                funding=peek_cache_file(self.data_dir / f"{sym}_funding.parquet"),
                trades=peek_cache_file(self.data_dir / f"{sym}_aggTrades.parquet"),
            )
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

    def symbol_stats(self, symbols: Sequence[str]) -> dict[str, SymbolCacheStats]:
        wanted = [str(s) for s in symbols]
        empty = {sym: empty_symbol_stats() for sym in wanted}
        if not wanted:
            return {}
        import psycopg

        # По символу (префікс PK). GROUP BY на всю таблицю + ANY(...)
        # дає seq-scan ~23M рядків — дашборд «висить», тести теж.
        try:
            self.ensure_schema()
            with self._connect() as conn, conn.cursor() as cur:
                return {sym: self._symbol_stats_one(cur, sym) for sym in wanted}
        except psycopg.OperationalError as exc:
            logger.warning("Postgres недоступний (%s) — порожній інвентар", exc)
            return empty

    def _symbol_stats_one(self, cur: Any, symbol: str) -> SymbolCacheStats:
        k1m = self._index_peek(cur, "klines", symbol, interval="1m")
        intervals = self._present_intervals(cur, symbol, has_1m=k1m.n_rows > 0)
        funding = self._agg_peek(cur, "funding", symbol)
        trades = self._agg_peek(cur, "agg_trades", symbol)
        return SymbolCacheStats(
            klines_1m=k1m,
            intervals=intervals,
            funding=funding,
            trades=trades,
        )

    def _present_intervals(self, cur: Any, symbol: str, *, has_1m: bool) -> tuple[str, ...]:
        # LIMIT 1 без ORDER BY на interval='1m' дає seq-scan ~23M рядків
        # (планувальник бачить 1.5M хітів). Інші ТФ — INDEX seek через ORDER BY ts.
        found: list[str] = ["1m"] if has_1m else []
        for interval in _TF_CANDIDATES:
            if interval == "1m":
                continue
            cur.execute(
                "SELECT 1 FROM klines WHERE symbol = %s AND interval = %s ORDER BY ts LIMIT 1",
                (symbol, interval),
            )
            if cur.fetchone():
                found.append(interval)
        return tuple(found)

    def _index_peek(self, cur: Any, table: str, symbol: str, *, interval: str) -> CachePeek:
        if table != "klines":
            raise ValueError(f"неочікувана таблиця {table}")
        cur.execute(
            "SELECT ts FROM klines WHERE symbol = %s AND interval = %s ORDER BY ts ASC LIMIT 1",
            (symbol, interval),
        )
        first = cur.fetchone()
        if not first:
            return _EMPTY_PEEK
        cur.execute(
            "SELECT ts FROM klines WHERE symbol = %s AND interval = %s ORDER BY ts DESC LIMIT 1",
            (symbol, interval),
        )
        last = cur.fetchone()
        start = _naive_utc(first[0])
        end = _naive_utc(last[0] if last else first[0])
        return CachePeek(_estimate_bars(start, end, interval), start, end)

    def _agg_peek(self, cur: Any, table: str, symbol: str) -> CachePeek:
        if table not in {"funding", "agg_trades"}:
            raise ValueError(f"неочікувана таблиця {table}")
        cur.execute(
            f"SELECT ts FROM {table} WHERE symbol = %s ORDER BY ts ASC LIMIT 1",
            (symbol,),
        )
        first = cur.fetchone()
        if not first:
            return _EMPTY_PEEK
        cur.execute(
            f"SELECT ts FROM {table} WHERE symbol = %s ORDER BY ts DESC LIMIT 1",
            (symbol,),
        )
        last = cur.fetchone()
        start = _naive_utc(first[0])
        end = _naive_utc(last[0] if last else first[0])
        if table == "funding":
            cur.execute(f"SELECT COUNT(*)::bigint FROM {table} WHERE symbol = %s", (symbol,))
            n_rows = int((cur.fetchone() or (0,))[0] or 0)
        else:
            # COUNT(*) по agg_trades сканує мільйони рядків (у проді ~9M лише BTC).
            n_rows = 1
        return CachePeek(n_rows, start, end)


def get_store() -> MarketDataStore:
    """Фабрика бекенду кешу за Settings.data_backend ('parquet' | 'postgres')."""
    from scalper_hft.config import get_settings

    backend = get_settings().data_backend
    if backend == "postgres":
        store = PostgresStore()
        store.ensure_schema()
        return store
    return ParquetStore()


__all__ = [
    "CachePeek",
    "MarketDataStore",
    "ParquetStore",
    "PostgresStore",
    "SymbolCacheStats",
    "empty_symbol_stats",
    "get_store",
    "peek_cache_file",
]
