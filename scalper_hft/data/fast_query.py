"""Високопродуктивний шар аналітичних запитів до Parquet-даних на базі DuckDB.

Забезпечує:
1. Миттєві SQL-запити безпосередньо над Parquet-файлами кешу без повного завантаження в RAM.
2. Гарантовано безпечні (no lookahead) ASOF JOIN для злиття розріджених подій (funding, depth, calendar) з klines.
3. Швидку агрегацію тікових потоків bookTicker / trades за довільними інтервалами.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False


def require_duckdb() -> None:
    """Перевірка наявності бібліотеки duckdb."""
    if not HAS_DUCKDB or duckdb is None:
        raise ImportError(
            "DuckDB не встановлено. Для використання швидких запитів встановіть: "
            "uv sync --extra research (або pip install duckdb)."
        )


def query_parquet(sql: str, params: dict[str, Any] | list[Any] | None = None) -> pd.DataFrame:
    """Виконання довільного SQL-запиту в DuckDB над локальними Parquet файлами.

    Приклад:
        df = query_parquet("SELECT symbol, AVG(close) FROM 'data/cache/*.parquet' GROUP BY symbol")
    """
    require_duckdb()
    conn = duckdb.connect(":memory:")
    try:
        if params:
            res = conn.execute(sql, params).df()
        else:
            res = conn.execute(sql).df()
        return res
    finally:
        conn.close()


def asof_join_parquet(
    base_parquet: str | Path,
    event_parquet: str | Path,
    time_col: str = "timestamp",
    by_col: str | None = None,
    select_event_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Безпечне (без lookahead) злиття часових рядів через ASOF JOIN в DuckDB.

    Правило:
        `base.{time_col} >= event.{time_col}` — кожному рядку з base зіставляється
        останній відомий запис з event станом на момент часу base.

    Args:
        base_parquet: шлях до основного Parquet (наприклад, klines).
        event_parquet: шлях до супутнього Parquet (наприклад, funding або depth).
        time_col: назва колонки мітки часу (наприклад, 'timestamp' або 'open_time').
        by_col: колонка групування (наприклад, 'symbol'), необов'язково.
        select_event_cols: список колонок з event, які потрібно додати (якщо None — усі, крім ключів).
    """
    require_duckdb()
    base_p = str(Path(base_parquet).resolve())
    event_p = str(Path(event_parquet).resolve())

    conn = duckdb.connect(":memory:")
    try:
        # Реєструємо як віртуальні таблиці для зручності
        conn.execute(f"CREATE VIEW base_tbl AS SELECT * FROM read_parquet('{base_p}')")
        conn.execute(f"CREATE VIEW event_tbl AS SELECT * FROM read_parquet('{event_p}')")

        # Отримуємо схему event_tbl
        event_schema = conn.execute("DESCRIBE event_tbl").fetchall()
        event_all_cols = [r[0] for r in event_schema]

        join_keys = [time_col]
        if by_col:
            join_keys.append(by_col)

        cols_to_add = select_event_cols or [c for c in event_all_cols if c not in join_keys]
        event_select_str = ", ".join(f"e.{c} AS event_{c}" for c in cols_to_add)

        by_clause = f"AND b.{by_col} = e.{by_col}" if by_col else ""

        query = f"""
        SELECT 
            b.*,
            {event_select_str}
        FROM base_tbl b
        ASOF LEFT JOIN event_tbl e
            ON b.{time_col} >= e.{time_col}
            {by_clause}
        ORDER BY b.{time_col} ASC
        """
        df = conn.execute(query).df()
        return df
    finally:
        conn.close()


def aggregate_bookticker_parquet(
    bookticker_parquet: str | Path,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """Високошвидкісна агрегація потоку bookTicker у часові інтервали.

    Обчислює:
    - mid_open, mid_high, mid_low, mid_close
    - avg_spread_bps, max_spread_bps
    - avg_bid_qty, avg_ask_qty
    """
    require_duckdb()
    p = str(Path(bookticker_parquet).resolve())
    conn = duckdb.connect(":memory:")
    try:
        # Перевірка наявності колонок у файлі
        conn.execute(f"CREATE VIEW bt AS SELECT * FROM read_parquet('{p}')")
        query = f"""
        WITH calculated AS (
            SELECT 
                to_timestamp(timestamp / 1000.0) AS ts,
                symbol,
                (bid_price + ask_price) / 2.0 AS mid_price,
                (ask_price - bid_price) / ((bid_price + ask_price) / 2.0) * 10000.0 AS spread_bps,
                bid_qty,
                ask_qty
            FROM bt
        )
        SELECT 
            time_bucket(INTERVAL '{interval_seconds} SECONDS', ts) AS bucket_time,
            symbol,
            first(mid_price) AS mid_open,
            max(mid_price) AS mid_high,
            min(mid_price) AS mid_low,
            last(mid_price) AS mid_close,
            avg(spread_bps) AS avg_spread_bps,
            max(spread_bps) AS max_spread_bps,
            avg(bid_qty) AS avg_bid_qty,
            avg(ask_qty) AS avg_ask_qty,
            count(*) AS tick_count
        FROM calculated
        GROUP BY bucket_time, symbol
        ORDER BY bucket_time ASC
        """
        return conn.execute(query).df()
    finally:
        conn.close()
