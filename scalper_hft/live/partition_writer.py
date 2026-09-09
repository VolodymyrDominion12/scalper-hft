"""Append-only daily partition writer for live recorders.

Writes `{base_stem}_{YYYYMMDD}.parquet` beside the logical base path
(e.g. ``BTCUSDT_bookTicker.parquet`` → ``BTCUSDT_bookTicker_20250909.parquet``).
Only the touched day partition is read/merged — never the full history.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scalper_hft.data.storage import _atomic_write, cache_write_lock


def partition_path(base_path: Path, day: pd.Timestamp) -> Path:
    """Path for one UTC calendar-day partition."""
    day_ts = pd.Timestamp(day)
    ymd = day_ts.strftime("%Y%m%d")
    return base_path.parent / f"{base_path.stem}_{ymd}{base_path.suffix}"


def append_rows(base_path: Path, rows: list[dict], index_col: str = "ts") -> None:
    """Atomically append rows into daily partition files grouped by ``index_col``."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index

    for day_key, part in df.groupby(df.index.normalize()):
        out = partition_path(base_path, pd.Timestamp(day_key))
        with cache_write_lock(out):
            if out.exists():
                existing = pd.read_parquet(out)
                if index_col in existing.columns:
                    existing = existing.set_index(index_col)
                existing.index = pd.to_datetime(existing.index)
                if existing.index.tz is not None:
                    existing.index = existing.index.tz_localize(None)
                merged = pd.concat([existing, part])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            else:
                merged = part
            _atomic_write(out, lambda tmp: merged.to_parquet(tmp, compression="zstd"))


__all__ = ["append_rows", "partition_path"]
