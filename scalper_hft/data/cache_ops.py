"""Операції над parquet-кешем символу: інвентар файлів, видалення, докачка.

Дашборд і CLI користуються тими самими правилами: символ лише [A-Z0-9],
старі бари не стираються при докачці, 1m — джерело істини для ресемплінгу.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,20}$")
MIN_CACHE_DAYS = 1
MAX_CACHE_DAYS = 1095
DEFAULT_CACHE_DAYS = 90

_ACTION_REFRESH = "refresh"
_ACTION_EXPAND = "expand"
_ACTION_DELETE = "delete"

_ROW_REFRESH = ":material/download: Оновити"
_ROW_DOWNLOAD = ":material/download: Завантажити"
_ROW_EXPAND = ":material/date_range: Розширити"
_ROW_DELETE = ":material/delete: Видалити"


@dataclass(frozen=True, slots=True)
class CacheDownloadResult:
    """Підсумок докачки одного символу (klines обов'язкові, решта — опційно)."""

    symbol: str
    klines_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    funding_rows: int | None = None
    trades_rows: int | None = None


def validate_symbol(symbol: str) -> str:
    """Нормалізувати тікер; відхилити шлях/сміття, щоб glob не вийшов за data_dir."""
    raw = str(symbol).strip().upper()
    if not _SYMBOL_RE.fullmatch(raw):
        raise ValueError(f"некоректний символ: {symbol!r}")
    return raw


def clamp_cache_days(days: int | float) -> int:
    """Обмежити глибину завантаження 1…1095 днів."""
    try:
        value = int(days)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"некоректна глибина днів: {days!r}") from exc
    if value < MIN_CACHE_DAYS or value > MAX_CACHE_DAYS:
        raise ValueError(f"глибина днів має бути {MIN_CACHE_DAYS}…{MAX_CACHE_DAYS}, а не {value}")
    return value


def suggested_cache_days(span_days: float | None, *, action: str, default: int = DEFAULT_CACHE_DAYS) -> int:
    """Типова глибина: оновлення — default; розширення — max(2×default, span+30)."""
    base = clamp_cache_days(default)
    if action != _ACTION_EXPAND:
        return base
    current = 0
    if span_days is not None and span_days == span_days and span_days > 0:
        current = int(span_days)
    return min(MAX_CACHE_DAYS, max(base * 2, current + 30))


def row_cache_actions(klines_1m: int) -> list[str]:
    """Підписи кнопок рядка: порожній кеш — лише завантажити."""
    if int(klines_1m) > 0:
        return [_ROW_REFRESH, _ROW_EXPAND, _ROW_DELETE]
    return [_ROW_DOWNLOAD]


def action_from_row_label(label: str) -> str:
    """Розпізнати дію з підпису ButtonColumn (іконка + текст)."""
    text = str(label)
    if "Видалити" in text:
        return _ACTION_DELETE
    if "Розширити" in text:
        return _ACTION_EXPAND
    return _ACTION_REFRESH


def list_symbol_cache_files(data_dir: Path, symbol: str) -> list[Path]:
    """Усі parquet-файли символу в data_dir (klines, spot, trades, funding, book)."""
    sym = validate_symbol(symbol)
    root = Path(data_dir)
    if not root.exists():
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    patterns = (
        f"{sym}_*_klines.parquet",
        f"{sym}_bookTicker*.parquet",
        f"{sym}_depth5.parquet",
        f"{sym}_depth.parquet",
    )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in seen and path.is_file():
                seen.add(path)
                found.append(path)
    for name in (f"{sym}_aggTrades.parquet", f"{sym}_funding.parquet"):
        path = root / name
        if path not in seen and path.is_file():
            seen.add(path)
            found.append(path)
    return found


def delete_symbol_cache(data_dir: Path, symbol: str) -> list[str]:
    """Видалити parquet-кеш символу. Повертає імена стертих файлів."""
    deleted: list[str] = []
    for path in list_symbol_cache_files(data_dir, symbol):
        path.unlink(missing_ok=True)
        deleted.append(path.name)
        logger.info("Видалено кеш %s", path.name)
    return deleted


def max_span_days(inv: pd.DataFrame, symbols: list[str]) -> float | None:
    """Найбільший span_days серед вибраних рядків інвентарю."""
    if inv is None or inv.empty or not symbols:
        return None
    if "symbol" not in inv.columns or "span_days" not in inv.columns:
        return None
    subset = inv[inv["symbol"].astype(str).isin(symbols)]["span_days"]
    numeric = pd.to_numeric(subset, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def download_symbol_cache(
    symbol: str,
    days: int,
    *,
    interval: str = "1m",
    force: bool = False,
    funding: bool = True,
    trades: bool = False,
) -> CacheDownloadResult:
    """Докачати 1m klines (і опційно funding/aggTrades) через той самий downloader, що CLI."""
    from scalper_hft.data.downloader import download_agg_trades, download_funding, download_klines

    sym = validate_symbol(symbol)
    depth = clamp_cache_days(days)
    if interval != "1m":
        raise ValueError("дашборд качає лише базовий інтервал 1m (інші ТФ — ресемплінг)")
    df = download_klines(sym, interval, depth, force=force)
    start = end = None
    n = 0
    if df is not None and not df.empty:
        n = int(len(df))
        start = pd.Timestamp(df.index[0])
        end = pd.Timestamp(df.index[-1])
    funding_n: int | None = None
    trades_n: int | None = None
    if funding:
        fu = download_funding(sym, depth, force=force)
        funding_n = int(len(fu)) if fu is not None and not fu.empty else 0
    if trades:
        tr = download_agg_trades(sym, min(depth, 2), force=force)
        trades_n = int(len(tr)) if tr is not None and not tr.empty else 0
    logger.info("кеш %s 1m: %d барів (%s … %s)", sym, n, start, end)
    return CacheDownloadResult(
        symbol=sym,
        klines_rows=n,
        start=start,
        end=end,
        funding_rows=funding_n,
        trades_rows=trades_n,
    )
