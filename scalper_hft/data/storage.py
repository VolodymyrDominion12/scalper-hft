"""Parquet-кеш даних: klines, aggTrades, funding.

Формат імен файлів:
    data/{symbol}_{interval}_klines.parquet
    data/{symbol}_aggTrades.parquet
    data/{symbol}_funding.parquet
Індекс — datetime64 (UTC). Parquet + zstd — компактно і швидко.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_KLINES_COLUMNS = ["open", "high", "low", "close", "volume"]
_TRADES_COLUMNS = ["trade_id", "price", "amount", "side"]


@contextmanager
def cache_write_lock(path: Path) -> Iterator[None]:
    """Serialize writers for one cache file across processes."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write[T](path: Path, writer: Callable[[Path], T]) -> T:
    """Write beside the target and atomically replace it after fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp)
    try:
        result = writer(tmp_path)
        with tmp_path.open("rb") as file_obj:
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    with cache_write_lock(path):
        _atomic_write(path, lambda tmp: df.to_parquet(tmp, compression="zstd"))


def _safe_load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося прочитати кеш %s: %s — ігнорую", path, exc)
        return None


def klines_path(data_dir: Path, symbol: str, interval: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_{interval}_klines.parquet"


def spot_klines_path(data_dir: Path, symbol: str, interval: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_{interval}_spot_klines.parquet"


def trades_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_aggTrades.parquet"


def funding_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_funding.parquet"


def load_klines(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    for col in _KLINES_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[_KLINES_COLUMNS].astype(float)


def load_trades(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    for col in _TRADES_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    out = df[_TRADES_COLUMNS].copy()
    if out["trade_id"].isna().any():
        # старі кеші без id — синтетичні унікальні ідентифікатори
        out["trade_id"] = pd.RangeIndex(1, len(out) + 1)
    return out


def load_funding(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    return df[["fundingRate"]]


def save_klines(path: Path, df: pd.DataFrame, *, strict: bool = True) -> None:
    """Зберегти klines у parquet. Fail-closed (strict=True, за замовчуванням):
    критичні дефекти (порожній датасет, немонотонний індекс, дублікати,
    OHLC-порушення, майбутні бари, НЕРИНКОВІ рухи/«плити») → ValueError,
    файл НЕ пишеться.
    Дірки (gaps) — лише warning: легітимні для тонких символів, downloader
    їх дозаповнює інкрементально."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from scalper_hft.data.validate import bars_are_critical, validate_bars

    report = validate_bars(df)
    if not report.ok:
        if strict and bars_are_critical(report):
            raise ValueError(f"Відмова у збереженні битих барів {path.name}: {report.summary()}")
        logger.warning("Якість барів %s: %s", path.name, report.summary())
    _atomic_parquet(path, df[_KLINES_COLUMNS].astype(float))
    logger.info("Збережено klines: %s (%d рядків)", path, len(df))


def save_trades(path: Path, df: pd.DataFrame, *, strict: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is not None and not df.empty:
        from scalper_hft.data.validate import stream_is_critical, validate_trades

        report = validate_trades(df)
        if not report.ok:
            if strict and stream_is_critical(report):
                raise ValueError(f"Відмова у збереженні битих aggTrades {path.name}: {report.summary()}")
            logger.warning("Якість aggTrades %s: %s", path.name, report.summary())
    _atomic_parquet(path, df[_TRADES_COLUMNS])
    logger.info("Збережено aggTrades: %s (%d рядків)", path, len(df))


def save_funding(path: Path, df: pd.DataFrame, *, strict: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is not None and not df.empty:
        from scalper_hft.data.validate import stream_is_critical, validate_funding

        report = validate_funding(df)
        if not report.ok:
            if strict and stream_is_critical(report):
                raise ValueError(f"Відмова у збереженні битого funding {path.name}: {report.summary()}")
            logger.warning("Якість funding %s: %s", path.name, report.summary())
    _atomic_parquet(path, df[["fundingRate"]])
    logger.info("Збережено funding: %s (%d рядків)", path, len(df))


def liquidations_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_liquidations.parquet"


def oi_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_oi.parquet"


def load_liquidations(path: Path) -> pd.DataFrame | None:
    return _safe_load(path)


def save_liquidations(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(path, df)
    logger.info("Збережено liquidations: %s (%d рядків)", path, len(df))


def load_oi(path: Path) -> pd.DataFrame | None:
    return _safe_load(path)


def save_oi(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(path, df)
    logger.info("Збережено open interest: %s (%d рядків)", path, len(df))
