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
    if list(df.columns) == _TRADES_COLUMNS:
        # Кадр уже має рівно потрібні колонки в потрібному порядку, тож `.copy()`
        # тут — зайва ПОВНА копія. На кеші aggTrades у 43 млн рядків це ~2 ГБ, і
        # саме накопичені копії призвели до OOM під час злиття (2026-09-12,
        # `Out of memory: Killed process ... anon-rss:7470996kB`).
        out = df
    else:
        for col in _TRADES_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        out = df[_TRADES_COLUMNS].copy()
    if out["trade_id"].isna().any():
        # старі кеші без id — синтетичні унікальні ідентифікатори
        out["trade_id"] = pd.RangeIndex(1, len(out) + 1)
    return out


def dedupe_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Прибрати дублікати aggTrades за `trade_id` (НЕ за мілісекундним індексом).

    Навіщо. Унікальний ключ aggTrade — `agg_trade_id` (колонка `trade_id`), а
    `transact_time` має мілісекундну роздільність, у якій регулярно опиняється
    кілька угод. Дедуп за індексом (`df.index.duplicated()`) знищує всі, крім
    однієї, у кожній такій мілісекунді: на кеші BTCUSDT 2026-09-11 лише в парах
    із Δt ≤ 1 мс так зникло ~0.6 млн id, а загальне покриття `trade_id` впало до
    39.3% (ETHUSDT 49.8%, DOGEUSDT 95.2%). Дефект був повністю тихий:
    `validate_trades` перевіряє дублікати індексу, а після дедупу їх немає.

    Синтетичні id (`_trade_id` fallback, від'ємні) унікальні в межах одного
    завантаження, але НЕ між завантаженнями — тому для них лишається дедуп за
    індексом (інакше докачка плодила б дублікати).
    """
    if df is None or df.empty:
        return df
    if "trade_id" not in df.columns or df["trade_id"].isna().any():
        logger.warning(
            "dedupe_trades: немає валідного trade_id — фолбек на дедуп за індексом "
            "(втрата угод у спільних мілісекундах неминуча)"
        )
        return df[~df.index.duplicated(keep="last")].sort_index()

    real = df["trade_id"] > 0  # від'ємні id — синтетичні (див. downloader._trade_id)
    if bool(real.all()):
        # Швидкий шлях (норма для vision-дампів): один прохід drop_duplicates
        # замість трьох повних копій (маска → зріз → concat). На 43 млн рядків
        # це різниця в кілька гігабайт пікової RAM.
        out = df.drop_duplicates(subset=["trade_id"], keep="last")
    else:
        real_part = df[real].drop_duplicates(subset=["trade_id"], keep="last")
        synth_part = df[~real]
        if synth_part.index.duplicated().any():
            synth_part = synth_part[~synth_part.index.duplicated(keep="last")]
        out = pd.concat([real_part, synth_part]) if not synth_part.empty else real_part
    if out.index.is_monotonic_increasing:
        return out
    return out.sort_index()


def load_funding(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    return df[["fundingRate"]]


def save_klines(path: Path, df: pd.DataFrame, *, strict: bool = True, interval: str | None = None) -> None:
    """Зберегти klines у parquet. Fail-closed (strict=True, за замовчуванням):
    критичні дефекти (порожній датасет, немонотонний індекс, дублікати,
    OHLC-порушення, майбутні бари, НЕРИНКОВІ рухи/«плити») → ValueError,
    файл НЕ пишеться.
    Дірки (gaps) — лише warning: легітимні для тонких символів, downloader
    їх дозаповнює інкрементально.
    interval: потрібен для порогів якості, що залежать від таймфрейму
    (частка «спайкових» барів на 1d інша, ніж на 1m — див.
    `validate.spike_rate_limit_for`). Без нього застосовується 1m-поріг, і
    справжня денна історія з 4 обвалами на 2500 барів відкидалась як «бита».
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    from scalper_hft.data.validate import bars_are_critical, validate_bars

    report = validate_bars(df, interval=interval)
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
    payload = df if list(df.columns) == _TRADES_COLUMNS else df[_TRADES_COLUMNS]
    _atomic_parquet(path, payload)
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
