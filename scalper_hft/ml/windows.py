"""Per-TF walk-forward вікна для ML (labeled samples, не сирі бари).

Triple-barrier відкидає timeout-и (label=0), тому 730 днів 1h (~17k свічок)
часто дають < 2500 підписів — менше за історичний дефолт 2000+500.
"""

from __future__ import annotations

import pandas as pd

# Локальна таблиця (не імпортуємо data.resample — цикл data → strategies → ml).
_INTERVAL_MINUTES: dict[str, float] = {
    "1s": 1 / 60,
    "5s": 5 / 60,
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}

# Вікна в кількості LABELED зразків (PT/SL), не OHLCV-барів.
ML_TRAIN_TEST: dict[str, tuple[int, int]] = {
    "1s": (2000, 500),
    "5s": (2000, 500),
    "1m": (2000, 500),
    "3m": (1500, 400),
    "5m": (1200, 400),
    "15m": (600, 200),
    "30m": (400, 150),
    "1h": (350, 120),
    "2h": (200, 80),
    "4h": (100, 40),
    "6h": (90, 35),
    "8h": (80, 30),
    "12h": (80, 30),
    "1d": (70, 25),
    "3d": (50, 20),
    "1w": (40, 15),
}
DEFAULT_ML_TRAIN_TEST: tuple[int, int] = (2000, 500)
MIN_ML_TRAIN = 50
MIN_ML_TEST = 20
# Авто-стискання лише на реальному горизонті, не на 300–500 барах spec-тестів.
_FIT_MIN_BARS = 700


def minutes_to_interval(minutes: float) -> str:
    """Найближчий ключ до тривалості бару."""
    return min(_INTERVAL_MINUTES, key=lambda key: abs(_INTERVAL_MINUTES[key] - minutes))


def infer_bar_interval(df: pd.DataFrame, explicit: str | None = None) -> str:
    """Таймфрейм з параметра стратегії або з медіани кроку індексу."""
    if explicit:
        return str(explicit)
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
        return "1m"
    diffs = pd.Series(df.index).diff().dropna()
    median = float(diffs.median().total_seconds()) / 60.0
    if median <= 0:
        return "1m"
    return minutes_to_interval(median)


def default_ml_train_test(interval: str) -> tuple[int, int]:
    """Дефолтні (train, test) labeled-вікна для таймфрейму."""
    return ML_TRAIN_TEST.get(interval, DEFAULT_ML_TRAIN_TEST)


def resolve_ml_windows(
    n_samples: int,
    n_bars: int,
    interval: str,
    train_bars: int | None,
    test_bars: int | None,
) -> tuple[int, int] | None:
    """Підібрати train/test під labeled-вибірку.

    Явні train_bars/test_bars не стискаються. Якщо дефолт не влазить і є
    достатньо сирих барів — вікно підганяється під n_samples (мін. 50/20).
    None = замало даних навіть для мінімального складки.
    """
    default_train, default_test = default_ml_train_test(interval)
    explicit = train_bars is not None or test_bars is not None
    train = default_train if train_bars is None else int(train_bars)
    test = default_test if test_bars is None else int(test_bars)
    if train <= 0 or test <= 0:
        return None
    if n_samples >= train + test:
        return train, test
    if explicit:
        return None
    if n_bars < _FIT_MIN_BARS or n_samples < MIN_ML_TRAIN + MIN_ML_TEST:
        return None
    test_fit = max(MIN_ML_TEST, min(test, n_samples // 5))
    train_fit = n_samples - test_fit
    if train_fit < MIN_ML_TRAIN:
        return None
    return train_fit, test_fit
