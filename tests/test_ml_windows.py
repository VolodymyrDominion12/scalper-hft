"""Per-TF labeled-вікна для MLStrategy."""

from __future__ import annotations

import pandas as pd
import pytest
from scalper_hft.ml.windows import (
    MIN_ML_TEST,
    MIN_ML_TRAIN,
    default_ml_train_test,
    infer_bar_interval,
    minutes_to_interval,
    resolve_ml_windows,
)


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"), (1440, "1d")],
)
def test_minutes_to_interval(minutes: float, expected: str) -> None:
    assert minutes_to_interval(minutes) == expected


def test_infer_bar_interval_from_index() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="4h")
    df = pd.DataFrame({"close": range(10)}, index=idx)
    assert infer_bar_interval(df) == "4h"


def test_infer_bar_interval_explicit_wins() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({"close": range(10)}, index=idx)
    assert infer_bar_interval(df, explicit="15m") == "15m"


def test_default_ml_train_test_known_and_fallback() -> None:
    assert default_ml_train_test("1m") == (2000, 500)
    assert default_ml_train_test("1h") == (350, 120)
    assert default_ml_train_test("4h") == (100, 40)
    assert default_ml_train_test("1d") == (70, 25)
    assert default_ml_train_test("unknown") == (2000, 500)


def test_resolve_1h_730d_under_old_2500_threshold() -> None:
    """730д 1h ≈ 1889 labeled < 2500, але ≥ 350+120."""
    assert resolve_ml_windows(1889, 17_520, "1h", None, None) == (350, 120)


def test_resolve_4h_and_1d_defaults_fit_730d() -> None:
    assert resolve_ml_windows(800, 4380, "4h", None, None) == (100, 40)
    assert resolve_ml_windows(130, 730, "1d", None, None) == (70, 25)


def test_resolve_explicit_windows_not_shrunk() -> None:
    assert resolve_ml_windows(100, 10_000, "1h", 2000, 500) is None
    assert resolve_ml_windows(800, 800, "1h", 50, 20) == (50, 20)


def test_resolve_spec_sized_hourly_does_not_shrink() -> None:
    """Spec-тести: 300–500 1h барів не повинні запускати LightGBM."""
    assert resolve_ml_windows(200, 500, "1h", None, None) is None
    assert resolve_ml_windows(120, 300, "1h", None, None) is None


def test_resolve_shrinks_on_long_history() -> None:
    got = resolve_ml_windows(200, 700, "1h", None, None)
    assert got is not None
    train, test = got
    assert train + test == 200
    assert train >= MIN_ML_TRAIN
    assert test >= MIN_ML_TEST


def test_resolve_rejects_tiny_sample() -> None:
    assert resolve_ml_windows(30, 10_000, "4h", None, None) is None
