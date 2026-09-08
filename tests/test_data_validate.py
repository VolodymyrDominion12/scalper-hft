"""Тести validate_bars: OHLC-інваріанти, дублікати, монотонність, дірки, майбутнє."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.data.validate import validate_bars


def _bars(n: int = 20, start: str = "2024-01-01", freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    c = np.full(n, 100.0)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)


def test_ok_on_clean_data() -> None:
    rep = validate_bars(_bars(), interval="1m")
    assert rep.ok and rep.summary().startswith("bars ok")


def test_empty_dataset_fails() -> None:
    rep = validate_bars(pd.DataFrame())
    assert not rep.ok and "порожній" in rep.issues[0]


def test_ohlc_violation_detected() -> None:
    df = _bars()
    df.loc[df.index[5], "low"] = 105.0  # low > close
    rep = validate_bars(df)
    assert not rep.ok and rep.n_ohlc_violations == 1


def test_duplicates_detected() -> None:
    df = pd.concat([_bars(), _bars().iloc[[3]]]).sort_index()
    rep = validate_bars(df)
    assert not rep.ok and rep.n_duplicates == 1


def test_non_monotonic_detected() -> None:
    df = _bars()
    df = df.iloc[::-1]  # розвертаємо індекс
    rep = validate_bars(df)
    assert not rep.ok and not rep.monotonic


def test_future_bars_detected() -> None:
    df = _bars(start="2999-01-01")
    rep = validate_bars(df)
    assert not rep.ok and rep.n_future == 20


def test_future_bars_respect_explicit_now() -> None:
    df = _bars(start="2999-01-01")
    rep = validate_bars(df, now=pd.Timestamp("2999-06-01"))
    assert rep.n_future == 0


def test_gaps_detected_with_interval() -> None:
    df = _bars(20, freq="1min")
    df = df.drop(df.index[10])  # дірка на 1 бар
    rep = validate_bars(df, interval="1m")
    assert not rep.ok and rep.n_gaps == 1


def test_gaps_ignored_without_interval() -> None:
    df = _bars(20, freq="1min").drop(_bars(20)[10:11].index)
    rep = validate_bars(df)  # без interval дірки не рахуються
    assert rep.n_gaps == 0 and rep.ok
