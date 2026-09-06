"""Triple-barrier: внутрішньобаровий (high/low) шлях і семантика міток."""

from __future__ import annotations

import pandas as pd
import pytest
from scalper_hft.ml.labeling import get_events, get_labels


@pytest.fixture()
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=5, freq="1min")


def _events(close, high, low, idx, pt=1.0, sl=1.0, vb=None):
    target = pd.Series(0.01, index=idx[:1])  # бар'єри ±1% від входу
    return get_events(
        close,
        t_events=idx[:1],
        pt_sl=(pt, sl),
        target=target,
        vertical_barrier_times=vb,
        high=high,
        low=low,
    )


def test_pt_hit_via_wick_not_close(idx) -> None:
    """PT торкається high-фітилем, close туди не доходить → barrier='pt'."""
    close = pd.Series([100.0, 100.1, 100.2, 100.15, 100.3], index=idx)
    high = pd.Series([100.0, 101.6, 100.25, 100.2, 100.35], index=idx)  # wick > 101.0 на барі 1
    low = pd.Series([99.9, 100.0, 100.1, 100.1, 100.25], index=idx)

    ev = _events(close, high, low, idx)
    assert ev["t1"].iloc[0] == idx[1]
    assert ev["barrier"].iloc[0] == "pt"
    labels = get_labels(ev, close)
    assert labels.iloc[0] == 1

    # close-to-close fallback (без wicks): PT не торкається взагалі
    ev_close = get_events(close, t_events=idx[:1], pt_sl=(1.0, 1.0), target=pd.Series(0.01, index=idx[:1]))
    assert pd.isna(ev_close["t1"].iloc[0])


def test_sl_hit_via_wick(idx) -> None:
    close = pd.Series([100.0, 100.2, 100.1, 100.2, 100.1], index=idx)
    high = pd.Series([100.1, 100.3, 100.2, 100.25, 100.15], index=idx)
    low = pd.Series([99.9, 99.9, 98.9, 100.0, 100.0], index=idx)  # low бару 2 < 99.0

    ev = _events(close, high, low, idx)
    assert ev["t1"].iloc[0] == idx[2]
    assert ev["barrier"].iloc[0] == "sl"
    assert get_labels(ev, close).iloc[0] == -1


def test_both_barriers_same_bar_pessimistic_sl(idx) -> None:
    """Один бар торкається і PT, і SL → песимістично SL."""
    close = pd.Series([100.0, 100.5, 100.2, 100.1, 100.0], index=idx)
    high = pd.Series([100.1, 101.2, 100.3, 100.2, 100.1], index=idx)  # бар 1: > 101.0
    low = pd.Series([99.9, 98.8, 100.1, 100.0, 99.9], index=idx)  # бар 1: < 99.0

    ev = _events(close, high, low, idx)
    assert ev["t1"].iloc[0] == idx[1]
    assert ev["barrier"].iloc[0] == "sl"
    assert get_labels(ev, close).iloc[0] == -1


def test_vertical_barrier_timeout_label_zero(idx) -> None:
    """Жоден бар'єр не торкнутий до вертикального → label 0 (timeout)."""
    close = pd.Series([100.0, 100.1, 100.05, 100.1, 100.0], index=idx)
    high = close + 0.05
    low = close - 0.05
    vb = pd.Series(idx[2], index=idx[:1])

    ev = _events(close, high, low, idx, vb=vb)
    assert ev["t1"].iloc[0] == idx[2]
    assert ev["barrier"].iloc[0] == "vb"
    assert get_labels(ev, close).iloc[0] == 0


def test_short_side_mirrored(idx) -> None:
    """Шорт: PT — падіння (low), SL — зростання (high)."""
    close = pd.Series([100.0, 99.0, 99.5, 99.0, 99.2], index=idx)
    high = pd.Series([100.1, 99.2, 99.6, 99.1, 99.3], index=idx)
    low = pd.Series([99.9, 98.3, 99.4, 98.9, 99.1], index=idx)  # бар 1: −1.7% → PT шорта

    target = pd.Series(0.01, index=idx[:1])
    ev = get_events(
        close,
        t_events=idx[:1],
        pt_sl=(1.0, 1.0),
        target=target,
        side=pd.Series(-1.0, index=idx[:1]),
        high=high,
        low=low,
    )
    assert ev["barrier"].iloc[0] == "pt"
    assert get_labels(ev, close).iloc[0] == 1
