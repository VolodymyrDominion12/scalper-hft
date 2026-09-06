"""Регресійні тести: AFML purge у walk-forward ML (t1 має реально застосовуватись).

Баг: train_from_ohlcv / train_walk_forward_meta не передавали t1 → train-зразки,
чиї triple-barrier лейбли заходять у test-вікно, потрапляли у навчання
(label leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_Xy(n: int = 400) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    X = pd.DataFrame({"f1": np.linspace(-1, 1, n), "f2": np.linspace(1, -1, n)}, index=idx)
    y = pd.Series(np.where(np.arange(n) % 2 == 0, 1, -1), index=idx)
    # t1 = t0 + 10 барів → останні ~10 train-зразків кожного вікна overlapping
    t1 = pd.Series(idx[10:], index=idx[:-10]).reindex(idx)
    return X, y, t1


class _FitSpy:
    """Записує індекси train-вибірки кожного fit."""

    def __init__(self) -> None:
        self.train_indices: list[pd.Index] = []


@pytest.fixture
def fit_spy(monkeypatch) -> _FitSpy:
    lgbm = pytest.importorskip("lightgbm")
    spy = _FitSpy()
    orig_fit = lgbm.LGBMClassifier.fit

    def _spy_fit(self, X, y, sample_weight=None, **kwargs):
        spy.train_indices.append(X.index if hasattr(X, "index") else pd.Index([]))
        return orig_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    monkeypatch.setattr(lgbm.LGBMClassifier, "fit", _spy_fit)
    return spy


def _assert_purged(spy: _FitSpy, t1: pd.Series, train_size: int, test_size: int) -> None:
    """Жоден train-зразок вікна не має t1 після початку test-вікна.

    Вікно визначається за позицією ПЕРШОГО train-зразка (primary, OOF і meta
    fit-и вікна починаються з одного offsets), тому працює і для meta-шляху
    з кількома fit на вікно.
    """
    assert spy.train_indices, "жодного fit — вікна не сформувались"
    pos = t1.index
    for tr_idx in spy.train_indices:
        if len(tr_idx) == 0:
            continue
        start = pos.get_loc(tr_idx[0])
        test_start_ts = pos[start + train_size]
        t1_tr = t1.reindex(tr_idx).dropna()
        violators = t1_tr[t1_tr > test_start_ts]
        assert len(violators) == 0, f"start={start}: {len(violators)} train-зразків з t1 > test_start (leakage)"


def test_train_walk_forward_purges_with_t1(fit_spy) -> None:
    from scalper_hft.ml.trainer import train_walk_forward

    X, y, t1 = _toy_Xy()
    train_walk_forward(X, y, train_size=100, test_size=50, t1=t1)
    _assert_purged(fit_spy, t1, train_size=100, test_size=50)


def test_train_walk_forward_meta_purges_with_t1(fit_spy) -> None:
    from scalper_hft.ml.trainer import train_walk_forward_meta

    X, y, t1 = _toy_Xy()
    train_walk_forward_meta(X, y, train_size=100, test_size=50, t1=t1)
    # meta-шлях робить кілька fit на вікно (primary + OOF + meta) — усі на purged train
    _assert_purged(fit_spy, t1, train_size=100, test_size=50)


def test_build_labeled_dataset_returns_aligned_t1() -> None:
    from scalper_hft.ml.features import build_labeled_dataset

    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.001, n))), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )
    X, y, w, t1 = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8, return_t1=True)
    # t1 вирівняний за X і завжди у майбутньому відносно t0
    assert list(t1.index) == list(X.index)
    assert (t1 > X.index).all()


def test_build_labeled_dataset_default_3tuple() -> None:
    """Зворотна сумісність: без return_t1 — 3-елементний кортеж."""
    from scalper_hft.ml.features import build_labeled_dataset

    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )
    out = build_labeled_dataset(df, mode="horizon", horizon=3)
    assert len(out) == 3
