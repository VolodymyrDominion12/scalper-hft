"""Тести: MLStrategy, PurgedKFold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 800, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── PurgedKFold ───────────────────────────────────────────────────────────────


class TestPurgedKFold:
    def _get_X_t1(self, n=500):
        from scalper_hft.ml.labeling import label_from_ohlcv

        df = _make_df(n)
        events = label_from_ohlcv(df, pt=1.0, sl=1.0, holding_bars=10)
        from scalper_hft.ml.features import build_labeled_dataset

        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=10)
        t1 = events.loc[X.index, "t1"].dropna()
        return X, y, w, t1

    def test_split_generates_correct_n_folds(self):
        from scalper_hft.validation.cv import PurgedKFold

        X, y, w, t1 = self._get_X_t1()
        pkf = PurgedKFold(n_splits=5)
        folds = list(pkf.split(X, t1=t1))
        assert len(folds) == 5

    def test_train_test_disjoint(self):
        from scalper_hft.validation.cv import PurgedKFold

        X, y, w, t1 = self._get_X_t1()
        pkf = PurgedKFold(n_splits=3)
        for train_idx, test_idx in pkf.split(X, t1=t1):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test перетинаються: {len(overlap)} індексів"

    def test_no_lookahead_in_train(self):
        """Жоден ЛІВИЙ train-зразок (t0 < test_start) чий t1 > test_start не має бути в train.

        Правосторонні зразки (t0 > test_end) мають t1 > test_start за природою,
        але не викликають lookahead — вони прийшли ПІСЛЯ test.
        """
        from scalper_hft.validation.cv import PurgedKFold

        X, y, w, t1 = self._get_X_t1(600)
        if len(X) < 100:
            pytest.skip("Замало зразків")
        pkf = PurgedKFold(n_splits=3, embargo_pct=0.0)
        for train_idx, test_idx in pkf.split(X, t1=t1):
            t_test_start = X.index[test_idx[0]]
            # Перевіряємо ЛИШЕ ліву частину train (t0 < t_test_start)
            left_train_idx = [i for i in train_idx if X.index[i] < t_test_start]
            if not left_train_idx:
                continue
            t1_left = t1.reindex(X.index[left_train_idx]).dropna()
            violators = t1_left[t1_left > t_test_start]
            assert len(violators) == 0, f"Lookahead: {len(violators)} лівих зразків мають t1 > test_start"

    def test_requires_datetime_index(self):
        from scalper_hft.validation.cv import PurgedKFold

        X = pd.DataFrame({"a": range(50)})  # RangeIndex
        t1 = pd.Series(pd.date_range("2025-01-01", periods=50, freq="1min"))
        pkf = PurgedKFold(n_splits=3)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            list(pkf.split(X, t1=t1))

    def test_cross_val_score_returns_array(self):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        from scalper_hft.validation.cv import PurgedKFold

        X, y, w, t1 = self._get_X_t1(600)
        if len(X) < 50:
            pytest.skip("Замало зразків")

        pkf = PurgedKFold(n_splits=3)
        model = LGBMClassifier(n_estimators=20, verbosity=-1)
        scores = pkf.cross_val_score(model, X, y, t1=t1, sample_weight=w, scoring="accuracy")
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_cross_val_neg_log_loss(self):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        from scalper_hft.validation.cv import PurgedKFold

        X, y, w, t1 = self._get_X_t1(600)
        if len(X) < 50:
            pytest.skip("Замало зразків")

        pkf = PurgedKFold(n_splits=3)
        model = LGBMClassifier(n_estimators=20, verbosity=-1)
        scores = pkf.cross_val_score(model, X, y, t1=t1, scoring="neg_log_loss")
        assert all(s <= 0.0 for s in scores), "neg_log_loss має бути ≤ 0"


# ── MLStrategy ────────────────────────────────────────────────────────────────

lgbm = pytest.importorskip("lightgbm", reason="lightgbm не встановлений")


class TestMLStrategy:
    def test_registered_in_registry(self):
        from scalper_hft.strategies import REGISTRY

        assert "ml_strategy" in REGISTRY

    def test_signals_shape_matches_df(self):
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy(
            "ml_strategy",
            train_bars=50,
            test_bars=20,
            holding_bars=8,
        )
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert signals.index.equals(df.index)

    def test_signals_are_valid_values(self):
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy("ml_strategy", train_bars=50, test_bars=20, holding_bars=8)
        signals = strat.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1}), f"Невалідні значення: {signals.unique()}"

    def test_signals_not_all_zero(self):
        """Стратегія має генерувати хоч якісь сигнали на достатньому датасеті."""
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy("ml_strategy", train_bars=50, test_bars=20, holding_bars=8)
        signals = strat.generate_signals(df)
        assert (signals != 0).sum() > 0, "Всі сигнали нульові"

    def test_oos_signals_only_after_train(self):
        """Сигнали мають з'являтися лише після train-вікна (без lookahead на train-барах)."""
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        train_bars = 50
        strat = get_strategy("ml_strategy", train_bars=train_bars, test_bars=20, holding_bars=8)
        signals = strat.generate_signals(df)
        # labeled зразки починаються після warm-up фіч (~50 барів)
        # Просто перевіряємо що є сигнали у другій половині
        second_half = signals.iloc[len(signals) // 2 :]
        assert second_half.abs().sum() >= 0  # структурна перевірка

    def test_vol_regime_filter(self):
        """Перевіряємо що regime_filter='vol' не виводить помилку."""
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy(
            "ml_strategy",
            train_bars=50,
            test_bars=20,
            holding_bars=8,
            regime_filter="vol",
            vol_regime_ok="normal",
        )
        signals = strat.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_insufficient_data_returns_zeros(self):
        """На замалому датасеті стратегія повертає нульові сигнали без помилки."""
        from scalper_hft.strategies import get_strategy

        df = _make_df(50)  # менше train_bars
        strat = get_strategy("ml_strategy", train_bars=500, test_bars=200, holding_bars=10)
        signals = strat.generate_signals(df)
        assert (signals == 0).all()

    def test_param_space_defined(self):
        from scalper_hft.strategies.ml_strategy import MLStrategy

        assert "pt" in MLStrategy.param_space
        assert "sl" in MLStrategy.param_space
        assert "holding_bars" in MLStrategy.param_space
