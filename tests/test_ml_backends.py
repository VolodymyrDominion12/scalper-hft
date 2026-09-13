"""Тести ML-бэкендів LightGBM / XGBoost (дослідження §2.1 bake-off).

Дослідження §2.1: XGBoost і LightGBM — обидва еталони. Bake-off — перевірка
чи XGBoost дає ΔSharpe > 0 на тій самій OOS-вибірці. Перевіряємо:
  - make_backend створює LightGBM/XGBoost
  - is_available детектить встановлені бекенди
  - train_walk_forward з backend="lightgbm" (backward-compat)
  - train_walk_forward з backend="xgboost" (skip якщо не встановлено)
  - обидва бекенди дають прогнози у [0,1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.ml.backends import is_available, make_backend

# ── make_backend / is_available ────────────────────────────────────────────


class TestBackends:
    def test_lightgbm_available(self) -> None:
        # lightgbm у [ml] extras — має бути встановлено в тестовому середовищі
        assert is_available("lightgbm") is True

    def test_lightgbm_make_backend(self) -> None:
        model = make_backend("lightgbm", {"n_estimators": 10, "verbose": -1})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Невідомий ML-бэкенд"):
            make_backend("garbage")

    def test_case_insensitive(self) -> None:
        model = make_backend("LightGBM", {"n_estimators": 5, "verbose": -1})
        assert hasattr(model, "fit")

    @pytest.mark.skipif(not is_available("xgboost"), reason="xgboost не встановлено")
    def test_xgboost_make_backend(self) -> None:
        model = make_backend("xgboost", {"n_estimators": 10, "verbosity": 0})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_xgboost_not_available_raises(self) -> None:
        if is_available("xgboost"):
            pytest.skip("xgboost встановлено — ImportError не спрацює")
        with pytest.raises(ImportError, match="xgboost"):
            make_backend("xgboost")


# ── train_walk_forward з backend ────────────────────────────────────────────


def _synthetic_dataset(n: int = 300, seed: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Синтетичний датасет з фічами, лейблами {-1,+1} та цінами."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
        },
        index=idx,
    )
    # лейбл: залежить від f1 (є edge)
    y = pd.Series(np.where(X["f1"] > 0, 1, -1), index=idx)
    close = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 100, index=idx)
    return X, y, close


class TestTrainWalkForwardBackend:
    def test_lightgbm_backend_backward_compat(self) -> None:
        """backend='lightgbm' (дефолт) = стара поведінка."""
        from scalper_hft.ml.trainer import train_walk_forward

        X, y, close = _synthetic_dataset(300)
        result = train_walk_forward(
            X,
            y,
            train_size=150,
            test_size=100,
            close=close,
            backend="lightgbm",
            params={"n_estimators": 10, "verbose": -1},
        )
        assert result.n_windows > 0
        assert 0.0 <= result.oos_accuracy <= 1.0
        assert len(result.predictions) > 0

    def test_default_backend_is_lightgbm(self) -> None:
        """Без параметра backend — дефолт lightgbm."""
        from scalper_hft.ml.trainer import train_walk_forward

        X, y, close = _synthetic_dataset(300)
        result = train_walk_forward(
            X,
            y,
            train_size=150,
            test_size=100,
            close=close,
            params={"n_estimators": 10, "verbose": -1},
        )
        assert result.n_windows > 0

    @pytest.mark.skipif(not is_available("xgboost"), reason="xgboost не встановлено")
    def test_xgboost_backend(self) -> None:
        """backend='xgboost' — XGBoost walk-forward."""
        from scalper_hft.ml.trainer import train_walk_forward

        X, y, close = _synthetic_dataset(300)
        result = train_walk_forward(
            X,
            y,
            train_size=150,
            test_size=100,
            close=close,
            backend="xgboost",
            params={"n_estimators": 10, "verbosity": 0, "use_label_encoder": False},
        )
        assert result.n_windows > 0
        assert 0.0 <= result.oos_accuracy <= 1.0
        # прогнози мають бути в {-1, +1} (не {0,1})
        assert set(result.predictions.unique()).issubset({-1, 0, 1})

    @pytest.mark.skipif(not is_available("xgboost"), reason="xgboost не встановлено")
    def test_both_backends_produce_predictions(self) -> None:
        """Обидва бекенди дають прогнози на тій самій вибірці."""
        from scalper_hft.ml.trainer import train_walk_forward

        X, y, close = _synthetic_dataset(300, seed=42)
        r_lgb = train_walk_forward(
            X,
            y,
            train_size=150,
            test_size=100,
            close=close,
            backend="lightgbm",
            params={"n_estimators": 10, "verbose": -1},
        )
        r_xgb = train_walk_forward(
            X,
            y,
            train_size=150,
            test_size=100,
            close=close,
            backend="xgboost",
            params={"n_estimators": 10, "verbosity": 0, "use_label_encoder": False},
        )
        # обидва дають стільки ж вікон/прогнозів
        assert r_lgb.n_windows == r_xgb.n_windows
        assert len(r_lgb.predictions) == len(r_xgb.predictions)
