"""Тести validation/sensitivity.py: плато vs пік, best_params, 2D-сітка."""

from __future__ import annotations

import pandas as pd
import pytest
from scalper_hft.strategies.base import Strategy


class _Plateau(Strategy):
    """Метрика не залежить від параметра (ідеальне плато)."""

    name = "plateau"
    param_space = {"x": (1.0, 5.0, 1.0)}

    def generate_signals(self, df, trades=None, funding=None):
        return pd.Series(1, index=df.index)


class _Spike(Strategy):
    """Метрика ненульова лише при x == 3 (ізольований пік)."""

    name = "spike"
    param_space = {"x": (1.0, 5.0, 1.0)}

    def generate_signals(self, df, trades=None, funding=None):
        if float(self.get("x", 1.0)) == 3.0:
            return pd.Series(1, index=df.index)
        return pd.Series(0, index=df.index)


def test_smoothness_high_on_plateau(make_ohlcv) -> None:
    from scalper_hft.validation.sensitivity import parameter_sensitivity

    res = parameter_sensitivity(make_ohlcv(120), _Plateau(), "x", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(res.grid) == 5
    assert res.smoothness == 1.0  # константна метрика → плато


def test_smoothness_low_on_spike(make_ohlcv) -> None:
    import numpy as np
    from scalper_hft.validation.sensitivity import parameter_sensitivity

    # дані з дрейфом вгору: always-long дає стійкий позитивний Sharpe
    df = make_ohlcv(120)
    idx = df.index
    close = pd.Series(100.0 * np.exp(np.linspace(0, 0.5, 120)), index=idx)
    df = df.assign(close=close, open=close.shift(1).fillna(100.0), high=close * 1.001, low=close * 0.999)

    res = parameter_sensitivity(df, _Spike(), "x", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert res.smoothness < 1.0
    # best — пік при x=3
    assert res.best_params["x"] == 3.0


def test_sensitivity_2d_grid(make_ohlcv) -> None:
    from scalper_hft.validation.sensitivity import parameter_sensitivity

    res = parameter_sensitivity(
        make_ohlcv(120),
        _Plateau(),
        "x",
        [1.0, 2.0],
        secondary_param="y",
        secondary_values=[10.0, 20.0],
    )
    assert len(res.grid) == 4
    assert {"x", "y", "metric"} <= set(res.grid.columns)
    assert "y" in res.best_params


def test_sensitivity_all_fail_raises(make_ohlcv) -> None:
    class _Boom(_Plateau):
        name = "boom"

        def generate_signals(self, df, trades=None, funding=None):
            raise RuntimeError("завжди падає")

    from scalper_hft.validation.sensitivity import parameter_sensitivity

    with pytest.raises(ValueError, match="комбінац"):
        parameter_sensitivity(make_ohlcv(120), _Boom(), "x", [1.0, 2.0])
