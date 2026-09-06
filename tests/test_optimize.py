"""Тести validation/optimize.py: часові зрізи trades/funding у CV, purge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy


def _ohlcv(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )


class _RecordingStrategy(Strategy):
    """needs_trades стратегія, що записує часові межі отриманих trades."""

    name = "recorder"
    param_space: dict = {}
    needs_trades = True
    needs_funding = False
    recorded: list[tuple[pd.Timestamp, pd.Timestamp] | None] = []

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df, trades=None, funding=None):
        if trades is not None and len(trades) > 0:
            type(self).recorded.append((trades.index.min(), trades.index.max()))
        else:
            type(self).recorded.append(None)
        return pd.Series(0, index=df.index)


def test_default_objective_slices_trades_by_test_window() -> None:
    """aggTrades мають індекс часу трейдів, не позиції барів: CV-бектест на
    test-вікні має отримувати trades саме цього вікна (не train-зріз)."""
    from scalper_hft.validation.optimize import _default_objective

    df = _ohlcv(200)
    # тики в 5 разів густіші за бари — позиційний iloc давав би абсолютно
    # інші (зсунуті в минуле) часові вікна
    tick_idx = pd.date_range("2025-01-01", periods=1000, freq="12s")
    trades = pd.DataFrame({"price": 100.0, "qty": 1.0, "is_buyer_maker": False}, index=tick_idx)

    _RecordingStrategy.recorded = []
    _default_objective(
        df,
        _RecordingStrategy,
        params={},
        cost=CostModel(),
        trades=trades,
        funding=None,
        n_splits=4,
        embargo=2,
        position_pct=0.01,
        purge=5,
    )

    assert len(_RecordingStrategy.recorded) == 4
    fold_size = 200 // 4
    for k, window in enumerate(_RecordingStrategy.recorded):
        assert window is not None
        t0, t1 = window
        te_start = df.index[k * fold_size]
        te_end = df.index[(k + 1) * fold_size - 1]
        assert t0 >= te_start, f"fold {k}: trades починаються {t0} раніше test-вікна {te_start}"
        assert t1 <= te_end + pd.Timedelta("1min"), f"fold {k}: trades виходять за test-вікно"


def test_effective_purge_from_param_space() -> None:
    from scalper_hft.validation.optimize import _effective_purge

    # явний purge перемагає
    assert _effective_purge({"holding_bars": (5, 30, 5)}, purge=7) == 7
    # holding_bars hi=30 < 50 → дефолт 50
    assert _effective_purge({"holding_bars": (5, 30, 5)}, None) == 50
    # довгий lookback → purge з нього
    assert _effective_purge({"lookback": (100, 480, 20)}, None) == 480
    # без horizon-параметрів → 50
    assert _effective_purge({"entry_z": (1.0, 3.0, 0.5)}, None) == 50


def test_purged_kfold_embargo_only_after_test() -> None:
    """AFML: purge ліворуч від test, embargo — лише праворуч."""
    from scalper_hft.validation.cv import purged_kfold_indices

    n = 100
    for tr, te in purged_kfold_indices(n, n_splits=5, purge=5, embargo=3):
        assert len(set(tr) & set(te)) == 0
        if len(te) == 0 or len(tr) == 0:
            continue
        te_start = te[0]
        left = tr[tr < te_start]
        if len(left):
            # ліворуч від test залишається gap рівно purge (без зайвого embargo)
            assert left.max() == te_start - 5 - 1
