"""Тести системи: рушій без lookahead, метрики, DSR, індикатори, downloader-кеш."""

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import compute_metrics


def make_klines(n: int = 300, seed: int = 1, start_price: float = 100.0) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz=None)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class AlwaysLong:
    """Стратегія-заглушка: завжди лонг."""

    name = "always_long"
    param_space = {}
    needs_trades = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df):
        import pandas as pd

        return pd.Series(1, index=df.index)


class FlipFlop:
    """Стратегія-заглушка: зміна позиції щобарно (тест turnover/fees)."""

    name = "flip_flop"
    param_space = {}
    needs_trades = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df):
        import numpy as np
        import pandas as pd

        return pd.Series(np.where(np.arange(len(df)) % 2 == 0, 1, -1), index=df.index)


def test_engine_no_lookahead():
    """Виконання з лагом 1: легальна lag-1 стратегія (сигнал з даних до бару t)
    на детермінованих даних з альтернуванням ЗОБОВ'ЯЗАНА програвати — це доводить,
    що позиція застосовується з бару t+1, а не на тому ж барі."""
    import numpy as np
    import pandas as pd

    n = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100.0 * np.where(np.arange(n) % 2 == 0, 1.01, 1.00), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 100.0},
        index=idx,
    )
    ret = close.pct_change().fillna(0.0)

    class Lag1Momentum:
        """Легальна стратегія: напрямок попереднього бару (дані до t)."""

        name = "lag1"
        param_space = {}
        needs_trades = False

        def __init__(self, **kwargs) -> None:
            pass

        def generate_signals(self, df):
            return pd.Series(np.sign(ret).fillna(0.0), index=df.index)

    res = run_backtest(df, Lag1Momentum(), position_pct=1.0)
    assert res.metrics.total_return < 0, f"lag-1 мав би програти, total_return={res.metrics.total_return}"


def test_engine_always_long_matches_buy_hold():
    df = make_klines()
    res = run_backtest(df, AlwaysLong(), initial_capital=10_000, position_pct=1.0)
    buy_hold = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    assert abs(res.metrics.total_return - buy_hold) < 0.01


def test_engine_fees_reduce_return():
    df = make_klines()
    res_no_fee = run_backtest(df, FlipFlop(), cost=CostModel(maker_fee=0, taker_fee=0, slippage_frac=0), position_pct=0.1)
    res_fee = run_backtest(
        df, FlipFlop(), cost=CostModel(maker_fee=0.0005, taker_fee=0.0005, slippage_frac=0.0002), position_pct=0.1
    )
    assert res_fee.metrics.total_return < res_no_fee.metrics.total_return


def test_metrics_basic():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2025-01-01", periods=100, freq="1min")
    equity = pd.Series((1.0 + 0.001 * np.arange(100)) * 10_000.0, index=idx)
    m = compute_metrics(equity)
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.max_drawdown <= 0


def test_deflated_sharpe():
    import numpy as np

    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio

    rng = np.random.default_rng(7)
    ret = rng.normal(0.001, 0.01, 500)
    dsr_1 = deflated_sharpe_ratio(ret, n_trials=1)
    dsr_1000 = deflated_sharpe_ratio(ret, n_trials=1000)
    assert dsr_1 >= dsr_1000, "Більше спроб → нижчий DSR"


def test_indicators_no_nan_at_tail():
    import pandas as pd

    df = make_klines(500)
    from scalper_hft.features.indicators import add_standard_features, rsi

    f = add_standard_features(df)
    for col in ["rsi_14", "ema_9", "atr_14", "bb_width", "vwap_20"]:
        assert f[col].iloc[-1] == f[col].iloc[-1], f"NaN у {col} на останньому барі"


def test_purged_kfold_covers_all():
    from scalper_hft.validation.cv import purged_kfold_indices

    n = 100
    seen = set()
    for tr, te in purged_kfold_indices(n, n_splits=5, purge=5, embargo=3):
        assert len(set(tr) & set(te)) == 0, "train/test перекриваються"
        seen.update(te.tolist())
    assert len(seen) == n - (n % 5), "тестові зрізи не покривають весь діапазон"


def test_downloader_cache_roundtrip(tmp_path):
    import pandas as pd

    from scalper_hft.data.storage import klines_path, load_klines, save_klines

    df = make_klines(50)
    path = klines_path(tmp_path, "TESTUSDT", "1m")
    save_klines(path, df)
    loaded = load_klines(path)
    assert loaded is not None and len(loaded) == len(df)
    assert abs(loaded["close"].iloc[-1] - df["close"].iloc[-1]) < 1e-9


def test_walk_forward_runs():
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = make_klines(1000)
    res = run_walk_forward(df, AlwaysLong(), train_bars=300, test_bars=100)
    assert len(res.windows) >= 4
    assert res.avg_oos_sharpe != float("nan")
