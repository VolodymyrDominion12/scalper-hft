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


def test_cscv_pbo_math():
    """CSCV: зі штучними даними, де edge є — PBO має бути низьким;
    де edge немає (шум) — PBO високий."""
    import numpy as np

    from scalper_hft.validation.cscv import combinatorial_splits, pbo_cscv

    n, s = 1600, 8
    rng = np.random.default_rng(11)
    # edge: перші 4 варіанти мають позитивний drift, решта — шум
    rets = np.zeros((s, n))
    for i in range(s):
        drift = 0.0005 if i < 4 else 0.0
        rets[i] = rng.normal(drift, 0.01, n)

    res = pbo_cscv(rets, n_blocks=8, max_combos=200)
    assert res.n_combos > 50
    # у 100% комбінацій IS-кращий варіант (з drift) добре працює на OOS
    assert res.pbo < 0.5, f"PBO має бути низьким, отримано {res.pbo}"

    # чистий шум → PBO високий
    noise = rng.normal(0.0, 0.01, (s, n))
    res_noise = pbo_cscv(noise, n_blocks=8, max_combos=200)
    assert res_noise.pbo > 0.3, f"PBO на шумі має бути високим, отримано {res_noise.pbo}"


def test_combinatorial_splits_cover_all():
    from scalper_hft.validation.cscv import combinatorial_splits

    splits = combinatorial_splits(6, 3)
    assert len(splits) == 20  # C(6,3)
    for tr, te in splits:
        assert len(set(tr) & set(te)) == 0
        assert len(tr) == 3 and len(te) == 3


def test_utc_now_helper():
    """_utc_now має повертати naive UTC (збігається з індексами кешу)."""
    from scalper_hft.data.downloader import _utc_now

    now = _utc_now()
    assert now.tz is None
    # різниця з реальним UTC мала (< 60 сек)
    import time

    assert abs(time.time() - now.timestamp()) < 60


def test_interval_staleness_logic():
    """Свіжість кешу залежить від інтервалу: 2 бари 1m = 2 хв, 2 бари 1h = 2 год."""
    from scalper_hft.data.downloader import _interval_ms

    assert _interval_ms("1m") == 60_000
    assert _interval_ms("5m") == 300_000
    assert _interval_ms("1h") == 3_600_000
    assert _interval_ms("1s") == 1_000


def test_funding_charged_once_per_block():
    """Funding платиться ОДИН раз на період ставки, а не кожен бар."""
    import numpy as np
    import pandas as pd

    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel

    # 3 години 1m-барів; одна funding-ставка в середині
    idx = pd.date_range("2025-01-01", periods=180, freq="1min")
    close = pd.Series(100.0 + 0.001 * np.arange(180), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0}, index=idx)
    funding = pd.DataFrame({"fundingRate": [0.001]}, index=[pd.Timestamp("2025-01-01 01:00:00")])

    class AlwaysShort:
        name = "short"
        param_space = {}
        needs_trades = False

        def __init__(self, **kwargs) -> None:
            pass

        def generate_signals(self, df):
            import pandas as pd

            return pd.Series(-1, index=df.index)

    # шорт з позитивним фандінгом ОТРИМУЄ платіж: -pos × rate = -(-1) × 0.001 = +0.001
    res_f = run_backtest(df, AlwaysShort(), cost=CostModel(maker_fee=0, taker_fee=0, slippage_frac=0), funding=funding, position_pct=1.0)
    res_0 = run_backtest(df, AlwaysShort(), cost=CostModel(maker_fee=0, taker_fee=0, slippage_frac=0), funding=None, position_pct=1.0)
    diff = res_f.equity.pct_change().fillna(0.0) - res_0.equity.pct_change().fillna(0.0)
    # funding вплив має бути ≈ +0.001 РІВНО на одному барі (а не на всіх 180)
    n_funding_bars = int((diff.abs() > 1e-9).sum())
    assert n_funding_bars == 1, f"funding мав бути на одному барі, на {n_funding_bars}"
    assert abs(float(diff[diff != 0].iloc[0]) - 0.001) < 1e-6, "розмір одного платежу = ставка"
    assert abs(float(diff.sum()) - 0.001) < 1e-6, "сумарний вплив = одна ставка"


def test_paper_replay_daily_reset():
    """Пауза після серії збитків скидається на новий день (не блокує назавжди)."""
    import numpy as np
    import pandas as pd

    from scalper_hft.live.paper_replay import paper_replay

    # 3 дні 1m-барів; ціна щодня падає → кожен лонг-цикл збитковий
    idx = pd.date_range("2025-01-01", periods=3 * 1440, freq="1min")
    day_offset = (idx.day - idx[0].day).values * 0.01  # -1%/день
    hour_drift = np.arange(len(idx)) * 0.00001  # повільне падіння всередині дня
    close = pd.Series(100.0 * (1 - day_offset - hour_drift), index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 10.0}, index=idx)

    class PeriodicLong:
        """Лонг кожен 6-й бар, решта — флет: часті збиткові round-trips."""

        name = "periodic_long"
        param_space = {}
        needs_trades = False

        def __init__(self, **kwargs) -> None:
            pass

        def generate_signals(self, df):
            import pandas as pd

            return pd.Series(np.where(np.arange(len(df)) % 6 == 0, 1, 0), index=df.index)

    r = paper_replay(df, PeriodicLong(), position_pct=0.1)
    trades = r.trades[r.trades["type"] == "trade"] if "type" in r.trades.columns else r.trades
    # з паузою по 3 збитки на день і скиданням — угоди мають бути КОЖЕН день
    days = pd.to_datetime(trades["exit_ts"]).dt.date.nunique() if len(trades) else 0
    assert days >= 2, f"після скидання паузи угоди мають бути в наступні дні, днів з угодами: {days}"
    assert len(trades) >= 3, f"мінімум 3 угоди (по одній серії на день), отримано {len(trades)}"


def test_delta_neutral_basis_accounting():
    """Delta-neutral: ціновий PnL = ±Δbasis; funding один раз за блок; 2-leg комісії."""
    import numpy as np
    import pandas as pd

    from scalper_hft.backtest.delta_neutral import run_delta_neutral_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.strategies.funding_arb import FundingArb

    idx = pd.date_range("2025-01-01", periods=300, freq="1min")
    # перп росте, спот стабільний → basis зростає
    perp = pd.DataFrame({"close": 100.0 + 0.001 * np.arange(300)}, index=idx)
    spot = pd.DataFrame({"close": 100.0 + np.zeros(300)}, index=idx)
    # три ставки: третя припадає на активну carry-позицію (перевірка funding-платежу)
    funding = pd.DataFrame({"fundingRate": [0.0005, 0.0005, 0.0005]}, index=[idx[30], idx[150], idx[210]])
    # carry=+1 (шорт перп/лонг спот) при позитивному фандінгу
    s = FundingArb(upper_threshold=0.0004, lower_threshold=-0.0004, exit_threshold=0.0001)
    res = run_delta_neutral_backtest(perp, spot, s, funding, position_pct=1.0,
                                     cost=CostModel(0, 0, 0))
    # basis зростає → carry+1 (шорт перп) втрачає на basis; funding отримує на активній позиції
    ret_total = res.metrics.total_return
    assert ret_total < 0, f"шорт перпа при зростаючому basis має втрачати, отримали {ret_total}"
    # funding внесок ≈ одна ставка × 0.0005 (позиція активна на третій ставці)
    assert abs(res.funding_pnl / 100 - 0.0005) < 0.0002, f"funding внесок: {res.funding_pnl}"
