"""Паритет векторизованого рушія з еталонними циклами + smoke-продуктивність."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

# ── Еталонні (попередні) реалізації — для parity-перевірки ──────────────────


def _ref_extract_trades(positions: pd.Series, ret: pd.Series, fees: pd.Series, close: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    cur_pos = 0.0
    entry_ts = None
    cum_ret = 0.0
    prev_close = close.shift(1)

    def fill_px(ts):
        px = prev_close.get(ts, np.nan)
        if not np.isfinite(px):
            px = close.get(ts, np.nan)
        return float(px) if np.isfinite(px) else float("nan")

    for ts, pos in positions.items():
        if pos != cur_pos:
            turnover = abs(pos - cur_pos)
            fee_bar = fees.get(ts, 0.0)
            rate_eff = fee_bar / turnover if turnover > 0 else 0.0
            exit_fee = rate_eff * abs(cur_pos) if cur_pos != 0 else 0.0
            if cur_pos != 0 and entry_ts is not None:
                rows.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "side": int(cur_pos / abs(cur_pos)),
                        "ret": cum_ret - exit_fee,
                        "entry_price": fill_px(entry_ts),
                        "exit_price": fill_px(ts),
                    }
                )
            if pos != 0:
                entry_ts = ts
                cum_ret = ret.get(ts, 0.0) * pos - (fee_bar - exit_fee)
            else:
                entry_ts = None
                cum_ret = 0.0
            cur_pos = pos
        elif cur_pos != 0 and entry_ts is not None:
            cum_ret += ret.get(ts, 0.0) * cur_pos
    if cur_pos != 0 and entry_ts is not None:
        rows.append(
            {
                "entry_ts": entry_ts,
                "exit_ts": positions.index[-1],
                "side": int(cur_pos / abs(cur_pos)),
                "ret": cum_ret,
                "entry_price": fill_px(entry_ts),
                "exit_price": fill_px(positions.index[-1]),
            }
        )
    return pd.DataFrame(rows)


def _ref_maker_fills(target_vals, close_vals, low_vals, high_vals, seed=42):
    n = len(target_vals)
    actual_pos = np.zeros(n)
    adverse = np.zeros(n)
    adverse_bps = 0.0001
    prob_touch = 0.5
    curr_pos = 0.0
    rng = np.random.default_rng(seed)
    rands = rng.random(n)
    for i in range(1, n):
        t_pos = target_vals[i]
        if t_pos != curr_pos:
            limit_px = close_vals[i - 1]
            filled = False
            if t_pos > curr_pos:
                if low_vals[i] < limit_px:
                    filled = True
                    adverse[i] += abs(t_pos - curr_pos) * adverse_bps
                elif low_vals[i] == limit_px and rands[i] < prob_touch:
                    filled = True
            elif t_pos < curr_pos:
                if high_vals[i] > limit_px:
                    filled = True
                    adverse[i] += abs(curr_pos - t_pos) * adverse_bps
                elif high_vals[i] == limit_px and rands[i] < prob_touch:
                    filled = True
            if filled:
                curr_pos = t_pos
        actual_pos[i] = curr_pos
    return actual_pos, adverse


# ── Тести ────────────────────────────────────────────────────────────────────


def _random_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Випадкові позиції: 0 / ±0.01 з фліпами і затримками."""
    out = np.zeros(n)
    pos = 0.0
    for i in range(n):
        u = rng.random()
        if u < 0.03:
            pos = 0.01 if rng.random() < 0.5 else -0.01
        elif u < 0.06:
            pos = 0.0
        out[i] = pos
    return out


def test_extract_trades_parity() -> None:
    from scalper_hft.backtest.engine import _extract_trades

    rng = np.random.default_rng(123)
    n = 5000
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    for _ in range(5):
        p = pd.Series(_random_positions(rng, n), index=idx)
        ret = pd.Series(rng.normal(0, 0.001, n), index=idx)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))), index=idx)
        # fees: ненульові лише на барах зміни позиції + випадковий adverse
        turnover = p.diff().abs().fillna(p.abs())
        fees = turnover * 0.0007 + pd.Series(np.where(rng.random(n) < 0.02, 1e-7, 0.0), index=idx)

        ref = _ref_extract_trades(p, ret, fees, close)
        new = _extract_trades(p, ret, fees, close)

        assert len(ref) == len(new), f"кількість угод: {len(ref)} vs {len(new)}"
        if len(ref) == 0:
            continue
        pd.testing.assert_series_equal(
            ref["entry_ts"].reset_index(drop=True), new["entry_ts"].reset_index(drop=True), check_names=False
        )
        pd.testing.assert_series_equal(
            ref["exit_ts"].reset_index(drop=True), new["exit_ts"].reset_index(drop=True), check_names=False
        )
        assert np.allclose(ref["side"], new["side"])
        assert np.allclose(ref["ret"], new["ret"], atol=1e-12)
        assert np.allclose(ref["entry_price"], new["entry_price"], atol=1e-9)
        assert np.allclose(ref["exit_price"], new["exit_price"], atol=1e-9)


def test_maker_fills_parity() -> None:
    from scalper_hft.backtest.engine import _simulate_maker_fills

    rng = np.random.default_rng(77)
    n = 3000
    for _ in range(5):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.0003, n)))
        high = close * (1 + np.abs(rng.normal(0, 0.0003, n)))
        target = _random_positions(rng, n)

        ref_pos, ref_adv = _ref_maker_fills(target, close, low, high)
        new_pos, new_adv = _simulate_maker_fills(target, close, low, high)

        assert np.array_equal(ref_pos, new_pos), "maker fill позиції розійшлись"
        assert np.allclose(ref_adv, new_adv), "adverse penalties розійшлись"


def test_maker_backtest_perf_smoke() -> None:
    """90 днів 1m (~130k барів) maker-бектест має завершуватись швидко."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.strategies.mean_reversion import MeanReversionScalper

    rng = np.random.default_rng(5)
    n = 130_000
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n))), index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    t0 = time.perf_counter()
    res = run_backtest(df, MeanReversionScalper(), is_maker=True)
    elapsed = time.perf_counter() - t0
    assert res.metrics is not None
    assert elapsed < 20.0, f"maker-бектест 130k барів: {elapsed:.1f}s — занадто повільно"


def _ref_run_trades(pos: pd.Series, strat_ret: pd.Series) -> pd.DataFrame:
    """Еталонний цикл pairs/delta_neutral (до векторизації)."""
    rows: list[dict] = []
    cur = 0.0
    entry_ts = None
    cum = 0.0
    for ts, p in pos.items():
        if p != cur:
            if cur != 0 and entry_ts is not None:
                exit_extra = strat_ret.get(ts, 0.0) if p == 0 else 0.0
                rows.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "side": int(cur / abs(cur)) if cur else 0,
                        "ret": cum + exit_extra,
                    }
                )
            entry_ts = ts if p != 0 else None
            cum = 0.0
            cur = p
        if cur != 0 and entry_ts is not None:
            cum += strat_ret.get(ts, 0.0)
    if cur != 0 and entry_ts is not None:
        rows.append({"entry_ts": entry_ts, "exit_ts": pos.index[-1], "side": int(cur / abs(cur)), "ret": cum})
    return pd.DataFrame(rows)


def test_extract_run_trades_parity() -> None:
    from scalper_hft.backtest.engine import _extract_run_trades

    rng = np.random.default_rng(31)
    n = 4000
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    for _ in range(5):
        p = pd.Series(_random_positions(rng, n), index=idx)
        strat_ret = pd.Series(rng.normal(0, 0.0005, n), index=idx)
        # на барах закриття у flat strat_ret = −fee (як у pairs/delta_neutral)
        flat_after = (p == 0) & (p.shift(1).fillna(0.0) != 0)
        strat_ret[flat_after] = -0.0007

        ref = _ref_run_trades(p, strat_ret)
        new = _extract_run_trades(p, strat_ret)
        assert len(ref) == len(new)
        if len(ref) == 0:
            continue
        pd.testing.assert_series_equal(
            ref["entry_ts"].reset_index(drop=True), new["entry_ts"].reset_index(drop=True), check_names=False
        )
        pd.testing.assert_series_equal(
            ref["exit_ts"].reset_index(drop=True), new["exit_ts"].reset_index(drop=True), check_names=False
        )
        assert np.allclose(ref["side"], new["side"])
        assert np.allclose(ref["ret"], new["ret"], atol=1e-12)


def _bars_df(n: int, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    close = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


def test_pairs_step_latency_smoke() -> None:
    """Synthetic pairs engine steps over 1000 bars should finish quickly."""
    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.pairs_runner import PairsEngine
    from scalper_hft.strategies.pairs_arb import PairsArb

    n = 1000
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close_a = 100.0 + np.arange(n, dtype=float)
    close_b = 50.0 + np.arange(n, dtype=float) * 0.1
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
    )
    # warmup spread history so the timed loop avoids ADF/coint entry checks
    for i in range(min(80, n)):
        ts = idx[i]
        eng.on_bar(ts, close_a[i] + 1, close_a[i] - 1, close_a[i], close_b[i] + 1, close_b[i] - 1, close_b[i], signal=0)

    t0 = time.perf_counter()
    for i, ts in enumerate(idx):
        eng.on_bar(ts, close_a[i] + 1, close_a[i] - 1, close_a[i], close_b[i] + 1, close_b[i] - 1, close_b[i], signal=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"pairs on_bar x{n}: {elapsed:.2f}s — too slow"


def test_storage_atomic_write_smoke(tmp_path) -> None:
    """save_klines twice then load succeeds (atomic parquet replace)."""
    from scalper_hft.data.storage import load_klines, save_klines

    path = tmp_path / "AAA_1m_klines.parquet"
    df1 = _bars_df(60)
    df2 = _bars_df(80, start="2025-01-02")
    save_klines(path, df1)
    save_klines(path, df2)
    loaded = load_klines(path)
    assert loaded is not None
    assert len(loaded) == len(df2)
    cols = ["open", "high", "low", "close", "volume"]
    expected = df2[cols].astype(float)
    assert list(loaded.columns) == cols
    np.testing.assert_allclose(loaded.values, expected.values)
