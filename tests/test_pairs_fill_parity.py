"""R1: paper PairsEngine і maker-бектест — одна модель філу (mid + seed)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scalper_hft.backtest.pairs import _maker_pair_positions
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import (
    decide_fill,
    dump_fill_rng_state,
    fill_probability,
    load_fill_rng_state,
    maker_fill_rng,
)
from scalper_hft.live.pairs_engine import PairsEngine, PendingOrder
from scalper_hft.strategies.pairs_arb import PairsArb


def _synthetic_common(n: int = 220, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    leg1 = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    leg2 = 50.0 + np.cumsum(rng.normal(0, 0.2, n))
    wobble1 = rng.uniform(0.2, 1.5, n)
    wobble2 = rng.uniform(0.1, 0.8, n)
    return pd.DataFrame(
        {
            "leg1": leg1,
            "leg2": leg2,
            "l1_high": leg1 + wobble1,
            "l1_low": leg1 - wobble1,
            "l2_high": leg2 + wobble2,
            "l2_low": leg2 - wobble2,
        },
        index=idx,
    )


def _synthetic_signals(index: pd.Index, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    raw = rng.integers(-1, 2, size=len(index))
    return pd.Series(raw.astype(float), index=index)


def _engine(*, fill_rng: np.random.Generator | None = None, fill_seed: int | None = 42) -> PairsEngine:
    acc = PaperAccount(1_000_000.0, taker_fee=0.0, maker_fee=0.0)
    return PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
        fill_rng=fill_rng,
        fill_seed=fill_seed,
    )


def _pending(side1: str, side2: str, lim1: float, lim2: float, ts: pd.Timestamp) -> tuple[PendingOrder, PendingOrder]:
    ps1 = "long" if side1 == "buy" else "short"
    ps2 = "long" if side2 == "buy" else "short"
    return (
        PendingOrder("AAA", "AAA/BBB:AAA", side1, ps1, 1.0, lim1, False, ts),
        PendingOrder("BBB", "AAA/BBB:BBB", side2, ps2, 1.0, lim2, False, ts),
    )


class _HighDraw:
    """Duck-typed rng: завжди random() > будь-яке p < 1."""

    def random(self) -> float:
        return 0.99


def test_library_decide_fill_without_mid_still_p1_on_touch() -> None:
    """Регресія контракту бібліотеки: без mid P=1 після touch."""
    hit = decide_fill("buy", 100.0, high=101.0, low=99.5)
    miss = decide_fill("buy", 100.0, high=102.0, low=100.5)
    assert hit.filled and hit.reason == "filled"
    assert not miss.filled and miss.reason == "unfilled_no_touch"


def test_paper_resolve_uses_mid_probability() -> None:
    limit = 100.0
    high, low = 110.0, 99.0
    mid = 0.5 * (high + low)
    p = fill_probability("buy", limit, mid)
    assert p < 0.2

    eng = _engine(fill_rng=_HighDraw())  # type: ignore[arg-type]
    ts = pd.Timestamp("2025-01-02")
    eng.pending = _pending("buy", "sell", limit, 60.0, ts)
    msg = eng._resolve_pending(ts, high, low, high2=61.0, low2=49.0)
    assert msg.startswith("unfilled")
    assert "unfilled_prob" in msg
    assert eng.n_filled == 0
    assert eng.pending is None


def test_pairs_engine_backtest_fill_decisions_match() -> None:
    common = _synthetic_common(n=220, seed=7)
    signals = _synthetic_signals(common.index, seed=11)
    position_pct = 0.1
    seed = 42
    bt = _maker_pair_positions(signals, common, position_pct, rng=np.random.default_rng(seed))

    eng = _engine(fill_rng=np.random.default_rng(seed))

    def _noop_fills(*args: object, **kwargs: object) -> None:
        return None

    setattr(eng, "_apply_fills", _noop_fills)

    target = (signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct).to_numpy()
    actual = np.zeros(len(common))
    curr = 0.0
    c1 = common["leg1"].to_numpy(dtype=float)
    c2 = common["leg2"].to_numpy(dtype=float)
    h1 = common["l1_high"].to_numpy(dtype=float)
    lo1 = common["l1_low"].to_numpy(dtype=float)
    h2 = common["l2_high"].to_numpy(dtype=float)
    lo2 = common["l2_low"].to_numpy(dtype=float)

    for i in range(1, len(common)):
        t = float(target[i])
        if t == curr:
            actual[i] = curr
            continue
        if t > 0:
            s1, s2 = "sell", "buy"
        elif t < 0:
            s1, s2 = "buy", "sell"
        else:
            s1, s2 = ("buy", "sell") if curr > 0 else ("sell", "buy")
        ts = pd.Timestamp(common.index[i])
        eng.pending = _pending(s1, s2, float(c1[i - 1]), float(c2[i - 1]), ts)
        msg = eng._resolve_pending(ts, float(h1[i]), float(lo1[i]), float(h2[i]), float(lo2[i]))
        if msg == "filled" or msg.startswith("filled:"):
            curr = t
        actual[i] = curr

    pd.testing.assert_series_equal(pd.Series(actual, index=common.index), bt)


def test_fill_rng_survives_snapshot() -> None:
    eng = _engine(fill_seed=42)
    ts = pd.Timestamp("2025-06-01")
    eng.pending = _pending("buy", "sell", 100.0, 50.0, ts)
    eng._resolve_pending(ts, 101.0, 99.0, 51.0, 49.0)
    snap = eng.to_snapshot()
    json.dumps(snap)
    next_a = float(eng._fill_rng.random())

    restored = _engine(fill_seed=99)
    restored.apply_snapshot(snap)
    next_b = float(restored._fill_rng.random())
    assert next_a == next_b
    assert restored._fill_seed == 42


def test_fill_rng_state_json_roundtrip() -> None:
    rng = maker_fill_rng(42)
    _ = float(rng.random())
    dumped = dump_fill_rng_state(rng)
    loaded = load_fill_rng_state(json.loads(json.dumps(dumped)))
    assert float(rng.random()) == float(loaded.random())
