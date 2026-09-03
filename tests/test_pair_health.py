"""Тести rolling ADF / half-life kill для paper pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.live.pair_health import pair_entry_allowed


def test_ou_spread_allows_entry() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.05)
    s = pd.Series(x)
    ok, reason = pair_entry_allowed(s, min_obs=80)
    assert ok is True
    assert reason == "ok"


def test_random_walk_blocks_entry() -> None:
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(0, 1.0, 250)))
    ok, reason = pair_entry_allowed(s, min_obs=80)
    assert ok is False
    assert "kill" in reason
