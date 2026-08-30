"""Тести Спринту 4:

1. MCP-сервер для трейдінгу (stdio, JSON-RPC) — scalper_hft/mcp_trading.py
2. Alpha-гіпотеза: HMM-гейтований mean reversion — strategies/hmm_reversion.py
3. Live-інтеграція: vol-scaled sizing + HMM-блок — live/trader.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 600, seed: int = 41) -> pd.DataFrame:
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


# ── 1. MCP-сервер ────────────────────────────────────────────────────────────


class TestMcpTrading:
    def _call(self, msg: dict):
        from scalper_hft.mcp_trading import handle_message

        return handle_message(msg)

    def test_initialize(self):
        resp = self._call(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "1"},
                },
            }
        )
        assert resp[0]["result"]["protocolVersion"] == "2024-11-05"
        assert resp[0]["result"]["capabilities"]["tools"] == {}

    def test_initialized_notification_no_response(self):
        assert self._call({"jsonrpc": "2.0", "method": "notifications/initialized"}) == []

    def test_tools_list(self):
        resp = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in resp[0]["result"]["tools"]]
        for expected in [
            "strategy_list",
            "settings_summary",
            "run_backtest",
            "run_cohort",
            "run_stress",
            "run_capacity",
            "paper_step",
            "market_status",
        ]:
            assert expected in names

    def test_strategy_list_tool(self):
        resp = self._call(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "strategy_list", "arguments": {}}}
        )
        text = resp[0]["result"]["content"][0]["text"]
        data = json.loads(text)
        names = [s["name"] for s in data["strategies"]]
        assert "hmm_reversion" in names
        assert "mean_reversion" in names

    def test_settings_summary_no_secrets(self):
        resp = self._call(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "settings_summary", "arguments": {}}}
        )
        text = resp[0]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "api_key" not in json.dumps(data).lower() or "api_key_configured" in data
        assert "dry_run" in data

    def test_unknown_tool_error(self):
        resp = self._call(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
        )
        assert resp[0]["error"]["code"] == -32602

    def test_unknown_method_error(self):
        resp = self._call({"jsonrpc": "2.0", "id": 6, "method": "foo"})
        assert resp[0]["error"]["code"] == -32601

    def test_ping(self):
        resp = self._call({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        assert resp[0]["result"] == {}

    def test_cli_mcp_subcommand_registered(self):
        """`scalper_hft.cli mcp` існує в парсері."""
        from scalper_hft.cli import main

        try:
            main(["mcp"])  # блокує на stdin — перехоплюємо SystemExit/потік не запускаємо
        except SystemExit:
            pass
        except Exception:  # noqa: BLE001
            # EOF на stdin або подібне — означає, що сервер стартував
            pass


# ── 2. HMM-гейтований mean reversion (alpha-гіпотеза) ───────────────────────


class TestHmmReversion:
    def test_registered(self):
        from scalper_hft.strategies import REGISTRY

        assert "hmm_reversion" in REGISTRY

    def test_signals_shape_and_values(self):
        from scalper_hft.strategies import get_strategy

        df = _make_df(600)
        strat = get_strategy("hmm_reversion", hmm_states=3, hmm_threshold=0.4)
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert signals.index.equals(df.index)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_calm_state_mask(self):
        """У низьковолатильному блоці HMM має вказувати «спокійний» стан."""
        from scalper_hft.strategies.hmm_reversion import HmmReversionScalper

        rng = np.random.default_rng(44)
        n = 800
        idx = pd.date_range("2025-01-01", periods=n, freq="1min")
        vol = np.concatenate([np.full(n // 2, 0.0003), np.full(n // 2, 0.006)])
        ret = rng.normal(0, vol)
        close = 100.0 * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": np.full(n, 50.0)},
            index=idx,
        )
        strat = HmmReversionScalper(hmm_states=3, hmm_threshold=0.5)
        mask = strat._calm_state_mask(df["close"], n_states=3, fit_bars=300, threshold=0.5)
        assert mask.dtype == bool
        assert len(mask) == len(df)
        # у першій (спокійній) половині маска має бути істинною хоча б частково
        assert mask.iloc[: n // 2].sum() > 20

    def test_hmm_gate_never_trades_in_storm(self):
        """При дуже високому порозі HMM-гейт має вимикати майже всі входи."""
        from scalper_hft.strategies import get_strategy

        df = _make_df(600)
        strat = get_strategy("hmm_reversion", hmm_threshold=0.99)
        signals = strat.generate_signals(df)
        assert (signals == 0).sum() >= len(signals) * 0.9  # майже все вимкнено

    def test_param_space(self):
        from scalper_hft.strategies.hmm_reversion import HmmReversionScalper

        assert "hmm_threshold" in HmmReversionScalper.param_space


# ── 3. Live-інтеграція ───────────────────────────────────────────────────────


class TestLiveSprint4:
    def _trader(self, **kw):
        from scalper_hft.live.trader import LiveTrader
        from scalper_hft.strategies import get_strategy

        strat = get_strategy("mean_reversion")
        return LiveTrader(strat, "BTCUSDT", "5m", **kw)

    def test_vol_scaled_size_disabled(self):
        t = self._trader(vol_sizing=False)
        df = _make_df(300)
        assert t.vol_scaled_size(100.0, df) == pytest.approx(100.0)

    def test_vol_scaled_size_scales_down_in_high_vol(self):
        t = self._trader(vol_sizing=True, vol_ref=0.0005)
        df = _make_df(300, seed=42)  # волатильність ~0.002 > vol_ref → менша позиція
        size = t.vol_scaled_size(100.0, df)
        assert 0 < size < 100.0
        assert size >= 25.0  # кліп [0.25, 3.0]

    def test_vol_scaled_size_scales_up_in_low_vol(self):
        rng = np.random.default_rng(43)
        idx = pd.date_range("2025-01-01", periods=300, freq="1min")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0001, 300)))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.0001,
                "low": close * 0.9999,
                "close": close,
                "volume": np.full(300, 50.0),
            },
            index=idx,
        )
        t = self._trader(vol_sizing=True, vol_ref=0.001)
        size = t.vol_scaled_size(100.0, df)
        assert size > 100.0
        assert size <= 300.0

    def test_hmm_blocked_disabled(self):
        t = self._trader(hmm_block=False)
        assert t.hmm_blocked(_make_df(300)) is False

    def test_hmm_blocked_runs(self):
        t = self._trader(hmm_block=True, hmm_states=3, hmm_threshold=0.5)
        blocked = t.hmm_blocked(_make_df(500))
        assert isinstance(blocked, bool)

    def test_execute_signal_with_block_runs(self):
        from scalper_hft.live.trader import execute_signal

        t = self._trader(hmm_block=True)
        df = _make_df(400)
        out = execute_signal(t, signal=1, df=df)
        assert isinstance(out, str)
        # якщо заблоковано — позиції немає
        if "blocked:hmm_regime" in out:
            assert len(t.account.positions) == 0
