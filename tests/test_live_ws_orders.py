"""Тести для WebSocket User Data Stream інтеграції в LiveTrader та PairsRunner."""

from __future__ import annotations

import dataclasses

import pandas as pd
from scalper_hft.config import get_settings, set_settings
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_runner import PairsEngine, PairsPortfolioRunner
from scalper_hft.live.pairs_runner import PendingOrder as PairsPendingOrder
from scalper_hft.live.trader import LiveTrader, PendingOrder
from scalper_hft.live.ws_user_stream import OrderTradeEvent
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.pairs_arb import PairsArb


class _DummyStrategy(Strategy):
    name = "dummy"
    param_space: dict = {}
    needs_trades = False
    needs_funding = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        return pd.Series(0, index=df.index)


class _MockExchangeClient:
    def __init__(self) -> None:
        self.created_key = "mock_listen_key_123"
        self.closed_key: str | None = None
        self.keepalive_key: str | None = None

    def create_listen_key(self) -> str:
        return self.created_key

    def keepalive_listen_key(self, key: str) -> None:
        self.keepalive_key = key

    def close_listen_key(self, key: str) -> None:
        self.closed_key = key

    def cancel_all_orders(self, symbol: str | None = None) -> list:
        return []


def test_trader_on_ws_order_filled() -> None:
    account = PaperAccount(initial_capital=10_000.0)
    trader = LiveTrader(strategy=_DummyStrategy(), symbol="BTCUSDT", account=account)

    po = PendingOrder(
        client_order_id="c1",
        order_id="101",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        reduce_only=False,
        kind="open",
        pos_side="long",
        placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    trader.pending_orders["c1"] = po

    event = OrderTradeEvent(
        event_time=12345,
        symbol="BTCUSDT",
        client_order_id="c1",
        order_id=101,
        side="BUY",
        order_type="LIMIT",
        status="FILLED",
        execution_type="TRADE",
        last_filled_qty=0.1,
        cumulative_filled_qty=0.1,
        last_filled_price=50010.0,
        commission=0.01,
        commission_asset="USDT",
        is_maker=True,
    )

    trader.on_ws_order_trade(event)

    assert "c1" not in trader.pending_orders
    assert "BTCUSDT" in trader.account.positions
    pos = trader.account.positions["BTCUSDT"]
    assert pos.size == 0.1
    assert pos.entry_price == 50010.0
    assert pos.side == "long"


def test_trader_on_ws_order_partial_fill() -> None:
    account = PaperAccount(initial_capital=10_000.0)
    trader = LiveTrader(strategy=_DummyStrategy(), symbol="BTCUSDT", account=account)

    po = PendingOrder(
        client_order_id="c2",
        order_id="102",
        symbol="BTCUSDT",
        side="buy",
        size=0.2,
        price=50000.0,
        reduce_only=False,
        kind="open",
        pos_side="long",
        placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    trader.pending_orders["c2"] = po

    event = OrderTradeEvent(
        event_time=12345,
        symbol="BTCUSDT",
        client_order_id="c2",
        order_id=102,
        side="BUY",
        order_type="LIMIT",
        status="PARTIALLY_FILLED",
        execution_type="TRADE",
        last_filled_qty=0.08,
        cumulative_filled_qty=0.08,
        last_filled_price=50000.0,
        commission=0.008,
        commission_asset="USDT",
        is_maker=True,
    )

    trader.on_ws_order_trade(event)

    assert "c2" in trader.pending_orders
    assert abs(trader.pending_orders["c2"].size - 0.12) < 1e-9
    assert "BTCUSDT" in trader.account.positions
    assert abs(trader.account.positions["BTCUSDT"].size - 0.08) < 1e-9


def test_trader_on_ws_order_canceled() -> None:
    account = PaperAccount(initial_capital=10_000.0)
    trader = LiveTrader(strategy=_DummyStrategy(), symbol="BTCUSDT", account=account)

    po = PendingOrder(
        client_order_id="c3",
        order_id="103",
        symbol="BTCUSDT",
        side="sell",
        size=0.1,
        price=51000.0,
        reduce_only=False,
        kind="open",
        pos_side="short",
        placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    trader.pending_orders["c3"] = po

    event = OrderTradeEvent(
        event_time=12345,
        symbol="BTCUSDT",
        client_order_id="c3",
        order_id=103,
        side="SELL",
        order_type="LIMIT",
        status="CANCELED",
        execution_type="CANCELED",
        last_filled_qty=0.0,
        cumulative_filled_qty=0.0,
        last_filled_price=0.0,
        commission=0.0,
        commission_asset="USDT",
        is_maker=True,
    )

    trader.on_ws_order_trade(event)

    assert "c3" not in trader.pending_orders
    assert "BTCUSDT" not in trader.account.positions


def test_trader_user_stream_lifecycle() -> None:
    orig = get_settings()
    mock_client = _MockExchangeClient()
    try:
        live_settings = dataclasses.replace(orig, dry_run=False, exchange="binance-testnet")
        set_settings(live_settings)

        account = PaperAccount(initial_capital=10_000.0)
        trader = LiveTrader(
            strategy=_DummyStrategy(),
            symbol="BTCUSDT",
            account=account,
            client=mock_client,  # type: ignore[arg-type]
        )

        stream = trader.start_user_stream()
        assert stream is not None
        assert stream.listen_key == "mock_listen_key_123"
        assert trader.ws_stream is stream

        trader.shutdown(reason="test_shutdown")
        assert trader.ws_stream is None
        assert mock_client.closed_key == "mock_listen_key_123"
    finally:
        set_settings(orig)


def test_pairs_engine_ws_fill_chase() -> None:
    account = PaperAccount(initial_capital=10_000.0)
    engine = PairsEngine("XRPUSDT", "BTCUSDT", PairsArb(), account, is_maker=True)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    o1 = PairsPendingOrder(
        symbol="XRPUSDT",
        key=engine._k("XRPUSDT"),
        side="buy",
        pos_side="long",
        size=100.0,
        limit_price=0.5,
        reduce_only=False,
        placed_ts=now,
    )
    o2 = PairsPendingOrder(
        symbol="BTCUSDT",
        key=engine._k("BTCUSDT"),
        side="sell",
        pos_side="short",
        size=0.001,
        limit_price=50000.0,
        reduce_only=False,
        placed_ts=now,
    )
    engine.pending = (o1, o2)
    engine.legging_mode = "chase_leg2"

    event = OrderTradeEvent(
        event_time=12345,
        symbol="XRPUSDT",
        client_order_id=o1.key,
        order_id=1,
        side="BUY",
        order_type="LIMIT",
        status="FILLED",
        execution_type="TRADE",
        last_filled_qty=100.0,
        cumulative_filled_qty=100.0,
        last_filled_price=0.501,
        commission=0.001,
        commission_asset="USDT",
        is_maker=True,
    )

    action = engine.on_ws_order_trade(event, now=now)
    assert action == "chase_leg2"
    assert engine.pending is None
    assert engine.n_filled == 1
    assert o1.key in account.positions
    assert o2.key in account.positions


def test_pairs_engine_ws_fill_strict_both() -> None:
    account = PaperAccount(initial_capital=10_000.0)
    engine = PairsEngine("XRPUSDT", "BTCUSDT", PairsArb(), account, is_maker=True)
    engine.legging_mode = "strict_both"

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    o1 = PairsPendingOrder(
        symbol="XRPUSDT",
        key=engine._k("XRPUSDT"),
        side="buy",
        pos_side="long",
        size=100.0,
        limit_price=0.5,
        reduce_only=False,
        placed_ts=now,
    )
    o2 = PairsPendingOrder(
        symbol="BTCUSDT",
        key=engine._k("BTCUSDT"),
        side="sell",
        pos_side="short",
        size=0.001,
        limit_price=50000.0,
        reduce_only=False,
        placed_ts=now,
    )
    engine.pending = (o1, o2)

    event1 = OrderTradeEvent(
        event_time=12345,
        symbol="XRPUSDT",
        client_order_id=o1.key,
        order_id=1,
        side="BUY",
        order_type="LIMIT",
        status="FILLED",
        execution_type="TRADE",
        last_filled_qty=100.0,
        cumulative_filled_qty=100.0,
        last_filled_price=0.5,
        commission=0.001,
        commission_asset="USDT",
        is_maker=True,
    )
    res1 = engine.on_ws_order_trade(event1, now=now)
    assert res1 == "waiting_other_leg"
    assert engine.pending is not None
    assert o1.key not in account.positions

    event2 = OrderTradeEvent(
        event_time=12346,
        symbol="BTCUSDT",
        client_order_id=o2.key,
        order_id=2,
        side="SELL",
        order_type="LIMIT",
        status="FILLED",
        execution_type="TRADE",
        last_filled_qty=0.001,
        cumulative_filled_qty=0.001,
        last_filled_price=50000.0,
        commission=0.005,
        commission_asset="USDT",
        is_maker=True,
    )
    res2 = engine.on_ws_order_trade(event2, now=now)
    assert res2 == "both_filled"
    assert engine.pending is None
    assert engine.n_filled == 1
    assert o1.key in account.positions
    assert o2.key in account.positions


def test_pairs_portfolio_ws_dispatch() -> None:
    configs = [{"leg1": "XRPUSDT", "leg2": "BTCUSDT", "entry_z": 2.0, "exit_z": 0.3, "lookback": 480}]
    portfolio = PairsPortfolioRunner(configs=configs)
    runner = portfolio.runners[0]
    engine = runner.engine

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    o1 = PairsPendingOrder(
        symbol="XRPUSDT",
        key=engine._k("XRPUSDT"),
        side="buy",
        pos_side="long",
        size=100.0,
        limit_price=0.5,
        reduce_only=False,
        placed_ts=now,
    )
    o2 = PairsPendingOrder(
        symbol="BTCUSDT",
        key=engine._k("BTCUSDT"),
        side="sell",
        pos_side="short",
        size=0.001,
        limit_price=50000.0,
        reduce_only=False,
        placed_ts=now,
    )
    engine.pending = (o1, o2)
    engine.legging_mode = "chase_leg2"

    event = OrderTradeEvent(
        event_time=12345,
        symbol="XRPUSDT",
        client_order_id=o1.key,
        order_id=1,
        side="BUY",
        order_type="LIMIT",
        status="FILLED",
        execution_type="TRADE",
        last_filled_qty=100.0,
        cumulative_filled_qty=100.0,
        last_filled_price=0.5,
        commission=0.001,
        commission_asset="USDT",
        is_maker=True,
    )

    dispatched = portfolio.on_ws_order_trade(event, now=now)
    assert dispatched == ["XRPUSDT/BTCUSDT:chase_leg2"]
    assert engine.pending is None
