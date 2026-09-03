"""Тести для Binance WebSocket User Data Stream парсера та клієнта."""

import json
import pytest

from scalper_hft.live.ws_user_stream import (
    BinanceUserDataStream,
    OrderTradeEvent,
    parse_order_trade_update,
)


def test_parse_order_trade_update():
    payload = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1568879465651,
        "T": 1568879465650,
        "o": {
            "s": "BTCUSDT",
            "c": "client_order_123",
            "S": "BUY",
            "o": "LIMIT",
            "f": "GTC",
            "q": "0.010",
            "p": "65000.0",
            "ap": "65000.0",
            "sp": "0",
            "x": "TRADE",
            "X": "FILLED",
            "i": 99887766,
            "l": "0.010",
            "z": "0.010",
            "L": "65000.0",
            "n": "0.13",
            "N": "USDT",
            "T": 1568879465651,
            "t": 12345,
            "b": "0",
            "a": "0",
            "m": True,
            "R": False,
            "wt": "CONTRACT_PRICE",
            "ot": "LIMIT",
            "ps": "BOTH",
        },
    }

    event = parse_order_trade_update(payload)
    assert event is not None
    assert event.symbol == "BTCUSDT"
    assert event.client_order_id == "client_order_123"
    assert event.side == "BUY"
    assert event.status == "FILLED"
    assert event.last_filled_qty == 0.010
    assert event.last_filled_price == 65000.0
    assert event.commission == 0.13
    assert event.is_maker is True


def test_user_data_stream_callback():
    received = []

    def on_update(e: OrderTradeEvent):
        received.append(e)

    stream = BinanceUserDataStream(
        listen_key="test_key_123", testnet=True, on_order_update=on_update
    )
    assert "test_key_123" in stream.ws_url
    assert "stream.binancefuture.com" in stream.ws_url

    msg = json.dumps(
        {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1000,
            "o": {
                "s": "ETHUSDT",
                "c": "order_abc",
                "S": "SELL",
                "ot": "LIMIT",
                "X": "PARTIALLY_FILLED",
                "x": "TRADE",
                "l": "0.5",
                "z": "0.5",
                "L": "3500.0",
                "n": "0.05",
                "N": "USDT",
                "m": True,
                "i": 112233,
            },
        }
    )

    event = stream.handle_raw_message(msg)
    assert event is not None
    assert len(received) == 1
    assert received[0].symbol == "ETHUSDT"
    assert received[0].status == "PARTIALLY_FILLED"
