"""Тести для Binance WebSocket User Data Stream парсера та клієнта."""

import json

from scalper_hft.live.ws_user_stream import (
    BinanceUserDataStream,
    ExponentialBackoff,
    OrderTradeEvent,
    parse_order_trade_update,
)


def test_exponential_backoff_progression():
    eb = ExponentialBackoff(base_delay=1.0, max_delay=10.0, factor=2.0, jitter_ratio=0.1)
    assert eb.attempts == 0
    # First attempt: nominal 1.0, jitter [0.9, 1.1]
    d1 = eb.next_delay()
    assert 0.9 <= d1 <= 1.1
    assert eb.attempts == 1

    # Second attempt: nominal 2.0, jitter [1.8, 2.2]
    d2 = eb.next_delay()
    assert 1.8 <= d2 <= 2.2
    assert eb.attempts == 2

    # Third attempt: nominal 4.0, jitter [3.6, 4.4]
    d3 = eb.next_delay()
    assert 3.6 <= d3 <= 4.4

    # Fourth attempt: nominal 8.0, jitter [7.2, 8.8]
    d4 = eb.next_delay()
    assert 7.2 <= d4 <= 8.8

    # Fifth attempt: nominal capped at max_delay=10.0, jitter [9.0, 11.0]
    d5 = eb.next_delay()
    assert 9.0 <= d5 <= 11.0
    assert eb.attempts == 5


def test_exponential_backoff_reset():
    eb = ExponentialBackoff(base_delay=2.0, max_delay=30.0, factor=2.0, jitter_ratio=0.0)
    eb.next_delay()  # 2.0 -> next nominal 4.0
    eb.next_delay()  # 4.0 -> next nominal 8.0
    assert eb.attempts == 2
    assert eb.current_delay == 8.0

    eb.reset()
    assert eb.attempts == 0
    assert eb.current_delay == 2.0
    assert eb.next_delay() == 2.0


def test_user_data_stream_custom_backoff():
    custom_backoff = ExponentialBackoff(base_delay=0.5, max_delay=5.0)
    stream = BinanceUserDataStream(listen_key="test_key", backoff=custom_backoff)
    assert stream.backoff is custom_backoff


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

    stream = BinanceUserDataStream(listen_key="test_key_123", testnet=True, on_order_update=on_update)
    assert "listenKey=test_key_123" in stream.ws_url
    assert "/private/ws" in stream.ws_url
    assert "/ws/test_key_123" not in stream.ws_url
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


def test_prod_ws_url_uses_private_query_listen_key() -> None:
    stream = BinanceUserDataStream(listen_key="prod_key")
    assert "/private/ws" in stream.ws_url
    assert "listenKey=prod_key" in stream.ws_url
    assert "fstream.binance.com" in stream.ws_url
    assert "/ws/prod_key" not in stream.ws_url
    assert "events=" in stream.ws_url


def test_maybe_keepalive_fires_after_interval() -> None:
    calls: list[str] = []
    stream = BinanceUserDataStream(
        listen_key="k1",
        keepalive=calls.append,
        keepalive_interval_sec=30.0,
    )
    assert stream.maybe_keepalive(0.0) is False
    assert calls == []
    assert stream.maybe_keepalive(10.0) is False
    assert stream.maybe_keepalive(30.0) is True
    assert calls == ["k1"]
    assert stream.maybe_keepalive(40.0) is False
    assert stream.maybe_keepalive(60.0) is True
    assert calls == ["k1", "k1"]


def test_listen_key_expired_regenerates() -> None:
    import asyncio

    stream = BinanceUserDataStream(
        listen_key="old",
        refresh_listen_key=lambda: "new_key",
    )
    event = stream.handle_raw_message({"e": "listenKeyExpired", "E": 1})
    assert event is None
    assert stream._listen_key_expired is True
    asyncio.run(stream.await_refresh_listen_key())
    assert stream.listen_key == "new_key"


def test_user_data_stream_handles_bytes_and_non_dict() -> None:
    stream = BinanceUserDataStream(listen_key="test_key")
    # Bytes payload
    payload_bytes = b'{"e": "listenKeyExpired", "E": 2}'
    ev = stream.handle_raw_message(payload_bytes)
    assert ev is None

    # Non-dict JSON payload (e.g. primitive string or array)
    ev_str = stream.handle_raw_message('"pong"')
    assert ev_str is None

    ev_bytes_num = stream.handle_raw_message(b"123")
    assert ev_bytes_num is None
