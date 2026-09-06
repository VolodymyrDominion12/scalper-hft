"""URL bookTicker / depth5 після міграції на /public/ws."""

from scalper_hft.live.ws_urls import bookticker_url, depth5_url


def test_bookticker_url_public_not_legacy_path() -> None:
    url = bookticker_url("BTCUSDT")
    assert url == "wss://fstream.binance.com/public/ws/btcusdt@bookTicker"


def test_depth5_url_public() -> None:
    url = depth5_url("ETHUSDT")
    assert url == "wss://fstream.binance.com/public/ws/ethusdt@depth5@100ms"


def test_bookticker_testnet_host() -> None:
    url = bookticker_url("BTCUSDT", testnet=True)
    assert url == "wss://stream.binancefuture.com/public/ws/btcusdt@bookTicker"


def test_mask_listen_key_hides_secret() -> None:
    from scalper_hft.live.ws_urls import mask_listen_key, private_user_stream_url

    url = private_user_stream_url("supersecretkey123", testnet=False)
    masked = mask_listen_key(url)
    assert "supersecretkey123" not in masked
    assert "listenKey=***" in masked
    assert "events=" in masked  # решта query залишається


def test_mask_listen_key_noop_without_key() -> None:
    from scalper_hft.live.ws_urls import mask_listen_key

    assert mask_listen_key("wss://fstream.binance.com/public/ws/btcusdt@bookTicker") == (
        "wss://fstream.binance.com/public/ws/btcusdt@bookTicker"
    )
