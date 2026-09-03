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
