"""USDT-M WebSocket URL після cut-off Binance 2026-04-23.

Legacy ``wss://fstream.binance.com/ws/{stream}`` вимкнено. Маршрути:

- ``/public/ws`` — bookTicker, depth (висока частота);
- ``/private/ws?listenKey=`` — user data (listenKey як query, не path).
"""

from __future__ import annotations

from urllib.parse import urlencode

PROD_FSTREAM = "wss://fstream.binance.com"
TESTNET_FSTREAM = "wss://stream.binancefuture.com"

USER_EVENTS = "ORDER_TRADE_UPDATE/ACCOUNT_UPDATE/listenKeyExpired"
KEEPALIVE_INTERVAL_SEC = 30 * 60


def _host(testnet: bool) -> str:
    return TESTNET_FSTREAM if testnet else PROD_FSTREAM


def public_stream_url(stream: str, *, testnet: bool = False) -> str:
    """``/public/ws/{stream}`` — stream без провідного слеша (напр. btcusdt@bookTicker)."""
    name = stream.lstrip("/")
    return f"{_host(testnet)}/public/ws/{name}"


def bookticker_url(symbol: str, *, testnet: bool = False) -> str:
    return public_stream_url(f"{symbol.lower()}@bookTicker", testnet=testnet)


def depth5_url(symbol: str, *, testnet: bool = False) -> str:
    return public_stream_url(f"{symbol.lower()}@depth5@100ms", testnet=testnet)


def force_order_url(symbol: str, *, testnet: bool = False) -> str:
    """Binance liquidation stream (forceOrder)."""
    return public_stream_url(f"{symbol.lower()}@forceOrder", testnet=testnet)


def private_user_stream_url(
    listen_key: str,
    *,
    testnet: bool = False,
    events: str = USER_EVENTS,
) -> str:
    """User data: listenKey лише як query-параметр, не сегмент шляху."""
    query = urlencode({"listenKey": listen_key, "events": events})
    return f"{_host(testnet)}/private/ws?{query}"


def mask_listen_key(url: str) -> str:
    """URL для логів: listenKey замінюється на *** (session hijack через логи)."""
    if "listenKey=" not in url:
        return url
    head, _, tail = url.partition("listenKey=")
    _, sep, rest = tail.partition("&")
    masked = head + "listenKey=***"
    if sep:
        masked += sep + rest
    return masked
