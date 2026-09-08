"""Тести ExchangeClient поверх мокового ccxt-об'єкта (без мережі)."""

from __future__ import annotations

from typing import Any

import pytest
from scalper_hft.data.client import ExchangeClient


class FakeExchange:
    """Мінімальний мок ccxt.Exchange для unit-тестів (без network)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    # precision helpers (ccxt-сумісні)
    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{amount:.3f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{price:.1f}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[Any]]:
        self.calls.append(("fetch_ohlcv", (symbol, timeframe), {"since": since, "limit": limit}))
        return [[since, 1.0, 2.0, 0.5, 1.5, 100.0]]

    def fetch_trades(self, symbol: str, since: int, limit: int) -> list[dict[str, Any]]:
        return [{"id": "1", "price": 1.0, "amount": 0.1, "side": "buy"}]

    def fetch_funding_rate_history(self, symbol: str, since: int, limit: int) -> list[dict[str, Any]]:
        return [{"timestamp": since, "fundingRate": 0.0001}]

    def create_order(self, symbol, order_type, side, amount, price, params) -> dict[str, Any]:
        self.calls.append(("create_order", (symbol, order_type, side, amount, price), dict(params)))
        return {"id": "oid-1", "status": "open"}

    def fetch_balance(self) -> dict[str, Any]:
        return {
            "info": {"totalWalletBalance": "1000.5", "totalUnrealizedProfit": "-50.25"},
            "total": {"USDT": 900.0},
        }


def _client(exchange: FakeExchange, market: dict[str, Any] | None = None) -> ExchangeClient:
    """ExchangeClient без __init__ (без реального ccxt/мережі)."""
    c = ExchangeClient.__new__(ExchangeClient)
    c.exchange = exchange  # type: ignore[assignment]
    c.exchange_id = "fake"
    c.market_type = "future"
    c._market_cache = {"BTCUSDT": market} if market else {}
    return c


_MARKET = {
    "precision": {"price": 0.1, "amount": 0.001},
    "limits": {"amount": {"min": 0.01, "max": 100.0}, "cost": {"min": 10.0}},
}


class TestSanitizeOrder:
    def test_ok_order_passes(self) -> None:
        c = _client(FakeExchange(), _MARKET)
        qty, px, err = c.sanitize_order("BTCUSDT", "buy", 0.1234, 50000.55)
        assert err is None
        assert qty == pytest.approx(0.123)  # округлено до amount precision
        assert px == pytest.approx(50000.6)  # tick 0.1

    def test_min_qty_rejected(self) -> None:
        c = _client(FakeExchange(), _MARKET)
        _, _, err = c.sanitize_order("BTCUSDT", "buy", 0.005, 50000.0)
        assert err is not None and "minQty" in err

    def test_max_qty_rejected(self) -> None:
        c = _client(FakeExchange(), _MARKET)
        _, _, err = c.sanitize_order("BTCUSDT", "buy", 150.0, 50000.0)
        assert err is not None and "maxQty" in err

    def test_min_notional_rejected(self) -> None:
        c = _client(FakeExchange(), _MARKET)
        _, _, err = c.sanitize_order("BTCUSDT", "buy", 0.02, 100.0)  # notional 2 < 10
        assert err is not None and "minNotional" in err

    def test_market_order_price_none(self) -> None:
        c = _client(FakeExchange(), _MARKET)
        qty, px, err = c.sanitize_order("BTCUSDT", "sell", 1.0, None)
        assert px is None and err is None and qty == pytest.approx(1.0)


class TestFetchPassthrough:
    def test_fetch_klines_delegates(self) -> None:
        ex = FakeExchange()
        c = _client(ex)
        out = c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)
        assert out == [[123, 1.0, 2.0, 0.5, 1.5, 100.0]]
        assert ex.calls[0][0] == "fetch_ohlcv"

    def test_fetch_agg_trades(self) -> None:
        c = _client(FakeExchange())
        out = c.fetch_agg_trades("BTCUSDT", since_ms=0)
        assert out[0]["side"] == "buy"

    def test_fetch_funding(self) -> None:
        c = _client(FakeExchange())
        out = c.fetch_funding_rate_history("BTCUSDT", since_ms=0)
        assert out[0]["fundingRate"] == pytest.approx(0.0001)


class TestCreateOrder:
    def test_post_only_flag(self) -> None:
        ex = FakeExchange()
        c = _client(ex)
        c.create_order("BTCUSDT", "limit", "buy", 0.1, 100.0, post_only=True, client_order_id="abc")
        _, _, params = ex.calls[0]
        assert params["postOnly"] is True
        assert params["newClientOrderId"] == "abc"

    def test_plain_market_order(self) -> None:
        ex = FakeExchange()
        c = _client(ex)
        c.create_order("BTCUSDT", "market", "sell", 0.1)
        _, _, params = ex.calls[0]
        assert "postOnly" not in params and "newClientOrderId" not in params


class TestEquity:
    def test_wallet_plus_unrealized(self) -> None:
        c = _client(FakeExchange())
        assert c.fetch_usdt_equity() == pytest.approx(1000.5 - 50.25)

    def test_fallback_to_total(self) -> None:
        ex = FakeExchange()

        def _bal() -> dict[str, Any]:
            return {"info": {}, "total": {"USDT": 777.0}}

        ex.fetch_balance = _bal  # type: ignore[method-assign]
        c = _client(ex)
        assert c.fetch_usdt_equity() == pytest.approx(777.0)
