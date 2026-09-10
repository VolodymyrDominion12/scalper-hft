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


class TestRateLimitRetry:
    def test_retries_on_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RateLimitExceeded → backoff і повтор; успіх на 3-й спробі."""
        import ccxt

        monkeypatch.setattr("time.sleep", lambda s: None)  # без реального сну
        ex = FakeExchange()
        fails = {"n": 0}

        def _flaky(*a, **kw) -> dict[str, Any]:
            fails["n"] += 1
            if fails["n"] < 3:
                raise ccxt.RateLimitExceeded("429")
            return {"id": "ok", "status": "open"}

        ex.create_order = _flaky  # type: ignore[method-assign]
        c = _client(ex)
        out = c.create_order("BTCUSDT", "limit", "buy", 0.1, 100.0, post_only=True)
        assert out["id"] == "ok"
        assert fails["n"] == 3

    def test_raises_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ccxt

        monkeypatch.setattr("time.sleep", lambda s: None)
        ex = FakeExchange()

        def _always_fail(*a, **kw) -> dict[str, Any]:
            raise ccxt.RateLimitExceeded("418 ban")

        ex.create_order = _always_fail  # type: ignore[method-assign]
        c = _client(ex)
        with pytest.raises(ccxt.RateLimitExceeded):
            c.create_order("BTCUSDT", "market", "sell", 0.1)

    def test_non_rate_limit_error_not_retried(self) -> None:
        """Інші помилки (відхилення ордера) НЕ ретраються — не маскуємо."""
        import ccxt

        ex = FakeExchange()
        calls = {"n": 0}

        def _bad(*a, **kw) -> dict[str, Any]:
            calls["n"] += 1
            raise ccxt.InsufficientFunds("margin")

        ex.create_order = _bad  # type: ignore[method-assign]
        c = _client(ex)
        with pytest.raises(ccxt.InsufficientFunds):
            c.create_order("BTCUSDT", "market", "buy", 0.1)
        assert calls["n"] == 1


class TestClockSync:
    def test_offset_computed(self) -> None:
        import time as _time

        ex = FakeExchange()
        ex.fetch_time = lambda: int(_time.time() * 1000) + 200  # type: ignore[attr-defined]
        c = _client(ex)
        offset = c.server_time_offset_ms()
        assert offset is not None and 100 < offset < 500

    def test_assert_passes_on_small_drift(self) -> None:
        import time as _time

        ex = FakeExchange()
        ex.fetch_time = lambda: int(_time.time() * 1000)  # type: ignore[attr-defined]
        c = _client(ex)
        assert abs(c.assert_clock_synced(max_drift_ms=1000.0)) < 1000.0

    def test_assert_raises_on_big_drift(self) -> None:
        import time as _time

        ex = FakeExchange()
        ex.fetch_time = lambda: int(_time.time() * 1000) + 60_000  # type: ignore[attr-defined]
        c = _client(ex)
        with pytest.raises(RuntimeError, match="Розсинхрон"):
            c.assert_clock_synced(max_drift_ms=1000.0)

    def test_no_fetch_time_graceful(self) -> None:
        c = _client(FakeExchange())  # без fetch_time
        assert c.server_time_offset_ms() is None
        assert c.assert_clock_synced() == 0.0  # warn, не падає


class TestDescribeSource:
    def test_describe_source_live_binanceusdm(self) -> None:
        c = ExchangeClient(exchange_id="binanceusdm")
        assert not c.is_testnet
        assert "fapi.binance.com" in c.api_url
        desc = c.describe_source()
        assert "binanceusdm" in desc
        assert "LIVE (НЕ testnet ✓)" in desc
        assert "https://fapi.binance.com" in desc

    def test_describe_source_live_spot(self) -> None:
        c = ExchangeClient(exchange_id="binance", market_type="spot")
        assert not c.is_testnet
        assert "api.binance.com" in c.api_url
        desc = c.describe_source()
        assert "binance" in desc
        assert "LIVE (НЕ testnet ✓)" in desc

    def test_describe_source_testnet(self) -> None:
        c = ExchangeClient(exchange_id="binanceusdm-testnet")
        assert c.is_testnet
        desc = c.describe_source()
        assert "TESTNET/SANDBOX" in desc

    def test_describe_source_fake_exchange_without_urls(self) -> None:
        c = _client(FakeExchange())
        assert c.api_url == ""
        assert not c.is_testnet
        desc = c.describe_source()
        assert "fake" in desc
        assert "LIVE (НЕ testnet ✓)" in desc

