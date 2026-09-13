"""Тести WeightBudget та інтеграції в ExchangeClient (без мережі).

Дослідження §7.2: Binance 6000 ваги/хв за IP; 429=backoff, 418=бан.
Перевіряємо парсинг заголовків, поріг throttle, рекомендований sleep,
та що ExchangeClient оновлює бюджет після ccxt-викликів.
"""

from __future__ import annotations

from typing import Any

import pytest
from scalper_hft.data.client import ExchangeClient
from scalper_hft.data.weight_budget import WeightBudget


class FakeExchange:
    """Мок ccxt.Exchange з керованими last_response_headers."""

    def __init__(self, headers: dict[str, Any] | None = None) -> None:
        self.last_response_headers: dict[str, Any] | None = headers
        self.calls: list[tuple[str, tuple, dict]] = []

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


def _client(exchange: FakeExchange) -> ExchangeClient:
    """ExchangeClient без __init__ (без реального ccxt/мережі)."""
    c = ExchangeClient.__new__(ExchangeClient)
    c.exchange = exchange  # type: ignore[assignment]
    c.exchange_id = "fake"
    c.market_type = "future"
    c._market_cache = {}
    c.weight_budget = WeightBudget()
    return c


# ── WeightBudget: парсинг заголовків ──────────────────────────────────────────


class TestUpdateFromHeaders:
    def test_parses_x_mbx_used_weight_1m(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "1234"})
        assert b.used_weight == 1234
        assert b.has_real_data is True
        assert b.last_updated_ms > 0

    def test_parses_lowercase_header(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"x-mbx-used-weight-1m": "500"})
        assert b.used_weight == 500

    def test_parses_legacy_header_without_1m(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT": "42"})
        assert b.used_weight == 42

    def test_missing_headers_noop(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"Content-Type": "application/json"})
        assert b.used_weight == 0
        assert b.has_real_data is False

    def test_none_headers_noop(self) -> None:
        b = WeightBudget()
        b.update_from_headers(None)
        assert b.used_weight == 0

    def test_invalid_value_noop(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "not-a-number"})
        assert b.used_weight == 0
        assert b.has_real_data is False

    def test_negative_value_ignored(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "-5"})
        assert b.used_weight == 0

    def test_overwrites_previous_value(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "100"})
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "3000"})
        assert b.used_weight == 3000


# ── WeightBudget: пороги та стани ─────────────────────────────────────────────


class TestThresholds:
    def test_defaults_binance_6000(self) -> None:
        b = WeightBudget()
        assert b.weight_limit == 6000
        assert b.throttle_threshold == 5520  # 92% of 6000

    def test_below_threshold_not_critical(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5000"})
        assert b.is_critical is False
        assert b.is_banned is False
        assert b.needs_throttle() is False

    def test_at_threshold_critical(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5520"})
        assert b.is_critical is True
        assert b.is_banned is False

    def test_at_limit_banned(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "6000"})
        assert b.is_banned is True

    def test_over_limit_banned(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "6500"})
        assert b.is_banned is True
        assert b.usage_pct == pytest.approx(6500 / 6000)

    def test_needs_throttle_requires_real_data(self) -> None:
        """Без валідних заголовків — не спимо «наосліп»."""
        b = WeightBudget()
        b.used_weight = 5900  # високий, але заголовків не бачили
        assert b.has_real_data is False
        assert b.needs_throttle() is False

    def test_remaining_clamped_at_zero(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "7000"})
        assert b.remaining == 0


# ── WeightBudget: рекомендований sleep ────────────────────────────────────────


class TestRecommendedSleep:
    def test_zero_when_not_needed(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "1000"})
        assert b.recommended_sleep_s() == 0.0

    def test_min_sleep_at_threshold(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5520"})
        s = b.recommended_sleep_s()
        assert 0.0 < s <= 0.3

    def test_max_sleep_near_limit(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5999"})
        s = b.recommended_sleep_s()
        assert 4.5 < s <= 5.0

    def test_monotone_increase(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5600"})
        low = b.recommended_sleep_s()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5900"})
        high = b.recommended_sleep_s()
        assert high > low


# ── WeightBudget: reset / mark / snapshot ────────────────────────────────────


class TestBookkeeping:
    def test_reset_clears_state(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5000"})
        b.mark_throttle()
        b.reset()
        assert b.used_weight == 0
        assert b.throttle_count == 0
        assert b.last_updated_ms == 0.0

    def test_mark_throttle_increments(self) -> None:
        b = WeightBudget()
        b.mark_throttle()
        b.mark_throttle()
        assert b.throttle_count == 2

    def test_snapshot_has_all_fields(self) -> None:
        b = WeightBudget()
        b.update_from_headers({"X-MBX-USED-WEIGHT-1M": "3000"})
        snap = b.snapshot()
        assert snap["used_weight"] == 3000
        assert snap["weight_limit"] == 6000
        assert snap["remaining"] == 3000
        assert snap["is_critical"] is False
        assert snap["has_real_data"] is True


# ── Інтеграція в ExchangeClient ──────────────────────────────────────────────


class TestClientIntegration:
    def test_fetch_klines_tracks_weight(self) -> None:
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "1500"})
        c = _client(ex)
        c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)
        assert c.weight_budget.used_weight == 1500
        assert c.weight_budget.has_real_data is True

    def test_fetch_agg_trades_tracks_weight(self) -> None:
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "800"})
        c = _client(ex)
        c.fetch_agg_trades("BTCUSDT", since_ms=0)
        assert c.weight_budget.used_weight == 800

    def test_fetch_funding_tracks_weight(self) -> None:
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "200"})
        c = _client(ex)
        c.fetch_funding_rate_history("BTCUSDT", since_ms=0)
        assert c.weight_budget.used_weight == 200

    def test_no_headers_no_weight_update(self) -> None:
        ex = FakeExchange(headers=None)
        c = _client(ex)
        c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)
        assert c.weight_budget.used_weight == 0
        assert c.weight_budget.has_real_data is False

    def test_throttle_if_needed_no_sleep_when_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = {"s": 0.0}
        monkeypatch.setattr("time.sleep", lambda s: slept.update(s=s))
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "1000"})
        c = _client(ex)
        c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)  # оновити вагу
        delay = c.throttle_if_needed()
        assert delay == 0.0
        assert slept["s"] == 0.0

    def test_throttle_if_needed_sleeps_when_critical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = {"calls": []}
        monkeypatch.setattr("time.sleep", lambda s: slept["calls"].append(s))
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "5800"})
        c = _client(ex)
        c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)  # оновити вагу
        delay = c.throttle_if_needed()
        assert delay > 0.0
        assert len(slept["calls"]) == 1
        assert slept["calls"][0] == delay
        assert c.weight_budget.throttle_count == 1

    def test_throttle_if_needed_dry_mode_no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = {"calls": []}
        monkeypatch.setattr("time.sleep", lambda s: slept["calls"].append(s))
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "5900"})
        c = _client(ex)
        c.fetch_klines("BTCUSDT", "1m", since_ms=123, limit=10)
        delay = c.throttle_if_needed(sleep=False)
        assert delay > 0.0
        assert slept["calls"] == []  # не спали
        assert c.weight_budget.throttle_count == 0  # не позначили

    def test_create_order_throttles_before_submit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = {"calls": []}
        monkeypatch.setattr("time.sleep", lambda s: slept["calls"].append(s))
        # Вага вже висока перед create_order.
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "5700"})
        c = _client(ex)
        # Симулюємо, що вже бачили заголовки (через попередній fetch).
        c.weight_budget.update_from_headers({"X-MBX-USED-WEIGHT-1M": "5700"})
        c.create_order("BTCUSDT", "limit", "buy", 0.1, 100.0, post_only=True)
        # throttle_if_needed мав спати перед submit.
        assert len(slept["calls"]) == 1
        assert slept["calls"][0] > 0.0

    def test_create_order_no_throttle_when_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = {"calls": []}
        monkeypatch.setattr("time.sleep", lambda s: slept["calls"].append(s))
        ex = FakeExchange(headers={"X-MBX-USED-WEIGHT-1M": "200"})
        c = _client(ex)
        c.weight_budget.update_from_headers({"X-MBX-USED-WEIGHT-1M": "200"})
        c.create_order("BTCUSDT", "limit", "buy", 0.1, 100.0, post_only=True)
        # throttle не спав (вага низька); ccxt RateLimitExceeded backoff теж не спрацював.
        assert slept["calls"] == []

    def test_weight_budget_initialized_in_real_init(self) -> None:
        """ExchangeClient.__init__ створює WeightBudget (без мережі — binanceusdm)."""
        c = ExchangeClient(exchange_id="binanceusdm")
        assert c.weight_budget is not None
        assert c.weight_budget.weight_limit == 6000
        assert c.weight_budget.used_weight == 0
