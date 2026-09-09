"""Phase 5.2: partial-fill політика (cancel|wait) + барові ключі IntentStore."""

from __future__ import annotations

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pending_orders import PendingOrder, PendingOrderManager
from scalper_hft.live.trader import _intent_key


class _FakeClient:
    """Мінімальний REST-клієнт: fetch_order керується тестом, cancel записується."""

    def __init__(self) -> None:
        self.status: dict[str, dict] = {}
        self.cancelled: list[str] = []

    def fetch_order(self, order_id, symbol):
        return dict(self.status.get(order_id, {"status": "open", "filled": 0.0}))

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        if order_id in self.status:
            self.status[order_id]["status"] = "canceled"
        return {}


def _manager(policy: str = "cancel") -> tuple[PendingOrderManager, _FakeClient]:
    client = _FakeClient()
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    mgr = PendingOrderManager(
        "BTCUSDT",
        acc,
        client,
        dry_run=False,
        interval="1h",
        maker_fill_wait_bars=1,
        partial_fill_policy=policy,
    )
    return mgr, client


def _register_partial(mgr: PendingOrderManager, placed_ts: pd.Timestamp | None = None) -> PendingOrder:
    po = PendingOrder(
        client_order_id="c1",
        order_id="ex-c1",
        symbol="BTCUSDT",
        side="buy",
        size=1.0,
        price=100.0,
        reduce_only=False,
        kind="open",
        pos_side="long",
        placed_ts=placed_ts or pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    mgr.register("c1", po)
    mgr.client.status["ex-c1"] = {"status": "open", "filled": 0.4, "average": 100.0}
    return po


def test_invalid_policy_raises() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    with pytest.raises(ValueError, match="partial_fill_policy"):
        PendingOrderManager("BTCUSDT", acc, _FakeClient(), dry_run=False, interval="1h", partial_fill_policy="requote")


def test_cancel_policy_books_and_cancels_remainder() -> None:
    """Дефолт: частковий філ → book дельти, cancel залишку, ордер знято."""
    mgr, client = _manager("cancel")
    po = _register_partial(mgr)
    events = mgr.poll()
    assert events == ["partial:c1"]
    assert client.cancelled == ["ex-c1"]
    assert "c1" not in mgr.pending_orders
    assert po.booked_qty == pytest.approx(0.4)
    # дельта забукана у позицію
    assert mgr.account.positions["BTCUSDT"].size == pytest.approx(0.4)


def test_wait_policy_keeps_order_resting() -> None:
    """wait: частковий філ → book дельти, ордер лишається resting без cancel."""
    mgr, client = _manager("wait")
    po = _register_partial(mgr)
    events = mgr.poll()
    assert events == ["partial_wait:c1"]
    assert client.cancelled == []
    assert "c1" in mgr.pending_orders
    assert po.booked_qty == pytest.approx(0.4)
    assert po.remaining_qty == pytest.approx(0.6)


def test_wait_policy_books_full_fill_later() -> None:
    """wait: повторний poll з повним філом → book решти, ордер закрито."""
    mgr, client = _manager("wait")
    _register_partial(mgr)
    mgr.poll()
    client.status["ex-c1"] = {"status": "closed", "filled": 1.0, "average": 100.0}
    events = mgr.poll()
    assert events == ["filled:c1"]
    assert "c1" not in mgr.pending_orders
    assert mgr.account.positions["BTCUSDT"].size == pytest.approx(1.0)


def test_wait_policy_timeout_cancels_remainder() -> None:
    """wait: ордер старший за maker_fill_wait_bars барів → cancel залишку."""
    mgr, client = _manager("wait")
    old_ts = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=3)
    _register_partial(mgr, placed_ts=old_ts)
    events = mgr.poll()
    assert events == ["timeout_cancel:c1"]
    assert client.cancelled == ["ex-c1"]
    assert "c1" not in mgr.pending_orders
    # частковий філ все одно забуканий
    assert mgr.account.positions["BTCUSDT"].size == pytest.approx(0.4)


# ── Барові ключі IntentStore ─────────────────────────────────────────────────


def test_intent_key_stable_within_bar() -> None:
    t1 = pd.Timestamp("2026-09-09 12:00:10", tz="UTC")
    t2 = pd.Timestamp("2026-09-09 12:00:55", tz="UTC")
    k1 = _intent_key("BTCUSDT", "buy", "open", False, "1m", now=t1)
    k2 = _intent_key("BTCUSDT", "buy", "open", False, "1m", now=t2)
    assert k1 == k2  # retry у межах бару — той самий coid


def test_intent_key_rolls_with_bar() -> None:
    t1 = pd.Timestamp("2026-09-09 12:00:55", tz="UTC")
    t2 = pd.Timestamp("2026-09-09 12:01:05", tz="UTC")
    k1 = _intent_key("BTCUSDT", "buy", "open", False, "1m", now=t1)
    k2 = _intent_key("BTCUSDT", "buy", "open", False, "1m", now=t2)
    assert k1 != k2  # новий бар — новий намір, stale coid не успадковується


def test_intent_key_distinguishes_side_kind_reduce_only() -> None:
    t = pd.Timestamp("2026-09-09 12:00:10", tz="UTC")
    base = _intent_key("BTCUSDT", "buy", "open", False, "1m", now=t)
    assert base != _intent_key("BTCUSDT", "sell", "open", False, "1m", now=t)
    assert base != _intent_key("BTCUSDT", "buy", "close", False, "1m", now=t)
    assert base != _intent_key("BTCUSDT", "buy", "open", True, "1m", now=t)
