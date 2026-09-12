"""Тести цілісності aggTrades: дедуп за `trade_id` і пагінація за `fromId`.

Контекст (аудит 2026-09-11, знахідка K5). Індекс aggTrades — `transact_time` у
мілісекундах, але унікальний ключ угоди — `agg_trade_id`. Дедуп за індексом
(`df.index.duplicated()`) знищував усі угоди, крім однієї, у кожній спільній
мілісекунді. На кеші BTCUSDT це дало 39.3% покриття `trade_id` (ETHUSDT 49.8%,
DOGEUSDT 95.2%); дефект був тихий, бо `validate_trades` перевіряє дублювання
індексу, а після дедупу його немає.

Другий бік тієї ж проблеми — курсор пагінації `since = last_ts + 1`: він
перестрибує цілу мілісекунду, тому угоди, що поділяють останню мс батча, не
довантажувались. Виправлено переходом на `fromId` (`fromId` ІНКЛЮЗИВНО).

Усі тести — офлайн, із фейковим клієнтом: жодного мережевого запиту.
"""

from __future__ import annotations

import pandas as pd
from scalper_hft.data.downloader import Downloader
from scalper_hft.data.storage import dedupe_trades


def _trades_frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """rows: [(trade_id, ts_ms), …] → фрейм у форматі кешу."""
    return pd.DataFrame(
        {
            "trade_id": [r[0] for r in rows],
            "price": [100.0] * len(rows),
            "amount": [1.0] * len(rows),
            "side": ["buy"] * len(rows),
        },
        index=pd.to_datetime([r[1] for r in rows], unit="ms"),
    )


# ── 1. dedupe_trades ────────────────────────────────────────────────────────


def test_dedupe_keeps_trades_sharing_a_millisecond():
    """Ключова регресія: кілька aggTrades в одній мс — це НЕ дублікати."""
    base = 1_700_000_000_000
    # 5 угод в одній мс, потім ще 5 у наступній.
    rows = [(10 + i, base + (i // 5)) for i in range(10)]
    df = _trades_frame(rows)
    assert df.index.duplicated().sum() > 0, "синтетика має містити спільні мілісекунди"

    out = dedupe_trades(df)

    assert len(out) == 10
    assert out["trade_id"].tolist() == list(range(10, 20))
    # Стара (багнута) логіка лишала б по одному рядку на мс.
    assert len(df[~df.index.duplicated(keep="last")]) == 2


def test_dedupe_removes_true_duplicates_by_trade_id():
    base = 1_700_000_000_000
    rows = [(10, base), (11, base), (12, base + 1), (10, base), (11, base)]  # 10 і 11 — двічі
    df = _trades_frame(rows)

    out = dedupe_trades(df)

    assert sorted(out["trade_id"].tolist()) == [10, 11, 12]
    assert len(out) == 3


def test_dedupe_fallback_for_synthetic_ids():
    """Синтетичні (від'ємні) id не унікальні між завантаженнями → фолбек за індексом."""
    base = 1_700_000_000_000
    df = _trades_frame([(-1, base), (-2, base), (-3, base + 1)])

    out = dedupe_trades(df)

    assert len(out) == 2  # у спільній мс лишається один рядок (інакше докачка плодила б дублі)
    assert out.index.is_monotonic_increasing


def test_dedupe_without_trade_id_column_warns_and_falls_back(caplog):
    base = 1_700_000_000_000
    df = _trades_frame([(1, base), (2, base), (3, base + 1)]).drop(columns=["trade_id"])

    with caplog.at_level("WARNING"):
        out = dedupe_trades(df)

    assert len(out) == 2
    assert any("trade_id" in rec.message for rec in caplog.records)


def test_dedupe_empty_and_none_are_passthrough():
    assert dedupe_trades(None) is None
    empty = pd.DataFrame(columns=["trade_id", "price", "amount", "side"])
    assert dedupe_trades(empty).empty


# ── 2. fetch_agg_trades(from_id=…) ─────────────────────────────────────────


def test_fetch_agg_trades_from_id_uses_fromId_without_start_time():
    """ccxt/Binance забороняють `fromId` разом із `startTime`/`endTime`."""
    from scalper_hft.data.client import ExchangeClient

    calls: list[tuple] = []

    class _Exchange:
        def fetch_trades(self, symbol, since=None, limit=None, params=None):
            calls.append((symbol, since, limit, params))
            return []

    client = ExchangeClient.__new__(ExchangeClient)
    client.exchange = _Exchange()

    client.fetch_agg_trades("BTC/USDT:USDT", None, limit=1000, from_id=555)
    assert calls[-1] == ("BTC/USDT:USDT", None, 1000, {"fromId": 555})

    client.fetch_agg_trades("BTC/USDT:USDT", 1_700_000_000_000, limit=1000)
    assert calls[-1] == ("BTC/USDT:USDT", 1_700_000_000_000, 1000, None)


# ── 3. Пагінація завантажувача ─────────────────────────────────────────────


class _FakeStore:
    def __init__(self) -> None:
        self.saved: list[pd.DataFrame] = []

    def load_trades(self, symbol: str):
        return None

    def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
        self.saved.append(df)


class _FakeClient:
    """Віддає задані aggTrades, поважаючи `fromId` або `startTime`."""

    def __init__(self, rows: list[tuple[int, int]], *, with_ids: bool = True) -> None:
        self.rows = rows
        self.with_ids = with_ids
        self.calls: list[tuple] = []

    def fetch_agg_trades(self, symbol, since_ms=None, limit=1000, *, from_id=None):
        self.calls.append((since_ms, from_id))
        if from_id is not None:
            sel = [r for r in self.rows if r[0] >= from_id]
        else:
            sel = [r for r in self.rows if r[1] >= (since_ms or 0)]
        out = []
        for tid, ts in sel[:limit]:
            item: dict = {"timestamp": ts, "price": 100.0, "amount": 1.0, "side": "buy"}
            if self.with_ids:
                item["id"] = tid
            out.append(item)
        return out


def _downloader(client) -> Downloader:
    return Downloader(client=client, store=_FakeStore(), retries=1, batch_delay=0, checkpoint_batches=0)


def _rows(n: int, *, per_ms: int = 5, base_ms: int | None = None) -> list[tuple[int, int]]:
    base_ms = base_ms if base_ms is not None else int(pd.Timestamp.now("UTC").timestamp() * 1000) - 30 * 60 * 1000
    return [(1 + i, base_ms + (i // per_ms)) for i in range(n)]


def test_downloader_paginates_by_id_and_keeps_all_trades():
    """Старий курсор `last_ts + 1` губив угоди, що ділять останню мс батча."""
    rows = _rows(2500, per_ms=5)
    client = _FakeClient(rows)
    dl = _downloader(client)

    out = dl.agg_trades("BTCUSDT", days=1, start_ms=rows[0][1] - 1)

    assert out["trade_id"].tolist() == [r[0] for r in rows], "втрачено угоди на межах батчів"
    assert any(from_id is not None for _since, from_id in client.calls), "має використовуватись fromId"
    # 3 батчі по 1000 + порожній/короткий — переконайся, що не зациклилось.
    assert len(client.calls) <= 5


def test_downloader_falls_back_to_time_cursor_without_ids():
    """Якщо API не дав aggTrade id — працює старий часовий курсор і цикл завершується."""
    rows = _rows(1200, per_ms=1)
    client = _FakeClient(rows, with_ids=False)
    dl = _downloader(client)

    out = dl.agg_trades("BTCUSDT", days=1, start_ms=rows[0][1] - 1)

    assert not out.empty
    assert (out["trade_id"] <= 0).all(), "без id мають бути синтетичні (від'ємні) id"
    assert len(client.calls) <= 6, "цикл має завершуватись"


def test_downloader_does_not_loop_when_ids_do_not_advance():
    """Незмінний id-курсор — вихід, а не нескінченний цикл."""

    class _StuckClient:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_agg_trades(self, symbol, since_ms=None, limit=1000, *, from_id=None):
            self.calls += 1
            ts = int(pd.Timestamp.now("UTC").timestamp() * 1000) - 60_000
            return [{"id": 42, "timestamp": ts, "price": 1.0, "amount": 1.0, "side": "buy"} for _ in range(1000)]

    client = _StuckClient()
    dl = _downloader(client)
    out = dl.agg_trades("BTCUSDT", days=1, start_ms=int(pd.Timestamp.now("UTC").timestamp() * 1000) - 3_600_000)

    assert client.calls <= 3, "курсор не просувається — маємо вийти"
    assert len(out) == 1  # та сама угода не дублюється за trade_id


def test_agg_trades_increment_resumes_from_cache_ids(tmp_path):
    """Докачка поверх кешу з реальними id продовжує з `max(id)+1` (fromId)."""
    base = int(pd.Timestamp.now("UTC").timestamp() * 1000) - 20 * 60 * 1000
    cached_rows = [(1 + i, base + (i // 5)) for i in range(1000)]
    cached = _trades_frame(cached_rows)
    new_rows = cached_rows + [(1001 + i, base + 200 + (i // 5)) for i in range(300)]

    class _Store(_FakeStore):
        def load_trades(self, symbol):
            return cached

    client = _FakeClient(new_rows)
    dl = Downloader(client=client, store=_Store(), retries=1, batch_delay=0, checkpoint_batches=0)

    out = dl.agg_trades("BTCUSDT", days=1, start_ms=base - 1)

    assert client.calls[0][1] == 1001, "перший батч має запитуватись за fromId = max(id)+1"
    assert out["trade_id"].tolist() == [r[0] for r in new_rows]
