"""Тести validate_bars: OHLC-інваріанти, дублікати, монотонність, дірки, майбутнє."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.validate import validate_bars


def _bars(n: int = 20, start: str = "2024-01-01", freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    c = np.full(n, 100.0)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)


def test_ok_on_clean_data() -> None:
    rep = validate_bars(_bars(), interval="1m")
    assert rep.ok and rep.summary().startswith("bars ok")


def test_empty_dataset_fails() -> None:
    rep = validate_bars(pd.DataFrame())
    assert not rep.ok and "порожній" in rep.issues[0]


def test_ohlc_violation_detected() -> None:
    df = _bars()
    df.loc[df.index[5], "low"] = 105.0  # low > close
    rep = validate_bars(df)
    assert not rep.ok and rep.n_ohlc_violations == 1


def test_duplicates_detected() -> None:
    df = pd.concat([_bars(), _bars().iloc[[3]]]).sort_index()
    rep = validate_bars(df)
    assert not rep.ok and rep.n_duplicates == 1


def test_non_monotonic_detected() -> None:
    df = _bars()
    df = df.iloc[::-1]  # розвертаємо індекс
    rep = validate_bars(df)
    assert not rep.ok and not rep.monotonic


def test_future_bars_detected() -> None:
    df = _bars(start="2999-01-01")
    rep = validate_bars(df)
    assert not rep.ok and rep.n_future == 20


def test_future_bars_respect_explicit_now() -> None:
    df = _bars(start="2999-01-01")
    rep = validate_bars(df, now=pd.Timestamp("2999-06-01"))
    assert rep.n_future == 0


def test_gaps_detected_with_interval() -> None:
    df = _bars(20, freq="1min")
    df = df.drop(df.index[10])  # дірка на 1 бар
    rep = validate_bars(df, interval="1m")
    assert not rep.ok and rep.n_gaps == 1


def test_gaps_ignored_without_interval() -> None:
    df = _bars(20, freq="1min").drop(_bars(20)[10:11].index)
    rep = validate_bars(df)  # без interval дірки не рахуються
    assert rep.n_gaps == 0 and rep.ok


def _trades(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame(
        {
            "trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "amount": np.full(n, 0.1),
            "side": ["buy", "sell"] * (n // 2),
        },
        index=idx,
    )


def test_validate_trades_ok() -> None:
    from scalper_hft.data.validate import validate_trades

    assert validate_trades(_trades()).ok


def test_validate_trades_rejects_nonpositive_price() -> None:
    from scalper_hft.data.validate import validate_trades

    df = _trades()
    df.loc[df.index[0], "price"] = 0.0
    rep = validate_trades(df)
    assert not rep.ok and rep.n_invalid >= 1


def test_validate_trades_rejects_bad_side() -> None:
    from scalper_hft.data.validate import validate_trades

    df = _trades()
    df.loc[df.index[1], "side"] = "hold"
    assert not validate_trades(df).ok


def test_validate_funding_ok_and_gap() -> None:
    from scalper_hft.data.validate import validate_funding

    idx = pd.date_range("2024-01-01", periods=3, freq="8h")
    df = pd.DataFrame({"fundingRate": [0.0001, 0.0002, -0.0001]}, index=idx)
    assert validate_funding(df).ok
    gappy = df.drop(df.index[1])
    rep = validate_funding(gappy)
    assert not rep.ok and rep.n_gaps == 1


def test_validate_funding_rejects_crazy_rate() -> None:
    from scalper_hft.data.validate import validate_funding

    idx = pd.date_range("2024-01-01", periods=2, freq="8h")
    df = pd.DataFrame({"fundingRate": [0.0001, 0.2]}, index=idx)
    assert not validate_funding(df).ok


def test_validate_bookticker_crossed() -> None:
    from scalper_hft.data.validate import validate_bookticker

    idx = pd.date_range("2024-01-01", periods=3, freq="1s")
    good = pd.DataFrame(
        {"bid": [100.0, 100.1, 100.2], "ask": [100.1, 100.2, 100.3], "bid_qty": 1.0, "ask_qty": 1.0},
        index=idx,
    )
    assert validate_bookticker(good).ok
    bad = good.copy()
    bad.loc[bad.index[1], "ask"] = 99.0
    assert not validate_bookticker(bad).ok


def _depth(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="100ms")
    mid = 100.0
    rows: dict[str, float] = {}
    for i in range(1, 6):
        rows[f"bid{i}"] = mid - i * 0.01
        rows[f"ask{i}"] = mid + i * 0.01
        rows[f"bid{i}_qty"] = 1.0
        rows[f"ask{i}_qty"] = 1.0
    return pd.DataFrame(rows, index=idx)


def test_validate_depth_ok_and_crossed() -> None:
    from scalper_hft.data.validate import validate_depth

    d = _depth()
    assert validate_depth(d).ok
    bad = d.copy()
    bad.loc[bad.index[0], "ask1"] = 99.0
    assert not validate_depth(bad).ok


def test_validate_depth_level_order() -> None:
    from scalper_hft.data.validate import validate_depth

    d = _depth()
    d.loc[d.index[2], "bid2"] = 100.5  # глибший bid дорожчий за bid1
    assert not validate_depth(d).ok


def test_save_trades_fail_closed(tmp_path) -> None:
    from scalper_hft.data.storage import save_trades

    path = tmp_path / "X_aggTrades.parquet"
    bad = _trades()
    bad.loc[bad.index[0], "price"] = -1.0
    with pytest.raises(ValueError, match="битих aggTrades"):
        save_trades(path, bad)
    assert not path.exists()


def test_save_funding_fail_closed(tmp_path) -> None:
    from scalper_hft.data.storage import save_funding

    path = tmp_path / "X_funding.parquet"
    idx = pd.date_range("2024-01-01", periods=2, freq="8h")
    bad = pd.DataFrame({"fundingRate": [0.0001, 0.5]}, index=idx)
    with pytest.raises(ValueError, match="битого funding"):
        save_funding(path, bad)
    assert not path.exists()


def _book(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame(
        {"bid": 100.0, "ask": 100.1, "bid_qty": 1.0, "ask_qty": 1.0},
        index=idx,
    )


def test_audit_l2_cache_empty_dir_fails(tmp_path: Path) -> None:
    from scalper_hft.data.validate import audit_l2_cache

    audit = audit_l2_cache(tmp_path)
    assert audit.quality_ok is False
    assert "quality_ok:** `false`" in audit.to_markdown()


def test_audit_l2_cache_ignores_klines_and_passes_clean(tmp_path: Path) -> None:
    from scalper_hft.data.validate import audit_l2_cache, list_l2_parquet

    klines = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )
    klines.to_parquet(tmp_path / "BTCUSDT_1h.parquet")
    _depth().to_parquet(tmp_path / "BTCUSDT_depth5.parquet")
    _book().to_parquet(tmp_path / "BTCUSDT_bookTicker.parquet")

    names = {p.name for p in list_l2_parquet(tmp_path)}
    assert names == {"BTCUSDT_depth5.parquet", "BTCUSDT_bookTicker.parquet"}

    audit = audit_l2_cache(tmp_path)
    assert audit.quality_ok is True
    md = audit.to_markdown()
    assert "quality_ok:** `true`" in md
    assert "BTCUSDT_1h.parquet" not in md


def test_audit_l2_cache_crossed_book_fails(tmp_path: Path) -> None:
    from scalper_hft.data.validate import audit_l2_cache

    _depth().to_parquet(tmp_path / "LINKUSDT_depth5.parquet")
    bad = _book()
    bad.loc[bad.index[0], "ask"] = 90.0
    bad.to_parquet(tmp_path / "LINKUSDT_bookTicker.parquet")

    audit = audit_l2_cache(tmp_path)
    assert audit.quality_ok is False
    assert any(not item.report.ok for item in audit.files)


def test_audit_l2_cache_unreadable_parquet_fails(tmp_path: Path) -> None:
    from scalper_hft.data.validate import audit_l2_cache

    (tmp_path / "ETHUSDT_depth5.parquet").write_bytes(b"not a parquet")
    audit = audit_l2_cache(tmp_path)
    assert audit.quality_ok is False
    assert audit.extra_issues


def test_cmd_depth_audit_writes_report_and_exits(tmp_path: Path) -> None:
    from argparse import Namespace

    from scalper_hft.cli.ops import cmd_depth_audit

    out = tmp_path / "quality_depth.md"
    with pytest.raises(SystemExit) as exc:
        cmd_depth_audit(Namespace(data_dir=str(tmp_path), out=str(out)))
    assert exc.value.code == 1
    text = out.read_text(encoding="utf-8")
    assert "quality_ok:** `false`" in text

    _depth().to_parquet(tmp_path / "XRPUSDT_depth5.parquet")
    _book().to_parquet(tmp_path / "XRPUSDT_bookTicker.parquet")
    cmd_depth_audit(Namespace(data_dir=str(tmp_path), out=str(out)))
    assert "quality_ok:** `true`" in out.read_text(encoding="utf-8")


# ── aggTrades: спільні мілісекунди — норма, а не дефект ─────────────────────


def _trades_multi_per_ms(n: int = 50, per_ms: int = 5) -> pd.DataFrame:
    """aggTrades, де кілька угод ділять одну мілісекунду (як у реальному потоці)."""
    idx = pd.to_datetime([1_700_000_000_000 + i // per_ms for i in range(n)], unit="ms")
    return pd.DataFrame(
        {
            "trade_id": np.arange(1, n + 1),
            "price": np.full(n, 100.0),
            "amount": np.full(n, 0.1),
            "side": ["buy", "sell"] * (n // 2),
        },
        index=idx,
    )


def test_validate_trades_allows_shared_milliseconds_when_ids_unique() -> None:
    """Регресія: дублікати міток часу НЕ блокують запис aggTrades.

    Реальний кейс (2026-09-12): завантаження SOLUSDT з Binance Vision впало з
    `FAIL: n=43289298 dup=1162183 invalid=0`. `invalid=0` означає, що з
    trade_id усе гаразд, а `dup` — це спільні мілісекунди, яких у потоці ~2.2%
    (на реальному архіві: 6085 на 279 778 рядків). До аудиту це не спрацьовувало
    лише тому, що старий дедуп за мілісекундним індексом ЗНИЩУВАВ ці угоди.
    """
    from scalper_hft.data.validate import stream_is_critical, validate_trades

    df = _trades_multi_per_ms()
    assert df.index.duplicated().sum() > 0, "синтетика має містити спільні мс"

    rep = validate_trades(df)

    assert rep.ok, rep.issues
    assert rep.n_duplicates == 0
    assert not stream_is_critical(rep)


def test_validate_trades_still_flags_duplicate_trade_id() -> None:
    from scalper_hft.data.validate import stream_is_critical, validate_trades

    df = _trades_multi_per_ms()
    dup = pd.concat([df, df.iloc[:10]])

    rep = validate_trades(dup)

    assert not rep.ok
    assert stream_is_critical(rep), "дублікати справжнього ключа — критично"
    assert any("trade_id" in i for i in rep.issues)


def test_validate_trades_flags_timestamp_dups_when_no_trade_id() -> None:
    """Без trade_id час — єдиний ключ, тож дублікати мс знову дефект."""
    from scalper_hft.data.validate import validate_trades

    df = _trades_multi_per_ms().drop(columns=["trade_id"])
    rep = validate_trades(df)

    assert not rep.ok
    assert rep.n_duplicates > 0


def test_save_trades_accepts_shared_milliseconds(tmp_path) -> None:
    """Запис кешу не має падати на кадрі зі спільними мс (fail-closed лишається для id)."""
    from scalper_hft.data.storage import load_trades, save_trades

    df = _trades_multi_per_ms()
    path = tmp_path / "SYN_aggTrades.parquet"

    save_trades(path, df)
    back = load_trades(path)

    assert back is not None and len(back) == len(df)
