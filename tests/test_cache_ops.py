"""Операції кешу даних: валідація символів, видалення parquet, докачка."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.data.cache_ops import (
    CacheDownloadResult,
    action_from_row_label,
    clamp_cache_days,
    delete_symbol_cache,
    download_symbol_cache,
    list_symbol_cache_files,
    max_span_days,
    row_cache_actions,
    suggested_cache_days,
    validate_symbol,
)


def test_validate_symbol_normalizes_and_rejects_paths() -> None:
    assert validate_symbol(" btcusdt ") == "BTCUSDT"
    with pytest.raises(ValueError):
        validate_symbol("../etc")
    with pytest.raises(ValueError):
        validate_symbol("BTC/USDT")
    with pytest.raises(ValueError):
        validate_symbol("")


def test_clamp_and_suggested_days() -> None:
    assert clamp_cache_days(90) == 90
    with pytest.raises(ValueError):
        clamp_cache_days(0)
    with pytest.raises(ValueError):
        clamp_cache_days(2000)
    assert suggested_cache_days(None, action="refresh") == 90
    assert suggested_cache_days(90.0, action="expand") == 180
    assert suggested_cache_days(200.0, action="expand") == 230
    assert suggested_cache_days(float("nan"), action="expand") == 180


def test_row_actions_and_labels() -> None:
    assert row_cache_actions(0) == [":material/download: Завантажити"]
    assert ":material/delete: Видалити" in row_cache_actions(10)
    assert action_from_row_label(":material/delete: Видалити") == "delete"
    assert action_from_row_label(":material/date_range: Розширити") == "expand"
    assert action_from_row_label(":material/download: Оновити") == "refresh"
    assert action_from_row_label(":material/download: Завантажити") == "refresh"


def _touch(path: Path) -> None:
    path.write_bytes(b"x")


def test_list_and_delete_only_that_symbol(tmp_path: Path) -> None:
    _touch(tmp_path / "ETHUSDT_1m_klines.parquet")
    _touch(tmp_path / "ETHUSDT_1h_klines.parquet")
    _touch(tmp_path / "ETHUSDT_1h_spot_klines.parquet")
    _touch(tmp_path / "ETHUSDT_aggTrades.parquet")
    _touch(tmp_path / "ETHUSDT_funding.parquet")
    _touch(tmp_path / "ETHUSDT_bookTicker.parquet")
    _touch(tmp_path / "ETHUSDT_depth5.parquet")
    _touch(tmp_path / "BTCUSDT_1m_klines.parquet")
    _touch(tmp_path / "WETHUSDT_1m_klines.parquet")

    names = {p.name for p in list_symbol_cache_files(tmp_path, "ETHUSDT")}
    assert "ETHUSDT_1m_klines.parquet" in names
    assert "ETHUSDT_funding.parquet" in names
    assert "ETHUSDT_depth5.parquet" in names
    assert "BTCUSDT_1m_klines.parquet" not in names
    assert "WETHUSDT_1m_klines.parquet" not in names

    deleted = delete_symbol_cache(tmp_path, "ethusdt")
    assert set(deleted) == names
    assert (tmp_path / "BTCUSDT_1m_klines.parquet").exists()
    assert (tmp_path / "WETHUSDT_1m_klines.parquet").exists()
    assert not (tmp_path / "ETHUSDT_1m_klines.parquet").exists()
    assert delete_symbol_cache(tmp_path, "ETHUSDT") == []


def test_max_span_days() -> None:
    inv = pd.DataFrame(
        {
            "symbol": ["ETHUSDT", "BTCUSDT", "SOLUSDT"],
            "span_days": [90.0, 12.5, float("nan")],
        }
    )
    assert max_span_days(inv, ["ETHUSDT", "BTCUSDT"]) == pytest.approx(90.0)
    assert max_span_days(inv, ["SOLUSDT"]) is None
    assert max_span_days(pd.DataFrame(), ["ETHUSDT"]) is None


def test_download_symbol_cache_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_klines(symbol: str, interval: str, days: int, force: bool = False, **_: object) -> pd.DataFrame:
        calls["klines"] = (symbol, interval, days, force)
        idx = pd.date_range("2026-01-01", periods=4, freq="1min")
        return pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=idx,
        )

    def fake_funding(symbol: str, days: int, force: bool = False, **_: object) -> pd.DataFrame:
        calls["funding"] = (symbol, days, force)
        return pd.DataFrame({"fundingRate": [0.0001]}, index=pd.date_range("2026-01-01", periods=1, freq="8h"))

    def fake_trades(symbol: str, days: int, force: bool = False, **_: object) -> pd.DataFrame:
        calls["trades"] = (symbol, days, force)
        return pd.DataFrame(
            {"trade_id": [1], "price": [1.0], "amount": [1.0], "side": ["buy"]},
            index=pd.date_range("2026-01-01", periods=1, freq="1min"),
        )

    monkeypatch.setattr("scalper_hft.data.downloader.download_klines", fake_klines)
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", fake_funding)
    monkeypatch.setattr("scalper_hft.data.downloader.download_agg_trades", fake_trades)

    result = download_symbol_cache("ethusdt", 90, force=True, funding=True, trades=True)
    assert isinstance(result, CacheDownloadResult)
    assert result.symbol == "ETHUSDT"
    assert result.klines_rows == 4
    assert result.funding_rows == 1
    assert result.trades_rows == 1
    assert calls["klines"] == ("ETHUSDT", "1m", 90, True)
    assert calls["trades"][1] == 2  # REST- treйди обмежені 2 добами

    with pytest.raises(ValueError):
        download_symbol_cache("ETHUSDT", 90, interval="1h")
