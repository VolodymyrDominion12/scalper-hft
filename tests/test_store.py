"""Тести кеш-бекендів: ParquetStore (завжди) і PostgresStore (якщо є TEST_POSTGRES_DSN).

Щоб прогнати Postgres-тести локально:
    docker compose up -d postgres
    TEST_POSTGRES_DSN="host=localhost port=5436 dbname=scalper user=scalper password=scalper" \
        .venv/bin/python -m pytest tests/test_store.py -q
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from scalper_hft.data.store import ParquetStore, PostgresStore, get_store


def _klines_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)


def _trades_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [10, 11, 12],
            "price": [100.0, 100.1, 100.2],
            "amount": [0.1, 0.2, 0.3],
            "side": ["buy", "sell", "buy"],
        },
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:02"]),
    )


def _funding_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"fundingRate": [0.0001, 0.0002]},
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 08:00:00"]),
    )


@pytest.fixture
def parquet_store(tmp_path):
    return ParquetStore(data_dir=tmp_path)


def test_parquet_klines_roundtrip(parquet_store) -> None:
    parquet_store.save_klines("BTCUSDT", "1m", _klines_df())
    out = parquet_store.load_klines("BTCUSDT", "1m")
    assert out is not None and len(out) == 100
    assert out["close"].iloc[-1] == 1.5
    assert ("BTCUSDT", "1m") in parquet_store.list_klines()


def test_parquet_trades_funding_roundtrip(parquet_store) -> None:
    parquet_store.save_trades("BTCUSDT", _trades_df())
    tr = parquet_store.load_trades("BTCUSDT")
    assert tr is not None and tr["trade_id"].tolist() == [10, 11, 12]

    parquet_store.save_funding("BTCUSDT", _funding_df())
    fu = parquet_store.load_funding("BTCUSDT")
    assert fu is not None and "fundingRate" in fu.columns and len(fu) == 2


def test_parquet_missing_returns_none(parquet_store) -> None:
    assert parquet_store.load_klines("NOPEUSDT", "1m") is None
    assert parquet_store.load_trades("NOPEUSDT") is None
    assert parquet_store.load_funding("NOPEUSDT") is None


def test_get_store_default_parquet(monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "parquet")
    monkeypatch.setattr("scalper_hft.config._settings", None)
    store = get_store()
    assert isinstance(store, ParquetStore)
    monkeypatch.setattr("scalper_hft.config._settings", None)


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_DSN"), reason="TEST_POSTGRES_DSN не задано — Postgres-тести пропущені")
def test_postgres_klines_roundtrip() -> None:
    store = PostgresStore(os.getenv("TEST_POSTGRES_DSN"))
    store.ensure_schema()
    store.save_klines("TESTUSDT", "1m", _klines_df())
    out = store.load_klines("TESTUSDT", "1m")
    assert out is not None and len(out) == 100
    # заміна: upsert-семантика — той самий діапазон з більшою кількістю барів
    extra = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.6, "volume": 5.0},
        index=[pd.Timestamp("2024-01-01 01:40")],
    )
    df2 = pd.concat([_klines_df(), extra])
    store.save_klines("TESTUSDT", "1m", df2)
    out2 = store.load_klines("TESTUSDT", "1m")
    assert out2 is not None and len(out2) == 101


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_DSN"), reason="TEST_POSTGRES_DSN не задано")
def test_postgres_trades_funding_roundtrip() -> None:
    store = PostgresStore(os.getenv("TEST_POSTGRES_DSN"))
    store.ensure_schema()
    store.save_trades("TESTUSDT", _trades_df())
    tr = store.load_trades("TESTUSDT")
    assert tr is not None and tr["trade_id"].tolist() == [10, 11, 12]

    store.save_funding("TESTUSDT", _funding_df())
    fu = store.load_funding("TESTUSDT")
    assert fu is not None and "fundingRate" in fu.columns and len(fu) == 2

    store.save_spot_klines("TESTUSDT", "1h", _klines_df(10))
    sp = store.load_spot_klines("TESTUSDT", "1h")
    assert sp is not None and len(sp) == 10

    assert ("TESTUSDT", "1m") in store.list_klines()


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_DSN"), reason="TEST_POSTGRES_DSN не задано")
def test_postgres_operational_error_returns_none() -> None:
    store = PostgresStore("host=localhost port=1 dbname=nope user=nope password=nope connect_timeout=2")
    assert store.load_klines("X", "1m") is None
    assert store.list_klines() == []
