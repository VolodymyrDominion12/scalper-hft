"""Тести для швидких аналітичних запитів та ASOF-злиття DuckDB."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.fast_query import (
    aggregate_bookticker_parquet,
    asof_join_parquet,
    query_parquet,
    require_duckdb,
)


def test_require_duckdb() -> None:
    """Перевірка, що duckdb встановлено та успішно імпортується."""
    require_duckdb()


def test_query_parquet_basic(tmp_path: Path) -> None:
    """Перевірка виконання базового SQL над Parquet."""
    df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"], "volume": [10.0, 20.0, 30.0]})
    file_p = tmp_path / "test_data.parquet"
    df.to_parquet(file_p)

    sql = f"SELECT symbol, SUM(volume) AS total_vol FROM '{file_p}' GROUP BY symbol ORDER BY symbol"
    res = query_parquet(sql)
    assert len(res) == 2
    assert res.loc[res["symbol"] == "BTCUSDT", "total_vol"].iloc[0] == 40.0
    assert res.loc[res["symbol"] == "ETHUSDT", "total_vol"].iloc[0] == 20.0


def test_asof_join_strict_no_lookahead(tmp_path: Path) -> None:
    """Перевірка, що ASOF JOIN не підглядає в майбутнє."""
    # Base: часовий ряд кожні 100 одиниць
    base_df = pd.DataFrame(
        {
            "timestamp": [100, 200, 300, 400],
            "price": [10.0, 11.0, 12.0, 13.0],
        }
    )
    base_p = tmp_path / "base.parquet"
    base_df.to_parquet(base_p)

    # Events: події у моменти 150 та 250
    event_df = pd.DataFrame(
        {
            "timestamp": [150, 250],
            "funding_rate": [0.0001, 0.0005],
        }
    )
    event_p = tmp_path / "events.parquet"
    event_df.to_parquet(event_p)

    joined = asof_join_parquet(base_p, event_p, time_col="timestamp")
    assert len(joined) == 4

    # При t=100 подій ще не було => NaN
    row_100 = joined[joined["timestamp"] == 100].iloc[0]
    assert np.isnan(row_100["event_funding_rate"])

    # При t=200 остання відома подія відбулася в t=150 (funding_rate=0.0001)
    row_200 = joined[joined["timestamp"] == 200].iloc[0]
    assert row_200["event_funding_rate"] == pytest.approx(0.0001)

    # При t=300 остання подія була в t=250 (funding_rate=0.0005)
    row_300 = joined[joined["timestamp"] == 300].iloc[0]
    assert row_300["event_funding_rate"] == pytest.approx(0.0005)

    # При t=400 подія все ще 0.0005
    row_400 = joined[joined["timestamp"] == 400].iloc[0]
    assert row_400["event_funding_rate"] == pytest.approx(0.0005)


def test_aggregate_bookticker_parquet(tmp_path: Path) -> None:
    """Перевірка бакет-агрегації тіків bookTicker."""
    # 3 тіки у першу хвилину (0-60с) і 1 тік у другу хвилину (60-120с)
    # timestamp у мілісекундах
    bt_df = pd.DataFrame(
        {
            "timestamp": [1000, 10000, 20000, 70000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "bid_price": [50000.0, 50010.0, 50020.0, 50050.0],
            "ask_price": [50002.0, 50012.0, 50022.0, 50052.0],
            "bid_qty": [1.5, 2.0, 1.0, 3.0],
            "ask_qty": [1.2, 1.8, 0.9, 2.5],
        }
    )
    bt_p = tmp_path / "bookticker.parquet"
    bt_df.to_parquet(bt_p)

    agg = aggregate_bookticker_parquet(bt_p, interval_seconds=60)
    assert len(agg) == 2
    # Перший бакет містить 3 тіки
    assert agg["tick_count"].iloc[0] == 3
    # Другий бакет містить 1 тік
    assert agg["tick_count"].iloc[1] == 1
    # Перевірка OHLC mid price
    # mid: 50001, 50011, 50021
    assert agg["mid_open"].iloc[0] == 50001.0
    assert agg["mid_close"].iloc[0] == 50021.0
    assert agg["mid_high"].iloc[0] == 50021.0
    assert agg["mid_low"].iloc[0] == 50001.0
