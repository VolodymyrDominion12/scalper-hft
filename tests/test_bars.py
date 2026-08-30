import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.bars import create_dollar_bars, create_tick_imbalance_bars, create_volume_bars


@pytest.fixture
def mock_trades():
    # 10 тіків, ціна зростає з 100 до 109, об'єм 1 для всіх
    dates = pd.date_range("2023-01-01", periods=10, freq="s")
    prices = np.arange(100, 110)
    amounts = np.ones(10) * 10

    df = pd.DataFrame({"price": prices, "amount": amounts, "side": ["buy"] * 10}, index=dates)
    return df


def test_volume_bars(mock_trades):
    # Кожен тік = 10 об'єм. Поріг 25 означає, що 3 тіки (30 об'єм) створять бар.
    # Тіки 0, 1, 2 -> бар 1
    # Тіки 3, 4, 5 -> бар 2
    # Тіки 6, 7, 8 -> бар 3
    # Тік 9 -> не дотягує до 25? Ні, cumulative sum:
    # 10, 20, 30(бар), 40, 50(бар), 60, 70, 80(бар), 90, 100(бар).
    # Group: 10//25 = 0
    # 20//25 = 0
    # 30//25 = 1 -> бар

    # Використовується // threshold
    bars = create_volume_bars(mock_trades, volume_threshold=25)

    assert not bars.empty
    assert len(bars) == 5

    assert bars.iloc[0]["open"] == 100
    assert bars.iloc[0]["close"] == 101
    assert bars.iloc[0]["volume"] == 20


def test_dollar_bars(mock_trades):
    bars = create_dollar_bars(mock_trades, dollar_threshold=2500)

    assert not bars.empty
    # First 2 ticks = 1000 + 1010 = 2010
    assert bars.iloc[0]["dollar_volume"] == 2010


def test_tick_imbalance_bars(mock_trades):
    # В mock_trades ціна постійно зростає, тому tick_rule = 1
    # expected_T = 2 (поріг 2). Значить кожен 2й тік - це бар.
    bars = create_tick_imbalance_bars(mock_trades, expected_imbalance_window=2, ewma_window=2)
    assert not bars.empty
    # Має бути близько 5 барів
    assert len(bars) > 0
