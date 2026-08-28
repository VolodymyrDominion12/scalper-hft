"""scalper-hft — високочастотна скальпінг-система для Binance USDT-M ф'ючерсів.

Архітектура за моделлю книги "Inside the Black Box" (R. Narang):
Alpha Model → Risk Model → Transaction Cost Model → Portfolio Construction → Execution,
поверх шарів Data та Research (бектест + анти-перенавчання).

Пакети:
    data        — завантаження/кешування даних Binance (klines, aggTrades, funding)
    features    — мікроструктурні фічі (OB imbalance, CVD, spread, волатильність)
    strategies  — альфа-моделі (стратегії) у єдиному інтерфейсі
    backtest    — рушій бектесту (векторизований + подієвий) та метрики
    validation  — walk-forward, CV, deflated Sharpe, sensitivity — анти-перенавчання
    ml          — ML-фічі та класифікатори напрямку
    live        — paper/live виконання через ccxt
"""

__version__ = "0.1.0"
