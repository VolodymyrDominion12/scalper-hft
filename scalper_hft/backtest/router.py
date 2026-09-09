"""Роутер рушія: векторний vs подієвий (market_maker).

Подієвий рушій (`event_engine`) — це симулятор ПАСИВНОГО market maker
(котирування обох сторін, інвентар, adverse selection): він доречний лише
для `market_maker`, сигнали якого (`generate_signals` → 0) не керують
напрямком — його облік живе в самому рушії, а параметри стратегії
(спред, інвентар-кап, haircut, розмір котирування) тепер реально
передаються в рушій.

`ob_imbalance` — НАПРЯМКОВА стратегія (власний `generate_signals` на
колонці imbalance/buy_ratio): для неї коректний векторний рушій
(`run_backtest`, сигнал на close t → виконання t+1 з maker/taker
комісіями). Раніше вона потрапляла в event_engine, який ігнорує сигнали —
результат був константою, що не залежить від її параметрів.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.event_engine import EventBacktestResult, run_event_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.micro_price import QueuePositionModel
from scalper_hft.portfolio.sizing import resolve_vol_target_ann
from scalper_hft.strategies.base import Strategy

if TYPE_CHECKING:
    from scalper_hft.overlay.policy import CellPolicy

# Стратегії, чиє виконання моделює ЛИШЕ подієвий рушій (пасивний MM).
EVENT_STRATEGIES = frozenset({"market_maker"})


def _settings_risk(
    max_leverage: float | None,
    vol_target_ann: float | None,
    apply_settings_risk: bool,
) -> tuple[float | None, float | None]:
    """Підставити max_leverage / vol-target з Settings, якщо caller не задав.

    apply_settings_risk=False — зворотна сумісність юніт-тестів (без кліпу).
    Явний max_leverage/vol_target_ann завжди перемагає.
    """
    if not apply_settings_risk:
        return max_leverage, vol_target_ann
    from scalper_hft.config import get_settings

    settings = get_settings()
    if max_leverage is None:
        max_leverage = float(settings.max_leverage)
    if vol_target_ann is None:
        vol_target_ann = resolve_vol_target_ann(
            None,
            enabled=bool(settings.enable_vol_target),
            target=float(settings.vol_target_ann),
        )
    return max_leverage, vol_target_ann


def run_strategy_backtest(
    df,
    strategy: Strategy,
    *,
    cost: CostModel | None = None,
    trades=None,
    funding=None,
    position_pct: float = 0.01,
    is_maker: bool = False,
    initial_capital: float = 10_000.0,
    trace: bool = False,
    overlay: CellPolicy | None = None,
    interval: str = "1m",
    queue_model: QueuePositionModel | None = None,
    spread_bps: float = 2.0,
    intrabar_exits: bool = False,
    signals=None,
    max_leverage: float | None = None,
    vol_target_ann: float | None = None,
    vol_lookback: int = 168,
    apply_settings_risk: bool = True,
    strict_data: bool = True,
    basket_df=None,
) -> BacktestResult | EventBacktestResult:
    max_leverage, vol_target_ann = _settings_risk(max_leverage, vol_target_ann, apply_settings_risk)
    name = getattr(strategy, "name", "")
    if name in EVENT_STRATEGIES:
        # Overlay: MM на OHLC невалідний — disabled клітинка лишається нулем
        # через векторний рушій, а не через подієвий філ-спам.
        if overlay is not None and not overlay.enabled:
            return run_backtest(
                df,
                strategy,
                initial_capital=initial_capital,
                cost=cost,
                position_pct=0.0,
                trades=trades,
                funding=funding,
                is_maker=is_maker,
                trace=trace,
                overlay=overlay,
                interval=interval,
                signals=signals,
                max_leverage=max_leverage,
                vol_target_ann=vol_target_ann,
                vol_lookback=vol_lookback,
                strict_data=False,  # no-op клітинка: не вимагаємо L2
                basket_df=basket_df,
            )
        # Параметри стратегії → параметри рушія (раніше лишались дефолти,
        # тож sweep/Optuna по market_maker повертали константу).
        return run_event_backtest(
            df,
            strategy,
            initial_capital=initial_capital,
            cost=cost,
            quote_size_pct=float(strategy.get("quote_size_pct", position_pct)),
            inventory_cap=float(strategy.get("inventory_cap", 1.0)),
            spread_offset_mult=float(strategy.get("spread_offset_mult", 0.5)),
            adverse_sel_haircut=float(strategy.get("adverse_sel_haircut", 0.5)),
        )
    return run_backtest(
        df,
        strategy,
        initial_capital=initial_capital,
        cost=cost,
        position_pct=position_pct,
        trades=trades,
        funding=funding,
        is_maker=is_maker,
        trace=trace,
        overlay=overlay,
        interval=interval,
        queue_model=queue_model,
        spread_bps=spread_bps,
        intrabar_exits=intrabar_exits,
        signals=signals,
        max_leverage=max_leverage,
        vol_target_ann=vol_target_ann,
        vol_lookback=vol_lookback,
        strict_data=strict_data,
        basket_df=basket_df,
    )
