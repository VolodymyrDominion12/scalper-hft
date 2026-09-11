"""Fit емпіричної карти «режим → стратегія» (Phase 2B, iter7/v2.0).

Призначення: замінити статичні таксономічні пріори (``preferred_regimes``)
ЕМПІРИЧНОЮ картою — «у якому режимі яка стратегія справді заробляла». iter7
показав, що гіпотетичні теги суперечать даним (supertrend мав тег trend_down,
який виявився його найгіршим режимом), і що на таких пріорах будь-який
``best_prior``/``regime_soft`` програє найкращій одиночній стратегії.

Пайплайн (без lookahead — карта будується на періоді, що СТРОГО передує
періоду застосування):

    1. для кожного символу вантажимо klines fit-періоду (readonly, з кешу);
    2. для кожної стратегії — один in-sample бектест (maker-модель витрат)
       → бар-дохідності;
    3. режимні мітки — каузальний RegimeDetector (structure|vol);
    4. дохідності всіх символів пуляться в один ряд → Sharpe per
       (regime × strategy) з урахуванням min_bars;
    5. карта: hard-off (Sharpe < порогу → вага 0), flat у high-vol,
       best_prior для режиму з однозначним лідером, інакше soft-ваги.

Метадані (fit-вікно, символи, стратегії) пишуться у той самий JSON: це дозволяє
аудитувати, що карта не «бачила» період застосування.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scalper_hft.validation.regime_map import (
    RegimePerfMatrix,
    RegimeStrategyMap,
    build_regime_strategy_map,
    pool_regime_perf_matrix,
)

logger = logging.getLogger(__name__)

DEFAULT_FIT_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT")
# Evidence-based пул слевів (iter7): моментум для trend_up + carry для range/trend_down.
DEFAULT_FIT_STRATEGIES: tuple[str, ...] = ("supertrend", "stoch_rsi", "funding_carry")


@dataclass(slots=True)
class RegimeFitResult:
    """Результат fit: карта, матриця Sharpe і метадані періоду."""

    map: RegimeStrategyMap
    matrix: RegimePerfMatrix
    meta: dict[str, Any] = field(default_factory=dict)
    per_symbol_returns: dict[str, pd.DataFrame] = field(default_factory=dict)


def _resolve_fit_end(fit_end: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Нормалізувати fit_end до tz-naive UTC (індекс кешу klines — tz-naive)."""
    if fit_end in (None, ""):
        return None
    ts = pd.Timestamp(fit_end)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _cell_table(rmap: RegimeStrategyMap) -> list[dict[str, Any]]:
    """Плоский список комірок карти (для логування/звіту)."""
    rows: list[dict[str, Any]] = []
    for regime in sorted(rmap.weights):
        weights = rmap.weights[regime]
        active = [s for s, w in weights.items() if w > 0.0]
        rows.append(
            {
                "regime": regime,
                "policy": rmap.policy.get(regime, ""),
                "active": ",".join(sorted(active)) if active else "FLAT",
                "weights": {s: round(float(w), 4) for s, w in sorted(weights.items())},
            }
        )
    return rows


def fit_regime_map(
    strategies: list[str] | tuple[str, ...] = DEFAULT_FIT_STRATEGIES,
    symbols: list[str] | tuple[str, ...] = DEFAULT_FIT_SYMBOLS,
    *,
    interval: str = "1h",
    fit_days: int = 365,
    fit_end: str | pd.Timestamp | None = None,
    base_interval: str = "1m",
    is_maker: bool = True,
    min_bars: int = 30,
    hard_off_sharpe: float = 0.0,
    high_vol_flat: bool = True,
    best_prior_min_gap: float = 0.1,
    position_pct: float | None = None,
    readonly: bool = True,
) -> RegimeFitResult:
    """Побудувати карту «режим → стратегія» на IS-періоді.

    Args:
        strategies: імена стратегій з REGISTRY (кандидати-слеви).
        symbols: символи, на яких пуляться дохідності (робастність по інструментах).
        interval: таймфрейм режиму/торгівлі.
        fit_days: глибина fit-періоду.
        fit_end: кінець fit-вікна (ISO-дата або Timestamp). None = останній бар
            кешу. Задавайте його для чесної валідації: карта будується на
            [fit_end − fit_days, fit_end], а застосовується на барах ПІСЛЯ
            ``meta["fit_end"]`` — інакше порушується no-lookahead контракт.
        base_interval: базовий ТФ для деривації (кеш).
        is_maker: maker-модель витрат (висновок проєкту: taker на суб-годинних барах не виживає).
        min_bars: мінімум барів у комірці (regime × strategy) для значущості.
        hard_off_sharpe: стратегії з Sharpe < порогу → вага 0.
        high_vol_flat: flat у режимах ``*|high`` (ризик-політика).
        best_prior_min_gap: відносний відрив лідера для політики best_prior.
        position_pct: розмір позиції (None = з налаштувань).
        readonly: лише читати кеш (без мережевих дозавантажень).

    Returns:
        RegimeFitResult (порожня карта, якщо не вдалося зібрати жодної комірки).
    """
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.store import get_store
    from scalper_hft.features.regime_detector import RegimeDetector
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    pct = float(settings.position_pct if position_pct is None else position_pct)
    cost = CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
    )
    store = get_store()
    cutoff = _resolve_fit_end(fit_end)
    # Скільки днів історії вантажити: fit_days + «хвіст» до сьогодні, якщо
    # fit_end у минулому (кеш завжди віддає ОСТАННІ N днів).
    load_days = int(fit_days)
    if cutoff is not None:
        tail_days = int((pd.Timestamp.now(tz="UTC").tz_localize(None) - cutoff).days) + 1
        load_days = int(fit_days) + max(0, tail_days)
    per_symbol: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    returns_by_symbol: dict[str, pd.DataFrame] = {}
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    skipped: dict[str, str] = {}

    for symbol in symbols:
        try:
            df = ensure_klines(symbol, interval, load_days, base_interval=base_interval, derive=True, readonly=readonly)
        except Exception as exc:  # noqa: BLE001
            logger.warning("regime-fit: %s — дані недоступні: %s", symbol, exc)
            skipped[symbol] = f"data: {exc}"[:160]
            continue
        if df is None or df.empty:
            skipped[symbol] = "порожній кеш"
            continue
        if cutoff is not None:
            df = df.loc[df.index <= cutoff]
            if len(df) < min_bars:
                skipped[symbol] = f"після fit_end={cutoff} лишилось {len(df)} барів"
                continue

        regime = RegimeDetector().detect(df["close"])[["structure", "vol", "label"]]
        returns: dict[str, pd.Series] = {}
        for name in strategies:
            try:
                strat = get_strategy(name)
                trades = store.load_trades(symbol) if strat.needs_trades else None
                funding = store.load_funding(symbol) if strat.needs_funding else None
                res = run_strategy_backtest(
                    df,
                    strat,
                    cost=cost,
                    trades=trades,
                    funding=funding,
                    position_pct=pct,
                    is_maker=is_maker,
                    interval=interval,
                )
                returns[name] = pd.Series(res.bar_returns, index=df.index).fillna(0.0).astype(float)
            except Exception as exc:  # noqa: BLE001
                logger.warning("regime-fit: %s/%s — бектест не вдався: %s", symbol, name, exc)
                skipped[f"{symbol}:{name}"] = f"backtest: {exc}"[:160]
        if not returns:
            skipped[symbol] = "жодної стратегії не пораховано"
            continue

        ret_df = pd.DataFrame(returns)
        per_symbol[symbol] = (ret_df, regime)
        returns_by_symbol[symbol] = ret_df
        windows.append((df.index[0], df.index[-1]))

    matrix = pool_regime_perf_matrix(per_symbol, min_bars=min_bars)
    rmap = build_regime_strategy_map(
        matrix,
        hard_off_sharpe=hard_off_sharpe,
        high_vol_flat=high_vol_flat,
        best_prior_min_gap=best_prior_min_gap,
    )

    meta: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "regime_strategy_map",
        "fit_symbols": [s for s in symbols if s in per_symbol],
        "fit_strategies": list(strategies),
        "fit_interval": interval,
        "fit_days": int(fit_days),
        "fit_start": min(w[0] for w in windows).isoformat() if windows else None,
        "fit_end": max(w[1] for w in windows).isoformat() if windows else None,
        "is_maker": bool(is_maker),
        "min_bars": int(min_bars),
        "hard_off_sharpe": float(hard_off_sharpe),
        "high_vol_flat": bool(high_vol_flat),
        "best_prior_min_gap": float(best_prior_min_gap),
        "skipped": skipped,
        "cells": _cell_table(rmap),
    }
    return RegimeFitResult(map=rmap, matrix=matrix, meta=meta, per_symbol_returns=returns_by_symbol)


def format_fit_summary(result: RegimeFitResult) -> str:
    """Людиночитаний підсумок fit (для CLI): fit-вікно, політики, активні стратегії."""
    meta = result.meta
    lines = [
        "Regime map fit",
        f"  fit-вікно:    {meta.get('fit_start')} → {meta.get('fit_end')} ({meta.get('fit_interval')})",
        f"  символи:      {', '.join(meta.get('fit_symbols', [])) or '—'}",
        f"  стратегії:    {', '.join(meta.get('fit_strategies', [])) or '—'}",
        f"  режимів:      {len(result.map.weights)}",
    ]
    for cell in meta.get("cells", []):
        lines.append(f"    {cell['regime']:>18}  {cell['policy']:<10} → {cell['active']}")
    if meta.get("skipped"):
        lines.append(f"  пропущено:    {len(meta['skipped'])} (деталі в meta.skipped)")
    lines.append(
        "  ⚠️ карту можна застосовувати лише на барах ПІСЛЯ fit_end (no-lookahead контракт)"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_FIT_STRATEGIES",
    "DEFAULT_FIT_SYMBOLS",
    "RegimeFitResult",
    "fit_regime_map",
    "format_fit_summary",
]
