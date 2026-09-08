"""Побудова labeled-датасету для ML з AFML-методологією.

Два режими лейблінгу:
  1. `mode='triple_barrier'` (рекомендовано, AFML Ch.3):
       - Profit-Take / Stop-Loss бар'єри на основі ATR-волатильності
       - Вертикальний бар'єр (max holding)
       - Повертає sample_weights (uniqueness + time-decay, AFML Ch.4)
       - Додає frac_diff фічі (AFML Ch.5) для збереження пам'яті

  2. `mode='horizon'` (legacy, простий):
       - Фіксований горизонт `horizon` барів
       - Рівні ваги

Без lookahead: фічі на барі t (закриття t) → таргет з t+1 до t+holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.features.indicators import add_standard_features, cvd_from_trades
from scalper_hft.ml.frac_diff import frac_diff_features
from scalper_hft.ml.labeling import label_from_ohlcv
from scalper_hft.ml.sample_weights import compute_sample_weights

# ── Список фіч ──────────────────────────────────────────────────────────────

_FEATURE_COLS_BASE = [
    "rsi_14",
    "ema_9",
    "ema_21",
    "ema_50",
    "atr_14",
    "bb_width",
    "vwap_20",
    "realized_vol_30",
    "ret_1",
    "ret_5",
    "vol_ratio",
    "cvd_mom",
    "buy_ratio",
]

# frac_diff фічі додаються динамічно (fd_close, fd_volume)
_FRAC_DIFF_COLS = ["fd_close", "fd_volume"]

# Спринт 2–3: мікроструктура (AFML Ch.19), HMM-режими (FSPML Ch.4.5), GARCH (FSPML Ch.7)
_MICRO_COLS = ["vpin", "kyle_t", "roll_spread", "amihud", "parkinson_vol", "corwin_schultz_spread", "signed_flow_ac"]
_HMM_COLS = ["hmm_state", "hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"]
_GARCH_COLS = ["garch_sigma"]
# Глибина стакана: depth5-снапшоти (depth_*) та bookTicker-attach (book_*)
_DEPTH_COLS = ["depth_imb", "depth_imb_l1", "depth_spread_bps"]
_BOOK_COLS = ["book_imb", "book_imb_chg"]


def _build_features(
    df: pd.DataFrame,
    trades: pd.DataFrame | None,
    add_frac: bool,
    frac_d: float,
    add_micro: bool = True,
    add_hmm: bool = False,
    add_garch: bool = False,
    hmm_states: int = 3,
    depth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Будує матрицю фіч (без lookahead)."""
    f = add_standard_features(df)

    # bookTicker imbalance, прикріплений до klines (load_research_data)
    if "imbalance" in df.columns:
        f["book_imb"] = df["imbalance"].astype(float).ffill().fillna(0.0)
        f["book_imb_chg"] = f["book_imb"].diff().fillna(0.0)

    # depth5-снапшоти стакана (VPS-рекордер) → барові фічі (ffill, без lookahead)
    if depth is not None and not depth.empty:
        from scalper_hft.features.depth_features import (
            depth_imbalance,
            depth_spread_bps,
            top_of_book_imbalance,
        )

        f["depth_imb"] = depth_imbalance(depth).reindex(f.index, method="ffill").fillna(0.0)
        f["depth_imb_l1"] = top_of_book_imbalance(depth).reindex(f.index, method="ffill").fillna(0.0)
        f["depth_spread_bps"] = depth_spread_bps(depth).reindex(f.index, method="ffill").fillna(0.0)

    # CVD з тікових угод (якщо є)
    if trades is not None and not trades.empty:
        cvd = cvd_from_trades(trades, resample=_infer(df))
        f = f.join(cvd[["cvd_mom", "buy_ratio"]], how="left")
        f["cvd_mom"] = f["cvd_mom"].ffill().fillna(0.0)
        f["buy_ratio"] = f["buy_ratio"].ffill().fillna(0.5)
    else:
        f["cvd_mom"] = 0.0
        f["buy_ratio"] = 0.5

    # Мікроструктурні фічі (AFML Ch.19) — лише з потоком угод
    if add_micro and trades is not None and not trades.empty:
        from scalper_hft.features.microstructure import add_microstructure_features

        micro = add_microstructure_features(df, trades, resample=_infer(df))
        for col in _MICRO_COLS:
            if col in micro.columns:
                f[col] = micro[col].reindex(f.index).ffill().fillna(0.0)

    # HMM-режими (FSPML Ch.4.5) — каузальна версія (без lookahead)
    if add_hmm:
        from scalper_hft.features.hmm_regime import hmm_regime_features

        hmm = hmm_regime_features(df["close"], n_states=hmm_states, causal=True, fit_window=2000)
        for col in _HMM_COLS:
            if col in hmm.columns:
                f[col] = hmm[col].reindex(f.index).fillna(0.0)

    # GARCH σ_{t+1} (FSPML Ch.7–8) — rolling прогноз (без lookahead)
    if add_garch:
        from scalper_hft.features.volatility import garch_forecast

        f["garch_sigma"] = (
            garch_forecast(f["close"].pct_change().fillna(0.0), window=500, refit_every=100, warmup=50)
            .reindex(f.index)
            .fillna(0.0)
        )

    # Fractional Differentiation (AFML Ch.5)
    if add_frac:
        f = frac_diff_features(f, cols=["close", "volume"], d=frac_d)
        # ffill NaN у warmup-зоні (не вносить lookahead — це константний decay)
        for col in _FRAC_DIFF_COLS:
            if col in f.columns:
                f[col] = f[col].ffill().fillna(0.0)

    return f


def build_labeled_dataset(
    df: pd.DataFrame,
    horizon: int = 3,
    noise_threshold: float = 0.0005,
    trades: pd.DataFrame | None = None,
    mode: str = "triple_barrier",
    # triple_barrier параметри
    pt: float = 1.0,
    sl: float = 1.0,
    holding_bars: int | None = None,
    vol_span: int = 100,
    decay: float = 0.9,
    # frac diff
    add_frac_diff: bool = True,
    frac_d: float = 0.4,
    # Спринт 3: micro/HMM/GARCH фічі
    add_micro: bool = True,
    add_hmm: bool = False,
    add_garch: bool = False,
    hmm_states: int = 3,
    event_filter: str = "all",
    cusum_threshold: float = 0.005,
    bar_type: str = "time",
    bar_threshold: float = 100_000.0,
    return_t1: bool = False,
    depth: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None] | tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.Series]:
    """Будує (X, y, sample_weights) для ML навчання.

    Args:
        df: OHLCV DataFrame з DatetimeIndex.
        horizon: горизонт для 'horizon' режиму (барів).
        noise_threshold: поріг фільтрації шуму для 'horizon' режиму.
        trades: aggTrades для CVD/micro-фіч (опціонально).
        mode: 'triple_barrier' (AFML) або 'horizon' (legacy).
        pt: profit-take множник (× ATR) для triple_barrier.
        sl: stop-loss множник (× ATR) для triple_barrier.
        holding_bars: максимальний горизонт тримання (default = horizon).
        vol_span: EWM span для волатильності (для triple_barrier).
        decay: time-decay коефіцієнт для ваг (1.0 = без decay).
        add_frac_diff: чи додавати FFD фічі.
        frac_d: ступінь frac diff.
        add_micro: мікроструктурні фічі VPIN/Kyle/Roll/... (AFML Ch.19; з trades).
        add_hmm: HMM-режими (каузальні, без lookahead).
        add_garch: GARCH σ_{t+1} (без lookahead).
        hmm_states: кількість HMM-станів (якщо add_hmm).
        depth: depth5-снапшоти стакана (data/{SYMBOL}_depth5.parquet) —
            барові depth_imb/depth_spread фічі (ffill, без lookahead).

    Returns:
        (X, y, w) або (X, y, w, t1) при return_t1=True:
            X — матриця фіч (pd.DataFrame)
            y — лейбли {-1, +1} (pd.Series)
            w — sample_weights (pd.Series або None для 'horizon' режиму)
            t1 — час завершення лейбла кожного зразка (для AFML purge у CV/WF)
    """
    if holding_bars is None:
        holding_bars = horizon

    if bar_type in {"dollar", "volume"} and trades is not None and not trades.empty:
        from scalper_hft.data.bars import create_dollar_bars, create_volume_bars

        if bar_type == "dollar":
            df = create_dollar_bars(trades, bar_threshold)
        else:
            df = create_volume_bars(trades, bar_threshold)
        if df is None or df.empty:
            raise ValueError(f"Не вдалося побудувати {bar_type} бари")

    f = _build_features(
        df,
        trades,
        add_frac=add_frac_diff,
        frac_d=frac_d,
        add_micro=add_micro,
        add_hmm=add_hmm,
        add_garch=add_garch,
        hmm_states=hmm_states,
        depth=depth,
    )

    if mode == "triple_barrier":
        X, y, w, t1 = _build_triple_barrier(
            df=df,
            f=f,
            pt=pt,
            sl=sl,
            holding_bars=holding_bars,
            vol_span=vol_span,
            decay=decay,
            event_filter=event_filter,
            cusum_threshold=cusum_threshold,
        )
    elif mode == "horizon":
        X, y, w, t1 = _build_horizon(df=df, f=f, horizon=horizon, noise_threshold=noise_threshold)
    else:
        raise ValueError(f"mode має бути 'triple_barrier' або 'horizon', отримано: {mode!r}")
    if return_t1:
        return X, y, w, t1
    return X, y, w


# ── Triple Barrier (AFML) ────────────────────────────────────────────────────


def _build_triple_barrier(
    df: pd.DataFrame,
    f: pd.DataFrame,
    pt: float,
    sl: float,
    holding_bars: int,
    vol_span: int,
    decay: float,
    event_filter: str = "all",
    cusum_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    close = df["close"]
    events = label_from_ohlcv(
        df,
        pt=pt,
        sl=sl,
        holding_bars=holding_bars,
        vol_span=vol_span,
        event_filter=event_filter,
        cusum_threshold=cusum_threshold,
    )
    # відкидаємо timeout-події (label=0) — занадто невизначені
    events = events[events["label"] != 0].dropna(subset=["label"])

    if events.empty:
        raise ValueError(
            "Triple-barrier labeling не дав жодної події. "
            "Спробуйте: зменшити pt/sl, збільшити holding_bars або датасет."
        )

    # ваги (uniqueness + time-decay)
    w = compute_sample_weights(events, close, decay=decay)

    # вибираємо фічі лише для барів подій
    feat_cols = _get_feat_cols(f)
    X = f[feat_cols].reindex(events.index).dropna(how="all")
    common = X.index.intersection(events.index)
    X = X.loc[common]
    y = events.loc[common, "label"].astype(int)
    w = w.reindex(common).fillna(0.0)
    # t1 (час завершення лейбла) — для AFML purge у walk-forward/CV
    t1 = events.loc[common, "t1"]

    # нормалізуємо ваги до [0, 1]
    if w.sum() > 0:
        w = w / w.sum()

    return X, y, w, t1


# ── Horizon (legacy) ─────────────────────────────────────────────────────────


def _build_horizon(
    df: pd.DataFrame,
    f: pd.DataFrame,
    horizon: int,
    noise_threshold: float,
) -> tuple[pd.DataFrame, pd.Series, None, pd.Series]:
    close = df["close"]
    fwd = close.shift(-horizon) / close - 1.0
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd > noise_threshold] = 1
    y[fwd < -noise_threshold] = -1

    feat_cols = _get_feat_cols(f)
    X = f[feat_cols].iloc[:-horizon]
    y = y.iloc[:-horizon]
    mask = y != 0
    # t1: бар, на якому завершується forward-горизонт (для purge)
    pos = np.arange(len(df) - horizon) + horizon
    t1 = pd.Series(df.index[pos], index=df.index[:-horizon])
    return X[mask], y[mask], None, t1.reindex(X[mask].index)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_feat_cols(f: pd.DataFrame) -> list[str]:
    """Повертає доступні фічі з пріоритетом frac_diff/micro/HMM/GARCH/depth."""
    cols = list(_FEATURE_COLS_BASE)
    for c in _FRAC_DIFF_COLS + _MICRO_COLS + _HMM_COLS + _GARCH_COLS + _DEPTH_COLS + _BOOK_COLS:
        if c in f.columns:
            cols.append(c)
    return [c for c in cols if c in f.columns]


def _infer(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "1min"
    delta = df.index[1] - df.index[0]
    minutes = delta.total_seconds() / 60.0
    return f"{int(minutes)}min" if minutes >= 1 else f"{int(max(1, minutes * 60))}s"
