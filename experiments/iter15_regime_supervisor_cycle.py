"""iter15 (RS-1) — чесний цикл оцінки regime-switching політик.

Pre-registration: docs/reports/hypothesis_iter15_rs.md
План циклу:      docs/reports/regime_supervisor_research.md

Що робить:
    1. Для кожного (символ, ТФ) рахує NET OOS-дохідності 5 рукавів (walk-forward,
       двигун сам лагає сигнал на 1 бар, комісії/slippage включені).
    2. Будує каузальні мітки стану ринку трьома детекторами (rule / vol / market).
    3. Оцінює 7 зареєстрованих політик + oracle на 4 фолдах: політика навчається
       ЛИШЕ на барах, що строго передують фолду; рішення на барі t використовує
       мітку стану на t-1 (ЛАГ НА РІШЕННЯ — граблі iter7, див. apply_choice_map).
    4. Агрегує портфельно (рівновага по символах), рахує t_Newey–West, stationary
       bootstrap CI, PBO (CSCV), DSR (чесний n_trials=63), switch/turnover.
    5. Оцінює pre-registered гейт на validation half і пише results/iter15/*.

Запуск:
    UV_CACHE_DIR=$PWD/.uvcache uv run python experiments/iter15_regime_supervisor_cycle.py \
        --intervals 1d,4h,1h --days 2500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter15"
SLEEVE_DIR = OUT / "sleeves"

CORE15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "UNIUSDT", "NEARUSDT", "DOTUSDT", "ATOMUSDT", "LTCUSDT", "AAVEUSDT",
]

# Рукави (5): incumbent-рукав першим.
SLEEVES: list[tuple[str, str, dict]] = [
    ("ts_long", "ts_momentum", {"lookback": 20, "allow_short": False}),
    ("ts_ls", "ts_momentum", {"lookback": 20, "allow_short": True}),
    ("cs_mom", "cross_momentum", {"lookback": 10}),
    ("carry", "funding_carry", {}),
    ("supertrend", "supertrend", {}),
]
SLEEVE_NAMES = [s[0] for s in SLEEVES]
INCUMBENT = 0

DETECTORS = ["det_rule", "det_vol", "det_mkt"]
POLICIES = [
    "incumbent", "equal", "best_single_train", "argmax", "gap_dwell", "soft_shrink",
    "riskoff_anchor", "oracle",
]
N_FOLDS = 5          # фолд 0 — burn-in для політики, 1..4 — оцінювані
MIN_COND_BARS = 30   # мінімум барів стану в train, щоб оцінювати умовний Sharpe
MIN_TRADES_CELL = 30
N_TRIALS_REGISTERED = 63  # 7 політик × 3 детектори × 3 ТФ (pre-registered)
NEEDS_FUNDING = True
GAP_DEFAULT = 0.5
DWELL_DEFAULT = 5
TAU_DEFAULT = 1.0
SHRINK_DEFAULT = 0.5
RISK_OFF_SCALE = 0.25

PPY = {"1h": 24.0 * 365.0, "4h": 6.0 * 365.0, "1d": 365.0}
WF_BARS = {"1h": (250, 125), "4h": (250, 125), "1d": (250, 125)}
DEFAULT_DAYS = {"1h": 1095, "4h": 2500, "1d": 2500}


# ─────────────────────────── статистика ───────────────────────────


def sharpe(r: pd.Series | np.ndarray, ppy: float, min_bars: int = MIN_COND_BARS) -> float:
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    if len(x) < min_bars:
        return float("nan")
    sd = float(x.std(ddof=0))
    if sd < 1e-12:
        return 0.0
    return float(x.mean() / sd * math.sqrt(ppy))


def max_dd(r: pd.Series) -> float:
    eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def newey_west_t(returns: np.ndarray, lags: int = 20) -> float:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return 0.0
    mu = float(r.mean())
    dev = r - mu
    var = float((dev @ dev) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float((dev[lag:] @ dev[:-lag]) / n)
    if var <= 0:
        return 0.0
    return float(mu / math.sqrt(var / n))


def bootstrap_sharpe_ci(r: np.ndarray, ppy: float, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(r, dtype=float)
    n = len(x)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    p = 1.0 / 10.0
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(n))
        for k in range(n):
            idx[k] = i
            i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
        s = x[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * math.sqrt(ppy) if sd > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ─────────────────────────── дані ───────────────────────────


def load_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Klines для ТФ: нативний довгий файл (1d/4h/1w) або з 1m-бази (1h і нижче).

    Пастка: `klines_from_store(..., base_interval=interval)` повертає `None`, якщо
    нативного файлу ТФ немає (або файл обрізаний, як SOLUSDT 1h на 37 барів), і
    `ensure_klines(..., readonly=True)` виводить ТФ із 1m-бази, яка в кеші покриває
    ~1095 днів. Тому перевіряємо довжину і маємо три рівні фолбеку.
    """
    from scalper_hft.data.access import ensure_klines, klines_from_store

    need = {"1d": days, "4h": days * 6, "1w": days // 7}.get(interval, days * 24)
    if interval in ("1d", "4h", "1w"):
        try:
            df = klines_from_store(symbol, interval, days, base_interval=interval)
            if df is not None and len(df) >= 0.5 * need:
                return df
        except Exception:  # noqa: BLE001
            pass
    try:
        df = ensure_klines(symbol, interval, days, derive=False, readonly=True)
        if df is not None and len(df) >= 0.5 * need:
            return df
    except Exception:  # noqa: BLE001
        pass
    return ensure_klines(symbol, interval, days, derive=False)


def symbol_regimes(df: pd.DataFrame, detector: str, mkt_states: pd.Series | None) -> pd.Series:
    """Каузальні мітки стану (обчислені на закритих барах)."""
    from scalper_hft.features.regimes import market_structure, volatility_regime

    if detector == "det_rule":
        return market_structure(df["close"]).astype(str)
    if detector == "det_vol":
        return volatility_regime(df["close"]).astype(str)
    if detector == "det_mkt":
        if mkt_states is None:
            return pd.Series("risk_on", index=df.index, dtype=object)
        return mkt_states.reindex(df.index).ffill().fillna("risk_on").astype(str)
    raise ValueError(detector)


def market_states(interval: str, days: int) -> pd.Series:
    """Ринковий risk-on/risk-off: BTC 1d close > EMA(200) і перцентиль rv(60) < 0.8."""
    btc = load_klines("BTCUSDT", "1d", days)
    close = btc["close"].astype(float)
    ema = close.ewm(span=200, adjust=False).mean()
    rv = np.log(close / close.shift(1)).rolling(60, min_periods=30).std()
    pct = rv.rolling(500, min_periods=100).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    on = (close > ema) & (pct < 0.8)
    out = pd.Series("risk_off", index=btc.index, dtype=object)
    out.loc[on] = "risk_on"
    return out


def sleeve_returns(
    symbol: str,
    interval: str,
    days: int,
    is_maker: bool,
    funding: pd.DataFrame | None,
    cache: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """NET OOS-дохідності рукавів (walk-forward, collect_oos_returns=True)."""
    key = (symbol, interval, days, is_maker)
    if key in cache:
        return cache[key]
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = load_klines(symbol, interval, days)
    train, test = WF_BARS[interval]
    cost = CostModel.from_settings(get_settings())
    series: dict[str, pd.Series] = {}
    trades: dict[str, int] = {}
    for label, strat_name, params in SLEEVES:
        strat = get_strategy(strat_name, **params)
        try:
            wf = run_walk_forward(
                df,
                strat,
                train_bars=train,
                test_bars=test,
                cost=cost,
                funding=funding if getattr(strat, "needs_funding", False) else None,
                is_maker=is_maker,
                interval=interval,
                collect_oos_returns=True,
                purge_bars=2,
                embargo_bars=2,
                strict_data=False,
            )
            r = wf.oos_returns.astype(float)
            r = r[~r.index.duplicated(keep="first")]
            series[label] = r
            trades[label] = int(sum(w.n_trades for w in wf.windows))
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {symbol} {interval} {label}: {exc}")
            series[label] = pd.Series(dtype=float)
            trades[label] = 0
    mat = pd.DataFrame(series).dropna(how="all")
    mat = mat.reindex(columns=SLEEVE_NAMES)
    # Кеш на диск: sensitivity/нові політики не мають перераховувати walk-forward.
    SLEEVE_DIR.mkdir(parents=True, exist_ok=True)
    mode = "maker" if is_maker else "taker"
    if not mat.empty:
        mat.to_parquet(SLEEVE_DIR / f"{interval}_{mode}_{symbol}.parquet")
    cache[key] = (mat, trades)
    return mat, trades


# ─────────────────────────── політики ───────────────────────────


def cond_sharpe_table(R: pd.DataFrame, state_dec: pd.Series, ppy: float) -> pd.DataFrame:
    """Умовний NET Sharpe рукавів по станах (стан = мітка бару t-1 → рішення бару t)."""
    states = sorted(set(state_dec.dropna().astype(str)))
    rows = {}
    for st in states:
        mask = (state_dec.astype(str) == st).to_numpy()
        vals = {}
        for col in R.columns:
            x = R[col].to_numpy(dtype=float)[mask]
            vals[col] = sharpe(x, ppy) if int(np.isfinite(x).sum()) >= MIN_COND_BARS else float("nan")
        rows[st] = vals
    rows["__ALL__"] = {c: sharpe(R[c].to_numpy(dtype=float), ppy) for c in R.columns}
    return pd.DataFrame(rows).T


def _weights_for(
    policy: str,
    cond: pd.DataFrame,
    state: str,
    cur: int | None,
    gap: float,
    dwell: int,
    tau: float,
    shrink: float,
) -> tuple[np.ndarray, int | None, str]:
    """Ваги на один бар. Повертає (w, новий cur, дія)."""
    n = len(SLEEVE_NAMES)
    row = cond.loc[state] if state in cond.index else cond.loc["__ALL__"]
    vals = row.astype(float).to_numpy()

    if policy == "incumbent":
        w = np.zeros(n)
        w[INCUMBENT] = 1.0
        return w, INCUMBENT, "hold"

    if policy == "equal":
        return np.full(n, 1.0 / n), None, "hold"

    finite = np.where(np.isfinite(vals), vals, -np.inf)
    best = int(np.argmax(finite))
    best_v = float(finite[best]) if np.isfinite(finite[best]) else -np.inf

    if policy in ("best_single_train", "argmax", "gap_dwell"):
        if policy == "best_single_train":
            row_all = cond.loc["__ALL__"].astype(float).to_numpy()
            finite_all = np.where(np.isfinite(row_all), row_all, -np.inf)
            best = int(np.argmax(finite_all))
            best_v = float(finite_all[best]) if np.isfinite(finite_all[best]) else -np.inf
        desired = best if best_v > 0 else None
        if policy != "gap_dwell":
            w = np.zeros(n)
            if desired is not None:
                w[desired] = 1.0
            return w, desired, "switch" if desired != cur else "hold"

        # gap_dwell: потрібен розрив Sharpe і підтвердження N барів
        if desired == cur:
            return _onehot(n, cur), cur, "hold"
        cur_v = float(vals[cur]) if cur is not None and np.isfinite(vals[cur]) else 0.0
        if desired is None:
            if cur is not None and cur_v <= -gap:
                return np.zeros(n), None, "exit"
            return _onehot(n, cur), cur, "hold"
        cand_v = float(vals[desired]) if np.isfinite(vals[desired]) else -np.inf
        if cand_v - cur_v < gap:
            return _onehot(n, cur), cur, "hold"
        # підтвердження виконується зовні (через dwell-лічильник у run_policy)
        if dwell <= 1:
            return _onehot(n, desired), desired, "switch"
        return _onehot(n, cur), cur, f"cand:{desired}"

    if policy == "soft_shrink":
        v = np.where(np.isfinite(vals), vals, np.nanmean(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0)
        z = v / max(tau, 1e-6)
        z = z - z.max()
        soft = np.exp(z)
        soft = soft / soft.sum()
        w = shrink * soft + (1.0 - shrink) * (1.0 / n)
        return w, int(np.argmax(w)), "hold"

    if policy == "riskoff_anchor":
        risk_on = state in ("risk_on", "trend_up", "low", "normal")
        if risk_on:
            return np.full(n, 1.0 / n), None, "hold"
        w = np.zeros(n)
        w[INCUMBENT] = RISK_OFF_SCALE
        return w, INCUMBENT, "de-risk"

    raise ValueError(policy)


def _onehot(n: int, idx: int | None) -> np.ndarray:
    w = np.zeros(n)
    if idx is not None:
        w[idx] = 1.0
    return w


def run_policy(
    R: pd.DataFrame,
    state_dec: pd.Series,
    policy: str,
    folds: list[tuple[int, int]],
    ppy: float,
    gap: float = GAP_DEFAULT,
    dwell: int = DWELL_DEFAULT,
    tau: float = TAU_DEFAULT,
    shrink: float = SHRINK_DEFAULT,
    pool: list[tuple[pd.DataFrame, pd.Series]] | None = None,
) -> dict:
    """Ковзний прогон політики: навчання на барах < fold_start, застосування у фолді.

    pool: додаткові (returns, state_dec) ІНШИХ символів — умовна статистика
    рахується на об'єднанні (cross-symbol transfer, HB-11). Train-вікно для пулу
    визначається за ЧАСОМ (`index < R.index[f_start]`), а не за позицією.
    """
    n = len(R)
    pnl = np.zeros(n)
    w_prev = np.zeros(len(SLEEVE_NAMES))
    switch_bars = 0
    turnover = 0.0
    per_fold: list[dict] = []
    cur: int | None = None
    cand: int | None = None
    run_len = 0

    for fold_idx, (f_start, f_end) in enumerate(folds):
        train = np.arange(0, f_start)
        test = np.arange(f_start, f_end)
        if len(train) < MIN_COND_BARS or len(test) == 0:
            continue
        cond_R = R.iloc[train]
        cond_sd = state_dec.iloc[train]
        if pool:
            cutoff = R.index[f_start]
            frames_R = [cond_R]
            frames_sd = [cond_sd]
            for pR, pS in pool:
                mask = pR.index < cutoff
                if int(mask.sum()) >= MIN_COND_BARS:
                    frames_R.append(pR[mask])
                    frames_sd.append(pS[mask])
            cond_R = pd.concat(frames_R)
            cond_sd = pd.concat(frames_sd)
        cond = cond_sharpe_table(cond_R, cond_sd, ppy)

        if policy == "oracle":
            # ex-post верхня межа: найкращий рукав на САМОМУ тесті (діагностика)
            orow = {c: sharpe(R[c].to_numpy(dtype=float)[test], ppy) for c in R.columns}
            best = max(orow, key=lambda k: (orow[k] if np.isfinite(orow[k]) else -np.inf))
            w_fold = np.zeros(len(SLEEVE_NAMES))
            if np.isfinite(orow[best]) and orow[best] > 0:
                w_fold[SLEEVE_NAMES.index(best)] = 1.0
            pnl[test] = R.to_numpy(dtype=float)[test] @ w_fold
            per_fold.append({"fold": fold_idx, "sharpe": sharpe(pnl[test], ppy), "n_bars": len(test)})
            continue

        for t in test:
            st = str(state_dec.iloc[t])
            if policy == "gap_dwell":
                w, cur_new, action = _weights_for(policy, cond, st, cur, gap, dwell, tau, shrink)
                if action.startswith("cand:"):
                    target = int(action.split(":")[1])
                    if cand == target:
                        run_len += 1
                    else:
                        cand, run_len = target, 1
                    if run_len >= dwell:
                        w = _onehot(len(SLEEVE_NAMES), target)
                        cur, cand, run_len = target, None, 0
                        action = "switch"
                    else:
                        w = _onehot(len(SLEEVE_NAMES), cur)
                else:
                    cand, run_len = None, 0
                    cur = cur_new
            else:
                w, cur_new, action = _weights_for(policy, cond, st, cur, gap, dwell, tau, shrink)
                cur = cur_new
            if action in ("switch", "exit", "de-risk"):
                switch_bars += 1
            turnover += float(np.abs(w - w_prev).sum())
            w_prev = w
            pnl[t] = float(R.to_numpy(dtype=float)[t] @ w)

        per_fold.append({"fold": fold_idx, "sharpe": sharpe(pnl[test], ppy), "n_bars": len(test)})

    return {
        "pnl": pd.Series(pnl, index=R.index),
        "switch_bars": switch_bars,
        "turnover": turnover,
        "per_fold": per_fold,
    }


# ─────────────────────────── IC детекторів ───────────────────────────


def detector_ic(R: pd.DataFrame, state_dec: pd.Series, folds: list[tuple[int, int]], ppy: float, h: int = 1) -> float:
    """IC: чи передбачає стан на t відносний PnL «обраного в стані» рукава на t+1..t+h (train-only).

    Передбачення: edge(r_t) = μ[best(r) | r] − μ_mean(r) (з train-статистики стану).
    Ціль: фактичний відносний дохід обраного рукава проти рівноваги на наступних h барах.
    """
    xs: list[float] = []
    ys: list[float] = []
    rets = R.to_numpy(dtype=float)
    for f_start, _f_end in folds:
        train = np.arange(0, f_start)
        if len(train) < MIN_COND_BARS * 2:
            continue
        cond = cond_sharpe_table(R.iloc[train], state_dec.iloc[train], ppy)
        eq = np.nanmean(rets[train], axis=1)
        for t in train[:-h]:
            st = str(state_dec.iloc[t])
            row = cond.loc[st] if st in cond.index else cond.loc["__ALL__"]
            vals = row.astype(float).to_numpy()
            finite = np.where(np.isfinite(vals), vals, -np.inf)
            best = int(np.argmax(finite))
            if not np.isfinite(finite[best]):
                continue
            mu_best = float(finite[best])
            mu_eq = float(np.nanmean(vals[np.isfinite(vals)]))
            pred = mu_best - mu_eq
            rel = float(np.nanmean(rets[t + 1 : t + 1 + h, best]) - np.nanmean(eq[t + 1 : t + 1 + h]))
            xs.append(pred)
            ys.append(rel)
    if len(xs) < 50:
        return float("nan")
    x = np.asarray(xs)
    y = np.asarray(ys)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ─────────────────────────── основний прогін ───────────────────────────


def compute_folds(index: pd.DatetimeIndex, n_folds: int = N_FOLDS) -> list[tuple[int, int]]:
    n = len(index)
    bounds = [int(round(i * n / n_folds)) for i in range(n_folds + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_folds)]


def selfcheck() -> int:
    """Mutation-тест ЛАГА: стан ідеально передбачає ПРОТИЛЕЖНИЙ рукав того ж бару.

    Синтетика: `s_a` чергує ±0.002 щобарно, `s_b = −s_a`, а мітка стану на барі t
    дорівнює «A_win», якщо `s_a[t] > 0` (тобто мітка того ж бару ідеально знає
    переможця ТОГО Ж бару). Правильна політика (рішення за міткою t−1) зобов'язана
    втрачати гроші (обирає вчорашнього переможця = сьогоднішнього лузера).
    Якщо хтось прибере `shift(1)` — знак PnL перевернеться на плюс, і тест впаде.
    """
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Переможець бару — i.i.d. випадковий; мітка стану = переможець ТОГО Ж бару.
    # Тому мітка бару t не передбачає переможця бару t+1 (незалежність), і:
    #   лаг на рішенні → E[PnL] = 0;
    #   без лага (leak) → E[PnL] = +0.003/бар.
    # Умови станів мають ненульову σ, інакше конвенція sharpe()=0 дає flat.
    rng = np.random.default_rng(7)
    win = rng.integers(0, 2, n).astype(bool)
    s_a = np.where(win, 0.003, -0.003) + rng.normal(0.0, 0.001, n)
    s_b = np.where(win, -0.003, 0.003) - rng.normal(0.0, 0.001, n)
    R = pd.DataFrame({"ts_long": s_a, "ts_ls": s_b, "cs_mom": 0.0, "carry": 0.0, "supertrend": 0.0}, index=idx)
    labels = pd.Series(np.where(win, "A_win", "B_win"), index=idx)
    folds = compute_folds(idx)
    res = run_policy(R, labels.shift(1), "argmax", folds, ppy=365.0)
    pnl = float(res["pnl"].sum())
    res_leak = run_policy(R, labels, "argmax", folds, ppy=365.0)
    leak = float(res_leak["pnl"].sum())
    ok = abs(pnl) < 0.25 and leak > 1.0
    print(f"selfcheck: pnl_lagged={pnl:+.4f} (має бути ≈0), pnl_no_lag={leak:+.4f} (має бути >1)")
    print("selfcheck:", "PASS ✅ лаг на рішенні застосовано" if ok else "FAIL ❌")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="1d,4h,1h")
    ap.add_argument("--symbols", default=",".join(CORE15))
    ap.add_argument("--days", type=int, default=0, help="0 = дефолт на ТФ")
    ap.add_argument("--cost-modes", default="maker", help="maker,taker — база витрат (taker = cost-sensitivity)")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true", help="лише mutation-тест лага")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    cost_modes = [c.strip() for c in args.cost_modes.split(",") if c.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    from scalper_hft.data.store import get_store
    from scalper_hft.strategies import get_strategy

    global NEEDS_FUNDING
    NEEDS_FUNDING = any(bool(getattr(get_strategy(n, **p), "needs_funding", False)) for _l, n, p in SLEEVES)
    store = get_store()
    cache: dict = {}
    mkt_cache: dict[str, pd.Series] = {}

    summary_rows: list[dict] = []
    per_symbol_rows: list[dict] = []
    per_fold_rows: list[dict] = []
    ic_rows: list[dict] = []
    port_store: dict[str, pd.DataFrame] = {}
    sleeve_trade_rows: list[dict] = []

    for cost_mode, interval in [(c, i) for c in cost_modes for i in intervals]:
        is_maker = cost_mode == "maker"
        days = args.days or DEFAULT_DAYS[interval]
        ppy = PPY[interval]
        if interval not in mkt_cache:
            mkt_cache[interval] = market_states(interval, max(days, 2500))
        mkt = mkt_cache[interval]

        per_symbol_pnl: dict[str, dict[str, pd.Series]] = {}
        sleeve_info: dict[str, dict] = {}

        for symbol in symbols:
            funding = None
            if NEEDS_FUNDING:
                try:
                    funding = store.load_funding(symbol)
                except Exception:  # noqa: BLE001
                    funding = None
            R, trades = sleeve_returns(symbol, interval, days, is_maker, funding, cache)
            if R.empty or len(R) < N_FOLDS * MIN_COND_BARS:
                print(f"  skip {symbol} {interval}: мало даних ({len(R)})")
                continue
            df = load_klines(symbol, interval, days)
            states = symbol_regimes(df, "det_rule", mkt).reindex(R.index)
            states_vol = symbol_regimes(df, "det_vol", mkt).reindex(R.index)
            states_mkt = symbol_regimes(df, "det_mkt", mkt).reindex(R.index)
            states_by_det = {"det_rule": states, "det_vol": states_vol, "det_mkt": states_mkt}
            state_dec = {k: v.shift(1) for k, v in states_by_det.items()}

            folds = compute_folds(R.index)
            sleeve_info[symbol] = trades
            for label, n_tr in trades.items():
                sleeve_trade_rows.append(
                    {
                        "symbol": symbol,
                        "cost_mode": cost_mode,
                        "interval": interval,
                        "sleeve": label,
                        "n_trades": n_tr,
                    }
                )

            per_symbol_pnl.setdefault(symbol, {})
            variants: list[tuple[str, str]] = [("incumbent", "-"), ("equal", "-"), ("best_single_train", "-"), ("oracle", "-")]
            for det in DETECTORS:
                for pol in ("argmax", "gap_dwell", "soft_shrink", "riskoff_anchor"):
                    variants.append((pol, det))

            for pol, det in variants:
                key = pol if det == "-" else f"{pol}@{det}"
                sd = state_dec[det] if det != "-" else state_dec["det_rule"]
                res = run_policy(R, sd, pol, folds, ppy)
                per_symbol_pnl[symbol][key] = res["pnl"]
                per_symbol_rows.append(
                    {
                        "symbol": symbol,
                        "cost_mode": cost_mode,
                        "interval": interval,
                        "variant": key,
                        "sharpe": sharpe(res["pnl"].to_numpy(), ppy, min_bars=60),
                        "maxdd": max_dd(res["pnl"]),
                        "switch_bars": res["switch_bars"],
                        "turnover": res["turnover"],
                        "n_bars": int((res["pnl"] != 0).sum()),
                    }
                )
                for pf in res["per_fold"]:
                    per_fold_rows.append(
                        {
                            "symbol": symbol,
                            "cost_mode": cost_mode,
                            "interval": interval,
                            "variant": key,
                            "fold": pf["fold"],
                            "sharpe": pf["sharpe"],
                        }
                    )

            for det in DETECTORS:
                ic_rows.append(
                    {
                        "symbol": symbol,
                        "cost_mode": cost_mode,
                        "interval": interval,
                        "detector": det,
                        "ic": detector_ic(R, state_dec[det], folds, ppy),
                    }
                )
            print(f"  ok {symbol} {interval}: bars={len(R)} sleeves={ {k: v for k, v in trades.items()} }")

        # ── агрегація по інтервалу ──
        if not per_symbol_pnl:
            continue
        keys = sorted({k for d in per_symbol_pnl.values() for k in d})
        port_mat = {}
        for key in keys:
            series_by_sym = {s: d[key] for s, d in per_symbol_pnl.items() if key in d}
            mat = pd.DataFrame(series_by_sym).sort_index()
            port_mat[key] = mat.mean(axis=1).dropna()
        port_df = pd.DataFrame(port_mat)
        port_store[(cost_mode, interval)] = port_df
        port_df.to_parquet(OUT / f"portfolio_returns_{interval}_{cost_mode}.parquet")

        n = len(port_df)
        split = n // 2
        val_index = port_df.index[split:]
        sel_index = port_df.index[:split]
        # Per-symbol × variant матриці (для честного перерахунку selection/validation)
        sym_panels = {
            key: pd.DataFrame({s: d[key] for s, d in per_symbol_pnl.items() if key in d}).sort_index()
            for key in keys
        }
        pd.concat(sym_panels, axis=1).to_parquet(OUT / f"symbol_returns_{interval}_{cost_mode}.parquet")
        for key in keys:
            s_all = port_df[key]
            s_sel = port_df[key].iloc[:split]
            s_val = port_df[key].iloc[split:]
            per_sym = pd.DataFrame(
                [
                    r
                    for r in per_symbol_rows
                    if r["interval"] == interval and r["cost_mode"] == cost_mode and r["variant"] == key
                ]
            )
            # Sharpe по символах САМЕ на validation half (не на всьому періоді)
            sym_val_sr = [
                sharpe(d[key].reindex(val_index).to_numpy(), ppy, min_bars=60)
                for s, d in per_symbol_pnl.items()
                if key in d
            ]
            sym_val_sr = [x for x in sym_val_sr if np.isfinite(x)]
            # ВИБІР переможця — ЛИШЕ за selection half (інакше вибір підглядає валідацію)
            sym_sel_sr = [
                sharpe(d[key].reindex(sel_index).to_numpy(), ppy, min_bars=60)
                for s, d in per_symbol_pnl.items()
                if key in d
            ]
            sym_sel_sr = [x for x in sym_sel_sr if np.isfinite(x)]
            folds_key = pd.DataFrame(
                [
                    r
                    for r in per_fold_rows
                    if r["interval"] == interval and r["cost_mode"] == cost_mode and r["variant"] == key
                ]
            )
            ci_lo, ci_hi = bootstrap_sharpe_ci(s_val.to_numpy(), ppy)
            q = max(len(s_val) / (ppy / (365.0 * 0.25)), 1e-9)  # кількість кварталів валідації
            summary_rows.append(
                {
                    "cost_mode": cost_mode,
                    "interval": interval,
                    "variant": key,
                    "n_symbols": len(per_sym),
                    "mean_sr_sel": float(np.mean(sym_sel_sr)) if sym_sel_sr else float("nan"),
                    "mean_sr_full": float(per_sym["sharpe"].mean()),
                    "mean_sym_sr_val": float(np.mean(sym_val_sr)) if sym_val_sr else float("nan"),
                    "frac_sym_pos_val": float(np.mean([x > 0 for x in sym_val_sr])) if sym_val_sr else float("nan"),
                    "port_sr_val": sharpe(s_val.to_numpy(), ppy, min_bars=60),
                    "port_sr_full": sharpe(s_all.to_numpy(), ppy, min_bars=60),
                    "t_nw_val": newey_west_t(s_val.to_numpy()),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "maxdd_val": max_dd(s_val),
                    "calmar_val": (
                        float(s_val.mean() * ppy / abs(max_dd(s_val))) if max_dd(s_val) < 0 else 0.0
                    ),
                    "switch_per_quarter": float(per_sym["switch_bars"].sum() / q),
                    "turnover": float(per_sym["turnover"].mean()),
                    "mean_fold_sr": float(folds_key["sharpe"].mean()) if len(folds_key) else float("nan"),
                    "min_fold_sr": float(folds_key["sharpe"].min()) if len(folds_key) else float("nan"),
                }
            )

    mode_tag = "_".join(cost_modes)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / f"summary_{mode_tag}.csv", index=False)
    pd.DataFrame(per_symbol_rows).to_csv(OUT / f"per_symbol_{mode_tag}.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(OUT / f"per_fold_{mode_tag}.csv", index=False)
    pd.DataFrame(ic_rows).to_csv(OUT / f"ic_{mode_tag}.csv", index=False)
    pd.DataFrame(sleeve_trade_rows).to_csv(OUT / f"sleeve_trades_{mode_tag}.csv", index=False)

    # ── PBO / DSR на цільовому ТФ ──
    stats: dict = {}
    for (cost_mode, interval), port_df in port_store.items():
        ppy = PPY[interval]
        tag = f"{interval}_{cost_mode}"
        try:
            from scalper_hft.validation.cscv import pbo_cscv

            # oracle — діагностична верхня межа, у PBO/вибір переможця НЕ входить
            mat = port_df.drop(columns=[c for c in port_df.columns if c == "oracle"], errors="ignore").dropna()
            if mat.shape[1] >= 3 and len(mat) > 100:
                res = pbo_cscv(mat.to_numpy().T, n_blocks=8, purge_bars=2, embargo_bars=2)
                stats[f"pbo_{tag}"] = float(getattr(res, "pbo", float("nan")))
        except Exception as exc:  # noqa: BLE001
            stats[f"pbo_{tag}_error"] = str(exc)
        sub = summary[
            (summary["interval"] == interval)
            & (summary["cost_mode"] == cost_mode)
            & (summary["variant"] != "oracle")
        ]
        if len(sub):
            best = sub.sort_values("mean_sr_sel", ascending=False).iloc[0]
            s_val = port_df[best["variant"]].iloc[len(port_df) // 2 :]
            try:
                from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio

                stats[f"dsr_{tag}"] = float(
                    deflated_sharpe_ratio(s_val.to_numpy(), n_trials=N_TRIALS_REGISTERED, sr_benchmark=0.0)
                )
            except Exception as exc:  # noqa: BLE001
                stats[f"dsr_{tag}_error"] = str(exc)
            stats[f"winner_sel_{tag}"] = str(best["variant"])
            stats[f"winner_val_sr_{tag}"] = float(best["port_sr_val"])

    (OUT / f"stats_{mode_tag}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    ic = pd.DataFrame(ic_rows)
    if not ic.empty:
        ic.groupby(["interval", "detector"])["ic"].mean().to_csv(OUT / f"ic_summary_{mode_tag}.csv")

    lines = ["# iter15 (RS-1) — результати прогону", ""]
    cols = [
        "variant", "mean_sr_sel", "mean_sym_sr_val", "port_sr_val", "t_nw_val", "ci_lo", "ci_hi",
        "frac_sym_pos_val", "maxdd_val", "turnover", "switch_per_quarter",
    ]
    if not summary.empty:
        for cost_mode in summary["cost_mode"].unique():
            for interval in summary["interval"].unique():
                sub = summary[(summary["cost_mode"] == cost_mode) & (summary["interval"] == interval)]
                if sub.empty:
                    continue
                lines.append(f"## ТФ {interval} · cost={cost_mode}")
                lines.append("")
                lines.append(sub[cols].round(3).to_markdown(index=False))
                lines.append("")
    lines.append("## PBO / DSR")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats, indent=2, ensure_ascii=False))
    lines.append("```")
    (OUT / f"report_{mode_tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nАртефакти: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
