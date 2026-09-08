"""Аудит однієї комірки (стратегія × символ × ТФ): WF + sensitivity + DSR (+ CSCV).

Критерії PASS — overfitting-audit skill:
    avg_oos_sharpe > 0.3, oos_pos_frac >= 0.5, DSR > 0.95 (на OOS-вікнах),
    smoothness > 0.30 (сітка на OOS-регіоні), n_trades_oos >= поріг таймфрейму,
    PBO < 0.5 (якщо CSCV запущено — with_cscv=True).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import pandas as pd

from scalper_hft.validation.walk_forward import WalkForwardWindow

VerdictLabel = Literal["PASS", "FAIL"]
AuditStatus = Literal["ok", "error"]

# Ті самі вікна, що у scripts/matrix_wf_sweeps.sh
WF_TRAIN_TEST: dict[str, tuple[int, int]] = {
    "1m": (4000, 2000),
    "5m": (2000, 1000),
    "15m": (1000, 500),
    "30m": (500, 250),
    "1h": (500, 200),
    "4h": (200, 100),
}
DEFAULT_TRAIN_TEST: tuple[int, int] = (2000, 500)
MIN_TRADES: dict[str, int] = {
    "1m": 100,
    "5m": 60,
    "15m": 40,
    "30m": 40,
    "1h": 30,
    "4h": 20,
}
DEFAULT_MIN_TRADES = 40
OOS_SHARPE_MIN = 0.3
OOS_POS_FRAC_MIN = 0.5
DSR_MIN = 0.95
SMOOTHNESS_MIN = 0.30
PBO_MAX = 0.5
DSR_BACKTESTS_PER_COMBO = 50
CSCV_VARIANTS = 20  # варіантів параметрів для PBO у audit_cell (with_cscv=True)

_SUMMARY_KEYS = (
    "symbol",
    "interval",
    "strategy",
    "status",
    "error",
    "n_windows",
    "avg_is_sharpe",
    "avg_oos_sharpe",
    "oos_pos_frac",
    "degradation",
    "n_trades_oos",
    "sens_param",
    "smoothness",
    "sens_n",
    "sens_error",
    "dsr",
    "n_trials_dsr",
    "pbo",
    "bt_total_return",
    "bt_sharpe",
    "bt_max_dd",
    "bt_n_trades",
    "bt_profit_factor",
    "bt_win_rate",
    "bt_trades_per_day",
)


def default_train_test(interval: str) -> tuple[int, int]:
    """Train/test барів для walk-forward за таймфреймом."""
    return WF_TRAIN_TEST.get(interval, DEFAULT_TRAIN_TEST)


def resolve_wf_windows(
    interval: str,
    train_bars: int | None = None,
    test_bars: int | None = None,
) -> tuple[int, int]:
    """Явні бари, або per-TF дефолт з `default_train_test`.

    None = узяти дефолт інтервалу. Так sweep з 1m і 4h в одній матриці
    не ставить усім train+test=3000 (4h тоді завжди error).
    """
    default_train, default_test = default_train_test(interval)
    train = default_train if train_bars is None else int(train_bars)
    test = default_test if test_bars is None else int(test_bars)
    if train <= 0 or test <= 0:
        raise ValueError(f"train/test мають бути > 0, отримано {train}/{test}")
    return train, test


def min_trades_for(interval: str) -> int:
    """Мінімум угод для статистичної значущості на цьому ТФ."""
    return MIN_TRADES.get(interval, DEFAULT_MIN_TRADES)


def _window_to_dict(w: WalkForwardWindow) -> dict[str, float | int]:
    return {
        "window_idx": w.window_idx,
        "train_start": w.train_start,
        "train_end": w.train_end,
        "test_start": w.test_start,
        "test_end": w.test_end,
        "is_sharpe": float(w.is_sharpe),
        "oos_sharpe": float(w.oos_sharpe),
        "oos_return": float(w.oos_return),
        "n_trades": int(w.n_trades),
    }


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        x = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _as_int(value: object) -> int | None:
    x = _as_float(value)
    if x is None:
        return None
    return int(x)


@dataclass(frozen=True, slots=True)
class CellAudit:
    """Результат аудиту однієї комірки (для JSON / CSV / UI)."""

    symbol: str
    interval: str
    strategy: str
    status: AuditStatus = "error"
    error: str = ""
    n_windows: int = 0
    avg_is_sharpe: float | None = None
    avg_oos_sharpe: float | None = None
    oos_pos_frac: float | None = None
    n_trades_oos: int | None = None
    degradation: float | None = None
    sens_param: str | None = None
    smoothness: float | None = None
    sens_n: int = 0
    sens_error: str | None = None
    dsr: float | None = None
    n_trials_dsr: int | None = None
    pbo: float | None = None  # CSCV PBO (лише якщо audit_cell з with_cscv=True)
    bt_total_return: float | None = None
    bt_sharpe: float | None = None
    bt_max_dd: float | None = None
    bt_n_trades: int | None = None
    bt_profit_factor: float | None = None
    bt_win_rate: float | None = None
    bt_trades_per_day: float | None = None
    windows: tuple[dict[str, float | int], ...] = ()
    sensitivity_grid: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_summary_dict(self) -> dict[str, Any]:
        """Плоский рядок без вікон/сітки (CSV матриці)."""
        d = asdict(self)
        return {k: d[k] for k in _SUMMARY_KEYS}

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CellAudit:
        status_raw = raw.get("status") or "error"
        status: AuditStatus = "ok" if status_raw == "ok" else "error"
        windows = raw.get("windows") or ()
        grid = raw.get("sensitivity_grid") or ()
        n_windows = _as_int(raw.get("n_windows")) or 0
        sens_n = _as_int(raw.get("sens_n")) or 0
        return cls(
            symbol=str(raw.get("symbol") or ""),
            interval=str(raw.get("interval") or ""),
            strategy=str(raw.get("strategy") or ""),
            status=status,
            error=str(raw.get("error") or ""),
            n_windows=n_windows,
            avg_is_sharpe=_as_float(raw.get("avg_is_sharpe")),
            avg_oos_sharpe=_as_float(raw.get("avg_oos_sharpe")),
            oos_pos_frac=_as_float(raw.get("oos_pos_frac")),
            n_trades_oos=_as_int(raw.get("n_trades_oos")),
            degradation=_as_float(raw.get("degradation")),
            sens_param=str(raw["sens_param"]) if raw.get("sens_param") else None,
            smoothness=_as_float(raw.get("smoothness")),
            sens_n=sens_n,
            sens_error=str(raw["sens_error"]) if raw.get("sens_error") else None,
            dsr=_as_float(raw.get("dsr")),
            n_trials_dsr=_as_int(raw.get("n_trials_dsr")),
            pbo=_as_float(raw.get("pbo")),
            bt_total_return=_as_float(raw.get("bt_total_return")),
            bt_sharpe=_as_float(raw.get("bt_sharpe")),
            bt_max_dd=_as_float(raw.get("bt_max_dd")),
            bt_n_trades=_as_int(raw.get("bt_n_trades")),
            bt_profit_factor=_as_float(raw.get("bt_profit_factor")),
            bt_win_rate=_as_float(raw.get("bt_win_rate")),
            bt_trades_per_day=_as_float(raw.get("bt_trades_per_day")),
            windows=tuple(windows),
            sensitivity_grid=tuple(grid),
        )


def cell_verdict(audit: CellAudit | Mapping[str, Any]) -> tuple[VerdictLabel, str]:
    """(PASS/FAIL, причини) за критеріями overfitting-audit."""
    row: Mapping[str, Any] = audit.to_summary_dict() if isinstance(audit, CellAudit) else audit
    interval = str(row.get("interval") or "")
    reasons: list[str] = []

    oos = _as_float(row.get("avg_oos_sharpe"))
    if oos is None or oos <= OOS_SHARPE_MIN:
        shown = "nan" if oos is None else f"{oos:.3f}"
        reasons.append(f"avg_oos_sharpe={shown}≤{OOS_SHARPE_MIN}")

    frac = _as_float(row.get("oos_pos_frac"))
    if frac is None or frac < OOS_POS_FRAC_MIN:
        shown = "nan" if frac is None else f"{frac:.0%}"
        reasons.append(f"oos_pos_frac={shown}<{OOS_POS_FRAC_MIN:.0%}")

    dsr = _as_float(row.get("dsr"))
    if dsr is None or dsr <= DSR_MIN:
        shown = "nan" if dsr is None else f"{dsr:.2f}"
        reasons.append(f"DSR={shown}≤{DSR_MIN}")

    sm = _as_float(row.get("smoothness"))
    if sm is None or sm <= SMOOTHNESS_MIN:
        shown = "nan" if sm is None else f"{sm:.2f}"
        reasons.append(f"smoothness={shown}≤{SMOOTHNESS_MIN}")

    # Гейт на OOS-активність (не full-sample бектест): комірка не може
    # пройти з "тонкою" OOS-торгівлею
    nt = _as_int(row.get("n_trades_oos"))
    min_nt = min_trades_for(interval)
    if nt is None or nt < min_nt:
        shown = "nan" if nt is None else str(nt)
        reasons.append(f"n_trades_oos={shown}<{min_nt}")

    # PBO перевіряється лише якщо CSCV дійсно запускався (pbo is not None)
    pbo = _as_float(row.get("pbo"))
    if pbo is not None and pbo > PBO_MAX:
        reasons.append(f"PBO={pbo:.2f}>{PBO_MAX}")

    if not reasons:
        return "PASS", ""
    return "FAIL", "; ".join(reasons)


def audit_cell(
    strategy_name: str,
    symbol: str,
    interval: str,
    days: int,
    *,
    train_bars: int | None = None,
    test_bars: int | None = None,
    with_cscv: bool = False,
    strategy_params: dict[str, Any] | None = None,
) -> CellAudit:
    """Повний аудит комірки. Помилки даних/рахунку — status=error, без raise.

    with_cscv: додатково рахувати CSCV PBO (~CSCV_VARIANTS додаткових
    бектестів — дорого для матричних прогонів, вмикати для фінального
    вердикту комірки, напр. CLI `overfit`).
    strategy_params: параметри конструктора стратегії (напр. use_kalman).
    """
    from scalper_hft.backtest.execution import CostModel

    # Роутер: market_maker аудитується подієвим рушієм, не zero-signal вектором
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_agg_trades, download_funding
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    try:
        settings = get_settings()
        cost = CostModel(
            maker_fee=settings.maker_fee,
            taker_fee=settings.taker_fee,
            slippage_frac=settings.slippage_frac,
        )
        strategy = get_strategy(strategy_name, **(strategy_params or {}))
        df = ensure_klines(symbol, interval, days)
        if df is None or df.empty:
            return CellAudit(
                symbol=symbol,
                interval=interval,
                strategy=strategy_name,
                status="error",
                error="немає даних",
            )
        trades = download_agg_trades(symbol, days) if strategy.needs_trades else None
        funding = download_funding(symbol, days) if strategy.needs_funding else None
        train, test = default_train_test(interval)
        if train_bars is not None:
            train = int(train_bars)
        if test_bars is not None:
            test = int(test_bars)

        wf = run_walk_forward(
            df,
            strategy,
            train,
            test,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            collect_oos_returns=True,
        )
        windows = tuple(_window_to_dict(w) for w in wf.windows)

        # Sensitivity на OOS-регіоні (від початку першого test-вікна):
        # сітка на повній історії забруднена IS-частиною
        df_oos = df.iloc[wf.windows[0].test_start :] if wf.windows else df
        if len(df_oos) < 200:
            df_oos = df

        smoothness: float | None = None
        sens_n = 0
        sens_param: str | None = None
        sens_error: str | None = None
        grid_rows: tuple[dict[str, Any], ...] = ()
        ps = getattr(strategy, "param_space", {}) or {}
        if ps:
            pname = next(iter(ps))
            lo, hi, step = ps[pname]
            step_f = float(step) if step else 1.0
            n_vals = min(int((float(hi) - float(lo)) / step_f) + 1, 15)
            values = [float(lo) + i * step_f for i in range(max(n_vals, 2))][:15]
            try:
                sres = parameter_sensitivity(df_oos, strategy, pname, values, cost=cost, trades=trades, funding=funding)
                smoothness = float(sres.smoothness)
                if smoothness != smoothness:
                    smoothness = None
                sens_n = len(sres.grid)
                sens_param = pname
                grid_rows = tuple({str(k): v for k, v in r.items()} for r in sres.grid.to_dict(orient="records"))
            except Exception as exc:  # noqa: BLE001
                sens_error = str(exc)[:120]

        res_full = run_strategy_backtest(
            df,
            strategy,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
        )
        m = res_full.metrics
        dsr: float | None = None
        n_trials: int | None = None
        # DSR на конкатенованих OOS-дохідностях walk-forward (не full-sample
        # equity — та забруднена IS-вікнами і завищує DSR)
        ret = wf.oos_returns.dropna() if wf.oos_returns is not None else pd.Series(dtype=float)
        if len(ret) >= 2:
            combos = 1
            for _lo, _hi, _s in ps.values():
                step_f = float(_s) if _s else 1.0
                combos *= max(int((float(_hi) - float(_lo)) / step_f) + 1, 1)
            combos = min(max(combos, 1), 100_000)
            n_trials = int(estimate_n_trials(param_combinations=combos, backtests_per_combo=DSR_BACKTESTS_PER_COMBO))
            dsr = float(deflated_sharpe_ratio(ret.to_numpy(dtype=float), n_trials=n_trials))

        # CSCV PBO (опційно): чи не є IS-кращий варіант перенавченим
        pbo: float | None = None
        if with_cscv and ps:
            from scalper_hft.validation.cscv import pbo_cscv, variant_returns

            try:
                vr = variant_returns(
                    df,
                    strategy,
                    n_variants=CSCV_VARIANTS,
                    cost=cost,
                    trades=trades,
                    funding=funding,
                    position_pct=settings.position_pct,
                )
                pbo = float(pbo_cscv(vr, n_blocks=8).pbo)
            except Exception:  # noqa: BLE001
                pbo = None

        return CellAudit(
            symbol=symbol,
            interval=interval,
            strategy=strategy_name,
            status="ok",
            n_windows=len(wf.windows),
            avg_is_sharpe=float(wf.avg_is_sharpe),
            avg_oos_sharpe=float(wf.avg_oos_sharpe),
            oos_pos_frac=float(wf.positive_windows_frac),
            n_trades_oos=int(sum(w.n_trades for w in wf.windows)),
            degradation=float(wf.degradation),
            sens_param=sens_param,
            smoothness=smoothness,
            sens_n=sens_n,
            sens_error=sens_error,
            dsr=dsr,
            n_trials_dsr=n_trials,
            pbo=pbo,
            bt_total_return=float(m.total_return),
            bt_sharpe=float(m.sharpe),
            bt_max_dd=float(m.max_drawdown),
            bt_n_trades=int(m.n_trades),
            bt_profit_factor=float(m.profit_factor),
            bt_win_rate=float(m.win_rate),
            bt_trades_per_day=float(m.trades_per_day),
            windows=windows,
            sensitivity_grid=grid_rows,
        )
    except Exception as exc:  # noqa: BLE001
        return CellAudit(
            symbol=symbol,
            interval=interval,
            strategy=strategy_name,
            status="error",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
