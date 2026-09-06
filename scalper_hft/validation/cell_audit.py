"""Аудит однієї комірки (стратегія × символ × ТФ): WF + sensitivity + DSR.

Критерії PASS — overfitting-audit skill:
    avg_oos_sharpe > 0.3, oos_pos_frac >= 0.5, DSR > 0.95,
    smoothness > 0.30, n_trades >= поріг таймфрейму.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

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
DSR_BACKTESTS_PER_COMBO = 50

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

    nt = _as_int(row.get("bt_n_trades"))
    min_nt = min_trades_for(interval)
    if nt is None or nt < min_nt:
        shown = "nan" if nt is None else str(nt)
        reasons.append(f"n_trades={shown}<{min_nt}")

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
) -> CellAudit:
    """Повний аудит комірки. Помилки даних/рахунку — status=error, без raise."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
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
        strategy = get_strategy(strategy_name)
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
        )
        windows = tuple(_window_to_dict(w) for w in wf.windows)

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
                sres = parameter_sensitivity(df, strategy, pname, values, cost=cost, trades=trades, funding=funding)
                smoothness = float(sres.smoothness)
                if smoothness != smoothness:
                    smoothness = None
                sens_n = len(sres.grid)
                sens_param = pname
                grid_rows = tuple({str(k): v for k, v in r.items()} for r in sres.grid.to_dict(orient="records"))
            except Exception as exc:  # noqa: BLE001
                sens_error = str(exc)[:120]

        res_full = run_backtest(
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
        ret = res_full.equity.pct_change().dropna()
        if len(ret) >= 2:
            combos = 1
            for _lo, _hi, _s in ps.values():
                step_f = float(_s) if _s else 1.0
                combos *= max(int((float(_hi) - float(_lo)) / step_f) + 1, 1)
            combos = min(max(combos, 1), 100_000)
            n_trials = int(estimate_n_trials(param_combinations=combos, backtests_per_combo=DSR_BACKTESTS_PER_COMBO))
            dsr = float(deflated_sharpe_ratio(ret.to_numpy(dtype=float), n_trials=n_trials))

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
