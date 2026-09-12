"""Paper vs backtest tracking-error (Phase 1 Paper Gate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.live.chase_shadow import ChaseShadowReport
from scalper_hft.live.is_report import ISReport
from scalper_hft.live.store import PaperStore
from scalper_hft.validation.forensics import ForensicsReport, analyze_trades, trades_from_paper_frames

DD_MULT_DEFAULT = 1.5

# Hard go/no-go критерії Paper Gate (Phase 6 / P6-A).
# Усі пороги мають бути пройдені одночасно: провал хоча б одного =
# заборона переходу з paper у live незалежно від прибутковості.
PAPER_GATE_THRESHOLDS: dict[str, float] = {
    # Tracking error (std diff per-bar returns) ≤ 3% на тиждень
    "tracking_error_weekly_pct": 0.03,
    # Maker fill rate ≥ 70% (нижче → ринок не підтримує post-only)
    "fill_rate_min": 0.70,
    # Paper max DD ≤ 1.5× backtest max DD
    "dd_ratio_max": DD_MULT_DEFAULT,
    # Blended TCA (Implementation Shortfall) ≤ 3 bps
    # Вище → реальні витрати перевищують модель → live = збиток
    "blended_tca_bps_max": 3.0,
}


def check_paper_gate(
    audit: PaperAudit,
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Hard go/no-go gate перед переходом з paper у live.

    Перевіряє всі критерії з PAPER_GATE_THRESHOLDS. Повертає (True, []) якщо
    всі пройдені, або (False, [список_провалів]) де кожен рядок пояснює причину.

    Args:
        audit: результат audit_paper_store / audit_paper_vs_backtest.
        thresholds: перевизначити порогові значення (для тестів / кастомізації).

    Returns:
        (passed, failures) де failures — список рядків з описом проблем.
    """
    t = thresholds if thresholds is not None else PAPER_GATE_THRESHOLDS
    failures: list[str] = []

    # 1. Tracking error
    te_limit = float(t.get("tracking_error_weekly_pct", 0.03))
    if audit.tracking_error is not None and audit.tracking_error > te_limit:
        failures.append(
            f"tracking_error={audit.tracking_error:.4%} > {te_limit:.4%} (max допустимий: {te_limit:.0%}/тиждень)"
        )

    # 2. Fill rate
    fr_min = float(t.get("fill_rate_min", 0.70))
    if audit.paper_fill_rate < fr_min:
        failures.append(
            f"fill_rate={audit.paper_fill_rate:.0%} < {fr_min:.0%} "
            f"(post-only ордери рідко філяться — ринок несприятливий)"
        )

    # 3. DD ratio (paper vs backtest)
    dd_max = float(t.get("dd_ratio_max", DD_MULT_DEFAULT))
    if audit.dd_gate_ok is False:
        bt_dd = audit.bt_max_dd or 0.0
        failures.append(
            f"paper_max_dd={audit.paper_max_dd:.2%} > backtest_max_dd×{dd_max:.1f} "
            f"(bt_max_dd={bt_dd:.2%}, threshold={bt_dd * dd_max:.2%})"
        )

    # 4. Blended TCA (IS) — ключовий hard criterion
    tca_max = float(t.get("blended_tca_bps_max", 3.0))
    if audit.is_report is not None and audit.is_report.coverage_ok:
        tca = audit.is_report.blended_tca_bps
        if tca > tca_max:
            failures.append(
                f"blended_tca_bps={tca:.2f} > {tca_max:.1f} bps (реальні T-costs перевищують модель → live = збиток)"
            )

    return len(failures) == 0, failures


def format_gate_result(passed: bool, failures: list[str]) -> str:
    """Форматований рядок для CLI/логів."""
    if passed:
        return "Paper Gate: ✅ PASS — всі критерії пройдені"
    lines = ["Paper Gate: ❌ FAIL — заборона live:"]
    for f in failures:
        lines.append(f"  • {f}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PaperAudit:
    n_bars: int
    paper_return: float
    bt_return: float | None
    tracking_error: float | None
    paper_max_dd: float
    bt_max_dd: float | None
    dd_gate_ok: bool | None
    paper_fill_rate: float
    bt_fill_rate: float | None
    fill_gap: float | None
    forensics: ForensicsReport
    is_report: ISReport | None = None
    chase_shadow: ChaseShadowReport | None = None

    def summary(self) -> str:
        te = "—" if self.tracking_error is None else f"{self.tracking_error:.4%}"
        bt = "—" if self.bt_return is None else f"{self.bt_return:+.2%}"
        dd = "—" if self.dd_gate_ok is None else ("ok" if self.dd_gate_ok else "FAIL")
        fill = "—" if self.fill_gap is None else f"{self.fill_gap:+.1%}"

        seed = get_settings().maker_fill_seed
        lines = (
            f"Paper audit: bars={self.n_bars} paper={self.paper_return:+.2%} bt={bt}\n"
            f"  tracking-error={te} maxDD paper={self.paper_max_dd:.2%} gate={dd}\n"
            f"  fill-rate paper={self.paper_fill_rate:.0%} gap={fill} MAKER_FILL_SEED={seed}\n"
            f"  {self.forensics.summary()}"
        )
        if self.is_report is not None:
            lines += "\n  " + self.is_report.summary().replace("\n", "\n  ")
        if self.chase_shadow is not None:
            from scalper_hft.live.chase_shadow import format_live_chase_gap

            paper_tca = None if self.is_report is None else self.is_report.blended_tca_bps
            lines += "\n  " + format_live_chase_gap(self.chase_shadow, paper_blended_tca_bps=paper_tca).replace(
                "\n", "\n  "
            )
        return lines


def max_drawdown(equity: pd.Series) -> float:
    """Максимальне просідання як додатна частка піку (0.1 = −10%)."""
    if equity is None or equity.empty:
        return 0.0
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(-dd.min()) if len(dd) else 0.0


def series_return(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2 or float(equity.iloc[0]) == 0.0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def tracking_error(paper: pd.Series, backtest: pd.Series) -> float:
    """Std різниці періодичних доходностей на спільному індексі."""
    p = _as_float_series(paper).pct_change()
    b = _as_float_series(backtest).pct_change()
    aligned = pd.concat([p, b], axis=1, keys=["paper", "bt"]).dropna()
    if aligned.empty:
        return 0.0
    diff = aligned["paper"] - aligned["bt"]
    return float(diff.std(ddof=0)) if len(diff) else 0.0


def fill_rate(stats: dict[str, int]) -> float:
    filled = int(stats.get("filled", 0))
    unfilled = int(stats.get("unfilled", 0))
    total = filled + unfilled
    if total <= 0:
        return 0.0
    return filled / total


def _as_float_series(equity: pd.Series) -> pd.Series:
    eq = equity.copy()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)
    return eq.sort_index().astype(float)


def equity_from_store(frame: pd.DataFrame) -> pd.Series:
    """Крива капіталу з таблиці equity (останнє значення на ts по всіх парах)."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    df = frame.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    grouped = df.groupby("ts", sort=True)["equity"].last()
    return grouped.astype(float)


def audit_paper_vs_backtest(
    paper_equity: pd.Series,
    *,
    paper_fill_rate: float,
    bt_equity: pd.Series | None = None,
    bt_fill_rate: float | None = None,
    trades: pd.DataFrame | None = None,
    dd_mult: float = DD_MULT_DEFAULT,
    is_report: ISReport | None = None,
    chase_shadow: ChaseShadowReport | None = None,
) -> PaperAudit:
    """Порівняння paper-кривої з бектестом за той самий (або вирівняний) період."""
    paper = _as_float_series(paper_equity) if paper_equity is not None else pd.Series(dtype=float)
    bt: pd.Series | None = None
    if bt_equity is not None and not bt_equity.empty:
        bt = _as_float_series(bt_equity)
        if not paper.empty:
            common = paper.index.intersection(bt.index)
            if len(common) >= 2:
                paper = paper.reindex(common).dropna()
                bt = bt.reindex(common).dropna()
    paper_dd = max_drawdown(paper)
    bt_dd = max_drawdown(bt) if bt is not None and len(bt) >= 2 else None
    dd_ok: bool | None = None
    if bt_dd is not None:
        dd_ok = paper_dd <= bt_dd * dd_mult + 1e-12
    te = tracking_error(paper, bt) if bt is not None and len(paper) >= 2 and len(bt) >= 2 else None
    fill_gap = None if bt_fill_rate is None else float(paper_fill_rate - bt_fill_rate)
    return PaperAudit(
        n_bars=int(len(paper)),
        paper_return=series_return(paper),
        bt_return=series_return(bt) if bt is not None else None,
        tracking_error=te,
        paper_max_dd=paper_dd,
        bt_max_dd=bt_dd,
        dd_gate_ok=dd_ok,
        paper_fill_rate=float(paper_fill_rate),
        bt_fill_rate=bt_fill_rate,
        fill_gap=fill_gap,
        forensics=analyze_trades(trades),
        is_report=is_report,
        chase_shadow=chase_shadow,
    )


def is_cost_hint(
    records: list[object],
    current_slippage_frac: float,
    *,
    warn_bps: float = 1.0,
    fallback: float = 0.0002,
) -> dict[str, float | bool]:
    """Порівняти CostModel slippage з медіаною IS. Не змінює комісії в loop.

    warn=True якщо |maker IS − current| або |chase IS − current| > warn_bps.
    """
    from scalper_hft.backtest.execution import calibrate_from_is

    maker = calibrate_from_is(records, kind="maker", fallback=fallback)
    chase = calibrate_from_is(records, kind="chase", fallback=fallback)
    current = float(current_slippage_frac)
    maker_bps = maker * 10_000.0
    chase_bps = chase * 10_000.0
    current_bps = current * 10_000.0
    delta = max(abs(maker_bps - current_bps), abs(chase_bps - current_bps))
    return {
        "maker_slippage_frac": maker,
        "chase_slippage_frac": chase,
        "current_slippage_frac": current,
        "maker_bps": maker_bps,
        "chase_bps": chase_bps,
        "current_bps": current_bps,
        "warn": delta > warn_bps,
    }


def format_cost_hint(hint: dict[str, float | bool]) -> str:
    """Один рядок для paper-audit CLI."""
    flag = "warn" if hint["warn"] else "ok"
    return (
        f"IS vs CostModel: maker={hint['maker_bps']:.2f} bps chase={hint['chase_bps']:.2f} bps "
        f"current={hint['current_bps']:.2f} bps [{flag}]"
    )


def audit_paper_store(
    store: PaperStore,
    *,
    bt_equity: pd.Series | None = None,
    bt_fill_rate: float | None = None,
    dd_mult: float = DD_MULT_DEFAULT,
) -> PaperAudit:
    equity = equity_from_store(store.all_equity())
    stats = store.fill_stats()
    orders = store.all_orders()
    trades = trades_from_paper_frames(store.all_trades(), orders)
    from scalper_hft.live.chase_shadow import build_from_store_frame
    from scalper_hft.live.is_report import build_from_orders

    is_rep = build_from_orders(orders, model_slippage_bps=get_settings().slippage_bps)
    shadow = build_from_store_frame(store.all_shadow_legging())
    return audit_paper_vs_backtest(
        equity,
        paper_fill_rate=fill_rate(stats),
        bt_equity=bt_equity,
        bt_fill_rate=bt_fill_rate,
        trades=trades,
        dd_mult=dd_mult,
        is_report=is_rep,
        chase_shadow=shadow,
    )


def load_equity_csv(path: Path) -> pd.Series:
    """CSV з колонками ts,equity (або індекс + одна колонка)."""
    df = pd.read_csv(path)
    if "ts" in df.columns and "equity" in df.columns:
        s = pd.Series(df["equity"].astype(float).values, index=pd.to_datetime(df["ts"]))
        return s.sort_index()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(df.iloc[:, 1].astype(float).values, index=df.iloc[:, 0]).sort_index()
