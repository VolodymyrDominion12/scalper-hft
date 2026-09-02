"""Paper vs backtest tracking-error (Phase 1 Paper Gate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalper_hft.live.store import PaperStore
from scalper_hft.validation.forensics import ForensicsReport, analyze_trades, trades_from_paper_frames

DD_MULT_DEFAULT = 1.5


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

    def summary(self) -> str:
        te = "—" if self.tracking_error is None else f"{self.tracking_error:.4%}"
        bt = "—" if self.bt_return is None else f"{self.bt_return:+.2%}"
        dd = "—" if self.dd_gate_ok is None else ("ok" if self.dd_gate_ok else "FAIL")
        fill = "—" if self.fill_gap is None else f"{self.fill_gap:+.1%}"
        return (
            f"Paper audit: bars={self.n_bars} paper={self.paper_return:+.2%} bt={bt}\n"
            f"  tracking-error={te} maxDD paper={self.paper_max_dd:.2%} gate={dd}\n"
            f"  fill-rate paper={self.paper_fill_rate:.0%} gap={fill}\n"
            f"  {self.forensics.summary()}"
        )


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
    trades = trades_from_paper_frames(store.all_trades(), store.all_orders())
    return audit_paper_vs_backtest(
        equity,
        paper_fill_rate=fill_rate(stats),
        bt_equity=bt_equity,
        bt_fill_rate=bt_fill_rate,
        trades=trades,
        dd_mult=dd_mult,
    )


def load_equity_csv(path: Path) -> pd.Series:
    """CSV з колонками ts,equity (або індекс + одна колонка)."""
    df = pd.read_csv(path)
    if "ts" in df.columns and "equity" in df.columns:
        s = pd.Series(df["equity"].astype(float).values, index=pd.to_datetime(df["ts"]))
        return s.sort_index()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(df.iloc[:, 1].astype(float).values, index=df.iloc[:, 0]).sort_index()
