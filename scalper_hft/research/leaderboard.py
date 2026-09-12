"""Агрегація результатів досліджень у єдиний лідерборд комірок.

Джерела (у порядку пріоритету метрик):
    - results/audit_verdicts.jsonl — повні аудити cell_audit
    - results/iter*/audit_*.csv — вибіркові аудити
    - results/iter*/screen_*.csv — скрини sweep
    - results/iter*/daily_long_history.csv — довга історія 1d
    - results/iter*/ts_momentum_portfolio*.md — портфельні тести
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from scalper_hft.validation.cell_audit import (
    DSR_MIN,
    OOS_POS_FRAC_MIN,
    OOS_SHARPE_MIN,
    PBO_MAX,
    cell_verdict,
)

Tier = Literal["paper", "monitoring", "candidate", "rejected", "validated_pairs"]

_TIER_ORDER = {"validated_pairs": 0, "paper": 1, "monitoring": 2, "candidate": 3, "rejected": 4}


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """Один рядок лідерборду — комірка або портфель."""

    rank: int
    tier: Tier
    strategy: str
    symbol: str
    interval: str
    avg_oos_sharpe: float | None
    oos_pos_frac: float | None
    dsr: float | None
    pbo: float | None
    n_trades_oos: float | None
    port_sharpe: float | None
    t_newey_west: float | None
    verdict: str
    notes: str
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "tier": self.tier,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "interval": self.interval,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "oos_pos_frac": self.oos_pos_frac,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "n_trades_oos": self.n_trades_oos,
            "port_sharpe": self.port_sharpe,
            "t_newey_west": self.t_newey_west,
            "verdict": self.verdict,
            "notes": self.notes,
            "source": self.source,
        }


def _safe_float(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _directional_verdict(row: dict) -> str:
    label, _ = cell_verdict(row, mode="exploratory")
    return label


def _normalize_symbol(symbol: str) -> str:
    """LINKUSDT/BTCUSDT і LINK/BTC → LINK-BTC; PORTFOLIO_* лишається як є."""
    if symbol.upper().startswith("PORTFOLIO"):
        return symbol.upper()
    cleaned = symbol.upper().replace("USDT", "")
    return cleaned.replace("/", "-").replace("_", "-")


def _is_link_btc_pair(strategy: str, symbol: str) -> bool:
    if "pairs_arb" not in strategy.lower():
        return False
    legs = [p for p in _normalize_symbol(symbol).split("-") if p]
    return "LINK" in legs and "BTC" in legs


def _is_pairs_single_symbol(strategy: str, symbol: str) -> bool:
    """pairs_arb на одному тікері — не пара, не paper-ready."""
    if "pairs_arb" not in strategy.lower():
        return False
    return len([p for p in _normalize_symbol(symbol).split("-") if p]) < 2


def _tier_for_row(
    strategy: str,
    symbol: str,
    interval: str,
    avg_oos: float | None,
    verdict: str,
    port_sharpe: float | None,
    t_nw: float | None,
    notes: str,
) -> Tier:
    del interval  # ключ комірки використовує caller; тут — лише класифікація
    if _is_link_btc_pair(strategy, symbol):
        return "validated_pairs"
    if _is_pairs_single_symbol(strategy, symbol):
        return "rejected"
    mined = f"{symbol} {notes}".lower()
    post_hoc_subset = "top10" in mined or "smooth3" in mined or "smooth=3" in mined
    rides_on_core = "full45" in mined or "h2;" in mined
    if port_sharpe is not None and t_nw is not None:
        if port_sharpe >= 0.5 and t_nw >= 2.0:
            return "candidate" if post_hoc_subset or rides_on_core else "monitoring"
        if port_sharpe >= 0.5 and t_nw >= 1.5:
            return "candidate"
    if verdict == "PASS":
        return "paper"
    if avg_oos is not None and avg_oos >= 0.15 and verdict in {"EXPLORATORY_PASS", "FAIL"}:
        return "candidate"
    if "monitoring" in notes.lower():
        return "monitoring"
    return "rejected"


def _dedupe_key(row: LeaderboardRow) -> tuple[str, str, str]:
    return (row.strategy.lower(), _normalize_symbol(row.symbol), row.interval.lower())


def _dedupe_rows(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    best: dict[tuple[str, str, str], LeaderboardRow] = {}
    for row in rows:
        key = _dedupe_key(row)
        prev = best.get(key)
        if prev is None or _score_row(row) > _score_row(prev):
            best[key] = row
    return list(best.values())


def _score_row(row: LeaderboardRow) -> float:
    """Композитний скор для ранжування (вище = краще)."""
    base = 0.0
    if row.port_sharpe is not None:
        base += row.port_sharpe * 2.0
    if row.avg_oos_sharpe is not None:
        base += row.avg_oos_sharpe
    if row.t_newey_west is not None:
        base += row.t_newey_west * 0.15
    if row.oos_pos_frac is not None:
        base += row.oos_pos_frac * 0.2
    if row.dsr is not None:
        base += row.dsr * 0.1
    if row.pbo is not None and row.pbo < PBO_MAX:
        base += 0.1
    tier_bonus = {t: v for t, v in zip(_TIER_ORDER, [5, 4, 3, 1, 0], strict=True)}
    base += tier_bonus.get(row.tier, 0)
    return base


def load_audit_frames(results_dir: Path = Path("results")) -> pd.DataFrame:
    """Зібрати аудити з jsonl і iter*/audit_*.csv."""
    frames: list[pd.DataFrame] = []
    verdicts = results_dir / "audit_verdicts.jsonl"
    if verdicts.exists():
        rows = [json.loads(line) for line in verdicts.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    for csv_path in sorted(results_dir.glob("iter*/audit_*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df["source"] = str(csv_path)
            frames.append(df)
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ("strategy", "symbol", "interval"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    out = out.drop_duplicates(subset=["strategy", "symbol", "interval"], keep="last")
    return out


def load_portfolio_summaries(results_dir: Path = Path("results")) -> list[dict]:
    """Парсити markdown-звіти портфельних тестів."""
    summaries: list[dict] = []
    pattern = re.compile(
        r"Sharpe\s+([+-]?\d+\.\d+).*?t_Newey–West\(20\)=([+-]?\d+\.\d+)",
        re.DOTALL,
    )
    for md_path in sorted(results_dir.glob("iter*/ts_momentum_portfolio*.md")):
        text = md_path.read_text(encoding="utf-8")
        m = pattern.search(text)
        if not m:
            continue
        name = md_path.stem.replace("ts_momentum_portfolio", "").strip("_") or "lb20"
        summaries.append(
            {
                "strategy": "ts_momentum",
                "symbol": "PORTFOLIO_15",
                "interval": "1d",
                "port_sharpe": float(m.group(1)),
                "t_newey_west": float(m.group(2)),
                "variant": name,
                "source": str(md_path),
            }
        )
    return summaries


def build_leaderboard(results_dir: Path | str = "results") -> list[LeaderboardRow]:
    """Побудувати відсортований лідерборд з усіх доступних артефактів."""
    root = Path(results_dir)
    rows: list[LeaderboardRow] = []

    # Валідована пара (статичний рядок — iter9 не чіпав pairs)
    rows.append(
        LeaderboardRow(
            rank=0,
            tier="validated_pairs",
            strategy="pairs_arb",
            symbol="LINK/BTC",
            interval="1h",
            avg_oos_sharpe=0.0064,
            oos_pos_frac=0.65,
            dsr=None,
            pbo=0.0,
            n_trades_oos=None,
            port_sharpe=None,
            t_newey_west=None,
            verdict="PASS",
            notes="VALIDATED_PAIRS; regime_scale=0.25; paper-v0.2.0 gate",
            source="docs/STRATEGY_STATUS.md",
        )
    )

    audit_df = load_audit_frames(root)
    for rec in audit_df.itertuples(index=False):
        d = rec._asdict() if hasattr(rec, "_asdict") else dict(rec)
        if str(d.get("status", "ok")) == "error":
            continue
        raw_label = d.get("label")
        if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
            verdict = _directional_verdict(d)
        else:
            verdict = str(raw_label)
        avg_oos = _safe_float(d.get("avg_oos_sharpe"))
        tier = _tier_for_row(
            str(d.get("strategy", "")),
            str(d.get("symbol", "")),
            str(d.get("interval", "")),
            avg_oos,
            verdict,
            None,
            None,
            "",
        )
        rows.append(
            LeaderboardRow(
                rank=0,
                tier=tier,
                strategy=str(d.get("strategy", "")),
                symbol=str(d.get("symbol", "")),
                interval=str(d.get("interval", "")),
                avg_oos_sharpe=avg_oos,
                oos_pos_frac=_safe_float(d.get("oos_pos_frac")),
                dsr=_safe_float(d.get("dsr")),
                pbo=_safe_float(d.get("pbo")),
                n_trades_oos=_safe_float(d.get("n_trades_oos")),
                port_sharpe=None,
                t_newey_west=None,
                verdict=verdict,
                notes=f"audit; DSR>{DSR_MIN}, OOS>{OOS_SHARPE_MIN}",
                source=str(d.get("source", "audit")),
            )
        )

    for ps in load_portfolio_summaries(root):
        variant = ps.get("variant", "")
        notes = f"портфель 15 символів; variant={variant}"
        tier = _tier_for_row(
            ps["strategy"],
            ps["symbol"],
            ps["interval"],
            None,
            "FAIL",
            ps["port_sharpe"],
            ps["t_newey_west"],
            notes,
        )
        rows.append(
            LeaderboardRow(
                rank=0,
                tier=tier,
                strategy=str(ps["strategy"]),
                symbol=str(ps["symbol"]),
                interval=str(ps["interval"]),
                avg_oos_sharpe=None,
                oos_pos_frac=None,
                dsr=None,
                pbo=None,
                n_trades_oos=None,
                port_sharpe=ps["port_sharpe"],
                t_newey_west=ps["t_newey_west"],
                verdict="MONITORING" if tier == "monitoring" else "CANDIDATE",
                notes=notes,
                source=str(ps["source"]),
            )
        )

    # iter*/variants.csv — портфельні порівняння (iter10, iter11, …)
    for csv_path in sorted(root.glob("iter*/variants.csv")):
        try:
            vdf = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            continue
        for rec in vdf.itertuples(index=False):
            d = rec._asdict()
            avg_oos = _safe_float(d.get("mean_oos_wf"))
            port_sr = _safe_float(d.get("port_sharpe"))
            t_nw = _safe_float(d.get("t_newey_west"))
            interval = str(d.get("interval") or "1d")
            variant = str(d.get("variant", csv_path.parent.name))
            strategy = str(d.get("strategy") or "ts_momentum")
            notes = str(d.get("notes", variant))
            tier = _tier_for_row(
                strategy,
                f"PORTFOLIO_{variant}",
                interval,
                avg_oos,
                "FAIL",
                port_sr,
                t_nw,
                notes,
            )
            rows.append(
                LeaderboardRow(
                    rank=0,
                    tier=tier,
                    strategy=strategy,
                    symbol=f"PORTFOLIO_{variant}",
                    interval=interval,
                    avg_oos_sharpe=avg_oos,
                    oos_pos_frac=_safe_float(d.get("symbols_pos_frac")),
                    dsr=None,
                    pbo=None,
                    n_trades_oos=_safe_float(d.get("n_trades_oos")),
                    port_sharpe=port_sr,
                    t_newey_west=t_nw,
                    verdict="MONITORING" if tier == "monitoring" else str(tier).upper(),
                    notes=notes,
                    source=str(csv_path),
                )
            )

    rows = _dedupe_rows(rows)
    rows.sort(key=lambda r: (_TIER_ORDER.get(r.tier, 9), -_score_row(r)))
    ranked: list[LeaderboardRow] = []
    for i, row in enumerate(rows, 1):
        ranked.append(
            LeaderboardRow(
                rank=i,
                tier=row.tier,
                strategy=row.strategy,
                symbol=row.symbol,
                interval=row.interval,
                avg_oos_sharpe=row.avg_oos_sharpe,
                oos_pos_frac=row.oos_pos_frac,
                dsr=row.dsr,
                pbo=row.pbo,
                n_trades_oos=row.n_trades_oos,
                port_sharpe=row.port_sharpe,
                t_newey_west=row.t_newey_west,
                verdict=row.verdict,
                notes=row.notes,
                source=row.source,
            )
        )
    return ranked


def leaderboard_dataframe(results_dir: Path | str = "results") -> pd.DataFrame:
    """Лідерборд як DataFrame."""
    rows = build_leaderboard(results_dir)
    return pd.DataFrame([r.as_dict() for r in rows])


_TIER_LABELS = {
    "validated_pairs": "✅ Validated (paper)",
    "paper": "🟢 Paper-ready",
    "monitoring": "🟡 Monitoring",
    "candidate": "🔵 Candidate",
    "rejected": "⛔ Rejected",
}


def render_leaderboard_markdown(
    results_dir: Path | str = "results",
    *,
    top_n: int = 30,
) -> str:
    """Markdown-звіт лідерборду для docs/reports/LEADERBOARD.md."""
    rows = build_leaderboard(results_dir)
    lines = [
        "# Лідерборд стратегій scalper-hft",
        "",
        f"Оновлено: автоматично з `{results_dir}/` · Топ-{top_n} комірок",
        "",
        "## Пороги гейту (directional)",
        "",
        f"- OOS Sharpe > {OOS_SHARPE_MIN}",
        f"- OOS позитивних вікон ≥ {OOS_POS_FRAC_MIN:.0%}",
        f"- DSR > {DSR_MIN}",
        f"- PBO < {PBO_MAX}",
        "",
        "## Тієри",
        "",
    ]
    for tier, label in _TIER_LABELS.items():
        n = sum(1 for r in rows if r.tier == tier)
        lines.append(f"- **{label}**: {n} комірок")
    lines.extend(["", "## Топ комірок", ""])

    table_rows = []
    for row in rows[:top_n]:
        oos = f"{row.avg_oos_sharpe:+.3f}" if row.avg_oos_sharpe is not None else "—"
        pos = f"{row.oos_pos_frac:.0%}" if row.oos_pos_frac is not None else "—"
        psr = f"{row.port_sharpe:+.2f}" if row.port_sharpe is not None else "—"
        tnw = f"{row.t_newey_west:+.2f}" if row.t_newey_west is not None else "—"
        table_rows.append(
            {
                "#": row.rank,
                "Тіер": _TIER_LABELS.get(row.tier, row.tier),
                "Стратегія": row.strategy,
                "Інструмент": row.symbol,
                "ТФ": row.interval,
                "OOS SR": oos,
                "OOS+": pos,
                "Port SR": psr,
                "t_NW": tnw,
                "Вердикт": row.verdict,
                "Нотатки": row.notes[:60],
            }
        )
    if table_rows:
        lines.append(pd.DataFrame(table_rows).to_markdown(index=False))
    else:
        lines.append("_Немає даних — запустіть `experiments/iter10_research_cycle.py`_")

    recs: list[str] = []
    for row in rows:
        if row.tier == "validated_pairs":
            recs.append("1. **pairs_arb LINK/BTC 1h** — єдиний validated; paper на `paper-v0.2.0`.")
            break
    mon = [
        r
        for r in rows
        if r.tier == "monitoring"
        and r.strategy == "ts_momentum"
        and "top10" not in r.symbol.lower()
        and "smooth3" not in r.symbol.lower()
    ]
    n = 2
    seen: set[str] = set()
    for best in mon:
        key = f"{best.interval}:{best.symbol}"
        if key in seen:
            continue
        seen.add(key)
        tnw = f"{best.t_newey_west:+.2f}" if best.t_newey_west is not None else "—"
        psr = f"{best.port_sharpe:+.2f}" if best.port_sharpe is not None else "—"
        recs.append(
            f"{n}. **ts_momentum {best.interval} {best.symbol}** — monitoring "
            f"(Port Sharpe {psr}, t_NW {tnw}); paper-моніторинг поруч із pairs, не live."
        )
        n += 1
        if n > 4:
            break
    recs.append(f"{n}. Directional single-symbol комірки **не проходять** гейт 0.3 — диверсифікація обов'язкова.")
    lines.extend(["", "## Рекомендації для paper", ""])
    lines.extend(recs)
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "LeaderboardRow",
    "build_leaderboard",
    "leaderboard_dataframe",
    "load_audit_frames",
    "render_leaderboard_markdown",
]
