"""Аудит кешу ринкових даних проти LIVE-біржі.

Навіщо: формальна валідація (`validate_bars`) перевіряє структуру, але не
відповідає на питання «це реальний ринок?». Кеш, скачаний з Binance **testnet**,
проходив її з `ok=True`, хоча ціни розходились із реальними на 1–17%, а 1m-бари
містили рухи +27% (BNBUSDT: 86 633 таких барів) і «плити» `O=H=L=C` з нульовим
обсягом. На такій історії sweep обирав `ml_strategy` з Sharpe 57 і «переможцем»
за Bailey–LdP haircut.

Тут — незалежна перевірка: беремо N випадкових барів з кешу і порівнюємо
закриття з тим, що віддає LIVE-біржа (`DATA_EXCHANGE`). Плюс похідні метрики:
частка неринкових рухів, найдовша «плита», покриття історії funding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from scalper_hft.data.validate import SPIKE_RATE_CRITICAL, validate_bars
from scalper_hft.data.downloader import funding_coverage_ratio

logger = logging.getLogger(__name__)

# Максимальна допустима розбіжність закриття з live. Закриті 1m-бари незмінні,
# тому правильний кеш дає рівно 0; допуск лишаємо для float-форматування.
LIVE_DIVERGENCE_MAX = 0.001
DEFAULT_LIVE_SAMPLES = 5


@dataclass
class SymbolAudit:
    """Результат аудиту одного символу."""

    symbol: str
    interval: str
    n_bars: int = 0
    first_ts: str = ""
    last_ts: str = ""
    n_spikes: int = 0
    spike_rate: float = 0.0
    max_flat_run: int = 0
    bars_ok: bool = False
    bars_issues: list[str] = field(default_factory=list)
    live_checked: int = 0
    live_mismatched: int = 0
    max_live_divergence: float = 0.0
    funding_rows: int = 0
    funding_coverage: float = 0.0
    reasons: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.reasons and not self.error

    def to_dict(self) -> dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "interval": self.interval,
            "n_bars": self.n_bars,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "n_spikes": self.n_spikes,
            "spike_rate": self.spike_rate,
            "max_flat_run": self.max_flat_run,
            "bars_ok": self.bars_ok,
            "bars_issues": list(self.bars_issues),
            "live_checked": self.live_checked,
            "live_mismatched": self.live_mismatched,
            "max_live_divergence": self.max_live_divergence,
            "funding_rows": self.funding_rows,
            "funding_coverage": self.funding_coverage,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "error": self.error,
        }
        return d


def _sample_timestamps(index: pd.DatetimeIndex, n: int) -> list[pd.Timestamp]:
    """N рівномірно розкиданих міток з середини кешу (без країв)."""
    if len(index) == 0:
        return []
    n = max(1, min(int(n), len(index)))
    lo, hi = int(len(index) * 0.05), int(len(index) * 0.95)
    if hi <= lo:
        lo, hi = 0, len(index) - 1
    positions = [int(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]
    return [index[p] for p in sorted(set(positions))]


def _live_close(client: Any, symbol: str, interval: str, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Закриття бару з LIVE-біржі за міткою `ts` (перший бар ≥ ts)."""
    since = int(ts.value // 1_000_000)
    candle = client.fetch_ohlcv(symbol, interval, since=since, limit=1)
    if not candle:
        return None
    row = candle[0]
    return pd.Timestamp(int(row[0]), unit="ms"), float(row[4])


def audit_symbol(
    symbol: str,
    *,
    interval: str = "1m",
    days: int = 1095,
    live_samples: int = DEFAULT_LIVE_SAMPLES,
    check_funding: bool = True,
    store: Any = None,
    client: Any = None,
) -> SymbolAudit:
    """Перевірити кеш символу: неринкові бари + звірка з LIVE + покриття funding."""
    from scalper_hft.config import get_settings
    from scalper_hft.data.store import get_store

    settings = get_settings()
    store = store if store is not None else get_store()
    out = SymbolAudit(symbol=symbol, interval=interval)
    try:
        klines = store.load_klines(symbol, interval)
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    if klines is None or klines.empty:
        out.error = "кеш порожній"
        return out

    report = validate_bars(klines, interval=interval)
    out.n_bars = report.n_rows
    out.n_spikes = report.n_spikes
    out.spike_rate = report.spike_rate
    out.max_flat_run = report.max_flat_run
    out.bars_ok = report.ok
    out.bars_issues = list(report.issues)
    out.first_ts = str(klines.index[0])
    out.last_ts = str(klines.index[-1])
    if not report.ok:
        out.reasons.extend(report.issues)
    if report.spike_rate > SPIKE_RATE_CRITICAL:
        out.reasons.append(f"spike_rate {report.spike_rate:.2%} > {SPIKE_RATE_CRITICAL:.2%}")

    # ── Звірка з live ────────────────────────────────────────────────────────
    if live_samples > 0:
        if client is None:
            from scalper_hft.data.downloader import _default_client

            client = _default_client()
        closes = klines["close"]
        for ts in _sample_timestamps(klines.index, live_samples):
            try:
                got = _live_close(client, symbol, interval, ts - pd.Timedelta(milliseconds=1))
            except Exception as exc:  # noqa: BLE001
                out.reasons.append(f"live-fetch {ts}: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            if got is None:
                continue
            live_ts, live_close = got
            if live_ts not in closes.index:
                continue
            cached_close = float(closes.loc[live_ts])
            if live_close <= 0:
                continue
            div = abs(cached_close / live_close - 1.0)
            out.live_checked += 1
            out.max_live_divergence = max(out.max_live_divergence, div)
            if div > LIVE_DIVERGENCE_MAX:
                out.live_mismatched += 1
        if out.live_checked == 0:
            out.reasons.append("не вдалось звірити жоден бар із live")
        elif out.live_mismatched:
            out.reasons.append(
                f"розбіжність із live: {out.live_mismatched}/{out.live_checked} барів, "
                f"max {out.max_live_divergence:.2%} (кеш не з цієї біржі?)"
            )

    # ── Funding ──────────────────────────────────────────────────────────────
    if check_funding:
        try:
            funding = store.load_funding(symbol)
        except Exception:  # noqa: BLE001
            funding = None
        out.funding_rows = 0 if funding is None or funding.empty else int(len(funding))
        out.funding_coverage = funding_coverage_ratio(funding, days)
        if out.funding_coverage < 0.90:
            out.reasons.append(f"покриття funding {out.funding_coverage:.0%} за {days} днів")

    return out


def audit_symbols(
    symbols: list[str],
    *,
    interval: str = "1m",
    days: int = 1095,
    live_samples: int = DEFAULT_LIVE_SAMPLES,
    check_funding: bool = True,
) -> list[SymbolAudit]:
    return [
        audit_symbol(
            s,
            interval=interval,
            days=days,
            live_samples=live_samples,
            check_funding=check_funding,
        )
        for s in symbols
    ]


def format_report(results: list[SymbolAudit], *, days: int) -> str:
    """Markdown-звіт по всіх символах + підсумок."""
    lines = [
        "| symbol | bars | spikes | max flat | live mismatch | funding cover | verdict |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.symbol} | — | — | — | — | — | ❌ {r.error} |")
            continue
        live = f"{r.max_live_divergence:.2%} ({r.live_checked} проб)" if r.live_checked else "—"
        verdict = "✅ ok" if r.ok else "❌ " + "; ".join(r.reasons)[:160]
        lines.append(
            f"| {r.symbol} | {r.n_bars} | {r.n_spikes} ({r.spike_rate:.2%}) | {r.max_flat_run} | "
            f"{live} | {r.funding_coverage:.0%} | {verdict} |"
        )
    bad = [r.symbol for r in results if not r.ok]
    lines.append("")
    lines.append(f"Перевірено символів: {len(results)}, провалено: {len(bad)}" + (f" → {', '.join(bad)}" if bad else ""))
    lines.append(f"Вікно funding: {days} днів")
    return "\n".join(lines)


__all__ = [
    "LIVE_DIVERGENCE_MAX",
    "SymbolAudit",
    "audit_symbol",
    "audit_symbols",
    "format_report",
]
