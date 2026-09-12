"""Валідація OHLCV-барів (UTC, OHLC-інваріанти, дірки, дублікати)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BarQualityReport:
    n_rows: int
    n_duplicates: int
    n_ohlc_violations: int
    n_gaps: int
    n_future: int
    monotonic: bool
    ok: bool
    issues: list[str] = field(default_factory=list)
    # Фальшиві рухи ціни (див. `_spike_stats`): testnet віддає серію з рухами
    # +27% за 1m і «плитами» O=H=L=C з нульовим обсягом. Такі бари не є
    # ринковими, і будь-який бектест на них дає фальшивий edge.
    n_spikes: int = 0
    spike_rate: float = 0.0
    max_flat_run: int = 0

    def summary(self) -> str:
        status = "ok" if self.ok else "FAIL"
        return (
            f"bars {status}: n={self.n_rows} dup={self.n_duplicates} "
            f"ohlc={self.n_ohlc_violations} gaps={self.n_gaps} future={self.n_future} "
            f"spikes={self.n_spikes}({self.spike_rate:.3%})"
        )


# ── Детекція неринкових барів ────────────────────────────────────────────────
# Поріг спайка: |r| > max(SPIKE_ABS_MIN, SPIKE_MAD_MULT × MAD(r)), де MAD —
# медіанне абсолютне відхилення дохідностей (стійке до викидів, на відміну від σ).
# Абсолютний флор потрібен, щоб дрібні ТФ з малою MAD не флагували нормальні рухи.
SPIKE_ABS_MIN = 0.05
SPIKE_MAD_MULT = 20.0
# Критично, якщо таких барів більше ніж ця частка: справжні «хвости» крипто
# дають ~0.0005% (ETHUSDT: 7 із 1.59M), а синтетична історія — 0.2–6%
# (BNBUSDT: 86633 із 1.59M = 5.5%).
SPIKE_RATE_CRITICAL = 0.001
# «Плита»: довгі серії O=H=L=C (ціна не рухається взагалі). Мертвий ринок
# такого не дає; testnet — дає тисячами барів підряд.
FLAT_RUN_CRITICAL = 240
_FLAT_RUN_WARN = 60


def _spike_stats(closes: pd.Series) -> tuple[int, float]:
    """(кількість спайків, їх частка) за стійким порогом MAD."""
    r = closes.astype(float).pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(r) < 30:
        return 0, 0.0
    mad = float((r - r.median()).abs().median())
    threshold = max(SPIKE_ABS_MIN, SPIKE_MAD_MULT * mad)
    n = int((r.abs() > threshold).sum())
    return n, n / len(r)


def _flat_run(closes: pd.Series) -> int:
    """Найдовша серія підряд однакових close (справжній ринок такого не дає)."""
    c = closes.to_numpy(dtype=float)
    if len(c) < 2:
        return 0
    change = np.empty(len(c), dtype=bool)
    change[0] = True
    change[1:] = c[1:] != c[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(c))
    return int((ends - starts).max())


def validate_bars(
    df: pd.DataFrame,
    *,
    interval: str | None = None,
    now: pd.Timestamp | None = None,
) -> BarQualityReport:
    """Перевіряє high/low, монотонний індекс, дублікати, майбутні мітки.

    Додатково — неринкові бари (спайки, «плити»): раніше кеш, скачаний з
    testnet, проходив валідацію з `ok=True` (усі перевірки формальні), і
    sweep рахував фальшивий edge.
    """
    issues: list[str] = []
    if df is None or df.empty:
        return BarQualityReport(0, 0, 0, 0, 0, True, False, ["порожній датасет"])

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        issues.append("індекс не DatetimeIndex")
        monotonic = False
    else:
        monotonic = bool(idx.is_monotonic_increasing)
        if not monotonic:
            issues.append("індекс не монотонний")

    n_dup = int(idx.duplicated().sum()) if len(idx) else 0
    if n_dup:
        issues.append(f"дублікати: {n_dup}")

    n_ohlc = 0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
        bad = (hi < lo) | (hi < op) | (hi < cl) | (lo > op) | (lo > cl)
        n_ohlc = int(bad.sum())
        if n_ohlc:
            issues.append(f"OHLC-порушення: {n_ohlc}")

    n_future = 0
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        last = idx[-1]
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_convert("UTC").tz_localize(None)
        if last > now_ts:
            n_future = int((idx > now_ts).sum())
            issues.append(f"майбутні бари: {n_future}")

    n_gaps = 0
    if interval and isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        from scalper_hft.data.downloader import _interval_ms

        expected = pd.Timedelta(milliseconds=_interval_ms(interval))
        deltas = idx.to_series().diff().iloc[1:]
        n_gaps = int((deltas > expected * 1.5).sum())
        if n_gaps:
            issues.append(f"дірки: {n_gaps}")

    # ── Неринкові бари (спайки / «плити») ───────────────────────────────────
    n_spikes, spike_rate, flat = 0, 0.0, 0
    if "close" in df.columns and len(df) > 30:
        n_spikes, spike_rate = _spike_stats(df["close"])
        if spike_rate > SPIKE_RATE_CRITICAL:
            issues.append(
                f"неринкові рухи ціни: {n_spikes} барів ({spike_rate:.2%}) "
                f"> |r| {SPIKE_ABS_MIN:.0%} — схоже на testnet/синтетичну історію"
            )
        flat = _flat_run(df["close"])
        if flat >= FLAT_RUN_CRITICAL:
            issues.append(f"«плита»: {flat} барів підряд з однаковим close (неринкові дані)")
        elif flat >= _FLAT_RUN_WARN:
            issues.append(f"довга серія однакових close: {flat} барів")

    ok = not issues
    return BarQualityReport(len(df), n_dup, n_ohlc, n_gaps, n_future, monotonic, ok, issues, n_spikes, spike_rate, flat)


def bars_are_critical(report: BarQualityReport) -> bool:
    """Критичні дефекти барів для fail-closed запису кешу (`save_klines`).

    Спільна точка для storage: формальні порушення + неринкові дані.
    """
    if report.n_rows == 0:
        return True
    return bool(
        not report.monotonic
        or report.n_duplicates
        or report.n_ohlc_violations
        or report.n_future
        or report.spike_rate > SPIKE_RATE_CRITICAL
        or report.max_flat_run >= FLAT_RUN_CRITICAL
    )


@dataclass
class StreamQualityReport:
    """Якість потоку trades / funding / depth / bookTicker (Phase 5.4)."""

    kind: str
    n_rows: int
    n_duplicates: int
    n_invalid: int
    n_gaps: int
    n_future: int
    monotonic: bool
    ok: bool
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "ok" if self.ok else "FAIL"
        return (
            f"{self.kind} {status}: n={self.n_rows} dup={self.n_duplicates} "
            f"invalid={self.n_invalid} gaps={self.n_gaps} future={self.n_future}"
        )


_FUNDING_MAX_ABS = 0.05  # 5% / 8h — вище за будь-який нормальний Binance USDT-M
_DEPTH_BID_PX = [f"bid{i}" for i in range(1, 6)]
_DEPTH_ASK_PX = [f"ask{i}" for i in range(1, 6)]
_DEPTH_BID_QTY = [f"bid{i}_qty" for i in range(1, 6)]
_DEPTH_ASK_QTY = [f"ask{i}_qty" for i in range(1, 6)]


def _empty_stream(kind: str) -> StreamQualityReport:
    return StreamQualityReport(kind, 0, 0, 0, 0, 0, True, False, ["порожній датасет"])


def _now_naive(now: pd.Timestamp | None) -> pd.Timestamp:
    if now is not None:
        ts = now
    else:
        ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return ts
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _index_quality(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    allow_duplicates: bool = False,
) -> tuple[bool, int, int, list[str]]:
    """Монотонність, дублікати, майбутні мітки. Не змінює validate_bars API.

    allow_duplicates: дублікати міток часу не вважаються дефектом — потік
    aggTrades, де унікальний ключ це `trade_id`, а `transact_time` має лише
    мілісекундну роздільність.
    """
    issues: list[str] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        issues.append("індекс не DatetimeIndex")
        return False, 0, 0, issues
    monotonic = bool(idx.is_monotonic_increasing)
    if not monotonic:
        issues.append("індекс не монотонний")
    n_dup = int(idx.duplicated().sum()) if len(idx) else 0
    if n_dup and not allow_duplicates:
        issues.append(f"дублікати: {n_dup}")
        n_dup_reported = n_dup
    else:
        n_dup_reported = 0
    n_future = 0
    now_ts = _now_naive(now)
    if len(idx):
        last = idx[-1]
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_convert("UTC").tz_localize(None)
        if last > now_ts:
            comparable = idx.tz_localize(None) if idx.tz is not None else idx
            n_future = int((comparable > now_ts).sum())
            issues.append(f"майбутні мітки: {n_future}")
    return monotonic, n_dup_reported, n_future, issues


def _gap_count(idx: pd.DatetimeIndex, expected: pd.Timedelta) -> int:
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return 0
    deltas = idx.to_series().diff().iloc[1:]
    return int((deltas > expected * 1.5).sum())


def validate_trades(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> StreamQualityReport:
    """aggTrades: price/amount > 0, side ∈ {buy,sell}, унікальний trade_id, монотонний час.

    Дублікати МІТОК ЧАСУ тут — норма, а не дефект. `transact_time` має
    мілісекундну роздільність, і кілька aggTrades регулярно ділять одну
    мілісекунду (на реальному архіві SOLUSDT: 6 085 дублів індексу на 279 778
    рядків ≈ 2.2%). Унікальний ключ угоди — `agg_trade_id`, і саме за ним
    дедуплікує `storage.dedupe_trades`.

    Тому перевірка «індекс без дублікатів» лишається чинною лише тоді, коли
    `trade_id` відсутній або сам має дублікати — тоді час є єдиним доступним
    ключем.

    Історична пастка: до аудиту 2026-09-11 дедуп ішов за мілісекундним індексом
    і ЗНИЩУВАВ ці угоди, тому перевірка ніколи не спрацьовувала. Після переходу
    на `trade_id` вона почала блокувати коректне завантаження: 43.3 млн рядків →
    `FAIL: n=43289298 dup=1162183 invalid=0`, хоч `invalid=0` означає, що з
    trade_id усе гаразд, а `dup` — це спільні мілісекунди.
    """
    if df is None or df.empty:
        return _empty_stream("trades")

    # Унікальність справжнього ключа визначає, чи взагалі має значення час.
    n_id_dup = 0
    id_is_key = False
    if "trade_id" in df.columns:
        n_id_dup = int(df["trade_id"].duplicated().sum())
        id_is_key = n_id_dup == 0
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now, allow_duplicates=id_is_key)
    if n_id_dup:
        issues.insert(0, f"дублікати trade_id: {n_id_dup}")

    n_invalid = n_id_dup
    price_col = "price" if "price" in df.columns else None
    amt_col = "amount" if "amount" in df.columns else ("qty" if "qty" in df.columns else None)
    if price_col is None or amt_col is None:
        issues.append("немає колонок price/amount")
        n_invalid = len(df)
    else:
        px = pd.to_numeric(df[price_col], errors="coerce")
        amt = pd.to_numeric(df[amt_col], errors="coerce")
        bad = px.isna() | amt.isna() | (px <= 0) | (amt <= 0)
        n_bad_px = int(bad.sum())
        n_invalid += n_bad_px
        if n_bad_px:
            issues.append(f"price/amount ≤0 або NaN: {n_bad_px}")
    if "side" in df.columns:
        side_norm = df["side"].astype(str).str.lower()
        bad_side = ~side_norm.isin({"buy", "sell", "1", "-1"})
        n_side = int(bad_side.sum())
        if n_side:
            n_invalid += n_side
            issues.append(f"невідомий side: {n_side}")
    else:
        issues.append("немає колонки side")
        n_invalid += len(df)
    if "trade_id" in df.columns:
        # Покриття за aggTrade id: `trade_id` — справжній ключ угоди, і він
        # монотонний з кроком 1, тож розриви = угоди, яких у кеші немає. Саме
        # так виявляється тиха втрата потоку (аудит 2026-09-11: BTCUSDT 39.3%,
        # ETHUSDT 49.8% — дедуп за мілісекундним індексом + часовий курсор).
        # Дублікати trade_id уже враховані вище (n_id_dup → n_invalid).
        ids = pd.to_numeric(df["trade_id"], errors="coerce")
        real = ids[ids > 0]
        if len(real) > 1:
            span = int(real.max() - real.min()) + 1
            coverage = float(len(real)) / span if span > 0 else 1.0
            if coverage < 0.99:
                issues.append(
                    f"розриви aggTrade id: покриття {coverage:.1%} "
                    f"({len(real)} із {span}; втрачено ~{span - len(real)})"
                )
    ok = not issues
    return StreamQualityReport("trades", len(df), n_dup, n_invalid, 0, n_future, monotonic, ok, issues)


def validate_funding(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    interval: str = "8h",
) -> StreamQualityReport:
    """Funding: колонка fundingRate, |rate| не божевільний, крок ~8h."""
    if df is None or df.empty:
        return _empty_stream("funding")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    n_invalid = 0
    if "fundingRate" not in df.columns:
        issues.append("немає колонки fundingRate")
        n_invalid = len(df)
    else:
        rate = pd.to_numeric(df["fundingRate"], errors="coerce")
        n_nan = int(rate.isna().sum())
        n_crazy = int((rate.abs() > _FUNDING_MAX_ABS).sum())
        n_invalid = n_nan + n_crazy
        if n_nan:
            issues.append(f"NaN fundingRate: {n_nan}")
        if n_crazy:
            issues.append(f"|fundingRate| > {_FUNDING_MAX_ABS}: {n_crazy}")
    n_gaps = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        n_gaps = _gap_count(df.index, pd.Timedelta(interval))
        if n_gaps:
            issues.append(f"дірки (>1.5×{interval}): {n_gaps}")
    ok = not issues
    return StreamQualityReport("funding", len(df), n_dup, n_invalid, n_gaps, n_future, monotonic, ok, issues)


def validate_bookticker(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> StreamQualityReport:
    """L1 bookTicker: bid/ask > 0, ask ≥ bid, qty ≥ 0."""
    if df is None or df.empty:
        return _empty_stream("book")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    needed = {"bid", "ask", "bid_qty", "ask_qty"}
    n_invalid = 0
    if not needed.issubset(df.columns):
        issues.append(f"немає колонок {sorted(needed - set(df.columns))}")
        n_invalid = len(df)
    else:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        bq = pd.to_numeric(df["bid_qty"], errors="coerce")
        aq = pd.to_numeric(df["ask_qty"], errors="coerce")
        bad = (
            bid.isna()
            | ask.isna()
            | (bid <= 0)
            | (ask <= 0)
            | (ask < bid)
            | bq.isna()
            | aq.isna()
            | (bq < 0)
            | (aq < 0)
        )
        n_invalid = int(bad.sum())
        if n_invalid:
            issues.append(f"перехрещений/невалідний L1: {n_invalid}")
    ok = not issues
    return StreamQualityReport("book", len(df), n_dup, n_invalid, 0, n_future, monotonic, ok, issues)


def validate_depth(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    max_gap_sec: float = 1.0,
) -> StreamQualityReport:
    """depth5: px>0, qty≥0, ask1≥bid1, рівні впорядковані, гепи ts."""
    if df is None or df.empty:
        return _empty_stream("depth")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    n_invalid = 0
    has_l1 = "bid1" in df.columns and "ask1" in df.columns
    if not has_l1:
        issues.append("немає bid1/ask1")
        n_invalid = len(df)
    else:
        bid1 = pd.to_numeric(df["bid1"], errors="coerce")
        ask1 = pd.to_numeric(df["ask1"], errors="coerce")
        crossed = bid1.isna() | ask1.isna() | (bid1 <= 0) | (ask1 <= 0) | (ask1 < bid1)
        n_cross = int(crossed.sum())
        if n_cross:
            n_invalid += n_cross
            issues.append(f"перехрещений/невалідний top-of-book: {n_cross}")
        # qty
        for col in _DEPTH_BID_QTY + _DEPTH_ASK_QTY:
            if col not in df.columns:
                continue
            q = pd.to_numeric(df[col], errors="coerce")
            n_bad_q = int((q.isna() | (q < 0)).sum())
            if n_bad_q:
                n_invalid += n_bad_q
                issues.append(f"{col} < 0 або NaN: {n_bad_q}")
        # рівні: bid спадний, ask зростаючий (де обидва валідні)
        for i in range(1, 5):
            b_lo, b_hi = f"bid{i}", f"bid{i + 1}"
            a_lo, a_hi = f"ask{i}", f"ask{i + 1}"
            if b_lo in df.columns and b_hi in df.columns:
                blo = pd.to_numeric(df[b_lo], errors="coerce")
                bhi = pd.to_numeric(df[b_hi], errors="coerce")
                n_ord = int(((bhi > blo) & blo.notna() & bhi.notna() & (bhi > 0)).sum())
                if n_ord:
                    n_invalid += n_ord
                    issues.append(f"bid рівні не спадні ({b_lo}<{b_hi}): {n_ord}")
            if a_lo in df.columns and a_hi in df.columns:
                alo = pd.to_numeric(df[a_lo], errors="coerce")
                ahi = pd.to_numeric(df[a_hi], errors="coerce")
                n_ord = int(((ahi < alo) & alo.notna() & ahi.notna() & (ahi > 0)).sum())
                if n_ord:
                    n_invalid += n_ord
                    issues.append(f"ask рівні не зростаючі ({a_lo}>{a_hi}): {n_ord}")
    n_gaps = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        n_gaps = _gap_count(df.index, pd.Timedelta(seconds=max_gap_sec))
        if n_gaps:
            issues.append(f"дірки ts (>{max_gap_sec}s×1.5): {n_gaps}")
    ok = not issues
    return StreamQualityReport("depth", len(df), n_dup, n_invalid, n_gaps, n_future, monotonic, ok, issues)


def stream_is_critical(report: StreamQualityReport) -> bool:
    """Критичні дефекти для fail-closed запису кешу (як у save_klines)."""
    if report.n_rows == 0:
        return False  # порожній checkpoint downloader-а не блокує
    return (not report.monotonic) or bool(report.n_duplicates) or bool(report.n_invalid) or bool(report.n_future)


def _l2_kind(path: Path) -> str | None:
    """depth5 → depth, bookTicker → book; інакше не L2-архів."""
    name = path.name.lower()
    if "depth5" in name:
        return "depth"
    if "bookticker" in name:
        return "book"
    return None


def list_l2_parquet(data_dir: Path | str) -> list[Path]:
    """`data/*depth5*.parquet` та `*bookTicker*` (регістр імен не важливий)."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.parquet") if p.is_file() and _l2_kind(p) is not None)


@dataclass(frozen=True, slots=True)
class L2FileReport:
    path: str
    report: StreamQualityReport


@dataclass(frozen=True, slots=True)
class L2CacheAudit:
    """Зведений quality-звіт depth5/bookTicker (W0-R5)."""

    data_dir: str
    files: tuple[L2FileReport, ...]
    extra_issues: tuple[str, ...] = ()

    @property
    def quality_ok(self) -> bool:
        if self.extra_issues or not self.files:
            return False
        return all(item.report.ok for item in self.files)

    def to_markdown(self) -> str:
        lines = [
            "# L2 quality (depth5 / bookTicker)",
            "",
            f"**quality_ok:** `{'true' if self.quality_ok else 'false'}`",
            "",
            f"Каталог: `{self.data_dir}`",
            "",
            "| Файл | kind | n | dup | invalid | gaps | future | ok | issues |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        if not self.files:
            lines.append("| — | — | 0 | — | — | — | — | false | немає файлів |")
        for item in self.files:
            rep = item.report
            issues = "; ".join(rep.issues) if rep.issues else "—"
            name = Path(item.path).name
            lines.append(
                f"| `{name}` | {rep.kind} | {rep.n_rows} | {rep.n_duplicates} | "
                f"{rep.n_invalid} | {rep.n_gaps} | {rep.n_future} | {rep.ok} | {issues} |"
            )
        if self.extra_issues:
            lines.extend(["", "## Проблеми", ""])
            lines.extend(f"- {issue}" for issue in self.extra_issues)
        lines.append("")
        return "\n".join(lines)


def audit_l2_cache(data_dir: Path | str) -> L2CacheAudit:
    """Прогнати `validate_depth` / `validate_bookticker` по локальному архіву."""
    root = Path(data_dir)
    extra: list[str] = []
    if not root.is_dir():
        extra.append(f"немає каталогу {root}")
        return L2CacheAudit(data_dir=str(root), files=(), extra_issues=tuple(extra))

    paths = list_l2_parquet(root)
    if not paths:
        extra.append(f"немає *depth5*.parquet / *bookTicker* у {root}")
        return L2CacheAudit(data_dir=str(root), files=(), extra_issues=tuple(extra))

    files: list[L2FileReport] = []
    for path in paths:
        kind = _l2_kind(path)
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            extra.append(f"{path.name}: не вдалося прочитати ({type(exc).__name__}: {exc})")
            files.append(
                L2FileReport(
                    path=str(path),
                    report=StreamQualityReport(
                        kind=kind or "unknown",
                        n_rows=0,
                        n_duplicates=0,
                        n_invalid=0,
                        n_gaps=0,
                        n_future=0,
                        monotonic=False,
                        ok=False,
                        issues=[f"read failed: {exc}"],
                    ),
                )
            )
            continue
        report = validate_depth(df) if kind == "depth" else validate_bookticker(df)
        files.append(L2FileReport(path=str(path), report=report))
    return L2CacheAudit(data_dir=str(root), files=tuple(files), extra_issues=tuple(extra))
