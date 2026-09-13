"""Сторінка «VPS Paper»: стан paper-ботів на VPS за локальним знімком журналів.

Дані привозить `scripts/sync_vps_paper.sh` (консистентні копії SQLite через
sqlite backup API + `manifest.json` зі станом systemd-юнітів). Сторінка читає
копії локально і **не** ходить на VPS по кожному rerun-у: 1 мережевий похід на
синк замість десятків запитів, і жодного ризику зачепити робочий процес бота.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from scalper_hft.live.vps_paper_snapshot import (
    BOTS,
    DEFAULT_SNAPSHOT_DIR,
    SnapshotReport,
    build_report,
    format_age,
    pairs_gate_status,
    read_equity_series,
    read_trades,
)

STRATEGIES_ORDER = [spec.key for spec in BOTS]


@st.cache_data(ttl=20, max_entries=4, show_spinner=False)
def _load_report(snapshot_dir: str) -> tuple[SnapshotReport, bool, str]:
    """Звіт по знімку + стан hard-гейта (кеш 20 с, щоб auto-refresh був живий)."""
    report = build_report(snapshot_dir)
    gate_ok, gate_msg = pairs_gate_status(snapshot_dir)
    return report, gate_ok, gate_msg


@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def _load_series(snapshot_dir: str, key: str) -> pd.DataFrame:
    spec = next(s for s in BOTS if s.key == key)
    rows = read_equity_series(Path(snapshot_dir) / spec.db, spec)
    if not rows:
        return pd.DataFrame(columns=["ts", "equity"])
    frame = pd.DataFrame(rows, columns=["ts", "equity"])
    return frame.set_index("ts")


@st.cache_data(ttl=60, max_entries=4, show_spinner=False)
def _load_trades(snapshot_dir: str) -> pd.DataFrame:
    rows = read_trades(Path(snapshot_dir) / "paper_pairs.sqlite")
    return pd.DataFrame(rows)


@st.fragment(run_every="30s")
def render_status(auto_refresh: bool) -> None:
    report, gate_ok, gate_msg = _load_report(snapshot_dir)
    if auto_refresh and not report.healthy:
        _load_report.clear()

    with st.container(horizontal=True):
        if report.sync_age_hours is None:
            st.badge("Знімок відсутній", icon=":material/cloud_off:", color="red")
        elif report.sync_age_hours > 2.0:
            st.badge(f"Знімок {format_age(report.sync_age_hours)}", icon=":material/history:", color="orange")
        else:
            st.badge(f"Знімок {format_age(report.sync_age_hours)}", icon=":material/cloud_done:", color="green")

        healthy_count = sum(1 for snap in report.snapshots if snap.healthy)
        st.badge(
            f"Ботів здорових: {healthy_count}/{len(report.snapshots)}",
            icon=":material/health_and_safety:",
            color="green" if healthy_count == len(report.snapshots) else "red",
        )

        control = report.control or {}
        if control.get("flatten"):
            st.badge("FLATTEN на VPS", icon=":material/emergency_home:", color="red")
        elif control.get("pause"):
            st.badge("Пауза на VPS", icon=":material/pause_circle:", color="orange")
        elif control.get("no_new_entries"):
            st.badge("Нові входи заборонено", icon=":material/block:", color="yellow")
        elif control:
            st.badge("Торгівля активна", icon=":material/check_circle:", color="green")

        st.badge(
            "Гейт pairs: PASS" if gate_ok else "Гейт pairs: НЕ PASS",
            icon=":material/verified:" if gate_ok else ":material/gpp_maybe:",
            color="green" if gate_ok else "red",
        )

    problems = list(report.issues)
    for snap in report.snapshots:
        problems.extend(f"{snap.spec.label} — {msg}" for msg in snap.issues)
    if problems:
        with st.container(border=True):
            st.markdown("**Проблеми**")
            for msg in problems:
                st.markdown(f":red[✗] {msg}")
    else:
        st.success(
            "Усі paper-боти пишуть журнали в межах інтервалу, юніти на VPS — active.",
            icon=":material/verified:",
        )

    with st.container(horizontal=True):
        for snap in report.snapshots:
            stats = snap.journal
            equity = "—" if stats.equity is None else f"{stats.equity:,.2f}"
            st.metric(
                snap.spec.label,
                equity,
                delta=None if stats.realized_pnl is None else f"PnL {stats.realized_pnl:+,.2f}",
                border=True,
                help=(
                    f"Юніт {snap.spec.unit}: {snap.unit_active}\n\n"
                    f"Останній запис: {format_age(snap.age_hours)} тому (інтервал {snap.spec.interval})\n"
                    f"Угод: {stats.trades} | Відкритих позицій: {stats.open_positions} | Ордерів: {stats.orders}"
                ),
            )

    ages = " · ".join(f"{snap.spec.key}: {format_age(snap.age_hours)}" for snap in report.snapshots)
    st.caption(f"Вік останнього запису в журналі — {ages}.")

    with st.expander("Деталі журналів і останні рядки логів", expanded=False):
        for snap in report.snapshots:
            stats = snap.journal
            st.markdown(f"**{snap.spec.label}** · `{snap.spec.db}` · `{snap.spec.unit}`")
            st.caption(
                f"рядків equity: {stats.equity_rows} · інструментів: {stats.symbols} · "
                f"угод: {stats.trades} · ордери: {stats.orders_by_status or '—'} · "
                f"розмір копії: {snap.db_bytes if snap.db_bytes is not None else '—'} Б"
            )
            if snap.unit_last_log:
                ts = snap.unit_last_log_ts.isoformat(timespec="seconds") if snap.unit_last_log_ts else "?"
                st.code(f"{ts}  {snap.unit_last_log}", language="text")
        st.caption(f"Гейт overfitting-аудиту для pairs: {gate_msg}")


snapshot_dir = str(Path(st.session_state.get("vps_snapshot_dir", DEFAULT_SNAPSHOT_DIR)))

st.title("VPS Paper")
st.caption(
    "Моніторинг трьох paper-ботів на VPS за локальним знімком журналів. "
    "Читання — read-only; керування ботами (пауза) робиться файлом `results/control.json` на VPS."
)

with st.container(horizontal=True):
    auto_refresh = st.toggle("Авто-оновлення (30 с)", value=True)
    if st.button("Синхронізувати зараз", icon=":material/sync:"):
        script = Path("scripts/sync_vps_paper.sh")
        if not script.exists():
            st.error(f"Немає {script} — запускайте дашборд із кореня репозиторію.")
        else:
            with st.spinner("Тягну консистентні копії журналів із VPS…"):
                proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=180)
            if proc.returncode == 0:
                _load_report.clear()
                _load_series.clear()
                _load_trades.clear()
                st.success("Знімок оновлено.", icon=":material/cloud_done:")
            else:
                st.error(f"Синк впав (exit {proc.returncode}): {proc.stderr.strip()[-800:]}")

render_status(auto_refresh)

with st.container(border=True):
    st.subheader("Equity портфелів")
    labels = {spec.label: spec.key for spec in BOTS}
    chosen = st.segmented_control(
        "Бот",
        options=list(labels),
        default=next(iter(labels), None),
        key="vps_paper_bot_choice",
    )
    if chosen:
        frame = _load_series(snapshot_dir, labels[chosen])
        if frame.empty:
            st.info("У журналі ще немає жодного запису equity.")
        else:
            st.line_chart(frame, y="equity", x_label="час (UTC)", y_label="equity")
            st.caption(f"Точок: {len(frame)} · останній запис: {frame.index[-1]}")

with st.container(border=True):
    st.subheader("Угоди pairs (з копії журналу)")
    trades = _load_trades(snapshot_dir)
    if trades.empty:
        st.caption("Угод ще немає — pairs тримає позицію або чекає сигналу.")
    else:
        st.dataframe(trades, hide_index=True)

st.caption(
    "Джерело даних: `results/vps/` (копії SQLite + `manifest.json` зі станом юнітів). "
    "Оновлення: `bash scripts/sync_vps_paper.sh` або цикл `scripts/vps_paper_watch.sh`, "
    "статус у терміналі: `uv run python scripts/vps_paper_status.py`. "
    "Деталі: `docs/RUNBOOK_PAPER_MONITORING.md`."
)
