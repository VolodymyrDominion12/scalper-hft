"""Фіналісти: комірки з overfit-вердиктом, порівняння equity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.app_pages._common import combo_prefill, latest_succeeded_job
from scalper_hft.app_pages._results import open_combo_details
from scalper_hft.research.job_artifacts import load_backtest_result
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, artifacts_dir
from scalper_hft.research.results_table import latest_cells
from scalper_hft.research.sweep_store import SweepStore
from scalper_hft.validation.verdict_store import latest_verdicts

st.title("Фіналісти")
st.caption(
    "Комірки з записаним overfitting-аудитом. EXPLORATORY_PASS не є допуском до paper. "
    "Криві — з останнього succeeded backtest, не з in-sample sweep."
)

verdicts = latest_verdicts()
if not verdicts:
    st.info("Ще немає записів у `results/audit_verdicts.jsonl`. Запустіть аудит з каталогу.")
    st.page_link("app_pages/research_hub.py", label="Каталог комірок", icon=":material/library_books:")
    st.stop()

frame = pd.DataFrame(verdicts)
frame["ts"] = pd.to_datetime(frame.get("ts"), errors="coerce", utc=True)

labels = ["Усі", "PASS", "FAIL", "EXPLORATORY_PASS"]
pick = st.segmented_control("Вердикт", labels, key="fin_label")
if pick is None:
    pick = "Усі"
view = frame if pick == "Усі" else frame[frame["label"] == pick]
if view.empty:
    st.warning("Немає комірок з цим вердиктом.")
    st.stop()

sweep_path = Path("results") / "sweep.db"
metrics = pd.DataFrame()
if sweep_path.exists():
    try:
        with SweepStore(sweep_path) as store:
            metrics = latest_cells(store.load())
    except Exception as exc:
        st.caption(f"sweep.db не прочитано: {exc}")

if not metrics.empty:
    keys = ["strategy", "symbol", "interval"]
    for k in keys:
        if k in view.columns:
            view[k] = view[k].astype(str)
        if k in metrics.columns:
            metrics[k] = metrics[k].astype(str)
    keep = [
        c
        for c in ("avg_oos_sharpe", "oos_positive_frac", "n_trades", "max_dd", "profit_factor", "days")
        if c in metrics.columns
    ]
    merged = view.merge(metrics[keys + keep], on=keys, how="left", suffixes=("", "_sw"))
else:
    merged = view.copy()

n_pass = int((frame["label"] == "PASS").sum())
n_fail = int((frame["label"] == "FAIL").sum())
n_exp = int((frame["label"] == "EXPLORATORY_PASS").sum())
with st.container(horizontal=True):
    st.metric("PASS", n_pass, border=True)
    st.metric("FAIL", n_fail, border=True)
    st.metric("Exploratory", n_exp, border=True)

show_cols = [
    c
    for c in (
        "strategy",
        "symbol",
        "interval",
        "label",
        "reasons",
        "avg_oos_sharpe",
        "oos_positive_frac",
        "n_trades",
        "max_dd",
        "profit_factor",
        "ts",
    )
    if c in merged.columns
]
shown = merged[show_cols].copy()
selection = st.dataframe(
    shown,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    key="fin_grid",
    column_config={
        "avg_oos_sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.3f"),
        "oos_positive_frac": st.column_config.NumberColumn("OOS+", format="percent"),
        "max_dd": st.column_config.NumberColumn("Max DD", format="percent"),
        "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
        "ts": st.column_config.DatetimeColumn("Вердикт", format="YYYY-MM-DD HH:mm"),
    },
)
st.caption(f"Показано **{len(shown)}** комірок.")

payloads = [row.to_dict() for _, row in merged.iterrows()]
sel_rows = list(selection.selection.rows) if selection is not None and selection.selection else []

with st.container(horizontal=True):
    if sel_rows and st.button("Відкрити першу як досьє", icon=":material/candlestick_chart:", key="fin_open"):
        open_combo_details(payloads[int(sel_rows[0])], enqueue=False)
    if sel_rows and st.button("До порівняння", icon=":material/compare:", key="fin_add"):
        bag = list(st.session_state.get("finalist_compare") or [])
        for i in sel_rows:
            combo = combo_prefill(payloads[int(i)])
            if combo not in bag:
                bag.append(combo)
        st.session_state["finalist_compare"] = bag
        st.rerun()
    if st.button("Очистити порівняння", icon=":material/delete:", key="fin_clear"):
        st.session_state["finalist_compare"] = []
        st.rerun()

compare: list[dict[str, Any]] = list(st.session_state.get("finalist_compare") or [])
if not compare and sel_rows:
    compare = [combo_prefill(payloads[int(i)]) for i in sel_rows]

st.subheader("Порівняння equity")
if len(compare) < 1:
    st.info("Виберіть рядки або додайте комірки з каталогу.")
    st.stop()

curves: dict[str, pd.Series] = {}
pending: list[str] = []
for combo in compare:
    label = f"{combo.get('strategy')} {combo.get('symbol')} {combo.get('interval')}"
    job = latest_succeeded_job(
        "backtest",
        strategy=str(combo.get("strategy") or ""),
        symbol=str(combo.get("symbol") or ""),
        interval=str(combo.get("interval") or ""),
    )
    if job is None:
        pending.append(label)
        continue
    try:
        res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, job.id))
        if res.equity is not None and not res.equity.empty:
            curves[label] = res.equity
    except Exception:
        pending.append(label)

if pending:
    st.caption("Немає backtest-артефактів: " + " · ".join(pending[:8]))

if len(curves) < 1:
    st.info("Поставте бектест для вибраних комірок (каталог → бектест + trace).")
    st.stop()

fig_eq = go.Figure()
daily: dict[str, pd.Series] = {}
for label, equity in curves.items():
    norm = equity / equity.iloc[0]
    fig_eq.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=label))
    daily[label] = equity.resample("1D").last().pct_change().dropna()
fig_eq.update_layout(title="Нормалізовані equity (база 1.0)", height=400, yaxis_title="Вартість")
st.plotly_chart(fig_eq, width="stretch")

if len(daily) >= 2:
    aligned = pd.DataFrame(daily).dropna(how="any")
    if not aligned.empty:
        corr = aligned.corr()
        st.subheader("Кореляція денних дохідностей")
        st.dataframe(corr.round(2), width="stretch")
        fig_rs = go.Figure()
        for label, ser in daily.items():
            if len(ser) >= 30:
                rs = (ser.rolling(30).mean() / ser.rolling(30).std()) * np.sqrt(365)
                fig_rs.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name=label))
        if fig_rs.data:
            fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_rs.update_layout(title="Rolling Sharpe (30 днів)", height=280)
            st.plotly_chart(fig_rs, width="stretch")
