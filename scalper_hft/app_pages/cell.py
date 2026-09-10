"""Досьє однієї комірки: KPI, свічки в робочому вікні, аудит, якість угод."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.app_pages._busy import busy
from scalper_hft.app_pages._charts import render_backtest_chart, render_pairs_chart
from scalper_hft.app_pages._common import (
    BT_INTERVALS,
    PAIR_CHOICES,
    RESEARCH_CELL_TAB,
    SYMBOLS,
    apply_research_cell_prefill,
    capacity_job_payload,
    job_status_caption,
    latest_succeeded_job,
    lookup_job,
    overfit_job_payload,
    single_backtest_payload,
    submit_research_job,
)
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import klines_from_store
from scalper_hft.research.dashboard_brief import hurdle_note
from scalper_hft.research.job_artifacts import (
    load_backtest_result,
    load_capacity_curve,
    load_cell_audit,
    load_pairs_result,
)
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore, artifacts_dir
from scalper_hft.strategies import REGISTRY
from scalper_hft.validation.cell_audit import cell_verdict, default_train_test
from scalper_hft.validation.verdict_store import latest_pair_verdict, latest_verdict

apply_research_cell_prefill(st.session_state)
if RESEARCH_CELL_TAB in st.session_state:
    raw_tab = str(st.session_state.pop(RESEARCH_CELL_TAB) or "Ціна")
    st.session_state["cell_section"] = {
        "price": "Ціна",
        "audit": "Аудит",
        "quality": "Якість",
        "stress": "Стрес / capacity",
        "Ціна": "Ціна",
        "Аудит": "Аудит",
        "Якість": "Якість",
        "Стрес / capacity": "Стрес / capacity",
    }.get(raw_tab, "Ціна")

settings = get_settings()
_SECTIONS = ("Ціна", "Аудит", "Якість", "Стрес / capacity")

st.title("Комірка")
st.caption("Досьє однієї комбінації. Важкі прогони — лише в чергу worker.")

if "cell_strategy" not in st.session_state:
    st.session_state["cell_strategy"] = sorted(REGISTRY)[0]
if "cell_symbol" not in st.session_state:
    st.session_state["cell_symbol"] = SYMBOLS[0]
if "cell_interval" not in st.session_state:
    st.session_state["cell_interval"] = BT_INTERVALS[min(1, len(BT_INTERVALS) - 1)]
if "cell_days" not in st.session_state:
    st.session_state["cell_days"] = 60

with st.container(horizontal=True, vertical_alignment="bottom"):
    strategy_name = st.selectbox("Стратегія", sorted(REGISTRY), key="cell_strategy")
    is_pairs = strategy_name == "pairs_arb"
    if is_pairs:
        pair_sel = st.selectbox("Пара", PAIR_CHOICES, key="cell_pair")
        symbol = str(pair_sel)
        interval = st.selectbox("Таймфрейм", ["1h", "15m", "5m"], key="cell_interval")
        leg1, leg2 = symbol.split("/")
    else:
        symbol = str(st.selectbox("Символ", SYMBOLS, key="cell_symbol"))
        interval = str(st.selectbox("Таймфрейм", BT_INTERVALS, key="cell_interval"))
        leg1, leg2 = "", ""
    days = int(st.number_input("Днів", 7, 1095, key="cell_days"))

if is_pairs:
    verdict = latest_pair_verdict("pairs_arb", symbol, interval)
else:
    verdict = latest_verdict(strategy_name, symbol, interval)

vlabel = str(verdict["label"]) if verdict else "немає аудиту"
st.badge(vlabel, color={"PASS": "green", "FAIL": "red", "EXPLORATORY_PASS": "orange"}.get(vlabel, "gray"))
if verdict and verdict.get("reasons"):
    st.caption(str(verdict["reasons"]))

bt_job = latest_succeeded_job(
    "pairs" if is_pairs else "backtest",
    strategy=strategy_name,
    symbol=symbol,
    interval=interval,
)
if bt_job is None and is_pairs:
    with JobStore(DEFAULT_JOBS_PATH) as js:
        for job in js.list_jobs(kind="pairs", limit=400):
            if job.status != "succeeded":
                continue
            p = job.params or {}
            if p.get("leg1") == leg1 and p.get("leg2") == leg2 and str(p.get("interval")) == interval:
                bt_job = job
                break

of_payload = None
if not is_pairs:
    train_b, test_b = default_train_test(interval)
    of_payload = overfit_job_payload(strategy_name, symbol, interval, days, train_bars=train_b, test_bars=test_b)
of_job, of_alive = lookup_job("overfit", of_payload) if of_payload else (None, True)
if of_job is None and not is_pairs:
    of_job = latest_succeeded_job("overfit", strategy=strategy_name, symbol=symbol, interval=interval)

with st.container(horizontal=True):
    if st.button("Поставити бектест", icon=":material/play_arrow:", key="cell_run_bt"):
        if is_pairs:
            submit_research_job(
                "pairs",
                {
                    "strategy": strategy_name,
                    "leg1": leg1,
                    "leg2": leg2,
                    "interval": interval,
                    "days": days,
                    "params": {},
                    "maker": True,
                    "position_pct": settings.pair_notional_pct,
                    "base_interval": "1m",
                },
            )
        else:
            submit_research_job(
                "backtest",
                single_backtest_payload(strategy_name, symbol, interval, days, trace=True),
            )
        st.rerun()
    if of_payload is not None and st.button("Поставити аудит", icon=":material/fact_check:", key="cell_run_of"):
        submit_research_job("overfit", of_payload)
        st.rerun()
    st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")
    st.page_link("app_pages/research_hub.py", label="Каталог", icon=":material/library_books:")

if bt_job is None:
    st.info("Немає succeeded бектесту. Sweep зберігає лише метрики — поставте бектест у чергу.")
    st.stop()

_job_dir = artifacts_dir(DEFAULT_JOBS_PATH, bt_job.id)
cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
res: Any = None
df_k: pd.DataFrame | None = None
title = f"{strategy_name} · {symbol} {interval}"
with busy("Завантаження артефактів…"):
    try:
        if is_pairs:
            res = load_pairs_result(_job_dir)
        else:
            res = load_backtest_result(_job_dir)
            df_k = klines_from_store(symbol, interval, days)
    except FileNotFoundError:
        res = None

if res is None:
    st.warning("Артефакти ще не записані.")
    st.stop()

m = res.metrics
with st.container(horizontal=True):
    st.metric("Дохідність", f"{m.total_return:.2%}", border=True)
    st.metric("Sharpe", f"{m.sharpe:.2f}", border=True)
    st.metric("Угоди", f"{m.n_trades}", border=True)
    st.metric("Win rate", f"{m.win_rate:.0%}", border=True)
    st.metric("Profit factor", f"{m.profit_factor:.2f}", border=True)
    st.metric("Max DD", f"{m.max_drawdown:.2%}", border=True)
if m.n_trades:
    st.caption(hurdle_note(m.avg_trade_return, cost.round_trip_maker(), n_legs=2 if is_pairs else 1))

section = st.segmented_control("Розділ", list(_SECTIONS), key="cell_section")
if section is None:
    section = _SECTIONS[0]

if section == "Ціна":
    if is_pairs:
        render_pairs_chart(res, title, key_prefix="cell_pairs")
    elif df_k is None or len(df_k) < 2:
        st.warning(f"Немає свічок {symbol} {interval} у кеші — оновіть дані на Моніторингу.")
    else:
        render_backtest_chart({"df": df_k, "res": res, "title": title}, key_prefix="cell_bt")
    if res.equity is not None and len(res.equity) > 2:
        eq = res.equity / res.equity.iloc[0]
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Стратегія"))
        if df_k is not None and "close" in df_k.columns:
            bh = df_k["close"].reindex(res.equity.index).ffill()
            if bh.notna().sum() > 2:
                bh_n = bh / bh.dropna().iloc[0]
                fig_eq.add_trace(go.Scatter(x=bh_n.index, y=bh_n.values, mode="lines", name="Buy & hold"))
        fig_eq.update_layout(title="Нормалізована equity vs buy & hold", height=280, yaxis_title="База 1.0")
        st.plotly_chart(fig_eq, width="stretch")

elif section == "Аудит":
    if is_pairs:
        st.info("Парний гейт пише вердикт окремо; directional overfit тут не застосовується.")
    elif of_job is None or of_job.status != "succeeded":
        _level, msg = job_status_caption(of_job)
        st.info(msg)
        if not of_alive:
            st.warning("Воркер не запущений.")
    else:
        try:
            audit = load_cell_audit(artifacts_dir(DEFAULT_JOBS_PATH, of_job.id))
        except (FileNotFoundError, KeyError, ValueError):
            st.warning("Артефакти аудиту ще не записані.")
        else:
            verdict_l, why = cell_verdict(audit)
            if verdict_l == "PASS":
                st.badge("PASS", icon=":material/verified:", color="green")
            else:
                st.badge(verdict_l, icon=":material/cancel:", color="red")
            if why:
                st.caption(f"Не пройдено: {why}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("OOS Sharpe", f"{audit.avg_oos_sharpe or 0:.3f}", border=True)
            c2.metric("OOS+", f"{(audit.oos_pos_frac or 0):.0%}", border=True)
            c3.metric("DSR", f"{audit.dsr if audit.dsr is not None else float('nan'):.3f}", border=True)
            c4.metric(
                "Smoothness",
                f"{audit.smoothness if audit.smoothness is not None else float('nan'):.2f}",
                border=True,
            )
            c5.metric("PBO", f"{audit.pbo if audit.pbo is not None else float('nan'):.2f}", border=True)
            c6.metric("Угод OOS", f"{audit.n_trades_oos or 0}", border=True)
            if audit.windows:
                wf_df = pd.DataFrame(list(audit.windows))
                fig_wf = go.Figure()
                fig_wf.add_trace(go.Bar(x=wf_df["window_idx"], y=wf_df["oos_sharpe"], name="OOS Sharpe"))
                fig_wf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_wf.update_layout(height=260, yaxis_title="OOS Sharpe", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_wf, width="stretch")
            if audit.sensitivity_grid and audit.sens_param:
                sens_df = pd.DataFrame(list(audit.sensitivity_grid))
                if "metric" in sens_df.columns and audit.sens_param in sens_df.columns:
                    fig_s = px.line(sens_df, x=audit.sens_param, y="metric", markers=True, title="Sensitivity")
                    fig_s.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_s, width="stretch")

elif section == "Якість":
    if res.trades is None or res.trades.empty or df_k is None:
        st.info("Потрібні угоди й свічки. Для пар дивіться графік спреду у розділі «Ціна».")
    else:
        from scalper_hft.research.filter_trace import filter_attribution, filter_pnl_impact
        from scalper_hft.research.session_analysis import mae_mfe_analysis, session_breakdown

        mae_df = mae_mfe_analysis(res.trades, df_k)
        if not mae_df.empty:
            fig_mf = px.scatter(
                mae_df,
                x="mfe",
                y="mae",
                color="ret",
                color_continuous_scale="RdYlGn",
                hover_data=["entry_ts", "side", "ret", "efficiency"],
                title="MAE vs MFE",
            )
            fig_mf.update_layout(height=360)
            st.plotly_chart(fig_mf, width="stretch")
        sess = session_breakdown(res.trades)
        if not sess.empty:
            st.dataframe(sess.reset_index(), width="stretch", hide_index=True)
        trace = getattr(res, "trace", None)
        if trace is not None and len(trace) > 0:
            st.subheader("Фільтри")
            attr_df = filter_attribution(trace)
            if not attr_df.empty:
                st.dataframe(attr_df, width="stretch", hide_index=True)
            pnl_df = filter_pnl_impact(trace, df_k["close"], horizon_bars=5)
            if not pnl_df.empty:
                st.dataframe(pnl_df, width="stretch", hide_index=True)
        else:
            st.caption("Немає filter trace — поставте бектест з трейсом з каталогу.")

else:
    from scalper_hft.validation.cohort import cohort_report
    from scalper_hft.validation.stress import stress_report

    ret = res.equity.pct_change().dropna()
    st.text(cohort_report(res.trades))
    st.dataframe(stress_report(ret).round(4), width="stretch")
    if is_pairs:
        st.caption("Capacity для пар не рахується.")
    else:
        cap_payload = capacity_job_payload(strategy_name, symbol, interval, days)
        if st.button("Поставити capacity", icon=":material/speed:", key="cell_cap"):
            submit_research_job("capacity", cap_payload)
            st.rerun()
        cap_job, _alive = lookup_job("capacity", cap_payload)
        _level, msg = job_status_caption(cap_job)
        if _level == "ok" and cap_job is not None:
            curve = load_capacity_curve(artifacts_dir(DEFAULT_JOBS_PATH, cap_job.id))
            st.dataframe(curve, width="stretch", hide_index=True)
        else:
            st.caption(msg)
