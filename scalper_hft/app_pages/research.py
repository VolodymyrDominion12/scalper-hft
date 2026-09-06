"""Сторінка «Дослідження»: sweep, filter attribution, порівняння, аудит, режими."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.app_pages._busy import busy
from scalper_hft.app_pages._common import (
    BT_INTERVALS,
    RESEARCH_SECTIONS,
    SYMBOLS,
    apply_research_audit_prefill,
    apply_research_section_prefill,
    job_status_caption,
    lookup_job,
    overfit_job_payload,
    single_backtest_payload,
    submit_research_job,
)
from scalper_hft.app_pages._results import (
    list_combo_jobs,
    open_combo_details,
    render_backtest_job_picker,
    render_sweep_explorer,
)
from scalper_hft.config import get_settings
from scalper_hft.research.job_artifacts import load_backtest_result, load_cell_audit
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore, artifacts_dir
from scalper_hft.research.results_table import trade_quality_blockers
from scalper_hft.research.sweep_store import SweepStore
from scalper_hft.strategies import REGISTRY
from scalper_hft.validation.cell_audit import cell_verdict, default_train_test
from scalper_hft.validation.sweep import DEFAULT_INTERVALS, default_strategies

_SWEEP_DB = Path("results/sweep.db")

apply_research_audit_prefill(st.session_state)
apply_research_section_prefill(st.session_state)
if "research_section" not in st.session_state:
    st.session_state["research_section"] = RESEARCH_SECTIONS[0]
if "rs_days" not in st.session_state:
    st.session_state["rs_days"] = 60

st.title("Дослідження")
st.caption(
    "Масовий sweep · filter attribution · аудит комірки · режими · порівняння кривих. Рахунок лише через чергу jobs."
)

with st.container(border=True):
    st.caption("Спільні параметри (sweep має власні списки стратегій/символів/ТФ)")
    from scalper_hft.data.exchange_registry import ExchangeRegistry

    exchanges = ExchangeRegistry.list_supported()

    rs0, rs1, rs2, rs3, rs4 = st.columns(5)
    with rs0:
        settings = get_settings()
        idx = exchanges.index(settings.exchange) if settings.exchange in exchanges else 0
        rs_ex = st.selectbox("Біржа", exchanges, index=idx, key="rs_exchange")
    with rs1:
        rs_strat = st.selectbox("Стратегія", sorted(REGISTRY), key="rs_strategy")
    with rs2:
        rs_sym = st.selectbox("Символ", SYMBOLS, key="rs_symbol")
    with rs3:
        rs_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=min(1, len(BT_INTERVALS) - 1), key="rs_interval")
    with rs4:
        rs_days = st.slider("Днів", 14, 730, key="rs_days")

section = st.segmented_control("Розділ", list(RESEARCH_SECTIONS), key="research_section")
if section is None:
    section = RESEARCH_SECTIONS[0]


def _job_banner(job: object, alive: bool) -> None:
    if not alive:
        st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
    st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")
    level, msg = job_status_caption(job)  # type: ignore[arg-type]
    if level == "empty":
        st.info(msg)
    elif level == "info":
        st.info(msg)
    elif level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep Matrix
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Sweep matrix":
    st.subheader("Sweep Matrix: стратегії × символи × таймфрейми")

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.caption("Налаштування прогону")

        sel_slow = st.toggle("Включити повільні (ML / ensemble)", value=False, key="sw_slow")
        all_strats = default_strategies(include_slow=sel_slow)
        n_full = len(all_strats) * len(SYMBOLS) * len(DEFAULT_INTERVALS)
        with st.container(horizontal=True):
            if st.button("Рекомендований старт", icon=":material/recommend:", key="sw_preset_reco"):
                st.session_state["sw_strats"] = list(all_strats)
                st.session_state["sw_syms"] = list(SYMBOLS[:3])
                st.session_state["sw_ivs"] = ["15m", "1h"]
                st.rerun()
            if st.button("Усі × усі × усі", icon=":material/select_all:", key="sw_preset_all"):
                st.session_state["sw_strats"] = list(all_strats)
                st.session_state["sw_syms"] = list(SYMBOLS)
                st.session_state["sw_ivs"] = list(DEFAULT_INTERVALS)
                st.rerun()
        st.caption(
            f"Повний універсум ≈ **{n_full}** клітинок (без `pairs_arb`). "
            "Спочатку оновіть кеш 1m на Моніторингу; walk-forward буде значно довше за backtest."
        )
        sel_strats = st.multiselect("Стратегії", all_strats, default=all_strats[:5], key="sw_strats")
        sel_symbols = st.multiselect("Символи", SYMBOLS, default=SYMBOLS[:3], key="sw_syms")
        sel_ivs = st.multiselect("Таймфрейми", DEFAULT_INTERVALS, default=["5m", "15m", "1h"], key="sw_ivs")
        sel_days = st.slider("Днів даних", 14, 730, 60, key="sw_days")
        sel_mode = st.segmented_control("Режим", ["backtest", "walkforward"], key="sw_mode")
        if sel_mode is None:
            sel_mode = "backtest"
        if sel_mode == "walkforward":
            _iv0 = sel_ivs[0] if sel_ivs else "1h"
            _dt, _dte = default_train_test(_iv0)
            if "sw_train" not in st.session_state:
                st.session_state["sw_train"] = int(_dt)
            if "sw_test" not in st.session_state:
                st.session_state["sw_test"] = int(_dte)
            sel_train = st.number_input("Train барів", 50, 20_000, key="sw_train")
            sel_test = st.number_input("Test барів (OOS)", 20, 10_000, key="sw_test")
        else:
            sel_train, sel_test = 2000, 500
        sel_workers = st.slider("Потоки", 1, 8, 2, key="sw_workers")
        sel_trace = st.toggle("Filter tracing (повільніше)", value=False, key="sw_trace")
        sel_resume = st.toggle("Resume (не повторювати виконані)", value=True, key="sw_resume")

        total_cells = len(sel_strats) * len(sel_symbols) * len(sel_ivs)
        st.caption(f"Клітинок: **{total_cells}**")

        run_sweep_btn = st.button(
            f"Запустити sweep ({total_cells} клітинок)",
            icon=":material/play_arrow:",
            disabled=not (sel_strats and sel_symbols and sel_ivs),
            key="sw_run_btn",
        )

    with col_right:

        @st.cache_data(ttl=30, show_spinner="Читання sweep.db…")
        def _load_sweep_df() -> pd.DataFrame:
            if not _SWEEP_DB.exists():
                return pd.DataFrame()
            try:
                with SweepStore(_SWEEP_DB) as store:
                    return store.load()
            except Exception:
                return pd.DataFrame()

        if run_sweep_btn and sel_strats and sel_symbols and sel_ivs:
            payload = {
                "strategies": list(sel_strats),
                "symbols": list(sel_symbols),
                "intervals": list(sel_ivs),
                "days": int(sel_days),
                "mode": sel_mode,
                "workers": int(sel_workers),
                "resume": bool(sel_resume),
                "enable_trace": bool(sel_trace),
                "include_slow": bool(sel_slow),
                "base_interval": "1m",
                "train_bars": int(sel_train),
                "test_bars": int(sel_test),
            }
            job, alive = submit_research_job("sweep", payload)
            st.success(f"Sweep у черзі як задача #{job.id} ({job.status})")
            if not alive:
                st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
            st.page_link("app_pages/jobs.py", label="Відкрити чергу задач", icon=":material/pending_actions:")

        df_all = _load_sweep_df()
        if df_all.empty:
            st.info(
                "Ще немає результатів sweep. Запустіть прогон зліва або виконайте:\n\n"
                "```\nuv run python -m scalper_hft.cli sweep --symbols BTCUSDT,ETHUSDT "
                "--intervals 5m,15m,1h --days 60 --workers 2\n```"
            )
        else:
            ok_df = (
                df_all[df_all.get("status", pd.Series("ok", index=df_all.index)) == "ok"]
                if "status" in df_all.columns
                else df_all
            )
            fc1, fc2, fc3 = st.columns(3)
            metric = fc1.selectbox(
                "Метрика heatmap",
                ["sharpe", "sortino", "calmar", "win_rate", "total_return", "max_dd"],
                key="sw_metric",
            )
            min_trades = fc2.number_input("Min угод", 0, 10000, 10, key="sw_min_trades")
            filter_mode = fc3.selectbox("Режим фільтру", ["Всі", "backtest", "walkforward"], key="sw_fmode")

            view = ok_df.copy()
            if "n_trades" in view.columns:
                view = view[view["n_trades"] >= min_trades]
            if filter_mode != "Всі" and "mode" in view.columns:
                view = view[view["mode"] == filter_mode]

            if view.empty:
                st.warning("Немає даних після фільтру heatmap")
            else:
                view["sym_iv"] = view.get("symbol", "") + " " + view.get("interval", "")
                pivot = view.pivot_table(index="strategy", columns="sym_iv", values=metric, aggfunc="mean")
                pivot = pivot.sort_index()
                colorscale = "RdYlGn" if metric not in ("max_dd",) else "RdYlGn_r"
                fig = px.imshow(
                    pivot,
                    color_continuous_scale=colorscale,
                    aspect="auto",
                    title=f"Sweep Heat-map: {metric}",
                    labels={"color": metric},
                )
                fig.update_layout(height=max(300, 50 * len(pivot)), font=dict(size=11))
                st.plotly_chart(fig, width="stretch")
            st.subheader("Таблиця результатів")
            render_sweep_explorer(df_all, key_prefix="sw_all")


# ═══════════════════════════════════════════════════════════════════════════════
# Filter Attribution
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Filter attribution":
    st.subheader("Filter Attribution — аналіз впливу фільтрів")
    st.caption(
        "Бектест з `trace=True` у черзі. Детальний трейс зараз у `mean_reversion`. "
        "Shadow-горизонт застосовується вже до збереженого трейсу."
    )
    fa_horizon = st.slider("Shadow-горизонт (барів)", 1, 20, 5, key="fa_horizon")
    fa_payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days), trace=True)
    if st.button("Поставити бектест з трейсингом", icon=":material/science:", key="fa_run"):
        submit_research_job("backtest", fa_payload)
        st.rerun()
    fa_job, fa_alive = lookup_job("backtest", fa_payload)
    _job_banner(fa_job, fa_alive)
    shown_job = False
    if fa_job is not None and fa_job.status == "succeeded":
        try:
            res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, fa_job.id))
        except FileNotFoundError:
            st.warning("Артефакти ще не записані.")
        else:
            shown_job = True
            from scalper_hft.data.access import klines_from_store
            from scalper_hft.research.filter_trace import filter_attribution, filter_pnl_impact

            trace = res.trace
            if trace is None or len(trace) == 0:
                st.info(f"Стратегія **{rs_strat}** не підтримує детальний трейсинг. Реалізовано для: `mean_reversion`.")
            else:
                n_raw = len(trace)
                n_blocked = trace.n_blocked()
                n_passed = trace.n_passed()
                m1, m2, m3 = st.columns(3)
                m1.metric("Raw сигналів", n_raw)
                m2.metric("Пройшли фільтри", n_passed, delta=f"{n_passed / n_raw:.0%}" if n_raw else "")
                m3.metric(
                    "Заблоковано",
                    n_blocked,
                    delta=f"-{n_blocked / n_raw:.0%}" if n_raw else "",
                    delta_color="inverse",
                )
                attr_df = filter_attribution(trace)
                if not attr_df.empty:
                    fig_attr = px.bar(
                        attr_df,
                        x="filter_name",
                        y="n_blocked",
                        title="Скільки сигналів заблокував кожен фільтр",
                        text="n_blocked",
                        color="pct_of_all_signals",
                        color_continuous_scale="Oranges",
                    )
                    fig_attr.update_layout(xaxis_title="", yaxis_title="Кількість сигналів", height=350)
                    st.plotly_chart(fig_attr, width="stretch")
                df_k = None
                with busy("Завантаження свічок для shadow PnL…"):
                    df_k = klines_from_store(str(rs_sym), str(rs_iv), int(rs_days))
                if df_k is not None and "close" in df_k.columns:
                    pnl_df = filter_pnl_impact(trace, df_k["close"], horizon_bars=int(fa_horizon))
                    if not pnl_df.empty:
                        st.subheader("Shadow PnL — що було б без фільтру")
                        st.caption(
                            f"Forward return за {fa_horizon} барів. "
                            "Якщо shadow_mean_ret > 0 → фільтр відкидає прибуткові угоди."
                        )
                        for _, row in pnl_df.iterrows():
                            harmful = row["shadow_mean_ret"] > 0 and row["shadow_win_rate"] > 0.5
                            verdict_badge = ":red-badge[Шкідливий]" if harmful else ":green-badge[Корисний]"
                            verdict_msg = (
                                "фільтр відкидає прибуткові угоди" if harmful else "фільтр відкидає збиткові угоди"
                            )
                            with st.container(border=True):
                                st.markdown(f"**{row['filter_name']}** {verdict_badge} — {verdict_msg}")
                                with st.container(horizontal=True):
                                    st.metric("Заблоковано", f"{int(row['n_blocked'])}", border=True)
                                    st.metric("Shadow win rate", f"{row['shadow_win_rate']:.0%}", border=True)
                                    st.metric("Shadow avg PnL", f"{row['shadow_mean_ret']:+.4%}", border=True)
                                    st.metric("Shadow total PnL", f"{row['shadow_total_pnl']:+.4%}", border=True)
    if not shown_job and _SWEEP_DB.exists():
        try:
            with SweepStore(_SWEEP_DB) as store:
                attr_store = store.load_filter_attribution()
            if not attr_store.empty:
                st.info("Дані filter attribution зі sweep-прогонів:")
                pivot_attr = attr_store.groupby(["filter_name", "strategy"])["n_blocked"].sum().reset_index()
                fig2 = px.bar(
                    pivot_attr,
                    x="filter_name",
                    y="n_blocked",
                    color="strategy",
                    barmode="stack",
                    title="Filter attribution (агрегований sweep)",
                )
                st.plotly_chart(fig2, width="stretch")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Equity Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Порівняння equity":
    st.subheader("Порівняння Equity Curves")
    st.caption("Кожна крива — окремий backtest у черзі. Нормалізація: база = 1.0")
    st.session_state.setdefault("eq_jobs", [])
    ec_label = st.text_input("Мітка", value=f"{rs_strat} {rs_sym} {rs_iv}", key="ec_label")
    with st.container(horizontal=True):
        if st.button("Додати криву", icon=":material/add:", key="ec_add"):
            payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days))
            submit_research_job("backtest", payload)
            items = [x for x in st.session_state["eq_jobs"] if x.get("label") != ec_label]
            items.append({"label": ec_label, "payload": payload})
            st.session_state["eq_jobs"] = items
            st.rerun()
        if st.button("Очистити всі", icon=":material/delete:", key="ec_clear"):
            st.session_state["eq_jobs"] = []
            st.rerun()

    pending: list[str] = []
    curves: dict[str, pd.Series] = {}
    for item in list(st.session_state.get("eq_jobs") or []):
        label = str(item.get("label") or "")
        payload = item.get("payload") or {}
        job, _alive = lookup_job("backtest", payload)
        if job is None or job.status in {"queued", "running"}:
            pending.append(f"{label}: {job.status if job else 'немає'}")
            continue
        if job.status != "succeeded":
            pending.append(f"{label}: {job.status}")
            continue
        try:
            res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, job.id))
        except FileNotFoundError:
            pending.append(f"{label}: артефакти ще не готові")
            continue
        curves[label] = res.equity
    if pending:
        st.caption("Очікуємо: " + " · ".join(pending))
        st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")
    if not curves:
        st.info("Додайте криві — кожна ставиться як backtest job.")
    else:
        fig_eq = go.Figure()
        for label, equity in curves.items():
            if equity is None or equity.empty:
                continue
            norm = equity / equity.iloc[0]
            fig_eq.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=label))
        fig_eq.update_layout(
            title="Нормалізовані Equity (база = 1.0)",
            xaxis_title="",
            yaxis_title="Нормалізована вартість",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_eq, width="stretch")
        if len(curves) > 1:
            daily = pd.DataFrame(
                {lbl: eq.resample("1D").last().pct_change().dropna() for lbl, eq in curves.items()}
            ).dropna()
            if len(daily) > 5:
                corr = daily.corr()
                fig_corr = px.imshow(
                    corr,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Кореляція денних дохідностей між стратегіями",
                    text_auto=".2f",
                )
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, width="stretch")
        st.subheader("Rolling Sharpe (30-денне вікно)")
        fig_rs = go.Figure()
        for label, equity in curves.items():
            daily_ret = equity.resample("1D").last().pct_change().dropna()
            if len(daily_ret) >= 30:
                rolling_sharpe = (daily_ret.rolling(30).mean() / daily_ret.rolling(30).std()) * np.sqrt(365)
                fig_rs.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name=label))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rs.update_layout(height=300, yaxis_title="Sharpe (30d rolling)", xaxis_title="")
        st.plotly_chart(fig_rs, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Quality
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Якість угод":
    st.subheader("Аналіз якості угод")
    st.caption(
        "MAE/MFE, heatmap і розподіл PnL читаються з **повного бектесту** (список угод). "
        "Sweep зберігає лише підсумок метрик — рядок матриці сам по собі тут не відкриється."
    )
    tq_payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days))
    combo_jobs = list_combo_jobs(strategy=str(rs_strat), symbol=str(rs_sym), interval=str(rs_iv))
    tq_job, tq_alive = lookup_job("backtest", tq_payload)
    override_id = st.session_state.get("tq_override_job_id")
    if override_id is not None:
        with JobStore(DEFAULT_JOBS_PATH) as js:
            override = js.get(int(override_id))
        if override is not None:
            tq_job = override
            st.caption(f"Показано задачу #{tq_job.id} ({tq_job.params.get('days')} днів), не обов'язково слайдер.")
    elif tq_job is None:
        succeeded = [j for j in combo_jobs if j.status == "succeeded"]
        if succeeded:
            tq_job = succeeded[0]
            st.caption(
                f"Точного збігу зі слайдером ({rs_days} днів) немає. "
                f"Показано останній succeeded бектест #{tq_job.id} на {tq_job.params.get('days')} днів."
            )
    with st.container(horizontal=True):
        if st.button("Поставити бектест", icon=":material/play_arrow:", key="tq_run"):
            st.session_state.pop("tq_override_job_id", None)
            submit_research_job("backtest", tq_payload)
            st.rerun()
        if st.button("Відкрити свічки й угоди", icon=":material/candlestick_chart:", key="tq_open_bt"):
            open_combo_details(
                {
                    "strategy": rs_strat,
                    "symbol": rs_sym,
                    "interval": rs_iv,
                    "days": int(tq_job.params.get("days", rs_days)) if tq_job else int(rs_days),
                }
            )
    _job_banner(tq_job, tq_alive)
    if combo_jobs:
        st.markdown("**Наявні бектести цієї комбінації** (будь-яка глибина днів)")
        render_backtest_job_picker(combo_jobs)

    from scalper_hft.data.access import klines_from_store
    from scalper_hft.research.session_analysis import hourly_heatmap_data, mae_mfe_analysis, session_breakdown

    df_k = klines_from_store(str(rs_sym), str(rs_iv), int(rs_days))
    has_klines = df_k is not None and not df_k.empty
    res = None
    n_trades = 0
    has_artifacts = False
    if tq_job is not None and tq_job.status == "succeeded":
        try:
            res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, tq_job.id))
            has_artifacts = True
            n_trades = 0 if res.trades is None or res.trades.empty else len(res.trades)
        except FileNotFoundError:
            has_artifacts = False
    blockers = trade_quality_blockers(
        job_status=None if tq_job is None else tq_job.status,
        has_artifacts=has_artifacts,
        n_trades=n_trades,
        has_klines=has_klines,
    )
    if res is None or n_trades <= 0:
        for reason in blockers:
            st.info(reason)
    else:
        trades = res.trades
        m = res.metrics
        with st.container(horizontal=True):
            st.metric("Угод", m.n_trades, border=True)
            st.metric("Win rate", f"{m.win_rate:.0%}", border=True)
            st.metric("Avg PnL/угода", f"{m.avg_trade_return:.4%}", border=True)
            st.metric("Угод/день", f"{m.trades_per_day:.2f}", border=True)
        if not has_klines:
            st.warning("Немає свічок у кеші — вкладка MAE/MFE буде порожня. Оновіть кеш на Моніторингу.")
        sub_tabs = st.tabs(["MAE/MFE", "Hourly Heatmap", "Розподіл PnL"])
        with sub_tabs[0]:
            if df_k is None:
                st.info("Немає klines для MAE/MFE")
            else:
                mae_df = mae_mfe_analysis(trades, df_k)
                if not mae_df.empty:
                    fig_mf = px.scatter(
                        mae_df,
                        x="mfe",
                        y="mae",
                        color="ret",
                        color_continuous_scale="RdYlGn",
                        hover_data=["entry_ts", "side", "ret", "efficiency"],
                        title="MAE vs MFE (кожна угода — точка)",
                        labels={"mfe": "MFE (макс. на користь)", "mae": "MAE (макс. проти)"},
                    )
                    fig_mf.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
                    fig_mf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                    fig_mf.update_layout(height=420)
                    st.plotly_chart(fig_mf, width="stretch")
                    avg_eff = mae_df["efficiency"].dropna().mean()
                    st.caption(
                        f"Avg efficiency = {avg_eff:.1%} "
                        "(частка MFE, що ми реально захопили). Якщо < 50% — виходимо занадто рано."
                    )
                else:
                    st.info("Не вдалось розрахувати MAE/MFE — потрібні high/low колонки")
        with sub_tabs[1]:
            hm = hourly_heatmap_data(trades)
            if not hm.empty:
                fig_hm = px.imshow(
                    hm.values,
                    x=list(hm.columns),
                    y=list(hm.index),
                    color_continuous_scale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    title="Win rate: день тижня × година UTC",
                    labels={"color": "Win rate"},
                )
                fig_hm.update_layout(height=300)
                st.plotly_chart(fig_hm, width="stretch")
                sess = session_breakdown(trades)
                if not sess.empty:
                    st.subheader("Розбивка по годинах UTC")
                    fig_sess = px.bar(
                        sess.reset_index(),
                        x="hour_utc",
                        y="n_trades",
                        color="win_rate",
                        color_continuous_scale="RdYlGn",
                        title="Кількість угод та win rate по годинах",
                        labels={"n_trades": "Угод", "win_rate": "Win rate"},
                    )
                    st.plotly_chart(fig_sess, width="stretch")
            else:
                st.info("Недостатньо угод для heatmap (потрібно ≥3 на клітинку)")
        with sub_tabs[2]:
            if "ret" in trades.columns:
                rets = trades["ret"].dropna()
                fig_hist = px.histogram(
                    rets,
                    nbins=50,
                    title="Розподіл PnL угод",
                    labels={"value": "PnL (відн.)", "count": "Кількість"},
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
                fig_hist.add_vline(
                    x=float(rets.mean()),
                    line_dash="dot",
                    line_color="green",
                    opacity=0.7,
                    annotation_text=f"mean={rets.mean():.4%}",
                )
                fig_hist.update_layout(height=380)
                st.plotly_chart(fig_hist, width="stretch")
                cum_pnl = (1 + rets).cumprod()
                fig_cum = px.line(
                    cum_pnl.values,
                    title="Кумулятивний PnL угод (по угодах, не по часу)",
                    labels={"index": "# Угоди", "value": "Кумулятивний PnL"},
                )
                st.plotly_chart(fig_cum, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# Best Combinations
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Топ комбінації":
    st.subheader("Топ комбінації зі sweep")
    st.caption("Найкращі комбінації (стратегія × символ × таймфрейм) з усіх sweep-прогонів")
    if not _SWEEP_DB.exists():
        st.info("Немає sweep результатів. Запустіть sweep у розділі «Sweep Matrix»")
    else:
        try:
            with SweepStore(_SWEEP_DB) as store:
                all_df = store.load()
        except Exception as e:
            st.error(f"Помилка читання sweep DB: {e}")
            all_df = pd.DataFrame()
        if all_df.empty:
            st.info("Немає результатів у sweep DB")
        else:
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc_min_sh = bc1.number_input("Min Sharpe", -10.0, 10.0, 0.5, step=0.1, key="bc_minsh")
            bc_min_tr = bc2.number_input("Min угод", 0, 10000, 20, key="bc_mintr")
            bc_min_wr = bc3.number_input("Min win rate", 0.0, 1.0, 0.4, step=0.05, key="bc_minwr")
            bc_show_n = bc4.number_input("Показати топ N", 5, 200, 30, key="bc_n")
            for col in [
                "sharpe",
                "n_trades",
                "win_rate",
                "sortino",
                "calmar",
                "total_return",
                "max_dd",
                "profit_factor",
                "trades_per_day",
                "avg_oos_sharpe",
                "oos_positive_frac",
            ]:
                if col in all_df.columns:
                    all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
            ok = (
                all_df[all_df.get("status", pd.Series("ok", index=all_df.index)) == "ok"]
                if "status" in all_df.columns
                else all_df
            )
            filtered = ok.copy()
            if "sharpe" in filtered.columns:
                filtered = filtered[filtered["sharpe"] >= bc_min_sh]
            if "n_trades" in filtered.columns:
                filtered = filtered[filtered["n_trades"] >= bc_min_tr]
            if "win_rate" in filtered.columns:
                filtered = filtered[filtered["win_rate"] >= bc_min_wr]
            if filtered.empty:
                st.warning("Немає комбінацій, що відповідають фільтрам")
            else:
                has_oos = "avg_oos_sharpe" in filtered.columns and filtered["avg_oos_sharpe"].notna().any()
                sort_col = "avg_oos_sharpe" if has_oos else "sharpe"
                if sort_col in filtered.columns:
                    valid = filtered.dropna(subset=[sort_col])
                    top = valid.nlargest(int(bc_show_n), sort_col) if not valid.empty else filtered.head(int(bc_show_n))
                else:
                    top = filtered.head(int(bc_show_n))
                top_view = top.reset_index(drop=True)
                render_sweep_explorer(top_view, key_prefix="sw_top")
                csv_bytes = top_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Завантажити CSV",
                    data=csv_bytes,
                    file_name="sweep_top.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key="bc_csv",
                )
                st.subheader("Зведена статистика по стратегіях")
                summary = (
                    filtered.groupby("strategy")
                    .agg(
                        cells=("sharpe", "count"),
                        mean_sharpe=("sharpe", "mean"),
                        max_sharpe=("sharpe", "max"),
                        mean_n_trades=("n_trades", "mean"),
                        mean_win_rate=("win_rate", "mean"),
                    )
                    .sort_values("mean_sharpe", ascending=False)
                )
                st.dataframe(
                    summary.reset_index(),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "mean_sharpe": st.column_config.NumberColumn("Avg Sharpe", format="%.3f"),
                        "max_sharpe": st.column_config.NumberColumn("Max Sharpe", format="%.3f"),
                        "mean_win_rate": st.column_config.NumberColumn("Avg Win rate", format="%.0%"),
                        "mean_n_trades": st.column_config.NumberColumn("Avg Trades", format="%.0f"),
                    },
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Cell audit
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Аудит комірки":
    st.subheader("Аудит комірки")
    st.caption(
        "Walk-forward + Deflated Sharpe + sensitivity. Вердикт PASS лише якщо всі пороги "
        "overfitting-audit виконано. Рахунок іде через чергу jobs (kind=overfit)."
    )
    _dt, _dte = default_train_test(str(rs_iv))
    if "au_train" not in st.session_state:
        st.session_state["au_train"] = int(_dt)
    if "au_test" not in st.session_state:
        st.session_state["au_test"] = int(_dte)
    au_train = st.number_input("Train барів", 50, 20_000, key="au_train")
    au_test = st.number_input("Test барів (OOS)", 20, 10_000, key="au_test")
    au_payload = overfit_job_payload(
        str(rs_strat),
        str(rs_sym),
        str(rs_iv),
        int(rs_days),
        train_bars=int(au_train),
        test_bars=int(au_test),
    )
    if st.button("Поставити аудит у чергу", icon=":material/fact_check:", key="au_run"):
        submit_research_job("overfit", au_payload)
        st.rerun()
    au_job, au_alive = lookup_job("overfit", au_payload)
    _job_banner(au_job, au_alive)
    if au_job is not None and au_job.status == "succeeded":
        try:
            audit = load_cell_audit(artifacts_dir(DEFAULT_JOBS_PATH, au_job.id))
        except FileNotFoundError:
            st.warning("Артефакти ще не записані.")
        else:
            verdict, why = cell_verdict(audit)
            if verdict == "PASS":
                st.badge("PASS", icon=":material/check_circle:", color="green")
            else:
                st.badge("FAIL", icon=":material/cancel:", color="red")
                if why:
                    st.caption(why)
            with st.container(horizontal=True):
                st.metric("OOS Sharpe", f"{audit.avg_oos_sharpe or 0:.3f}", border=True)
                st.metric("OOS+", f"{(audit.oos_pos_frac or 0):.0%}", border=True)
                st.metric("DSR", f"{audit.dsr if audit.dsr is not None else float('nan'):.3f}", border=True)
                st.metric(
                    "Smoothness",
                    f"{audit.smoothness if audit.smoothness is not None else float('nan'):.2f}",
                    border=True,
                )
                st.metric("Угод (BT)", f"{audit.bt_n_trades or 0}", border=True)
            if audit.windows:
                wf_df = pd.DataFrame(list(audit.windows))
                st.subheader("Walk-forward вікна")
                st.dataframe(wf_df, width="stretch", hide_index=True)
                fig_wf = go.Figure()
                fig_wf.add_trace(go.Bar(x=wf_df["window_idx"], y=wf_df["oos_sharpe"], name="OOS Sharpe"))
                fig_wf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_wf.update_layout(height=280, yaxis_title="OOS Sharpe", xaxis_title="Вікно")
                st.plotly_chart(fig_wf, width="stretch")
            if audit.sensitivity_grid and audit.sens_param:
                sens_df = pd.DataFrame(list(audit.sensitivity_grid))
                st.subheader(f"Sensitivity ({audit.sens_param})")
                if "metric" in sens_df.columns and audit.sens_param in sens_df.columns:
                    fig_s = px.line(
                        sens_df,
                        x=audit.sens_param,
                        y="metric",
                        markers=True,
                        title="Sharpe по сітці параметра",
                    )
                    fig_s.update_layout(height=280)
                    st.plotly_chart(fig_s, width="stretch")
                st.dataframe(sens_df, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Regimes
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Режими":
    st.subheader("Режими ринку")
    st.caption(
        "Розбивка угод по волатильності, структурі та складеному стану structure|vol. "
        "Це in-sample атрибуція, не OOS-вердикт."
    )
    from scalper_hft.data.access import klines_from_store
    from scalper_hft.features.regimes import named_market_state
    from scalper_hft.research.session_analysis import named_regime_breakdown, regime_breakdown, structure_breakdown

    rg_payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days))
    if st.button("Поставити бектест у чергу", icon=":material/play_arrow:", key="rg_run"):
        submit_research_job("backtest", rg_payload)
        st.rerun()
    rg_job, rg_alive = lookup_job("backtest", rg_payload)
    _job_banner(rg_job, rg_alive)
    if rg_job is not None and rg_job.status == "succeeded":
        try:
            rg_res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, rg_job.id))
        except FileNotFoundError:
            st.warning("Артефакти ще не записані.")
        else:
            trades = rg_res.trades
            if trades is None or trades.empty:
                st.info("Угод немає — немає що розкладати по режимах.")
            else:
                df_k = klines_from_store(str(rs_sym), str(rs_iv), int(rs_days))
                if df_k is None or df_k.empty or "close" not in df_k.columns:
                    st.warning(f"Немає klines {rs_sym} {rs_iv} для класифікації режиму.")
                else:
                    state = named_market_state(df_k["close"])
                    vol = regime_breakdown(trades, state["vol"])
                    struct = structure_breakdown(trades, state["structure"])
                    named = named_regime_breakdown(trades, state)
                    if not vol.empty:
                        st.markdown("**Волатильність**")
                        st.dataframe(vol.reset_index(), width="stretch", hide_index=True)
                    if not struct.empty:
                        st.markdown("**Структура**")
                        st.dataframe(struct.reset_index(), width="stretch", hide_index=True)
                    if not named.empty:
                        st.markdown("**Складений стан (structure|vol)**")
                        st.dataframe(named.reset_index(), width="stretch", hide_index=True)
