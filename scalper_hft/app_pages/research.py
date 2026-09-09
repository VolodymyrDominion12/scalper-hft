"""Сторінка «Дослідження»: масовий пошук, повний цикл (аналіз угод, аудит, режими), порівняння."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    render_sweep_explorer,
)
from scalper_hft.config import get_settings
from scalper_hft.research.job_artifacts import load_backtest_result, load_cell_audit
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, artifacts_dir
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

# --- Спільні параметри у Sidebar ---
with st.sidebar:
    st.subheader("Спільні параметри")
    st.caption("Ці налаштування використовуються для Повного циклу та Порівняння.")
    from scalper_hft.data.exchange_registry import ExchangeRegistry

    exchanges = ExchangeRegistry.list_supported()
    settings = get_settings()
    idx = exchanges.index(settings.exchange) if settings.exchange in exchanges else 0
    rs_ex = st.selectbox("Біржа", exchanges, index=idx, key="rs_exchange")
    rs_strat = st.selectbox("Стратегія", sorted(REGISTRY), key="rs_strategy")
    rs_sym = st.selectbox("Символ", SYMBOLS, key="rs_symbol")
    rs_iv = st.pills("Таймфрейм", BT_INTERVALS, default=BT_INTERVALS[min(1, len(BT_INTERVALS) - 1)], key="rs_interval")
    if rs_iv is None:
        rs_iv = BT_INTERVALS[1]
    rs_days = st.slider("Днів", 14, 730, key="rs_days")


section = st.segmented_control("Розділ", list(RESEARCH_SECTIONS), key="research_section", default=RESEARCH_SECTIONS[0])
if section is None:
    section = RESEARCH_SECTIONS[0]


def _job_banner(job: object, alive: bool) -> None:
    if not alive:
        st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
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
# Масовий пошук (Sweep + Top)
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Масовий пошук":
    tab_sweep, tab_top = st.tabs(["Sweep Matrix", "Топ комбінації"])

    with tab_sweep:
        st.subheader("Sweep Matrix: стратегії × символи × таймфрейми")
        col_left, col_right = st.columns([1, 3])
        with col_left:
            with st.container(border=True):
                st.caption("Налаштування прогону")
                sel_slow = st.toggle("Включити повільні (ML / ensemble)", value=False, key="sw_slow")
                all_strats = default_strategies(include_slow=sel_slow)

                with st.container(horizontal=True):
                    if st.button("Рекомендований", icon=":material/recommend:", key="sw_preset_reco"):
                        st.session_state["sw_strats"] = list(all_strats)
                        st.session_state["sw_syms"] = list(SYMBOLS[:3])
                        st.session_state["sw_ivs"] = ["15m", "1h"]
                        st.rerun()
                    if st.button("Усі × усі × усі", icon=":material/select_all:", key="sw_preset_all"):
                        st.session_state["sw_strats"] = list(all_strats)
                        st.session_state["sw_syms"] = list(SYMBOLS)
                        st.session_state["sw_ivs"] = list(DEFAULT_INTERVALS)
                        st.rerun()

                sel_strats = st.multiselect("Стратегії", all_strats, default=all_strats[:5], key="sw_strats")
                sel_symbols = st.multiselect("Символи", SYMBOLS, default=SYMBOLS[:3], key="sw_syms")
                sel_ivs = st.multiselect("Таймфрейми", DEFAULT_INTERVALS, default=["5m", "15m", "1h"], key="sw_ivs")
                sel_days = st.slider("Днів даних", 14, 730, 60, key="sw_days")

                with st.expander("⚙️ Розширені налаштування (режим, OOS)"):
                    sel_mode = st.segmented_control(
                        "Режим", ["backtest", "walkforward"], key="sw_mode", default="backtest"
                    )
                    if sel_mode is None:
                        sel_mode = "backtest"
                    sel_train: int | None = None
                    sel_test: int | None = None
                    if sel_mode == "walkforward":
                        use_per_tf = st.toggle("Авто-вікна по ТФ", value=True, key="sw_per_tf")
                        if not use_per_tf:
                            _iv0 = sel_ivs[0] if sel_ivs else "1h"
                            _dt, _dte = default_train_test(_iv0)
                            if "sw_train" not in st.session_state:
                                st.session_state["sw_train"] = int(_dt)
                            if "sw_test" not in st.session_state:
                                st.session_state["sw_test"] = int(_dte)
                            sel_train = int(st.number_input("Train барів", 50, 20_000, key="sw_train"))
                            sel_test = int(st.number_input("Test барів (OOS)", 20, 10_000, key="sw_test"))
                    sel_workers = st.slider("Потоки", 1, 8, 2, key="sw_workers")
                    sel_trace = st.toggle("Filter tracing (повільніше)", value=False, key="sw_trace")
                    sel_resume = st.toggle("Resume (не повторювати виконані)", value=True, key="sw_resume")

                total_cells = len(sel_strats) * len(sel_symbols) * len(sel_ivs)

                run_sweep_btn = st.button(
                    f"Запустити sweep ({total_cells} клітинок)",
                    icon=":material/play_arrow:",
                    disabled=not (sel_strats and sel_symbols and sel_ivs),
                    key="sw_run_btn",
                    width="stretch",
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
                }
                if sel_train is not None and sel_test is not None:
                    payload["train_bars"] = int(sel_train)
                    payload["test_bars"] = int(sel_test)
                job, alive = submit_research_job("sweep", payload)
                st.success(f"Sweep у черзі як задача #{job.id} ({job.status})")
                if not alive:
                    st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")

            df_all = _load_sweep_df()
            if df_all.empty:
                st.info("Ще немає результатів sweep.")
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

    with tab_top:
        st.subheader("Топ комбінації зі sweep")
        if df_all.empty:
            st.info("Немає sweep результатів.")
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
                if col in df_all.columns:
                    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

            ok = (
                df_all[df_all.get("status", pd.Series("ok", index=df_all.index)) == "ok"]
                if "status" in df_all.columns
                else df_all
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
# Повний цикл (Аналіз однієї конфігурації)
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Повний цикл":
    st.subheader(f"Повний цикл: {rs_strat} | {rs_sym} | {rs_iv} | {rs_days}d")
    st.caption("Цей розділ об'єднує результати бектесту (з увімкненим трейсингом) та overfit-аудиту.")

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Запустити повний цикл дослідження", type="primary", width="stretch"):
                bt_payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days), trace=True)
                submit_research_job("backtest", bt_payload)

                _dt, _dte = default_train_test(str(rs_iv))
                of_payload = overfit_job_payload(
                    str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days), train_bars=int(_dt), test_bars=int(_dte)
                )
                submit_research_job("overfit", of_payload)

                st.rerun()
        with col2:
            st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")

    bt_payload = single_backtest_payload(str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days), trace=True)
    bt_job, bt_alive = lookup_job("backtest", bt_payload)

    _dt, _dte = default_train_test(str(rs_iv))
    of_payload = overfit_job_payload(
        str(rs_strat), str(rs_sym), str(rs_iv), int(rs_days), train_bars=int(_dt), test_bars=int(_dte)
    )
    of_job, of_alive = lookup_job("overfit", of_payload)

    # Status indicators
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**Backtest Status**")
        _job_banner(bt_job, bt_alive)
    with status_col2:
        st.markdown("**Overfit Audit Status**")
        _job_banner(of_job, of_alive)

    bt_ready = bt_job is not None and bt_job.status == "succeeded"
    of_ready = of_job is not None and of_job.status == "succeeded"

    if bt_ready or of_ready:
        st.divider()
        st.subheader("Результати аналізу")

        # Load artifacts
        res_bt = None
        has_klines = False
        df_k = None
        if bt_ready:
            try:
                res_bt = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, bt_job.id))
                from scalper_hft.data.access import klines_from_store

                df_k = klines_from_store(str(rs_sym), str(rs_iv), int(rs_days))
                has_klines = df_k is not None and not df_k.empty
            except Exception:
                pass

        audit = None
        if of_ready:
            try:
                audit = load_cell_audit(artifacts_dir(DEFAULT_JOBS_PATH, of_job.id))
            except Exception:
                pass

        # Summary Header
        sum_c1, sum_c2, sum_c3, sum_c4, sum_c5 = st.columns(5)
        if audit is not None:
            verdict, why = cell_verdict(audit)
            with sum_c1:
                if verdict == "PASS":
                    st.metric("Аудит", "PASS", delta="ОК", delta_color="normal")
                else:
                    st.metric("Аудит", "FAIL", delta=why, delta_color="inverse")
            sum_c2.metric("OOS Sharpe", f"{audit.avg_oos_sharpe or 0:.3f}")
            sum_c3.metric("OOS+", f"{(audit.oos_pos_frac or 0):.0%}")
            sum_c4.metric("DSR", f"{audit.dsr if audit.dsr is not None else float('nan'):.3f}")

        if res_bt is not None and res_bt.metrics is not None:
            m = res_bt.metrics
            if audit is None:
                sum_c1.metric("Угод", m.n_trades)
                sum_c2.metric("Win rate", f"{m.win_rate:.0%}")
                sum_c3.metric("Avg PnL/угода", f"{m.avg_trade_return:.4%}")
                sum_c4.metric("Угод/день", f"{m.trades_per_day:.2f}")
            else:
                sum_c5.metric("Win rate (BT)", f"{m.win_rate:.0%}")

        tab_perf, tab_rob, tab_ctx = st.tabs(["Performance", "Robustness (Audit)", "Context (Regimes/Filters)"])

        with tab_perf:
            if res_bt is not None and res_bt.trades is not None and not res_bt.trades.empty:
                trades = res_bt.trades

                perf_col1, perf_col2 = st.columns([1, 1])
                with perf_col1:
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
                        fig_hist.update_layout(height=350)
                        st.plotly_chart(fig_hist, width="stretch")
                with perf_col2:
                    if "ret" in trades.columns:
                        cum_pnl = (1 + rets).cumprod()
                        fig_cum = px.line(
                            cum_pnl.values,
                            title="Кумулятивний PnL угод",
                            labels={"index": "# Угоди", "value": "Кумулятивний PnL"},
                        )
                        fig_cum.update_layout(height=350)
                        st.plotly_chart(fig_cum, width="stretch")

                st.subheader("MAE/MFE")
                if has_klines:
                    from scalper_hft.research.session_analysis import mae_mfe_analysis

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
                        )
                        fig_mf.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
                        fig_mf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                        fig_mf.update_layout(height=420)
                        st.plotly_chart(fig_mf, width="stretch")
                        avg_eff = mae_df["efficiency"].dropna().mean()
                        st.caption(
                            f"Avg efficiency = {avg_eff:.1%} (частка MFE, що ми реально захопили). Якщо < 50% — виходимо занадто рано."
                        )
                else:
                    st.info("Немає klines для розрахунку MAE/MFE.")
            else:
                st.info("Бектест не завершено або немає угод.")

        with tab_rob:
            if audit is not None:
                rob_c1, rob_c2 = st.columns(2)
                with rob_c1:
                    if audit.windows:
                        wf_df = pd.DataFrame(list(audit.windows))
                        st.markdown("**Walk-forward вікна**")
                        fig_wf = go.Figure()
                        fig_wf.add_trace(go.Bar(x=wf_df["window_idx"], y=wf_df["oos_sharpe"], name="OOS Sharpe"))
                        fig_wf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_wf.update_layout(
                            height=280, yaxis_title="OOS Sharpe", xaxis_title="Вікно", margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig_wf, width="stretch")
                        with st.expander("Деталі вікон"):
                            st.dataframe(wf_df, width="stretch", hide_index=True)
                with rob_c2:
                    if audit.sensitivity_grid and audit.sens_param:
                        sens_df = pd.DataFrame(list(audit.sensitivity_grid))
                        st.markdown(f"**Sensitivity ({audit.sens_param})**")
                        if "metric" in sens_df.columns and audit.sens_param in sens_df.columns:
                            fig_s = px.line(
                                sens_df, x=audit.sens_param, y="metric", markers=True, title="Sharpe по сітці"
                            )
                            fig_s.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_s, width="stretch")
                        with st.expander("Деталі Sensitivity"):
                            st.dataframe(sens_df, width="stretch", hide_index=True)
            else:
                st.info("Аудит не завершено або немає даних.")

        with tab_ctx:
            if res_bt is not None and res_bt.trades is not None and not res_bt.trades.empty:
                trades = res_bt.trades

                # Regimes
                st.subheader("Режими ринку")
                if has_klines:
                    from scalper_hft.features.regimes import named_market_state
                    from scalper_hft.research.session_analysis import (
                        named_regime_breakdown,
                        regime_breakdown,
                        structure_breakdown,
                    )

                    state = named_market_state(df_k["close"])
                    vol = regime_breakdown(trades, state["vol"])
                    struct = structure_breakdown(trades, state["structure"])
                    named = named_regime_breakdown(trades, state)

                    reg_c1, reg_c2, reg_c3 = st.columns(3)
                    with reg_c1:
                        if not vol.empty:
                            st.markdown("**Волатильність**")
                            st.dataframe(vol.reset_index(), width="stretch", hide_index=True)
                    with reg_c2:
                        if not struct.empty:
                            st.markdown("**Структура**")
                            st.dataframe(struct.reset_index(), width="stretch", hide_index=True)
                    with reg_c3:
                        if not named.empty:
                            st.markdown("**Складений стан**")
                            st.dataframe(named.reset_index(), width="stretch", hide_index=True)

                # Sessions
                from scalper_hft.research.session_analysis import hourly_heatmap_data

                st.subheader("Сесії (UTC)")
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

                # Filter Trace
                st.subheader("Filter Attribution")
                trace = res_bt.trace
                if trace is None or len(trace) == 0:
                    st.info(f"Стратегія **{rs_strat}** не підтримує детальний трейсинг (або trace=False).")
                else:
                    from scalper_hft.research.filter_trace import filter_attribution, filter_pnl_impact

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

                        st.caption("Shadow PnL — що було б без фільтру")
                        fa_horizon = st.slider("Shadow-горизонт (барів)", 1, 20, 5, key="fa_horizon_fc")
                        if has_klines:
                            pnl_df = filter_pnl_impact(trace, df_k["close"], horizon_bars=int(fa_horizon))
                            for _, row in pnl_df.iterrows():
                                harmful = row["shadow_mean_ret"] > 0 and row["shadow_win_rate"] > 0.5
                                verdict_badge = ":red-badge[Шкідливий]" if harmful else ":green-badge[Корисний]"
                                with st.container(border=True):
                                    st.markdown(f"**{row['filter_name']}** {verdict_badge}")
                                    st.metric("Shadow avg PnL", f"{row['shadow_mean_ret']:+.4%}")
            else:
                st.info("Бектест не завершено або немає угод.")

# ═══════════════════════════════════════════════════════════════════════════════
# Порівняння Equity
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Порівняння equity":
    st.subheader("Порівняння Equity Curves")
    st.caption("Кожна крива — окремий backtest у черзі. Нормалізація: база = 1.0")
    st.session_state.setdefault("eq_jobs", [])
    ec_label = st.text_input("Мітка", value=f"{rs_strat} {rs_sym} {rs_iv}", key="ec_label")
    with st.container(horizontal=True):
        if st.button("Додати поточну криву", icon=":material/add:", key="ec_add"):
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
            curves[label] = res.equity
        except Exception:
            pending.append(f"{label}: помилка артефактів")

    if pending:
        st.caption("Очікуємо: " + " · ".join(pending))

    if not curves:
        st.info("Додайте криві.")
    else:
        fig_eq = go.Figure()
        for label, equity in curves.items():
            if equity is None or equity.empty:
                continue
            norm = equity / equity.iloc[0]
            fig_eq.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=label))
        fig_eq.update_layout(
            title="Нормалізовані Equity", xaxis_title="", yaxis_title="Нормалізована вартість", height=400
        )
        st.plotly_chart(fig_eq, width="stretch")

        st.subheader("Rolling Sharpe (30-денне вікно)")
        fig_rs = go.Figure()
        for label, equity in curves.items():
            daily_ret = equity.resample("1D").last().pct_change().dropna()
            if len(daily_ret) >= 30:
                rolling_sharpe = (daily_ret.rolling(30).mean() / daily_ret.rolling(30).std()) * np.sqrt(365)
                fig_rs.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name=label))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rs.update_layout(height=300)
        st.plotly_chart(fig_rs, width="stretch")
