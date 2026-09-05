"""Сторінка «Дослідження»: масовий sweep, filter attribution, порівняння стратегій.

Секції:
  1. Sweep Matrix — запуск масового прогону та heatmap результатів
  2. Filter Attribution — аналіз які фільтри скільки угод відкидають і чи це вигідно
  3. Equity Comparison — порівняння equity кривих декількох комбінацій
  4. Trade Quality — MAE/MFE, hourly heatmap, розподіл PnL
  5. Best Combinations — топ результатів з переходом до бектесту / аудиту
  6. Cell audit — walk-forward + DSR + sensitivity + PASS/FAIL
  7. Regimes — розбивка угод по vol / structure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scalper_hft.app_pages._common import (
    BT_INTERVALS,
    RESEARCH_AUDIT_PREFILL,
    RESEARCH_BT_PREFILL,
    SYMBOLS,
    apply_research_audit_prefill,
    combo_prefill,
    overfit_job_payload,
    single_backtest_payload,
)
from scalper_hft.config import get_settings
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore, artifacts_dir, fingerprint
from scalper_hft.research.sweep_store import SweepStore
from scalper_hft.validation.cell_audit import default_train_test
from scalper_hft.validation.sweep import DEFAULT_INTERVALS, default_strategies

settings = get_settings()
_SWEEP_DB = Path("results/sweep.db")

apply_research_audit_prefill(st.session_state)

st.title("Дослідження")
st.caption(
    "Масовий sweep стратегій × символів × таймфреймів · filter attribution · аудит комірки · режими · порівняння кривих"
)

tabs = st.tabs(
    [
        "Sweep matrix",
        "Filter attribution",
        "Порівняння equity",
        "Якість угод",
        "Топ комбінації",
        "Аудит комірки",
        "Режими",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sweep Matrix
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Sweep Matrix: стратегії × символи × таймфрейми")

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.caption("Налаштування прогону")

        sel_slow = st.toggle("Включити повільні (ML / ensemble)", value=False, key="sw_slow")
        all_strats = default_strategies(include_slow=sel_slow)
        sel_strats = st.multiselect("Стратегії", all_strats, default=all_strats[:5], key="sw_strats")
        sel_symbols = st.multiselect("Символи", SYMBOLS, default=SYMBOLS[:3], key="sw_syms")
        sel_ivs = st.multiselect("Таймфрейми", DEFAULT_INTERVALS, default=["5m", "15m", "1h"], key="sw_ivs")
        sel_days = st.slider("Днів даних", 14, 180, 60, key="sw_days")
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
            f"▶ Запустити sweep ({total_cells} клітинок)",
            disabled=not (sel_strats and sel_symbols and sel_ivs),
            key="sw_run_btn",
        )

    with col_right:
        # Завантажити наявні результати зі store
        @st.cache_data(ttl=30, show_spinner=False)
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
            with JobStore(DEFAULT_JOBS_PATH) as js:
                job = js.submit("sweep", payload)
                alive = js.worker_is_alive()
            st.success(f"Sweep у черзі як задача #{job.id} ({job.status})")
            if not alive:
                st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
            st.page_link("app_pages/jobs.py", label="Відкрити чергу задач", icon=":material/pending_actions:")

        # Завантажити для відображення
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

            # Фільтри heatmap
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
                st.warning("Немає даних після фільтру")
            else:
                # Pivot: рядки=стратегії, колонки=символ_TF
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

                # Таблиця з деталями
                with st.expander("📋 Всі результати", expanded=False):
                    display_cols = [
                        c
                        for c in [
                            "strategy",
                            "symbol",
                            "interval",
                            "mode",
                            "n_trades",
                            "total_return",
                            "sharpe",
                            "sortino",
                            "calmar",
                            "max_dd",
                            "win_rate",
                            "profit_factor",
                            "trades_per_day",
                            "avg_oos_sharpe",
                            "oos_positive_frac",
                        ]
                        if c in view.columns
                    ]
                    st.dataframe(
                        view[display_cols].sort_values("sharpe", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Filter Attribution
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Filter Attribution — аналіз впливу фільтрів")
    st.caption(
        "Показує скільки сигналів заблокував кожен фільтр і яким був би PnL без нього. "
        "Натисніть «Запустити з трейсингом» для свіжих даних."
    )

    fa_col1, fa_col2 = st.columns([1, 2])
    with fa_col1:
        from scalper_hft.strategies import REGISTRY

        fa_strat = st.selectbox("Стратегія", sorted(REGISTRY), key="fa_strat")
        fa_sym = st.selectbox("Символ", SYMBOLS, key="fa_sym")
        fa_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=1, key="fa_iv")
        fa_days = st.slider("Днів", 7, 180, 60, key="fa_days")
        fa_horizon = st.slider("Shadow-горизонт (барів)", 1, 20, 5, key="fa_horizon")
        run_trace_btn = st.button("🔬 Запустити з трейсингом", key="fa_run")

    with fa_col2:
        if run_trace_btn:
            from scalper_hft.backtest.engine import run_backtest
            from scalper_hft.backtest.execution import CostModel
            from scalper_hft.data.access import klines_from_store
            from scalper_hft.research.filter_trace import filter_attribution, filter_pnl_impact
            from scalper_hft.strategies import get_strategy

            df_k = klines_from_store(fa_sym, fa_iv, fa_days)
            if df_k is None or len(df_k) < 50:
                st.warning(f"Немає даних {fa_sym} {fa_iv} — завантажте через download")
            else:
                strat = get_strategy(fa_strat)
                cost = CostModel(
                    maker_fee=settings.maker_fee,
                    taker_fee=settings.taker_fee,
                    slippage_frac=settings.slippage_frac,
                )
                with st.spinner("Бектест з трейсингом..."):
                    res = run_backtest(df_k, strat, cost=cost, trace=True, position_pct=settings.position_pct)

                trace = res.trace
                if trace is None or len(trace) == 0:
                    st.info(
                        f"Стратегія **{fa_strat}** не підтримує детальний трейсинг. "
                        "Трейсинг реалізовано для: `mean_reversion`. "
                        "Інші стратегії повертають порожній FilterTrace."
                    )
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

                    # Filter attribution bar chart
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

                    # Shadow PnL impact
                    pnl_df = filter_pnl_impact(trace, df_k["close"], horizon_bars=fa_horizon)
                    if not pnl_df.empty:
                        st.subheader("Shadow PnL — що було б без фільтру")
                        st.caption(
                            f"Forward return за {fa_horizon} барів від моменту заблокованого сигналу × "
                            "напрямок. Якщо shadow_mean_ret > 0 → фільтр відкидає прибуткові угоди."
                        )
                        for _, row in pnl_df.iterrows():
                            verdict_msg = (
                                "⚠️ фільтр відкидає прибуткові угоди"
                                if row["shadow_mean_ret"] > 0 and row["shadow_win_rate"] > 0.5
                                else "✅ фільтр відкидає збиткові угоди"
                            )
                            verdict_badge = ":red-badge[Шкідливий]" if "⚠️" in verdict_msg else ":green-badge[Корисний]"

                            with st.container(border=True):
                                st.markdown(f"**{row['filter_name']}** {verdict_badge} — {verdict_msg}")
                                with st.container(horizontal=True):
                                    st.metric("Заблоковано", f"{int(row['n_blocked'])}", border=True)
                                    st.metric("Shadow win rate", f"{row['shadow_win_rate']:.0%}", border=True)
                                    st.metric("Shadow avg PnL", f"{row['shadow_mean_ret']:+.4%}", border=True)
                                    st.metric("Shadow total PnL", f"{row['shadow_total_pnl']:+.4%}", border=True)
        else:
            # Завантажити з sweep store якщо є
            if _SWEEP_DB.exists():
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
                    else:
                        st.info("Запустіть sweep з опцією «Filter tracing» або натисніть «Запустити з трейсингом»")
                except Exception:
                    st.info("Запустіть трейсинг або sweep з filter tracing")
            else:
                st.info("Натисніть «Запустити з трейсингом» для аналізу фільтрів")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Equity Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Порівняння Equity Curves")
    st.caption("Запустіть декілька бектестів і порівняйте нормалізовані криві на одному графіку")

    if "eq_curves" not in st.session_state:
        st.session_state["eq_curves"] = {}

    # Додати поточний бектест
    ec_col1, ec_col2 = st.columns([1, 2])
    with ec_col1:
        from scalper_hft.strategies import REGISTRY as _REG

        ec_strat = st.selectbox("Стратегія", sorted(_REG), key="ec_strat")
        ec_sym = st.selectbox("Символ", SYMBOLS, key="ec_sym")
        ec_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=1, key="ec_iv")
        ec_days = st.slider("Днів", 14, 180, 60, key="ec_days")
        ec_label = st.text_input("Мітка", value=f"{ec_strat} {ec_sym} {ec_iv}", key="ec_label")
        add_curve_btn = st.button("➕ Додати криву", key="ec_add")
        clear_curves_btn = st.button("🗑 Очистити всі", key="ec_clear")

        if clear_curves_btn:
            st.session_state["eq_curves"] = {}

        if add_curve_btn:
            from scalper_hft.backtest.engine import run_backtest
            from scalper_hft.backtest.execution import CostModel
            from scalper_hft.data.access import klines_from_store
            from scalper_hft.strategies import get_strategy

            df_k = klines_from_store(ec_sym, ec_iv, ec_days)
            if df_k is None or len(df_k) < 50:
                st.warning(f"Немає даних {ec_sym} {ec_iv}")
            else:
                strat = get_strategy(ec_strat)
                cost = CostModel(
                    maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac
                )
                with st.spinner("Бектест..."):
                    res = run_backtest(df_k, strat, cost=cost, position_pct=settings.position_pct)
                st.session_state["eq_curves"][ec_label] = res.equity
                st.success(f"Додано: {ec_label}")

    with ec_col2:
        curves = st.session_state.get("eq_curves", {})
        if not curves:
            st.info("Додайте криві зліва")
        else:
            # Нормалізовані (починають з 1.0)
            fig_eq = go.Figure()
            for label, equity in curves.items():
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

            # Correlation matrix
            if len(curves) > 1:
                # Ресемплінг до спільного денного індексу
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

            # Rolling Sharpe
            st.subheader("Rolling Sharpe (30-денне вікно)")
            fig_rs = go.Figure()
            for label, equity in curves.items():
                daily_ret = equity.resample("1D").last().pct_change().dropna()
                if len(daily_ret) >= 30:
                    rolling_sharpe = (daily_ret.rolling(30).mean() / daily_ret.rolling(30).std()) * np.sqrt(365)
                    fig_rs.add_trace(
                        go.Scatter(
                            x=rolling_sharpe.index,
                            y=rolling_sharpe.values,
                            mode="lines",
                            name=label,
                        )
                    )
            fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_rs.update_layout(
                height=300,
                yaxis_title="Sharpe (30d rolling)",
                xaxis_title="",
            )
            st.plotly_chart(fig_rs, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Trade Quality
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Аналіз якості угод")
    st.caption("Запустіть бектест щоб бачити MAE/MFE, hourly heatmap та розподіл PnL")

    tq_col1, tq_col2 = st.columns([1, 2])
    with tq_col1:
        from scalper_hft.strategies import REGISTRY as _REG2

        tq_strat = st.selectbox("Стратегія", sorted(_REG2), key="tq_strat")
        tq_sym = st.selectbox("Символ", SYMBOLS, key="tq_sym")
        tq_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=1, key="tq_iv")
        tq_days = st.slider("Днів", 14, 180, 60, key="tq_days")
        run_tq_btn = st.button("▶ Запустити аналіз", key="tq_run")

    with tq_col2:
        if run_tq_btn:
            from scalper_hft.backtest.engine import run_backtest
            from scalper_hft.backtest.execution import CostModel
            from scalper_hft.data.access import klines_from_store
            from scalper_hft.research.session_analysis import (
                hourly_heatmap_data,
                mae_mfe_analysis,
                session_breakdown,
            )
            from scalper_hft.strategies import get_strategy

            df_k = klines_from_store(tq_sym, tq_iv, tq_days)
            if df_k is None or len(df_k) < 50:
                st.warning(f"Немає даних {tq_sym} {tq_iv}")
            else:
                strat = get_strategy(tq_strat)
                cost = CostModel(
                    maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac
                )
                with st.spinner("Бектест..."):
                    res = run_backtest(df_k, strat, cost=cost, position_pct=settings.position_pct)

                trades = res.trades
                if trades is None or trades.empty:
                    st.info("Угод немає — спробуйте іншу стратегію/інструмент")
                else:
                    st.session_state["tq_res"] = {"res": res, "df": df_k}

                    # Метрики
                    m = res.metrics
                    with st.container(horizontal=True):
                        st.metric("Угод", m.n_trades, border=True)
                        st.metric("Win rate", f"{m.win_rate:.0%}", border=True)
                        st.metric("Avg PnL/угода", f"{m.avg_trade_return:.4%}", border=True)
                        st.metric("Угод/день", f"{m.trades_per_day:.2f}", border=True)

                    sub_tabs = st.tabs(["📉 MAE/MFE", "🕐 Hourly Heatmap", "📊 Розподіл PnL"])

                    with sub_tabs[0]:
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
                                "(частка MFE, що ми реально захопили). "
                                "Якщо < 50% — виходимо занадто рано."
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
                            st.caption("Комірки з <3 угодами — порожні (статистично незначущі)")

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

                            # Cumulative PnL
                            cum_pnl = (1 + rets).cumprod()
                            fig_cum = px.line(
                                cum_pnl.values,
                                title="Кумулятивний PnL угод (по угодах, не по часу)",
                                labels={"index": "# Угоди", "value": "Кумулятивний PnL"},
                            )
                            st.plotly_chart(fig_cum, width="stretch")
        else:
            st.info("Натисніть «Запустити аналіз» для відображення деталей угод")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Best Combinations
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Топ комбінації зі sweep")
    st.caption("Найкращі комбінації (стратегія × символ × таймфрейм) з усіх sweep-прогонів")

    if not _SWEEP_DB.exists():
        st.info("Немає sweep результатів. Запустіть sweep на вкладці «Sweep Matrix»")
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
            # Фільтри
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc_min_sh = bc1.number_input("Min Sharpe", -10.0, 10.0, 0.5, step=0.1, key="bc_minsh")
            bc_min_tr = bc2.number_input("Min угод", 0, 10000, 20, key="bc_mintr")
            bc_min_wr = bc3.number_input("Min win rate", 0.0, 1.0, 0.4, step=0.05, key="bc_minwr")
            bc_show_n = bc4.number_input("Показати топ N", 5, 200, 30, key="bc_n")

            # Приведення числових колонок
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

                disp_cols = [
                    c
                    for c in [
                        "strategy",
                        "symbol",
                        "interval",
                        "mode",
                        "n_trades",
                        "sharpe",
                        "sortino",
                        "calmar",
                        "total_return",
                        "max_dd",
                        "win_rate",
                        "profit_factor",
                        "trades_per_day",
                        "avg_oos_sharpe",
                        "oos_positive_frac",
                        "n_raw_signals",
                        "n_filtered",
                        "days",
                    ]
                    if c in top.columns
                ]
                top_view = top[disp_cols].reset_index(drop=True)
                st.dataframe(
                    top_view,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "total_return": st.column_config.NumberColumn("Return", format="%.2%"),
                        "max_dd": st.column_config.NumberColumn("Max DD", format="%.2%"),
                        "win_rate": st.column_config.NumberColumn("Win rate", format="%.0%"),
                        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.3f"),
                        "sortino": st.column_config.NumberColumn("Sortino", format="%.3f"),
                        "calmar": st.column_config.NumberColumn("Calmar", format="%.3f"),
                        "trades_per_day": st.column_config.NumberColumn("T/day", format="%.1f"),
                        "avg_oos_sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.3f"),
                        "oos_positive_frac": st.column_config.NumberColumn("OOS+ frac", format="%.0%"),
                    },
                )
                labels = [
                    f"{r.strategy} · {r.symbol} {r.interval}" + (f" ({r.mode})" if "mode" in top_view.columns else "")
                    for r in top_view.itertuples(index=False)
                ]
                picked = st.selectbox(
                    "Комбінація",
                    list(range(len(top_view))),
                    format_func=lambda i: labels[i],
                    key="bc_pick",
                )
                row = top_view.iloc[int(picked)]
                combo = combo_prefill(row.to_dict())
                train_b, test_b = default_train_test(combo["interval"])
                csv_bytes = top_view.to_csv(index=False).encode("utf-8")
                with st.container(horizontal=True):
                    if st.button("Відкрити бектест", icon=":material/query_stats:", key="bc_open_bt"):
                        st.session_state[RESEARCH_BT_PREFILL] = combo
                        st.switch_page("app_pages/backtest.py")
                    if st.button("Поставити аудит у чергу", icon=":material/fact_check:", key="bc_audit"):
                        st.session_state[RESEARCH_AUDIT_PREFILL] = combo
                        payload = overfit_job_payload(
                            combo["strategy"],
                            combo["symbol"],
                            combo["interval"],
                            combo["days"],
                            train_bars=train_b,
                            test_bars=test_b,
                        )
                        with JobStore(DEFAULT_JOBS_PATH) as js:
                            job = js.submit("overfit", payload)
                            alive = js.worker_is_alive()
                        st.session_state["au_last_job"] = job.id
                        if not alive:
                            st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
                        st.rerun()
                    st.download_button(
                        "Завантажити CSV",
                        data=csv_bytes,
                        file_name="sweep_top.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        key="bc_csv",
                    )

                # Зведена статистика по стратегіях
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
# TAB 6 — Cell audit
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Аудит комірки")
    st.caption(
        "Walk-forward + Deflated Sharpe + sensitivity. Вердикт PASS лише якщо всі пороги "
        "overfitting-audit виконано. Рахунок іде через чергу jobs (kind=overfit)."
    )
    from scalper_hft.research.job_artifacts import load_cell_audit
    from scalper_hft.strategies import REGISTRY as _REG_AU
    from scalper_hft.validation.cell_audit import cell_verdict

    au_col1, au_col2 = st.columns([1, 2])
    with au_col1:
        au_strat = st.selectbox("Стратегія", sorted(_REG_AU), key="au_strat")
        au_sym = st.selectbox("Символ", SYMBOLS, key="au_sym")
        au_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=min(1, len(BT_INTERVALS) - 1), key="au_iv")
        au_days = st.slider("Днів", 14, 365, 60, key="au_days")
        _dt, _dte = default_train_test(str(au_iv))
        if "au_train" not in st.session_state:
            st.session_state["au_train"] = int(_dt)
        if "au_test" not in st.session_state:
            st.session_state["au_test"] = int(_dte)
        au_train = st.number_input("Train барів", 50, 20_000, key="au_train")
        au_test = st.number_input("Test барів (OOS)", 20, 10_000, key="au_test")
        au_payload = overfit_job_payload(
            str(au_strat),
            str(au_sym),
            str(au_iv),
            int(au_days),
            train_bars=int(au_train),
            test_bars=int(au_test),
        )
        au_fp = fingerprint("overfit", au_payload)
        with JobStore(DEFAULT_JOBS_PATH) as _js:
            au_job = _js.get_by_fingerprint(au_fp)
            au_alive = _js.worker_is_alive()
        run_audit = st.button("Поставити аудит у чергу", icon=":material/fact_check:", key="au_run")
        if run_audit:
            with JobStore(DEFAULT_JOBS_PATH) as _js:
                au_job = _js.submit("overfit", au_payload)
                au_alive = _js.worker_is_alive()
            st.rerun()
        if not au_alive:
            st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
        st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")

    with au_col2:
        if au_job is None:
            st.info("Ще немає аудиту з цими параметрами.")
        elif au_job.status in {"queued", "running"}:
            prog = f"{au_job.progress_done}/{au_job.progress_total}" if au_job.progress_total else au_job.status
            st.info(f"Задача #{au_job.id} · {au_job.status} · {prog}")
        elif au_job.status == "failed":
            st.error(f"Задача #{au_job.id} провалилась: {au_job.error}")
        elif au_job.status == "cancelled":
            st.warning(f"Задача #{au_job.id} скасована.")
        elif au_job.status == "succeeded":
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
# TAB 7 — Regimes
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Режими ринку")
    st.caption(
        "Розбивка угод по волатильності, структурі та складеному стану structure|vol. "
        "Це in-sample атрибуція, не OOS-вердикт."
    )
    from scalper_hft.data.access import klines_from_store
    from scalper_hft.features.regimes import named_market_state
    from scalper_hft.research.job_artifacts import load_backtest_result
    from scalper_hft.research.session_analysis import (
        named_regime_breakdown,
        regime_breakdown,
        structure_breakdown,
    )
    from scalper_hft.strategies import REGISTRY as _REG_RG

    rg_col1, rg_col2 = st.columns([1, 2])
    with rg_col1:
        rg_strat = st.selectbox("Стратегія", sorted(_REG_RG), key="rg_strat")
        rg_sym = st.selectbox("Символ", SYMBOLS, key="rg_sym")
        rg_iv = st.selectbox("Таймфрейм", BT_INTERVALS, index=min(1, len(BT_INTERVALS) - 1), key="rg_iv")
        rg_days = st.slider("Днів", 14, 365, 60, key="rg_days")
        rg_payload = single_backtest_payload(str(rg_strat), str(rg_sym), str(rg_iv), int(rg_days))
        rg_fp = fingerprint("backtest", rg_payload)
        with JobStore(DEFAULT_JOBS_PATH) as _js:
            rg_job = _js.get_by_fingerprint(rg_fp)
            rg_alive = _js.worker_is_alive()
        if st.button("Поставити бектест у чергу", icon=":material/play_arrow:", key="rg_run"):
            with JobStore(DEFAULT_JOBS_PATH) as _js:
                rg_job = _js.submit("backtest", rg_payload)
            st.rerun()
        if not rg_alive:
            st.warning("Воркер не запущений — `uv run python -m scalper_hft.cli job worker`")
        st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")

    with rg_col2:
        if rg_job is None:
            st.info("Немає бектесту з цими параметрами — поставте задачу зліва.")
        elif rg_job.status in {"queued", "running"}:
            st.info(f"Задача #{rg_job.id} · {rg_job.status}")
        elif rg_job.status == "failed":
            st.error(f"Задача #{rg_job.id} провалилась: {rg_job.error}")
        elif rg_job.status != "succeeded":
            st.warning(f"Задача #{rg_job.id}: {rg_job.status}")
        else:
            try:
                rg_res = load_backtest_result(artifacts_dir(DEFAULT_JOBS_PATH, rg_job.id))
            except FileNotFoundError:
                st.warning("Артефакти ще не записані.")
            else:
                trades = rg_res.trades
                if trades is None or trades.empty:
                    st.info("Угод немає — немає що розкладати по режимах.")
                else:
                    df_k = klines_from_store(str(rg_sym), str(rg_iv), int(rg_days))
                    if df_k is None or df_k.empty or "close" not in df_k.columns:
                        st.warning(f"Немає klines {rg_sym} {rg_iv} для класифікації режиму.")
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
