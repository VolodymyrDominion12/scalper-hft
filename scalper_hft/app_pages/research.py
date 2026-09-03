"""Сторінка «🔬 Дослідження»: масовий sweep, filter attribution, порівняння стратегій.

Секції:
  1. Sweep Matrix — запуск масового прогону та heatmap результатів
  2. Filter Attribution — аналіз які фільтри скільки угод відкидають і чи це вигідно
  3. Equity Comparison — порівняння equity кривих декількох комбінацій
  4. Trade Quality — MAE/MFE, hourly heatmap, розподіл PnL
  5. Best Combinations — топ результатів з кнопкою переходу до бектесту
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scalper_hft.app_pages._common import SYMBOLS
from scalper_hft.config import get_settings
from scalper_hft.research.sweep_store import SweepStore

settings = get_settings()
_SWEEP_DB = Path("results/sweep.db")

st.title("🔬 Дослідження")
st.caption(
    "Масовий sweep всіх стратегій × символів × таймфреймів · Filter attribution · Аналіз угод · Порівняння кривих"
)

tabs = st.tabs(
    [
        "📊 Sweep Matrix",
        "🔍 Filter Attribution",
        "📈 Порівняння Equity",
        "🎯 Якість Угод",
        "🏆 Топ Комбінації",
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
        from scalper_hft.validation.sweep import DEFAULT_INTERVALS, default_strategies

        all_strats = default_strategies(include_slow=False)
        sel_strats = st.multiselect("Стратегії", all_strats, default=all_strats[:5], key="sw_strats")
        sel_symbols = st.multiselect("Символи", SYMBOLS, default=SYMBOLS[:3], key="sw_syms")
        sel_ivs = st.multiselect("Таймфрейми", DEFAULT_INTERVALS, default=["5m", "15m", "1h"], key="sw_ivs")
        sel_days = st.slider("Днів даних", 14, 180, 60, key="sw_days")
        sel_mode = st.radio("Режим", ["backtest", "walkforward"], key="sw_mode", horizontal=True)
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
            from scalper_hft.validation.sweep import run_sweep

            with st.spinner(f"Sweep {total_cells} клітинок..."):
                store = SweepStore(_SWEEP_DB)
                df_sw = run_sweep(
                    strategies=sel_strats,
                    symbols=sel_symbols,
                    intervals=sel_ivs,
                    days=sel_days,
                    mode=sel_mode,
                    workers=sel_workers,
                    enable_trace=sel_trace,
                    store=store,
                    resume=sel_resume,
                )
                store.close()
            st.cache_data.clear()
            ok = df_sw[df_sw.get("status", "ok") == "ok"] if "status" in df_sw.columns else df_sw
            st.success(f"Готово: {len(ok)} / {len(df_sw)} клітинок успішно")

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
        fa_iv = st.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h", "4h"], index=1, key="fa_iv")
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

                        # Зберегти у session_state для Tab 5
                        st.session_state["fa_result"] = {
                            "strat": fa_strat,
                            "sym": fa_sym,
                            "iv": fa_iv,
                            "res": res,
                            "trace": trace,
                            "pnl_df": pnl_df,
                        }
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
        ec_iv = st.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h"], index=1, key="ec_iv")
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
        tq_iv = st.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h"], index=1, key="tq_iv")
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
    st.subheader("🏆 Топ комбінації зі sweep")
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
                sort_col = "avg_oos_sharpe" if "avg_oos_sharpe" in filtered.columns else "sharpe"
                top = filtered.nlargest(int(bc_show_n), sort_col)

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
                    ]
                    if c in top.columns
                ]
                st.dataframe(
                    top[disp_cols].reset_index(drop=True),
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
