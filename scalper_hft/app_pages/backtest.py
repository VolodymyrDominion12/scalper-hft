"""Сторінка «Бектест»: запуск, інтерактивний графік угод, діагностика.

Клік по маркеру угоди (вхід/вихід) на графіку відкриває деталі угоди:
свічки навколо входу, SL/TP-сегмент, позицію та equity.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scalper_hft.app_pages._common import BT_INTERVALS, PAIR_CHOICES, SYMBOLS, apply_research_bt_prefill
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import klines_from_store
from scalper_hft.data.storage import load_funding, load_trades
from scalper_hft.features.indicators import add_standard_features
from scalper_hft.research.dashboard_brief import dsr_verdict, hurdle_note
from scalper_hft.research.job_artifacts import load_backtest_result, load_pairs_result
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore, artifacts_dir, fingerprint
from scalper_hft.research.strategy_book import select_label
from scalper_hft.strategies import REGISTRY, get_strategy
from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
from scalper_hft.visualization import (
    auto_indicator_columns,
    find_trade_by_ts,
    make_backtest_figure,
    trade_detail_figure,
    trades_table,
)

settings = get_settings()
data_dir = settings.data_dir_abs

apply_research_bt_prefill(st.session_state)

st.title("Бектест")

st.sidebar.header("Параметри", divider=False)
_strat_names = sorted(REGISTRY)
strategy_name = st.sidebar.selectbox(
    "Стратегія",
    _strat_names,
    index=_strat_names.index("pairs_arb") if "pairs_arb" in _strat_names else 0,
    format_func=select_label,
    key="bt_strategy",
)
is_pairs = strategy_name == "pairs_arb"
if is_pairs:
    pair_sel = st.sidebar.selectbox("Пара", PAIR_CHOICES, key="bt_pair")
    leg1, leg2 = pair_sel.split("/")
    interval = st.sidebar.selectbox("Таймфрейм", ["1h", "15m", "5m"], index=0, key="bt_interval_pairs")
else:
    symbol = st.sidebar.selectbox("Символ", SYMBOLS, key="bt_symbol")
    interval = st.sidebar.selectbox("Таймфрейм", BT_INTERVALS, index=1, key="bt_interval_single")
days = st.sidebar.slider("Глибина даних, днів", 7, 365, 90 if is_pairs else 30, key="bt_days")
run_bt = st.sidebar.button("▶ Запустити бектест")
run_rerun = st.sidebar.button("Перезапустити задачу")
run_diag = st.sidebar.button("🩺 Діагностика (cohort/stress)")


def _clear_trade_selection() -> None:
    """Скинути вибір угоди: і наш стан, і selection widget-стану графіка."""
    st.session_state["bt_sel_ts"] = None
    st.session_state.pop("bt_fig", None)


@st.fragment
def _render_bt_chart(view: dict) -> None:
    """Графік угод + клік по маркеру + таблиця (окремий фрагмент — не перезапускає бектест)."""
    df, res, title = view["df"], view["res"], view["title"]
    st.subheader("Графік угод")
    left, right = st.columns([1, 4])
    with left:
        ts0 = df.index[0].to_pydatetime()
        ts1 = df.index[-1].to_pydatetime()
        st.caption("Вікно графіка")
        window = st.slider(
            "Час",
            min_value=ts0,
            max_value=ts1,
            value=(ts0, ts1),
            format="%d.%m %H:%M",
            key="bt_window",
        )
        with_trades = st.toggle("Точки входу/виходу", value=True, key="bt_trades")
        with_sl_tp = st.toggle("Рівні SL / TP", value=True, key="bt_sl_tp")
        with_inds = st.toggle("Індикатори", value=True, key="bt_inds")
        max_bars = st.select_slider(
            "Максимум барів",
            options=[1_000, 5_000, 20_000, 100_000, 500_000],
            value=20_000,
            key="bt_max_bars",
        )
        st.caption(f"Угод: {len(res.trades)} · Max DD: {res.metrics.max_drawdown:.1%}")
        st.caption("💡 Клік по маркеру ▲/▼/× — деталі угоди")
    with right:
        fdf = add_standard_features(df)  # фічі лише для графіка
        fig = make_backtest_figure(
            fdf,
            res,
            symbol=title,
            start=window[0],
            end=window[1],
            max_bars=max_bars,
            with_trades=with_trades,
            with_sl_tp=with_sl_tp,
            indicators=auto_indicator_columns(fdf) if with_inds else [],
        )
        sel = st.plotly_chart(fig, width="stretch", key="bt_fig", on_select="rerun", selection_mode="points")
        sel_state: Any = getattr(sel, "selection", None)
        if sel_state:
            # шукаємо угоду серед усіх вибраних точок (клік може зачепити
            # лінію індикатора на тому ж барі — ts все одно співпаде)
            points = sel_state.get("points") if isinstance(sel_state, dict) else getattr(sel_state, "points", None)
            for p in points or []:
                if isinstance(p, dict) and p.get("x") is not None and find_trade_by_ts(res.trades, p["x"]) is not None:
                    st.session_state["bt_sel_ts"] = p["x"]
                    break

    # ── Деталі вибраної угоди ──────────────────────────────────────────────
    sel_ts = st.session_state.get("bt_sel_ts")
    trade = find_trade_by_ts(res.trades, sel_ts) if sel_ts is not None else None
    if trade is not None:
        t0 = pd.Timestamp(trade["entry_ts"])
        with st.container(border=True):
            st.subheader(f"Деталі угоди · {t0:%d.%m.%Y %H:%M}")
            sl = trade["sl_price"] if "sl_price" in trade.index else float("nan")
            tp = trade["tp_price"] if "tp_price" in trade.index else float("nan")

            with st.container(horizontal=True):
                st.metric("Сторона", "Лонг" if trade["side"] == 1 else "Шорт", border=True)
                st.metric("Вхід → Вихід", f"{trade['entry_price']:.2f} → {trade['exit_price']:.2f}", border=True)
                st.metric("PnL", f"{trade['ret']:.3%}", border=True)
                st.metric("SL / TP", f"{sl:.2f} / {tp:.2f}" if pd.notna(sl) and pd.notna(tp) else "—", border=True)

            st.plotly_chart(trade_detail_figure(fdf, res, trade["entry_ts"]), width="stretch", key="bt_detail_fig")
            if st.button("✕ Закрити деталі", key="bt_clear_sel", on_click=_clear_trade_selection):
                st.rerun(scope="fragment")

    st.subheader("Угоди")
    if res.trades is not None and not res.trades.empty:
        st.dataframe(
            trades_table(res, initial_capital=10_000.0),
            width="stretch",
            hide_index=True,
            column_config={
                "Вхід": st.column_config.DatetimeColumn("Вхід", format="DD.MM.YYYY HH:mm"),
                "Вихід": st.column_config.DatetimeColumn("Вихід", format="DD.MM.YYYY HH:mm"),
                "Сторона": st.column_config.TextColumn("Сторона"),
                "Ціна входу": st.column_config.NumberColumn("Ціна входу", format="%.2f"),
                "Ціна виходу": st.column_config.NumberColumn("Ціна виходу", format="%.2f"),
                "SL": st.column_config.NumberColumn("SL", format="%.2f"),
                "TP": st.column_config.NumberColumn("TP", format="%.2f"),
                "PnL, %": st.column_config.NumberColumn("PnL, %", format="%.3f"),
                "PnL, $": st.column_config.NumberColumn("PnL, $", format="%.2f"),
            },
        )
    else:
        st.info("Угод за цей період немає — спробуйте іншу стратегію/період.")


st.header("Запуск бектесту")
if is_pairs:
    _kind = "pairs"
    _payload = {
        "strategy": strategy_name,
        "leg1": leg1,
        "leg2": leg2,
        "interval": interval,
        "days": days,
        "params": {},
        "maker": True,
        "position_pct": settings.pair_notional_pct,
        "base_interval": "1m",
    }
    _title = f"pairs_arb · {leg1}/{leg2} {interval} maker"
else:
    _kind = "backtest"
    _payload = {
        "strategy": strategy_name,
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "params": {},
        "maker": False,
        "trace": False,
        "base_interval": "1m",
        "breakeven_gate": False,
    }
    _title = f"{strategy_name} · {symbol} {interval}"

_fp = fingerprint(_kind, _payload)
with JobStore(DEFAULT_JOBS_PATH) as _js:
    _job = _js.get_by_fingerprint(_fp)
    _alive = _js.worker_is_alive()

if run_bt:
    with JobStore(DEFAULT_JOBS_PATH) as _js:
        _job = _js.submit(_kind, _payload)
    st.rerun()
if run_rerun:
    with JobStore(DEFAULT_JOBS_PATH) as _js:
        _job = _js.submit(_kind, _payload, force=True)
    st.rerun()

if not _alive:
    st.warning("Research worker не запущений. `uv run python -m scalper_hft.cli job worker`")
st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")

if _job is None:
    st.caption("Ще немає задачі з цими параметрами.")
elif _job.status in {"queued", "running"}:
    prog = f"{_job.progress_done}/{_job.progress_total}" if _job.progress_total else _job.status
    st.info(f"Задача #{_job.id} · {_job.status} · {prog}")
elif _job.status == "failed":
    st.error(f"Задача #{_job.id} провалилась: {_job.error}")
elif _job.status == "cancelled":
    st.warning(f"Задача #{_job.id} скасована.")
elif _job.status == "succeeded":
    _job_dir = artifacts_dir(DEFAULT_JOBS_PATH, _job.id)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    if is_pairs:
        st.session_state.pop("bt_view", None)
        try:
            pairs_res = load_pairs_result(_job_dir)
        except FileNotFoundError:
            st.warning("Артефакти ще не записані.")
        else:
            m = pairs_res.metrics
            with st.container(horizontal=True):
                st.metric("Дохідність", f"{m.total_return:.2%}", border=True)
                st.metric("Sharpe (год.)", f"{m.sharpe_hourly:.2f}", border=True)
                st.metric("Угоди", f"{m.n_trades}", border=True)
                st.metric("Max DD", f"{m.max_drawdown:.2%}", border=True, delta_color="inverse")
                st.metric("Profit factor", f"{m.profit_factor:.2f}", border=True)
            if m.n_trades:
                st.caption(hurdle_note(m.avg_trade_return, cost.round_trip_maker(), n_legs=2))
            fig = go.Figure(
                go.Scatter(x=pairs_res.equity.index, y=pairs_res.equity.values, mode="lines", name="Equity")
            )
            fig.update_layout(title=_title, height=350)
            st.plotly_chart(fig, width="stretch")
            with st.expander("Повні метрики"):
                st.text(m.summary())
    else:
        strategy = get_strategy(strategy_name)
        df = klines_from_store(symbol, interval, days)
        try:
            res = load_backtest_result(_job_dir)
        except FileNotFoundError:
            st.warning("Артефакти ще не записані.")
        else:
            if df is None or len(df) < 2:
                st.warning(f"Немає даних {symbol} {interval} для графіка — download спершу")
            else:
                st.session_state["bt_view"] = {"df": df, "res": res, "title": _title}
                st.session_state.pop("bt_sel_ts", None)
                m = res.metrics
                ret = res.equity.pct_change().dropna()
                n_trials = estimate_n_trials(max(len(strategy.param_space), 1), 40)
                dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)
                verdict = dsr_verdict(dsr)
                dsr_label = {"significant": "значущий", "weak": "слабкий", "none": "немає edge"}[verdict]
                with st.container(horizontal=True):
                    st.metric("Дохідність", f"{m.total_return:.2%}", border=True)
                    st.metric("Sharpe (год.)", f"{m.sharpe_hourly:.2f}", border=True)
                    st.metric("Угоди", f"{m.n_trades}", border=True)
                    st.metric("Win rate", f"{m.win_rate:.0%}", border=True)
                    st.metric("Profit factor", f"{m.profit_factor:.2f}", border=True)
                    st.metric(
                        "Deflated Sharpe",
                        f"{dsr:.3f}",
                        delta=dsr_label,
                        delta_color="normal" if verdict == "significant" else "inverse",
                        border=True,
                        help=f"> 0.95 = значущий edge після поправки на trials={n_trials}",
                    )
                if m.n_trades:
                    st.caption(hurdle_note(m.avg_trade_return, cost.round_trip_maker(), n_legs=1))
                with st.expander("Повні метрики", icon=":material/analytics:"):
                    st.text(m.summary())

if st.session_state.get("bt_fp") != _fp:
    if not (_job and _job.status == "succeeded" and not is_pairs):
        st.session_state.pop("bt_view", None)
st.session_state["bt_fp"] = _fp

if st.session_state.get("bt_view") is not None:
    _render_bt_chart(st.session_state["bt_view"])

    # ── Розширена аналітика після основного графіка ─────────────────────────
    bt_res = st.session_state["bt_view"].get("res")
    bt_df = st.session_state["bt_view"].get("df")
    if bt_res is not None and bt_df is not None:
        st.header("Розширена аналітика")
        anal_tabs = st.tabs(["📊 Метрики", "🔍 Фільтри", "🎯 MAE/MFE", "📅 Сесійний аналіз"])

        with anal_tabs[0]:
            m = bt_res.metrics
            with st.container(horizontal=True):
                st.metric("Sharpe (річн.)", f"{m.sharpe:.3f}", border=True)
                st.metric("Sortino", f"{m.sortino:.3f}", border=True)
                st.metric("Calmar", f"{m.calmar:.3f}", border=True)
                st.metric("Max DD", f"{m.max_drawdown:.1%}", border=True)
                st.metric("Угод/день", f"{m.trades_per_day:.2f}", border=True)
                st.metric("P(розорення)", f"{m.risk_of_ruin:.4f}", border=True)

            # Rolling Sharpe
            if bt_res.equity is not None and len(bt_res.equity) > 30:
                daily_ret = bt_res.equity.resample("1D").last().pct_change().dropna()
                if len(daily_ret) >= 14:
                    import numpy as np

                    rs = (daily_ret.rolling(14).mean() / daily_ret.rolling(14).std()) * np.sqrt(365)
                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name="Sharpe 14d rolling"))
                    fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_rs.update_layout(title="Rolling Sharpe (14-денне вікно)", height=280, yaxis_title="Sharpe")
                    st.plotly_chart(fig_rs, width="stretch")

        with anal_tabs[1]:
            trace = getattr(bt_res, "trace", None)
            if trace is not None and len(trace) > 0:
                from scalper_hft.research.filter_trace import filter_attribution, filter_pnl_impact

                n_raw = len(trace)
                n_blocked = trace.n_blocked()
                fa1, fa2, fa3 = st.columns(3)
                fa1.metric("Raw сигналів", n_raw)
                fa2.metric("Пройшли фільтри", trace.n_passed())
                fa3.metric(
                    "Заблоковано", n_blocked, delta=f"-{n_blocked / n_raw:.0%}" if n_raw else "", delta_color="inverse"
                )

                attr_df = filter_attribution(trace)
                if not attr_df.empty:
                    fig_a = go.Figure(
                        go.Bar(
                            x=attr_df["filter_name"],
                            y=attr_df["n_blocked"],
                            text=attr_df["n_blocked"],
                            textposition="outside",
                            marker_color="#f97316",
                        )
                    )
                    fig_a.update_layout(title="Скільки сигналів заблокував кожен фільтр", height=320)
                    st.plotly_chart(fig_a, width="stretch")

                pnl_df = filter_pnl_impact(trace, bt_df["close"], horizon_bars=5)
                if not pnl_df.empty:
                    st.subheader("Shadow PnL (без фільтру)")
                    st.dataframe(
                        pnl_df.style.background_gradient(subset=["shadow_mean_ret", "shadow_win_rate"], cmap="RdYlGn"),
                        width="stretch",
                    )
            else:
                st.info(
                    "Трейсинг фільтрів не активовано. Запустіть бектест з параметром `trace=True` "
                    "або перейдіть на сторінку **🔬 Дослідження → Filter Attribution**."
                )
                if st.button("🔬 Поставити бектест з трейсингом у чергу", key="bt_run_trace"):
                    trace_payload = dict(_payload)
                    trace_payload["trace"] = True
                    with JobStore(DEFAULT_JOBS_PATH) as _js:
                        _js.submit("backtest", trace_payload)
                    st.info("Задачу з трейсингом поставлено в чергу (інший fingerprint).")
                    st.rerun()

        with anal_tabs[2]:
            if bt_res.trades is not None and not bt_res.trades.empty:
                from scalper_hft.research.session_analysis import mae_mfe_analysis

                mae_df = mae_mfe_analysis(bt_res.trades, bt_df)
                if not mae_df.empty:
                    import plotly.express as px

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
                    m1e, m2e, m3e = st.columns(3)
                    m1e.metric("Avg MAE", f"{mae_df['mae'].mean():+.4%}")
                    m2e.metric("Avg MFE", f"{mae_df['mfe'].mean():+.4%}")
                    m3e.metric("Avg Efficiency", f"{avg_eff:.1%}" if avg_eff == avg_eff else "N/A")
                    st.caption("Efficiency = ret / MFE. Якщо < 50% — виходимо занадто рано.")
                else:
                    st.info("Не вдалось розрахувати MAE/MFE")
            else:
                st.info("Немає угод для MAE/MFE аналізу")

        with anal_tabs[3]:
            if bt_res.trades is not None and not bt_res.trades.empty:
                from scalper_hft.research.session_analysis import (
                    hourly_heatmap_data,
                    session_breakdown,
                    weekday_breakdown,
                )

                sub_s1, sub_s2 = st.columns(2)
                with sub_s1:
                    sess = session_breakdown(bt_res.trades)
                    if not sess.empty:
                        fig_sess = go.Figure(
                            go.Bar(
                                x=sess.index,
                                y=sess["n_trades"],
                                marker_color=[f"hsl({int(wr * 120)},70%,45%)" for wr in sess["win_rate"].fillna(0.5)],
                                text=[f"{wr:.0%}" for wr in sess["win_rate"].fillna(0)],
                                textposition="outside",
                            )
                        )
                        fig_sess.update_layout(
                            title="Угоди по годинах UTC (колір = win rate)",
                            xaxis_title="Година UTC",
                            yaxis_title="Кількість угод",
                            height=320,
                        )
                        st.plotly_chart(fig_sess, width="stretch")

                with sub_s2:
                    wd = weekday_breakdown(bt_res.trades)
                    if not wd.empty:
                        fig_wd = go.Figure(
                            go.Bar(
                                x=wd["day"],
                                y=wd["n_trades"],
                                marker_color=[f"hsl({int(wr * 120)},70%,45%)" for wr in wd["win_rate"].fillna(0.5)],
                                text=[f"{wr:.0%}" for wr in wd["win_rate"].fillna(0)],
                                textposition="outside",
                            )
                        )
                        fig_wd.update_layout(
                            title="Угоди по днях тижня (колір = win rate)",
                            xaxis_title="",
                            yaxis_title="Кількість угод",
                            height=320,
                        )
                        st.plotly_chart(fig_wd, width="stretch")

                hm = hourly_heatmap_data(bt_res.trades)
                if not hm.empty:
                    import plotly.express as px

                    fig_hm = px.imshow(
                        hm.values,
                        x=list(hm.columns),
                        y=list(hm.index),
                        color_continuous_scale="RdYlGn",
                        zmin=0,
                        zmax=1,
                        title="Win rate: день тижня × година UTC",
                    )
                    fig_hm.update_layout(height=300)
                    st.plotly_chart(fig_hm, width="stretch")
            else:
                st.info("Немає угод для сесійного аналізу")


st.header("Діагностика (cohort / stress / capacity)")
if run_diag:
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    try:
        strategy = get_strategy(strategy_name)
        if is_pairs:
            st.warning("Діагностика працює для одиночних стратегій (не pairs_arb).")
        else:
            df = klines_from_store(symbol, interval, days)
            if df is None or len(df) < 100:
                st.warning(f"Немає даних {symbol} {interval} — download спершу")
            else:
                trades = (
                    load_trades(data_dir / f"{symbol}_aggTrades.parquet")
                    if getattr(strategy, "needs_trades", False)
                    else None
                )
                funding = (
                    load_funding(data_dir / f"{symbol}_funding.parquet")
                    if getattr(strategy, "needs_funding", False)
                    else None
                )
                diag_res = run_backtest(
                    df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct
                )
                ret = diag_res.equity.pct_change().dropna()

                with st.expander("Cohort decay (Predictive Marketing)", expanded=False):
                    from scalper_hft.validation.cohort import cohort_report

                    st.text(cohort_report(diag_res.trades))
                with st.expander("Стрес-тест (Narang гл. 10)", expanded=False):
                    from scalper_hft.validation.stress import stress_report

                    rep = stress_report(ret)
                    st.dataframe(rep.round(4), width="stretch")
                    st.caption(
                        "crash = найгірше вікно ×2; liquidity = витрати ×10; "
                        "vol_spike = волатильність ×2; funding_shock = per-bar 0.1%"
                    )
                with st.expander("Capacity (share of wallet)", expanded=False):
                    from scalper_hft.validation.capacity import capacity_curve, saturation_scale

                    curve = capacity_curve(
                        df, strategy, scales=[1.0, 2.0, 5.0, 10.0], cost=cost, position_pct=settings.position_pct
                    )
                    fig = go.Figure(go.Bar(x=curve["scale"], y=curve["sharpe"]))
                    fig.update_layout(
                        title=f"Sharpe при масштабі позицій ×(1..10) — насичення ×{saturation_scale(curve):g}",
                        xaxis_title="scale",
                        yaxis_title="Sharpe",
                        height=320,
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(curve.round(4), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Діагностика не вдалася: {exc}")
