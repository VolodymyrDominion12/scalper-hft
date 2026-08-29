"""Streamlit-дашборд для моніторингу стратегій scalper-hft.

Запуск:
    uv pip install streamlit
    .venv/bin/streamlit run scalper_hft/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.pairs import run_pairs_backtest
from scalper_hft.config import get_settings
from scalper_hft.data.storage import klines_path, load_funding, load_klines, load_trades
from scalper_hft.live.store import PaperStore
from scalper_hft.strategies import REGISTRY, get_strategy
from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials

st.set_page_config(page_title="scalper-hft", page_icon="📈", layout="wide")
st.title("scalper-hft — моніторинг стратегій")

settings = get_settings()
data_dir = settings.data_dir_abs
_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
_PAIR_CHOICES = ["XRPUSDT/BTCUSDT", "BTCUSDT/ETHUSDT", "LINKUSDT/BTCUSDT", "LINKUSDT/ETHUSDT"]

st.sidebar.header("Параметри")
strategy_name = st.sidebar.selectbox("Стратегія", sorted(REGISTRY), index=sorted(REGISTRY).index("pairs_arb") if "pairs_arb" in REGISTRY else 0)
is_pairs = strategy_name == "pairs_arb"
if is_pairs:
    pair_sel = st.sidebar.selectbox("Пара", _PAIR_CHOICES)
    leg1, leg2 = pair_sel.split("/")
    interval = st.sidebar.selectbox("Таймфрейм", ["1h", "15m", "5m"], index=0)
else:
    symbol = st.sidebar.selectbox("Символ", _SYMBOLS)
    interval = st.sidebar.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h"], index=1)
days = st.sidebar.slider("Глибина даних, днів", 7, 365, 90 if is_pairs else 30)
run_bt = st.sidebar.button("Запустити бектест")

st.header("1. Дані (кеш)")
rows = []
for sym in _SYMBOLS:
    k = load_klines(klines_path(data_dir, sym, "1h" if is_pairs else "1m"))
    t = load_trades(data_dir / f"{sym}_aggTrades.parquet")
    f = load_funding(data_dir / f"{sym}_funding.parquet")
    bt = data_dir / f"{sym}_bookTicker.parquet"
    bt_df = pd.read_parquet(bt) if bt.exists() else None
    rows.append(
        {
            "symbol": sym,
            "klines": len(k) if k is not None else 0,
            "aggTrades": len(t) if t is not None else 0,
            "funding": len(f) if f is not None else 0,
            "bookTicker": len(bt_df) if bt_df is not None else 0,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.header("2. Бектест")
if run_bt:
    with st.spinner("Бектест..."):
        cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
        strategy = get_strategy(strategy_name)
        if is_pairs:
            d1 = load_klines(klines_path(data_dir, leg1, interval))
            d2 = load_klines(klines_path(data_dir, leg2, interval))
            if d1 is None or d2 is None or len(d1) < 100 or len(d2) < 100:
                st.warning(f"Немає даних {leg1}/{leg2} {interval} — download спершу")
            else:
                f1 = load_funding(data_dir / f"{leg1}_funding.parquet")
                f2 = load_funding(data_dir / f"{leg2}_funding.parquet")
                res = run_pairs_backtest(
                    d1, d2, strategy, f1, f2,
                    position_pct=settings.pair_notional_pct, cost=cost, maker_execution=True,
                )
                m = res.metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Дохідність", f"{m.total_return:.2%}")
                c2.metric("Sharpe (год.)", f"{m.sharpe_hourly:.2f}")
                c3.metric("Угоди", f"{m.n_trades}")
                c4.metric("Max DD", f"{m.max_drawdown:.2%}")
                fig = go.Figure(go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Equity"))
                fig.update_layout(title=f"pairs_arb · {leg1}/{leg2} {interval} maker", height=350)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Повні метрики"):
                    st.text(m.summary())
        else:
            df = load_klines(klines_path(data_dir, symbol, interval))
            if df is None or len(df) < 100:
                st.warning(f"Немає даних {symbol} {interval} — запустіть download спершу")
            else:
                trades = load_trades(data_dir / f"{symbol}_aggTrades.parquet") if getattr(strategy, "needs_trades", False) else None
                funding = load_funding(data_dir / f"{symbol}_funding.parquet") if getattr(strategy, "needs_funding", False) else None
                res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
                m = res.metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Дохідність", f"{m.total_return:.2%}")
                c2.metric("Sharpe (год.)", f"{m.sharpe_hourly:.2f}")
                c3.metric("Угоди", f"{m.n_trades}")
                c4.metric("Win rate", f"{m.win_rate:.0%}")
                fig = go.Figure(go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Equity"))
                fig.update_layout(title=f"{strategy_name} · {symbol} {interval}", height=350)
                st.plotly_chart(fig, use_container_width=True)
                ret = res.equity.pct_change().dropna()
                n_trials = estimate_n_trials(max(len(strategy.param_space), 1), 40)
                dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)
                st.info(f"Deflated Sharpe: **{dsr:.3f}** (trials={n_trials}) — edge значущий якщо > 0.95")
                with st.expander("Повні метрики"):
                    st.text(m.summary())

st.header("3. Paper pairs (SQLite)")
db = Path("results") / "paper_pairs.sqlite"
if db.exists():
    store = PaperStore(db)
    stats = store.fill_stats()
    st.caption("fill / unfilled / pending — модель post-only")
    st.write(stats)
    eq = store.recent_equity()
    if not eq.empty:
        eq["ts"] = pd.to_datetime(eq["ts"])
        fig = go.Figure()
        for pair, g in eq.sort_values("ts").groupby("pair"):
            fig.add_trace(go.Scatter(x=g["ts"], y=g["equity"], mode="lines", name=str(pair)))
        fig.update_layout(title="Paper equity by pair", height=320)
        st.plotly_chart(fig, use_container_width=True)
    orders = store.recent_orders()
    if not orders.empty:
        st.subheader("Останні ордери")
        st.dataframe(orders.head(40), use_container_width=True)
    store.close()
else:
    st.info("Немає results/paper_pairs.sqlite — запустіть: python -m scalper_hft.cli paper-replay-pairs")

st.header("4. Paper-run CSV")
results_dir = Path("results")
if results_dir.exists():
    eq_files = sorted(results_dir.glob("paper_equity_*.csv"))
    if eq_files:
        for f in eq_files:
            eq = pd.read_csv(f, parse_dates=["ts"]).set_index("ts")
            fig = go.Figure(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name=f.stem))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Немає CSV paper-run.")
else:
    st.info("Немає results/")

st.header("5. Cohort / Stress / Capacity (Спринт 4)")
run_diag = st.sidebar.button("Запустити діагностику (cohort+stress+capacity)")
if run_diag:
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    try:
        strategy = get_strategy(strategy_name)
        if is_pairs:
            st.warning("Діагностика працює для одиночних стратегій (не pairs_arb).")
        else:
            df = load_klines(klines_path(data_dir, symbol, interval))
            if df is None or len(df) < 100:
                st.warning(f"Немає даних {symbol} {interval} — download спершу")
            else:
                trades = load_trades(data_dir / f"{symbol}_aggTrades.parquet") if getattr(strategy, "needs_trades", False) else None
                funding = load_funding(data_dir / f"{symbol}_funding.parquet") if getattr(strategy, "needs_funding", False) else None
                res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
                ret = res.equity.pct_change().dropna()

                with st.expander("Cohort decay (Predictive Marketing)", expanded=False):
                    from scalper_hft.validation.cohort import cohort_report

                    st.text(cohort_report(res.trades))
                with st.expander("Стрес-тест (Narang гл. 10)", expanded=False):
                    from scalper_hft.validation.stress import stress_report

                    rep = stress_report(ret)
                    st.dataframe(rep.round(4), use_container_width=True)
                    st.caption("crash = найгірше вікно ×2; liquidity = витрати ×10; "
                               "vol_spike = волатильність ×2; funding_shock = per-bar 0.1%")
                with st.expander("Capacity (share of wallet)", expanded=False):
                    from scalper_hft.validation.capacity import capacity_curve, saturation_scale

                    curve = capacity_curve(df, strategy, scales=[1.0, 2.0, 5.0, 10.0],
                                           cost=cost, position_pct=settings.position_pct)
                    fig = go.Figure(go.Bar(x=curve["scale"], y=curve["sharpe"]))
                    fig.update_layout(title=f"Sharpe при масштабі позицій ×(1..10) — насичення ×{saturation_scale(curve):g}",
                                      xaxis_title="scale", yaxis_title="Sharpe", height=320)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(curve.round(4), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Діагностика не вдалася: {exc}")
