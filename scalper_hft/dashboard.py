"""Streamlit-дашборд для моніторингу стратегій scalper-hft.

Запуск:
    uv pip install streamlit
    .venv/bin/streamlit run scalper_hft/dashboard.py

Секції:
    1. Дані: обсяг кешу (klines/trades/funding/bookTicker) по символах;
    2. Бектест: метрики обраної стратегії на обраному символі/таймфреймі;
    3. Аудит: walk-forward + Deflated Sharpe (стисло);
    4. Paper-run: результати з results/paper_* (equity, угоди).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.storage import klines_path, load_klines, load_funding, load_trades
from scalper_hft.strategies import REGISTRY, get_strategy
from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials

st.set_page_config(page_title="scalper-hft", page_icon="📈", layout="wide")
st.title("📈 scalper-hft — моніторинг стратегій")

settings = get_settings()
data_dir = settings.data_dir_abs

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Параметри")
symbol = st.sidebar.selectbox("Символ", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
interval = st.sidebar.selectbox("Таймфрейм", ["1m", "5m", "15m", "1h"], index=1)
strategy_name = st.sidebar.selectbox("Стратегія", sorted(REGISTRY))
days = st.sidebar.slider("Глибина даних, днів", 7, 120, 30)
run_bt = st.sidebar.button("Запустити бектест")

# ── Секція 1: дані ─────────────────────────────────────────────────────────
st.header("1. Дані (кеш)")
rows = []
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    k = load_klines(klines_path(data_dir, sym, "1m"))
    t = load_trades(data_dir / f"{sym}_aggTrades.parquet")
    f = load_funding(data_dir / f"{sym}_funding.parquet")
    bt = data_dir / f"{sym}_bookTicker.parquet"
    bt_df = pd.read_parquet(bt) if bt.exists() else None
    rows.append(
        {
            "symbol": sym,
            "klines_1m": len(k) if k is not None else 0,
            "aggTrades": len(t) if t is not None else 0,
            "funding": len(f) if f is not None else 0,
            "bookTicker": len(bt_df) if bt_df is not None else 0,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Секція 2: бектест ──────────────────────────────────────────────────────
st.header("2. Бектест")
if run_bt:
    with st.spinner("Бектест..."):
        df = load_klines(klines_path(data_dir, symbol, interval))
        if df is None or len(df) < 100:
            st.warning(f"Немає даних {symbol} {interval} — запустіть download спершу")
        else:
            strategy = get_strategy(strategy_name)
            trades = load_trades(data_dir / f"{symbol}_aggTrades.parquet") if getattr(strategy, "needs_trades", False) else None
            funding = load_funding(data_dir / f"{symbol}_funding.parquet") if getattr(strategy, "needs_funding", False) else None
            cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
            res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
            m = res.metrics

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Дохідність", f"{m.total_return:.2%}")
            c2.metric("Sharpe (год.)", f"{m.sharpe_hourly:.2f}")
            c3.metric("Угоди", f"{m.n_trades}")
            c4.metric("Win rate", f"{m.win_rate:.0%}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Max DD", f"{m.max_drawdown:.2%}")
            c6.metric("Profit factor", f"{m.profit_factor:.2f}")
            c7.metric("Експозиція", f"{m.exposure:.0%}")
            c8.metric("Угод/день", f"{m.trades_per_day:.1f}")

            # equity curve
            fig = go.Figure(go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Equity"))
            fig.update_layout(title=f"{strategy_name} · {symbol} {interval}", height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Deflated Sharpe
            ret = res.equity.pct_change().dropna()
            n_trials = estimate_n_trials(max(len(strategy.param_space), 1), 40)
            dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)
            st.info(f"Deflated Sharpe: **{dsr:.3f}** (trials={n_trials}) — edge значущий якщо > 0.95")

            with st.expander("Повні метрики"):
                st.text(m.summary())

# ── Секція 3: paper-run результати ─────────────────────────────────────────
st.header("3. Paper-run")
results_dir = Path("results")
if results_dir.exists():
    eq_files = sorted(results_dir.glob("paper_equity_*.csv"))
    if eq_files:
        st.subheader("Equity")
        for f in eq_files:
            eq = pd.read_csv(f, parse_dates=["ts"]).set_index("ts")
            fig = go.Figure(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name=f.stem))
            st.plotly_chart(fig, use_container_width=True)
        tr_files = sorted(results_dir.glob("paper_trades_*.csv"))
        for f in tr_files:
            tr = pd.read_csv(f)
            st.subheader(f"Угоди: {f.stem}")
            st.dataframe(tr.tail(20), use_container_width=True)
    else:
        st.info("Немає paper-run результатів — запустіть: python -m scalper_hft.cli paper-run ...")
else:
    st.info("Немає results/ — запустіть paper-run спершу")
