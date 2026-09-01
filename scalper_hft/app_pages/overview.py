"""Сторінка «Моніторинг»: кеш даних та paper-результати."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scalper_hft.app_pages._common import SYMBOLS
from scalper_hft.config import get_settings
from scalper_hft.data.storage import klines_path, load_funding, load_klines, load_trades
from scalper_hft.live.store import PaperStore

settings = get_settings()
data_dir = settings.data_dir_abs

st.title("Моніторинг")

st.header("1. Дані (кеш)")
rows = []
for sym in SYMBOLS:
    k = load_klines(klines_path(data_dir, sym, "1m"))
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
st.dataframe(pd.DataFrame(rows), width="stretch")

st.header("2. Paper pairs (SQLite)")
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
        st.plotly_chart(fig, width="stretch")
    orders = store.recent_orders()
    if not orders.empty:
        st.subheader("Останні ордери")
        st.dataframe(orders.head(40), width="stretch")
    store.close()
else:
    st.info("Немає results/paper_pairs.sqlite — запустіть: python -m scalper_hft.cli paper-replay-pairs")

st.header("3. Paper-run CSV")
results_dir = Path("results")
if results_dir.exists():
    eq_files = sorted(results_dir.glob("paper_equity_*.csv"))
    if eq_files:
        for f in eq_files:
            eq = pd.read_csv(f, parse_dates=["ts"]).set_index("ts")
            fig = go.Figure(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name=f.stem))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Немає CSV paper-run.")
else:
    st.info("Немає results/")
