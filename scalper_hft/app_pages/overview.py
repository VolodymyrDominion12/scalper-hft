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

st.header("1. Дані (кеш)", divider=False)
rows = []
total_klines = 0
total_trades = 0
for sym in SYMBOLS:
    k = load_klines(klines_path(data_dir, sym, "1m"))
    t = load_trades(data_dir / f"{sym}_aggTrades.parquet")
    funding_df = load_funding(data_dir / f"{sym}_funding.parquet")
    bt = data_dir / f"{sym}_bookTicker.parquet"
    bt_df = pd.read_parquet(bt) if bt.exists() else None
    
    k_len = len(k) if k is not None else 0
    t_len = len(t) if t is not None else 0
    total_klines += k_len
    total_trades += t_len
    
    rows.append(
        {
            "symbol": sym,
            "klines": k_len,
            "aggTrades": t_len,
            "funding": len(funding_df) if funding_df is not None else 0,
            "bookTicker": len(bt_df) if bt_df is not None else 0,
        }
    )

with st.container(horizontal=True):
    st.metric("Символів", len(SYMBOLS), border=True)
    st.metric("Всього свічок", f"{total_klines:,}", border=True)
    st.metric("Всього трейдів", f"{total_trades:,}", border=True)

df_rows = pd.DataFrame(rows)
st.dataframe(
    df_rows,
    width="stretch",
    hide_index=True,
    column_config={
        "symbol": st.column_config.TextColumn("Символ"),
        "klines": st.column_config.ProgressColumn("Klines (1m)", format="%d", min_value=0, max_value=int(df_rows["klines"].max() if not df_rows.empty else 100000)),
        "aggTrades": st.column_config.NumberColumn("aggTrades", format="%d"),
        "funding": st.column_config.NumberColumn("Funding", format="%d"),
        "bookTicker": st.column_config.NumberColumn("Book Ticker", format="%d"),
    }
)

st.header("2. Paper pairs (SQLite)", divider=False)
db = Path("results") / "paper_pairs.sqlite"
if db.exists():
    store = PaperStore(db)
    stats = store.fill_stats()
    
    st.subheader("Статистика ордерів (post-only)")
    with st.container(horizontal=True):
        st.metric("Filled", stats.get("filled", 0), border=True)
        st.metric("Unfilled", stats.get("unfilled", 0), border=True)
        st.metric("Pending", stats.get("pending", 0), border=True)
        
    eq = store.recent_equity()
    if not eq.empty:
        eq["ts"] = pd.to_datetime(eq["ts"])
        st.subheader("Paper equity by pair")
        
        # Використовуємо st.line_chart для чистішого вигляду
        pivot_eq = eq.pivot(index="ts", columns="pair", values="equity")
        st.line_chart(pivot_eq, height=320)
        
    orders = store.recent_orders()
    if not orders.empty:
        with st.expander("Останні ордери", icon=":material/list:"):
            st.dataframe(orders.head(40), width="stretch", hide_index=True)
    store.close()
else:
    st.caption("Немає `results/paper_pairs.sqlite` — запустіть: `uv run python -m scalper_hft.cli paper-replay-pairs`")

st.header("3. Paper-run CSV", divider=False)
results_dir = Path("results")
if results_dir.exists():
    eq_files = sorted(results_dir.glob("paper_equity_*.csv"))
    if eq_files:
        combined_eq = {}
        for eq_file in eq_files:
            eq = pd.read_csv(eq_file, parse_dates=["ts"]).set_index("ts")
            combined_eq[eq_file.stem] = eq["equity"]
            
        st.line_chart(pd.DataFrame(combined_eq), height=320)
    else:
        st.caption("Немає CSV paper-run.")
else:
    st.caption("Немає директорії `results/`")
