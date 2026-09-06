"""Порівняльні графіки Multi-Exchange: баланси, PnL по біржах, активні боти."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scalper_hft.live.store import PaperStore

st.title("Multi-Exchange Порівняння")
st.caption("Аналіз балансів та PnL розрізі різних бірж/акаунтів (live/paper).")

_PAPER_DB = Path("results") / "paper_pairs.sqlite"

if not _PAPER_DB.exists():
    st.info("База даних paper_pairs.sqlite не знайдена. Спочатку запустіть paper/live торгівлю.")
    st.stop()

# --- Дані ---
@st.cache_data(ttl="10s")
def load_all_accounts() -> pd.DataFrame:
    try:
        store = PaperStore(_PAPER_DB)
        conn = store._conn
        df = pd.read_sql_query("SELECT ts, exchange, mode, balance, unrealized_pnl FROM accounts ORDER BY ts", conn)
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception as exc:
        st.error(f"Помилка завантаження accounts: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl="10s")
def load_bots() -> pd.DataFrame:
    try:
        store = PaperStore(_PAPER_DB)
        return store.all_bots()
    except Exception as exc:
        st.error(f"Помилка завантаження bots: {exc}")
        return pd.DataFrame()

accounts_df = load_all_accounts()
bots_df = load_bots()

if accounts_df.empty:
    st.warning("Немає записів у таблиці accounts.")
    st.stop()

# --- Фільтри ---
col1, col2 = st.columns(2)
with col1:
    exchanges = accounts_df["exchange"].unique().tolist()
    selected_exchanges = st.multiselect("Біржі", options=exchanges, default=exchanges)
with col2:
    modes = accounts_df["mode"].unique().tolist()
    selected_modes = st.multiselect("Режим", options=modes, default=modes)

filtered_accounts = accounts_df[
    (accounts_df["exchange"].isin(selected_exchanges)) &
    (accounts_df["mode"].isin(selected_modes))
]

if filtered_accounts.empty:
    st.info("Немає даних для вибраних фільтрів.")
    st.stop()

# --- Порівняльний графік балансів ---
st.subheader("Динаміка балансів")

fig_bal = go.Figure()
for ex_mode, group in filtered_accounts.groupby(["exchange", "mode"]):
    ex, mode = ex_mode
    label = f"{ex} ({mode})"
    fig_bal.add_trace(go.Scatter(
        x=group["ts"], y=group["balance"],
        mode="lines",
        name=label,
        line=dict(width=2),
    ))

fig_bal.update_layout(
    xaxis_title="Час",
    yaxis_title="Баланс (USDT)",
    hovermode="x unified",
    template="plotly_dark",
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_bal, use_container_width=True)

# --- Активні боти ---
st.subheader("Реєстр ботів (Multi-Exchange)")
if bots_df.empty:
    st.write("Немає активних ботів у реєстрі.")
else:
    filtered_bots = bots_df[
        (bots_df["exchange"].isin(selected_exchanges)) &
        (bots_df["mode"].isin(selected_modes))
    ]
    st.dataframe(filtered_bots, use_container_width=True, hide_index=True)
