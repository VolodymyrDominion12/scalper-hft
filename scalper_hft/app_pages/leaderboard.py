"""Лідерборд: рейтинг комірок і портфелів з audit/sweep/iter-циклів."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scalper_hft.research.leaderboard import (
    _TIER_LABELS,
    leaderboard_dataframe,
    render_leaderboard_markdown,
)

st.title("Лідерборд")
st.caption("Агрегація audit, sweep і портфельних тестів. Оновлення: `uv run python -m scalper_hft.cli leaderboard`")


@st.cache_data(ttl=30, max_entries=4)
def _cached_frame(results_dir: str) -> pd.DataFrame:
    return leaderboard_dataframe(Path(results_dir))


results_dir = Path("results")
df = _cached_frame(str(results_dir))

if df.empty:
    st.info("Немає даних. Запустіть дослідницький цикл або `cli leaderboard`.")
    st.stop()

tier_counts = df["tier"].value_counts()
with st.container(horizontal=True):
    for tier, label in _TIER_LABELS.items():
        n = int(tier_counts.get(tier, 0))
        st.metric(label.split(" ", 1)[-1], n, border=True)

hide_rejected = st.toggle("Сховати rejected", value=True)
filters = st.container(horizontal=True)
with filters:
    intervals = ["Усі"] + sorted(df["interval"].dropna().astype(str).unique().tolist())
    interval_pick = st.selectbox("Таймфрейм", intervals, index=0)
    strategies = ["Усі"] + sorted(df["strategy"].dropna().unique().tolist())
    strategy_pick = st.selectbox("Стратегія", strategies)

view = df.copy()
if hide_rejected:
    view = view[view["tier"] != "rejected"]
if interval_pick != "Усі":
    view = view[view["interval"].astype(str) == interval_pick]
if strategy_pick != "Усі":
    view = view[view["strategy"] == strategy_pick]

paper_rows = df[df["tier"].isin(["validated_pairs", "paper", "monitoring"])].head(6)
if not paper_rows.empty:
    st.subheader("Рекомендації для paper")
    for rec in paper_rows.itertuples(index=False):
        label = _TIER_LABELS.get(str(rec.tier), rec.tier)
        notes = str(rec.notes or "")[:120]
        st.markdown(f"- **{rec.strategy}** `{rec.symbol}` {rec.interval} — {label}. {notes}")

port = view.dropna(subset=["port_sharpe"]) if "port_sharpe" in view.columns else view.iloc[0:0]
if not port.empty:
    st.subheader("Портфельний Sharpe")
    chart_df = (
        port.assign(label=port["symbol"].astype(str) + " " + port["interval"].astype(str))
        .set_index("label")[["port_sharpe"]]
        .sort_values("port_sharpe", ascending=False)
        .head(12)
    )
    st.bar_chart(chart_df, y="port_sharpe")
    st.caption("Source: results/iter*/variants.csv · річний Sharpe рівноважного портфеля, maker.")

st.subheader("Таблиця")
show_cols = [
    "rank",
    "tier",
    "strategy",
    "symbol",
    "interval",
    "avg_oos_sharpe",
    "oos_pos_frac",
    "port_sharpe",
    "t_newey_west",
    "verdict",
    "notes",
]
display = view[[c for c in show_cols if c in view.columns]].copy()
if "avg_oos_sharpe" in display.columns:
    display["avg_oos_sharpe"] = display["avg_oos_sharpe"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
if "oos_pos_frac" in display.columns:
    display["oos_pos_frac"] = display["oos_pos_frac"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
if "port_sharpe" in display.columns:
    display["port_sharpe"] = display["port_sharpe"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
if "t_newey_west" in display.columns:
    display["t_newey_west"] = display["t_newey_west"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
display["tier"] = display["tier"].map(lambda t: _TIER_LABELS.get(t, t))
st.dataframe(display, width="stretch", hide_index=True)

with st.expander("Повний markdown-звіт"):
    st.markdown(render_leaderboard_markdown(results_dir, top_n=40))

if st.button("Оновити лідерборд з диска", type="primary"):
    _cached_frame.clear()
    out = Path("docs/reports/LEADERBOARD.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_leaderboard_markdown(results_dir), encoding="utf-8")
    leaderboard_dataframe(results_dir).to_csv(results_dir / "leaderboard.csv", index=False)
    st.success(f"Збережено {out}")
