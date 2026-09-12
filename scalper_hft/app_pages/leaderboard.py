"""Лідерборд: рейтинг комірок і портфелів з audit/sweep/iter10."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scalper_hft.research.leaderboard import (
    _TIER_LABELS,
    build_leaderboard,
    leaderboard_dataframe,
    render_leaderboard_markdown,
)

st.title("Лідерборд")
st.caption("Агрегація audit, sweep і портфельних тестів. Оновлення: `uv run python -m scalper_hft.cli leaderboard`")

results_dir = Path("results")
rows = build_leaderboard(results_dir)

if not rows:
    st.info("Немає даних. Запустіть дослідницький цикл або `cli leaderboard`.")
    st.stop()

df = leaderboard_dataframe(results_dir)

# KPI по тієрах
tier_counts = df["tier"].value_counts()
with st.container(horizontal=True):
    for tier, label in _TIER_LABELS.items():
        n = int(tier_counts.get(tier, 0))
        st.metric(
            label.replace("✅ ", "").replace("🟢 ", "").replace("🟡 ", "").replace("🔵 ", "").replace("⛔ ", ""),
            n,
            border=True,
        )

# Фільтри
tiers_avail = ["Усі"] + [label for label in _TIER_LABELS.values()]
tier_pick = st.selectbox("Тіер", tiers_avail, index=0)
strategy_pick = st.selectbox("Стратегія", ["Усі"] + sorted(df["strategy"].dropna().unique().tolist()))

view = df.copy()
if tier_pick != "Усі":
    tier_key = next(k for k, v in _TIER_LABELS.items() if v == tier_pick)
    view = view[view["tier"] == tier_key]
if strategy_pick != "Усі":
    view = view[view["strategy"] == strategy_pick]

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

st.dataframe(display, use_container_width=True, hide_index=True)

# Paper-ready блок
paper_rows = [r for r in rows if r.tier in ("validated_pairs", "paper", "monitoring")][:5]
if paper_rows:
    st.subheader("Рекомендації для paper")
    for r in paper_rows:
        st.markdown(
            f"- **{r.strategy}** `{r.symbol}` {r.interval} — {_TIER_LABELS.get(r.tier, r.tier)}: {r.notes[:80]}"
        )

with st.expander("Повний markdown-звіт"):
    st.markdown(render_leaderboard_markdown(results_dir, top_n=40))

if st.button("Оновити лідерборд з диска", type="primary"):
    out = Path("docs/reports/LEADERBOARD.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_leaderboard_markdown(results_dir), encoding="utf-8")
    df.to_csv(results_dir / "leaderboard.csv", index=False)
    st.success(f"Збережено {out}")
