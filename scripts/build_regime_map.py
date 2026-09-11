#!/usr/bin/env python3
"""Побудувати RegimeStrategyMap з iter7 OOS parquet або fallback JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scalper_hft.validation.regime_map import (
    build_regime_strategy_map,
    compute_regime_perf_matrix,
)


def _load_iter7_oos(oos_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    files = list(oos_dir.glob("*.parquet"))
    if not files:
        return None
    frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp)
        if "ret" not in df.columns or "regime" not in df.columns:
            continue
        variant = fp.stem
        tmp = df[["ret", "regime"]].copy()
        tmp.columns = ["ret", "label"]
        tmp["strategy"] = variant
        frames.append(tmp.reset_index())
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(index="ts", columns="strategy", values="ret", aggfunc="first")
    regime = long.drop_duplicates("ts").set_index("ts")[["label"]]
    return pivot, regime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-dir", default="results/iter7_oos")
    parser.add_argument("--out", default="data/regime_strategy_map.json")
    parser.add_argument("--fallback", default="data/regime_strategy_map.json")
    args = parser.parse_args()

    oos_dir = Path(args.oos_dir)
    out = Path(args.out)
    loaded = _load_iter7_oos(oos_dir)
    if loaded is None:
        print(f"iter7 OOS не знайдено в {oos_dir}, лишаю fallback {args.fallback}")
        return
    returns_df, regime_df = loaded
    matrix = compute_regime_perf_matrix(returns_df, regime_df)
    rmap = build_regime_strategy_map(matrix)
    rmap.to_json(out)
    print(f"Записано {out} з {len(matrix.regimes())} режимів")


if __name__ == "__main__":
    main()
