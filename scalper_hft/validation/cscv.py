"""Combinatorial Purged Cross-Validation (CSCV) та PBO (López de Prado).

Повна методологія оцінки імовірності перенавчання (Probability of Backtest
Overfitting, Bailey–Borwein–López de Prado–Zhu): замість одного train/test —
ВСІ комбінації блоків:
    - ряди прибутковостей розбиваються на N блоків;
    - для кожної комбінації N/2 блоків як "train" обчислюється, який варіант
      стратегії найкращий на train;
    - OOS-ранг IS-кращого варіанта серед усіх варіантів ЦЬОГО Ж спліту
      переводиться у logit λ_c;
    - PBO = частка комбінацій з λ_c < 0 (IS-кращий нижче медіани OOS).

Вхід: матриця прибутковостей (S варіантів × N спостережень) — кожен рядок —
результат однієї комбінації параметрів на одному часовому ряду.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.strategies.base import Strategy


@dataclass
class CscvResult:
    pbo: float
    n_combos: int
    n_variants: int
    n_blocks: int
    is_best_oos_sharpes: np.ndarray
    oos_sharpe_median: float
    logits: np.ndarray = field(default_factory=lambda: np.array([]))
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"CSCV: {self.n_combos} комбінацій, {self.n_variants} варіантів, {self.n_blocks} блоків\n"
            f"PBO = {self.pbo:.3f}  ({'⚠ перенавчання ймовірне' if self.pbo > 0.5 else '✅ PBO прийнятний'})\n"
            f"OOS Sharpe IS-кращих: медіана {np.median(self.is_best_oos_sharpes):.3f}, "
            f"частка < 0: {(self.is_best_oos_sharpes < 0).mean():.0%}"
        )


def _sharpe(x: np.ndarray) -> float:
    if x.std(ddof=0) == 0 or len(x) < 2:
        return 0.0
    return float(x.mean() / x.std(ddof=0))


def variant_returns(
    df: pd.DataFrame,
    strategy: Strategy,
    n_variants: int = 30,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    seed: int = 42,
    position_pct: float = 0.01,
) -> np.ndarray:
    """Матриця прибутковостей (n_variants × n_bars) для випадкових комбінацій
    параметрів з param_space стратегії (для CSCV/PBO)."""
    from scalper_hft.backtest.engine import run_backtest

    rng = np.random.default_rng(seed)
    space = strategy.param_space
    if not space:
        raise ValueError("Стратегія без param_space — неможливо згенерувати варіанти")

    rows: list[np.ndarray] = []
    for _ in range(n_variants):
        params: dict = {}
        for pname, (lo, hi, step) in space.items():
            if float(step) == int(step) and float(lo) == int(lo) and float(hi) == int(hi):
                params[pname] = int(rng.integers(int(lo), int(hi) + 1))
            else:
                params[pname] = float(rng.uniform(lo, hi))
        try:
            res = run_backtest(
                df,
                type(strategy)(**params),
                cost=cost,
                trades=trades,
                funding=funding,
                position_pct=position_pct,
            )
            rows.append(res.equity.pct_change().fillna(0.0).to_numpy(dtype=float))
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        raise RuntimeError("Жоден варіант не відпрацював")
    return np.vstack(rows)


def combinatorial_splits(n_blocks: int, n_train_blocks: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Усі комбінації блоків: (train_blocks, test_blocks).

    n_blocks: кількість блоків; n_train_blocks: скільки з них — "train"
    (за замовчуванням n_blocks // 2). Кількість комбінацій = C(n, k).
    """
    if n_train_blocks is None:
        n_train_blocks = n_blocks // 2
    blocks = np.arange(n_blocks)
    splits = []
    for train in combinations(blocks, n_train_blocks):
        train_arr = np.array(train, dtype=int)
        test_arr = np.setdiff1d(blocks, train_arr)
        splits.append((train_arr, test_arr))
    return splits


def pbo_cscv(
    strategy_returns: np.ndarray,
    n_blocks: int = 8,
    n_train_blocks: int | None = None,
    threshold: float = 0.0,
    max_combos: int = 200,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> CscvResult:
    """PBO за методом CSCV (Bailey, Borwein, López de Prado, Zhu).

    Для кожної комбінації train/test блоків:
      1. IS-кращий варіант n* = argmax Sharpe на train-блоках;
      2. ω_c — зростаючий ранг OOS Sharpe варіанта n* серед OOS Sharpe УСІХ
         варіантів цього ж спліту, нормований на (S+1): ω→1, коли IS-кращий
         також OOS-кращий;
      3. λ_c = logit(ω_c); λ_c < 0 ⟺ IS-кращий нижче медіани OOS.

    PBO = частка комбінацій з λ_c < 0.

    strategy_returns: (S × N) — кожен рядок — прибутковості варіанта стратегії
    на спільному часовому ряду (однакові дати по колонках).
    threshold: лише для details["frac_below_threshold"] (частка IS-кращих
        OOS Sharpe нижче порога); на PBO не впливає.
    max_combos: обмеження кількості комбінацій (при великих C(n,k)).
    purge_bars: скільки барів ПЕРЕД кожним test-блоком викинути з train
        (overlap лейблів/фіч); embargo_bars — скільки барів ПІСЛЯ test-блоку.
    """
    from scipy.stats import rankdata

    arr = np.asarray(strategy_returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("strategy_returns має бути (S × N)")
    s, n = arr.shape
    if s < 2:
        raise ValueError("Потрібно щонайменше 2 варіанти стратегії")

    splits = combinatorial_splits(n_blocks, n_train_blocks)
    if len(splits) > max_combos:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(splits), size=max_combos, replace=False)
        splits = [splits[i] for i in idx]

    block_size = n // n_blocks
    block_idx = np.arange(n) // block_size
    block_idx = np.minimum(block_idx, n_blocks - 1)

    is_best_oos: list[float] = []
    logits: list[float] = []
    for train_blocks, test_blocks in splits:
        train_mask = np.isin(block_idx, train_blocks)
        test_mask = np.isin(block_idx, test_blocks)
        if purge_bars > 0 or embargo_bars > 0:
            # межі train↔test: purge перед test-блоком, embargo після нього
            for b in test_blocks:
                b_start = int(b) * block_size
                b_end = min((int(b) + 1) * block_size, n)
                if purge_bars > 0:
                    train_mask[max(0, b_start - purge_bars) : b_start] = False
                if embargo_bars > 0:
                    train_mask[b_end : min(n, b_end + embargo_bars)] = False
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        # IS Sharpe по кожному варіанту
        is_sharpes = np.array([_sharpe(arr[i, train_mask]) for i in range(s)])
        if np.all(np.isnan(is_sharpes)):
            continue
        best_i = int(np.nanargmax(is_sharpes))
        # OOS Sharpe УСІХ варіантів → відносний ранг IS-кращого.
        # Зростаючий ранг (1 = найгірший OOS, S = найкращий): ω→1, коли
        # IS-кращий також OOS-кращий; ω<0.5 (λ<0) — нижче медіани OOS.
        oos_sharpes = np.array([_sharpe(arr[i, test_mask]) for i in range(s)])
        ranks = rankdata(oos_sharpes, method="average")
        omega = float(ranks[best_i]) / (s + 1.0)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(float(np.log(omega / (1.0 - omega))))
        is_best_oos.append(float(oos_sharpes[best_i]))

    if not is_best_oos:
        return CscvResult(
            pbo=1.0,
            n_combos=0,
            n_variants=s,
            n_blocks=n_blocks,
            is_best_oos_sharpes=np.array([]),
            oos_sharpe_median=0.0,
        )

    oos_arr = np.array(is_best_oos)
    logits_arr = np.array(logits)
    median = float(np.median(oos_arr))
    pbo = float(np.mean(logits_arr < 0.0))
    return CscvResult(
        pbo=pbo,
        n_combos=len(oos_arr),
        n_variants=s,
        n_blocks=n_blocks,
        is_best_oos_sharpes=oos_arr,
        oos_sharpe_median=median,
        logits=logits_arr,
        details={"frac_below_threshold": float(np.mean(oos_arr < threshold))},
    )
