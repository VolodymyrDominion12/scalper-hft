"""Validated regime→strategy map (Phase 2B).

Замінює статичні гіпотетичні теги ``preferred_regimes`` емпіричною таблицею
«у якому режимі яка стратегія справді заробляла OOS». Будується з walk-forward
результатів: для кожної комірки (regime × strategy) рахується OOS Sharpe, і
стратегії з Sharpe нижче порогу значущості hard-off'аються (вага 0).

Політика hard vs soft (Narang: regime-conditional allocation):
    - stable regime (low/normal vol, чіткий structure): best_prior — одна
      найкраща стратегія (менше churn, чистіший сигнал);
    - transition / mixed regime: soft blend — ваги ∝ додатній OOS Sharpe;
    - high-vol regime: flat — усі ваги 0 (не торгуємо у буремному ринку).

Без lookahead: матриця будуеться лише на OOS-дохідностях walk-forward і
застосовується на майбутніх барах (окремий fit-період → окремий apply-період).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RegimePerfMatrix:
    """OOS-метрики per (regime × strategy): Sharpe, n_bars, mean_ret."""

    # matrix[regime][strategy] = {"sharpe": float, "n_bars": int, "mean_ret": float}
    data: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def sharpe(self, regime: str, strategy: str) -> float:
        cell = self.data.get(regime, {}).get(strategy)
        return float(cell["sharpe"]) if cell else 0.0

    def regimes(self) -> list[str]:
        return sorted(self.data)

    def strategies(self) -> list[str]:
        s: set[str] = set()
        for r in self.data.values():
            s.update(r)
        return sorted(s)

    def to_dict(self) -> dict[str, dict[str, dict[str, float]]]:
        return {r: dict(v) for r, v in self.data.items()}

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> RegimePerfMatrix:
        data = json.loads(Path(path).read_text())
        m = cls()
        m.data = {r: dict(v) for r, v in data.items()}
        return m


def compute_regime_perf_matrix(
    returns_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    min_bars: int = 30,
    periods_per_year: int = 8760,
) -> RegimePerfMatrix:
    """Побудувати OOS-матрицю (regime × strategy) → Sharpe.

    Args:
        returns_df: (T × N) пербарні дохідності стратегій (OOS, після комісій).
            Індекс — час, колонки — імена стратегій.
        regime_df: DataFrame з колонкою ``label`` (structure|vol), індексований
            як returns_df.
        min_bars: мінімум барів у комірці, щоб вважати метрику значущою.
        periods_per_year: для ануаліза Sharpe (8760 для 1h).

    Returns:
        RegimePerfMatrix з Sharpe / n_bars / mean_ret per (regime, strategy).
    """
    ret = returns_df.reindex(regime_df.index)
    labels = regime_df["label"].reindex(ret.index).fillna("range|normal")
    matrix = RegimePerfMatrix()
    ann = float(periods_per_year) ** 0.5
    for strat in ret.columns:
        for regime in sorted(labels.unique()):
            mask = labels == regime
            r = ret.loc[mask, strat].dropna()
            if len(r) < min_bars:
                continue
            mu = float(r.mean())
            sd = float(r.std(ddof=0))
            sharpe = (mu / sd * ann) if sd > 1e-12 else 0.0
            matrix.data.setdefault(str(regime), {})[str(strat)] = {
                "sharpe": sharpe,
                "n_bars": float(len(r)),
                "mean_ret": mu,
            }
    return matrix


@dataclass(slots=True)
class RegimeStrategyMap:
    """Валідована таблиця regime → {strategy: вага} з hard-off політикою.

    Attributes:
        weights: regime → {strategy: вага} (сума ≤ 1; 0 = hard-off).
        policy: regime → "best_prior" | "soft" | "flat".
    """

    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    policy: dict[str, str] = field(default_factory=dict)

    def active_strategies(self, regime: str) -> list[str]:
        w = self.weights.get(regime, {})
        return [s for s, v in w.items() if v > 0.0]

    def weights_for(self, regime: str) -> dict[str, float]:
        return dict(self.weights.get(regime, {}))

    def is_flat(self, regime: str) -> bool:
        return self.policy.get(regime) == "flat" or not self.active_strategies(regime)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"weights": self.weights, "policy": self.policy}, indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> RegimeStrategyMap:
        d = json.loads(Path(path).read_text())
        return cls(weights=d.get("weights", {}), policy=d.get("policy", {}))


def build_regime_strategy_map(
    matrix: RegimePerfMatrix,
    *,
    hard_off_sharpe: float = 0.0,
    high_vol_flat: bool = True,
    best_prior_min_gap: float = 0.1,
) -> RegimeStrategyMap:
    """Конвертувати OOS-матрицю у вагову таблицю з hard/soft політикою.

    Політика:
        - high-vol regime (label закінчується на |high) → flat (усі ваги 0),
          якщо ``high_vol_flat``.
        - stable regime (low/normal vol) з однозначним лідером (відносний відрив ≥
          ``best_prior_min_gap`` над другим) → best_prior: одна стратегія, вага 1.
        - інакше → soft: ваги ∝ max(sharpe - hard_off, 0), нормовані; якщо всі
          нижче порогу → flat.

    Args:
        matrix: OOS-матриця з compute_regime_perf_matrix.
        hard_off_sharpe: стратегії з Sharpe < цього порогу hard-off (вага 0).
        high_vol_flat: flat у high-vol режимах.
        best_prior_min_gap: відносний відрив лідера (частка top-Sharpe) для
            best_prior політики (default 0.1 = 10%).
    """
    rmap = RegimeStrategyMap()
    for regime in matrix.regimes():
        cells = matrix.data[regime]
        is_high_vol = regime.endswith("|high")
        if is_high_vol and high_vol_flat:
            rmap.weights[regime] = {s: 0.0 for s in cells}
            rmap.policy[regime] = "flat"
            continue
        # додатні Sharpe після hard-off
        positives = {s: float(c["sharpe"]) for s, c in cells.items() if c["sharpe"] >= hard_off_sharpe}
        if not positives:
            rmap.weights[regime] = {s: 0.0 for s in cells}
            rmap.policy[regime] = "flat"
            continue
        # best_prior якщо чіткий лідер (відносний відрив ≥ best_prior_min_gap)
        ranked = sorted(positives.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[0][1]
        if len(ranked) == 1 or (top > 0 and (ranked[0][1] - ranked[1][1]) >= best_prior_min_gap * top):
            best = ranked[0][0]
            rmap.weights[regime] = {s: (1.0 if s == best else 0.0) for s in cells}
            rmap.policy[regime] = "best_prior"
        else:
            vals = np.array([v for _, v in ranked], dtype=float)
            w = vals / vals.sum() if vals.sum() > 0 else vals
            rmap.weights[regime] = {s: 0.0 for s in cells}
            for (s, _), wi in zip(ranked, w, strict=False):
                rmap.weights[regime][s] = float(wi)
            rmap.policy[regime] = "soft"
    return rmap


def preferred_regimes_from_matrix(
    matrix: RegimePerfMatrix,
    strategy: str,
    *,
    min_sharpe: float = 0.0,
) -> frozenset[str]:
    """OOS-теги PreferredRegime зі комірок, де стратегія мала Sharpe ≥ порогу.

    Ключі матриці — ``structure|vol`` (наприклад ``range|normal``). Повертає
    об'єднання structure/vol міток з додатнім OOS. Порожній frozenset =
    немає підтвердженої комірки (не записувати в клас як «усі режими»).
    """
    from scalper_hft.features.regimes import STRUCTURE_LABELS, VOL_LABELS

    allowed = STRUCTURE_LABELS | VOL_LABELS
    tags: set[str] = set()
    for regime, cells in matrix.data.items():
        cell = cells.get(strategy)
        if not cell or float(cell.get("sharpe", 0.0)) < min_sharpe:
            continue
        for part in str(regime).split("|"):
            label = part.strip()
            if label in allowed:
                tags.add(label)
    return frozenset(tags)


def preferred_regimes_book_from_matrix(
    matrix: RegimePerfMatrix,
    *,
    min_sharpe: float = 0.0,
) -> dict[str, frozenset[str]]:
    """preferred_regimes для кожної стратегії в OOS-матриці."""
    return {s: preferred_regimes_from_matrix(matrix, s, min_sharpe=min_sharpe) for s in matrix.strategies()}


def apply_oos_preferred_regimes(
    strategies: list,
    matrix: RegimePerfMatrix,
    *,
    min_sharpe: float = 0.0,
) -> None:
    """Записати OOS-теги на інстанси. Порожній результат не затирає гіпотезу класу."""
    for strat in strategies:
        name = str(getattr(strat, "name", ""))
        if not name:
            continue
        tags = preferred_regimes_from_matrix(matrix, name, min_sharpe=min_sharpe)
        if tags:
            strat.preferred_regimes = tags


def pool_regime_perf_matrix(
    per_symbol: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    min_bars: int = 30,
    periods_per_year: int = 8760,
) -> RegimePerfMatrix:
    """OOS-матриця, об'єднана по символах (один пул замість per-symbol).

    Селектор «режим → стратегія» має бути робастним ПО ІНСТРУМЕНТАХ, а не
    підігнаним під один символ: карти, навчені на одному символі, не
    переносяться (iter7). Тому дохідності та режимні мітки всіх символів
    конкатенуються в один ряд (порядок символів зберігається), і матриця
    рахується один раз — Sharpe кожної комірки вже усереднений по пулу.

    Args:
        per_symbol: {symbol: (returns_df (T×N), regime_df з колонкою ``label``)}.
            Індекси returns_df і regime_df мають збігатися в межах символу.
        min_bars: мінімум барів у комірці (regime × strategy) для значущості.
        periods_per_year: ануалізація Sharpe (8760 для 1h).

    Returns:
        RegimePerfMatrix по пулу; порожня матриця, якщо немає валідних даних.
    """
    ret_parts: list[pd.DataFrame] = []
    label_parts: list[pd.Series] = []
    for symbol in sorted(per_symbol):
        ret, regime = per_symbol[symbol]
        if ret is None or ret.empty or regime is None or regime.empty:
            continue
        common = ret.index.intersection(regime.index)
        if len(common) < min_bars:
            continue
        r = ret.loc[common].apply(pd.to_numeric, errors="coerce")
        r.index = pd.RangeIndex(len(r))
        lab = regime.loc[common, "label"].astype(str)
        lab.index = r.index
        ret_parts.append(r)
        label_parts.append(lab.rename("label"))
    if not ret_parts:
        return RegimePerfMatrix()

    pooled_ret = pd.concat(ret_parts, axis=0, ignore_index=True)
    pooled_regime = pd.concat(label_parts, axis=0, ignore_index=True).to_frame()
    return compute_regime_perf_matrix(
        pooled_ret,
        pooled_regime,
        min_bars=min_bars,
        periods_per_year=periods_per_year,
    )


def save_regime_map(
    rmap: RegimeStrategyMap,
    path: str | Path,
    *,
    meta: dict | None = None,
) -> None:
    """Записати карту у JSON разом із метаданими fit-вікна (no-lookahead аудит).

    Структура: ``{"weights": ..., "policy": ..., "meta": {...}}``. ``from_json``
    читає лише weights/policy, тож метадані зворотно сумісні.
    """
    payload: dict = {"weights": rmap.weights, "policy": rmap.policy}
    if meta:
        payload["meta"] = meta
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_regime_map_meta(path: str | Path) -> dict:
    """Метадані карти (fit-вікно, символи, стратегії); {} якщо їх немає."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    meta = d.get("meta", {})
    return dict(meta) if isinstance(meta, dict) else {}


__all__ = [
    "RegimePerfMatrix",
    "RegimeStrategyMap",
    "compute_regime_perf_matrix",
    "pool_regime_perf_matrix",
    "build_regime_strategy_map",
    "save_regime_map",
    "load_regime_map_meta",
    "preferred_regimes_from_matrix",
    "preferred_regimes_book_from_matrix",
    "apply_oos_preferred_regimes",
]
