"""RegimeSupervisor — мета-стратегія з детектором режиму та онлайн-адаптацією.

Архітектура (Narang «Inside the Black Box», гл. 3/6/8):
    1. RegimeDetector: визначає поточний ринковий режим (HMM + rule-based).
    2. Static prior:   taxonomy.preferred_regimes → м'яка базова вага [0.25, 1.0].
    3. Dynamic adapt:  ContextualHedgeBlend (або Exp3Bandit) → ваги адаптуються
                       онлайн per-regime без перетренування.
    4. Сигнал:         зважена сума сигналів суб-стратегій, кліпнута до [-1, 1].

Три режими blend_mode:
    "regime_soft"       — лише static prior із taxonomy (найпростіший, прозорий).
    "contextual_hedge"  — ContextualHedgeBlend (рекомендований; онлайн, per-regime).
    "exp3"              — Exp3Bandit (вибирає ONE best strategy per bar, не зважує).

Без lookahead:
    - HMM: filtered_proba (forward-only), навчання лише на перших hmm_fit_bars.
    - Hedge/Exp3: оновлення після спостереження бару t, вага діє на t+1.
    - Rule-based: EMA/vol — каузальні за конструкцією.
    - Сигнал суб-стратегій: generate_signals() кожної — також без lookahead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.regime_detector import RegimeDetector
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.taxonomy import DEFAULT_UNFAVORABLE_WEIGHT

logger = logging.getLogger(__name__)

# Структурні режими для ContextualHedgeBlend
_STRUCTURE_REGIMES = ["range", "trend_up", "trend_down"]


class RegimeSupervisor(Strategy):
    """Supervisor: детектор режиму + онлайн-зважування сигналів стратегій.

    Параметри (через **params):
        strategies:         рядок через кому, наприклад "mean_reversion,supertrend,hmm_reversion".
                            Підтримує формат name:param1=val1:param2=val2.
        blend_mode:         "regime_soft" | "contextual_hedge" | "exp3" (default: "contextual_hedge").
        n_hmm_states:       кількість HMM станів (default 3).
        hmm_fit_bars:       скільки перших барів для навчання HMM (default 2000).
        unfavorable_weight: вага в несприятливому режимі для regime_soft (default 0.25).
        hedge_eta:          параметр швидкості навчання Hedge (default: адаптивний).
        exp3_gamma:         exploration rate для Exp3 (default 0.05).
    """

    name = "regime_supervisor"
    family = "meta"
    preferred_regimes: frozenset[str] = frozenset()
    param_space: dict[str, tuple[float, float, float]] = {
        "hmm_fit_bars": (500.0, 5000.0, 500.0),
        "unfavorable_weight": (0.1, 0.5, 0.05),
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        # Список суб-стратегій
        strat_str = str(self.get("strategies", "mean_reversion,supertrend,hmm_reversion"))
        self._strat_names: list[str] = []
        self._strats: list[Strategy] = []
        self._load_strategies(strat_str)

        self.blend_mode: str = str(self.get("blend_mode", "contextual_hedge"))
        self.needs_trades = any(s.needs_trades for s in self._strats)
        self.needs_funding = any(s.needs_funding for s in self._strats)

        # RegimeDetector
        self._detector = RegimeDetector(
            n_hmm_states=int(self.get("n_hmm_states", 3)),
            hmm_fit_bars=int(self.get("hmm_fit_bars", 2000)),
        )

        # Онлайн-блендер (ініціалізується при першому виклику)
        self._contextual_hedge: Any = None  # ContextualHedgeBlend
        self._exp3_bandit: Any = None  # Exp3Bandit
        self._prev_sigs: np.ndarray | None = None  # сигнали минулого бару для hedge update


    @classmethod
    def from_config(cls, config_path: str) -> RegimeSupervisor:
        from scalper_hft.live.supervisor_config import SupervisorConfig
        from scalper_hft.strategies import get_strategy
        
        cfg = SupervisorConfig.from_yaml(config_path)
        
        # Створюємо базовий Supervisor
        sup = cls(blend_mode="contextual_hedge")
        sup._strat_names = []
        sup._strats = []
        
        # Ініціалізуємо суб-стратегії
        for s in cfg.strategies:
            strat_cls = get_strategy(s.family, **s.params)  # базове ім'я - це family
            if not strat_cls:
                logger.warning(f"Стратегію {s.family} ({s.id}) не знайдено, пропускаємо.")
                continue
                
            inst = strat_cls
            # Перевизначаємо preferred_regimes з конфігу
            if s.preferred_regimes:
                inst.preferred_regimes = frozenset(s.preferred_regimes)
                
            sup._strat_names.append(s.id)
            sup._strats.append(inst)
            
        sup.needs_trades = any(s.needs_trades for s in sup._strats)
        sup.needs_funding = any(s.needs_funding for s in sup._strats)
        
        return sup


    # ────────────────────────────────────────────────────────────────────────
    # Batch (бектест)
    # ────────────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Генерує сигнали на основі детекції режиму та зважування суб-стратегій.

        Логіка (без lookahead):
            1. Детектуємо режим для всього ряду (RegimeDetector.detect).
            2. Отримуємо сигнали від кожної суб-стратегії.
            3. Зважуємо за обраним blend_mode:
               - regime_soft: static prior із taxonomy.
               - contextual_hedge: HedgeBlend per-regime (барне оновлення).
               - exp3: Exp3 вибирає одну стратегію per-bar.
        """
        if not self._strats:
            return pd.Series(0.0, index=df.index)

        # 1. Сигнали суб-стратегій
        sig_df = self._collect_signals(df, trades, funding)

        # 2. Детекція режиму (batch, каузальна)
        regime_df = self._detector.detect(df["close"])

        # 3. Зважування
        if self.blend_mode == "regime_soft":
            return self._blend_regime_soft(sig_df, regime_df)
        elif self.blend_mode == "exp3":
            return self._blend_exp3(sig_df, df["close"])
        else:  # contextual_hedge (default)
            return self._blend_contextual_hedge(sig_df, regime_df, df["close"])

    # ────────────────────────────────────────────────────────────────────────
    # Blend implementations
    # ────────────────────────────────────────────────────────────────────────

    def _blend_regime_soft(self, sig_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.Series:
        """Static soft-weights: taxonomy.preferred_regimes → вага в [unfavorable, 1].

        Найпростіший і найпрозоріший режим. Ваги не змінюються після навчання.
        """
        from scalper_hft.strategies.taxonomy import regime_capital_weight

        unfavorable = float(self.get("unfavorable_weight", DEFAULT_UNFAVORABLE_WEIGHT))
        result = pd.Series(0.0, index=sig_df.index)

        for i, (idx, row) in enumerate(sig_df.iterrows()):
            structure = str(regime_df.loc[idx, "structure"]) if idx in regime_df.index else "range"
            vol = str(regime_df.loc[idx, "vol"]) if idx in regime_df.index else "normal"
            weights = np.array([
                regime_capital_weight(
                    frozenset(s.preferred_regimes),
                    structure,
                    vol,
                    unfavorable=unfavorable,
                )
                for s in self._strats
            ])
            total_w = weights.sum()
            if total_w > 0:
                result.iloc[i] = float(np.dot(row.values, weights / total_w))

        return result.clip(-1.0, 1.0)

    def _blend_contextual_hedge(
        self,
        sig_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        close: pd.Series,
    ) -> pd.Series:
        """ContextualHedgeBlend: per-regime онлайн ваги (рекомендований режим).

        Алгоритм (без lookahead):
            На барі t:
                - оновлюємо ваги за прибутковістю бару t-1 (вже спостережена).
                - вираховуємо комбінований сигнал вагами бару t.
        """
        from scalper_hft.strategies.blend import ContextualHedgeBlend
        from scalper_hft.strategies.taxonomy import regime_capital_weight

        n = len(self._strats)
        unfavorable = float(self.get("unfavorable_weight", DEFAULT_UNFAVORABLE_WEIGHT))
        eta = self.get("hedge_eta", None)
        eta_val = float(eta) if eta is not None else None

        if self._contextual_hedge is None:
            self._contextual_hedge = ContextualHedgeBlend(
                n_experts=n,
                regimes=_STRUCTURE_REGIMES,
                eta=eta_val,
            )
        blend = self._contextual_hedge

        ret = close.pct_change().fillna(0.0)
        sigs = sig_df.values  # (T, N)
        regimes_arr = regime_df.reindex(sig_df.index)["structure"].fillna("range").values
        rets = ret.reindex(sig_df.index).values
        result_vals = np.zeros(len(sig_df))

        for t in range(len(sig_df)):
            regime = str(regimes_arr[t])

            # Оновлення ваг за ПОПЕРЕДНІМ баром (без lookahead)
            if t > 0:
                prev_regime = str(regimes_arr[t - 1])
                strat_rets = sigs[t - 1] * float(rets[t])  # sig_{t-1} * ret_t
                blend.step(strat_rets, prev_regime)

            # Базові ваги з taxonomy (static prior) — модифікуємо Hedge-ваги
            prior = np.array([
                regime_capital_weight(
                    frozenset(s.preferred_regimes),
                    regime,
                    str(regime_df.iloc[t].get("vol", "normal") if t < len(regime_df) else "normal"),
                    unfavorable=unfavorable,
                )
                for s in self._strats
            ])

            # Фінальні ваги: Hedge-ваги × static prior
            hedge_w = blend.weights(regime)
            final_w = hedge_w * prior
            total_w = final_w.sum()
            if total_w > 0:
                final_w = final_w / total_w

            result_vals[t] = float(np.dot(sigs[t], final_w))

        return pd.Series(result_vals, index=sig_df.index).clip(-1.0, 1.0)

    def _blend_exp3(self, sig_df: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Exp3 Bandit: вибирає одну найкращу стратегію per-bar (exploration)."""
        from scalper_hft.strategies.bandit import exp3_select_signals

        ret = close.pct_change().fillna(0.0)
        returns_df = sig_df.shift(1).fillna(0.0).mul(ret, axis=0)
        gamma = float(self.get("exp3_gamma", 0.05))
        return exp3_select_signals(sig_df, returns_df, gamma=gamma)

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _load_strategies(self, strat_str: str) -> None:
        """Парсинг і завантаження суб-стратегій."""
        from scalper_hft.strategies import get_strategy

        for s in strat_str.split(","):
            s = s.strip()
            if not s:
                continue
            parts = s.split(":")
            name = parts[0].strip()
            p: dict[str, Any] = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        p[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        p[k] = v
            try:
                strategy = get_strategy(name, **p)
                self._strats.append(strategy)
                self._strat_names.append(name)
            except KeyError as exc:
                logger.warning("RegimeSupervisor: невідома стратегія %s: %s", name, exc)

    def _collect_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None,
        funding: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Отримати сигнали від кожної суб-стратегії."""
        import inspect

        sigs: list[pd.Series] = []
        for strat in self._strats:
            try:
                sig_params = inspect.signature(strat.generate_signals).parameters
                kwargs: dict[str, Any] = {}
                if "trades" in sig_params:
                    kwargs["trades"] = trades
                if "funding" in sig_params:
                    kwargs["funding"] = funding
                sig = strat.generate_signals(df, **kwargs).fillna(0.0)
            except Exception as exc:
                logger.warning(
                    "RegimeSupervisor: суб-стратегія %s помилка: %s",
                    strat.name,
                    exc,
                )
                sig = pd.Series(0.0, index=df.index)
            sigs.append(sig)

        cols = [s.name for s in self._strats]
        return pd.concat(sigs, axis=1).reindex(df.index).fillna(0.0).set_axis(cols, axis=1)

    def regime_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Зведення по режимах: частота, avg signal per strategy.

        Корисно для аналізу: яка стратегія домінує в якому режимі.
        """
        regime_df = self._detector.detect(df["close"])
        sig_df = self._collect_signals(df, None, None)
        combined = pd.concat([regime_df[["structure", "vol", "label"]], sig_df], axis=1)
        return combined.groupby("label")[sig_df.columns.tolist()].agg(["mean", "std", "count"])

    @property
    def sub_strategies(self) -> list[str]:
        """Імена суб-стратегій."""
        return list(self._strat_names)


__all__ = ["RegimeSupervisor"]
