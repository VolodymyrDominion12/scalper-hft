"""Micro-Price та оцінка позиції в черзі стакана (Narang Ch. 7, 15 & AFML Ch. 10.6, 19).

Micro-Price коригує Mid-Price на основі дисбалансу обсягів найкращих Bid і Ask:
    P_micro = Mid + ((V_bid - V_ask) / (V_bid + V_ask)) * (Spread / 2)

QueuePositionModel оцінює ймовірність заповнення лімітного ордера та ризик
adverse selection з урахуванням токсичності потоку (VPIN) та дисбалансу (OFI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def estimate_order_book_imbalance(
    bid_qty: float | pd.Series | np.ndarray,
    ask_qty: float | pd.Series | np.ndarray,
) -> float | pd.Series | np.ndarray:
    """Розрахунок відносного дисбалансу стакана L1/L2: (V_b - V_a) / (V_b + V_a).

    Значення у діапазоні [-1.0, 1.0]. Додатне -> тиск покупців.
    """
    total_qty = bid_qty + ask_qty
    if isinstance(total_qty, pd.Series):
        denom = total_qty.replace(0, np.nan)
        return ((bid_qty - ask_qty) / denom).fillna(0.0).clip(-1.0, 1.0)
    elif isinstance(total_qty, np.ndarray):
        denom = np.where(total_qty == 0, 1e-12, total_qty)
        return np.clip((bid_qty - ask_qty) / denom, -1.0, 1.0)
    else:
        if total_qty <= 0:
            return 0.0
        return float(np.clip((bid_qty - ask_qty) / total_qty, -1.0, 1.0))


def calculate_micro_price(
    bid_price: float | pd.Series | np.ndarray,
    ask_price: float | pd.Series | np.ndarray,
    bid_qty: float | pd.Series | np.ndarray,
    ask_qty: float | pd.Series | np.ndarray,
) -> float | pd.Series | np.ndarray:
    """Розрахунок справедливої мікро-ціни (Micro-Price).

    P_micro = P_bid * (V_ask / (V_bid + V_ask)) + P_ask * (V_bid / (V_bid + V_ask))
            = Mid + Imbalance * (Spread / 2)
    """
    mid = (bid_price + ask_price) / 2.0
    spread = ask_price - bid_price
    imbalance = estimate_order_book_imbalance(bid_qty, ask_qty)
    return mid + imbalance * (spread / 2.0)


@dataclass
class QueuePositionModel:
    """Модель черги лімітного ордера та ризику несприятливого вибору (Adverse Selection).

    Оцінює ймовірність філу лімітного ордера в черзі:
        P(fill) = 1 / (1 + exp(gamma * (queue_ahead / ADV) - beta * vol + alpha * distance))
    та коригує її на токсичність потоку (VPIN).
    """

    alpha_dist: float = 2.0  # штраф за відстань від спреду
    gamma_queue: float = 5.0  # штраф за глибину черги попереду
    beta_vol: float = 1.5  # бонус волатильності (висока волатильність швидше заповнює чергу)
    vpin_penalty: float = 3.0  # штраф за токсичність потоку

    def estimate_fill_prob(
        self,
        distance_bps: float,
        queue_ahead_ratio: float = 0.5,
        vol_frac: float = 0.001,
        vpin: float = 0.5,
    ) -> float:
        """Оцінка ймовірності заповнення лімітного ордера (0.0 .. 1.0).

        Args:
            distance_bps: відстань ордера від mid ціни в базисних пунктах (> 0 для пасивного).
            queue_ahead_ratio: частка обсягу в черзі перед нами відносно типового бару [0..1].
            vol_frac: волатильність як частка ціни.
            vpin: показник токсичності потоку ордерів [0..1].
        """
        exponent = (
            self.alpha_dist * (distance_bps / 10.0)
            + self.gamma_queue * queue_ahead_ratio
            - self.beta_vol * (vol_frac * 1000.0)
            + self.vpin_penalty * (vpin - 0.5)
        )
        exponent = float(np.clip(exponent, -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(exponent)))

    def adverse_selection_risk(
        self,
        side: int,
        imbalance: float,
        vpin: float = 0.5,
    ) -> float:
        """Оцінка ризику несприятливого вибору (Adverse Selection Score) у [0.0, 1.0].

        Для лонг-ордера (side=1) ризик зростає, коли стакан тисне вниз (imbalance < 0)
        і потік токсичний (VPIN високий).
        """
        # side: +1 (Buy limit), -1 (Sell limit)
        directional_pressure = -side * imbalance  # > 0 якщо ринок тисне проти нашого лімітного ордера
        risk_score = 0.5 + 0.3 * directional_pressure + 0.2 * (vpin - 0.5) * 2.0
        return float(np.clip(risk_score, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class QueueCalibration:
    """Емпіричний спред/глибина з depth5 для QueuePositionModel.

    Не вмикає MM/OBI. Лише числа: `spread_bps` у run_backtest і типовий L1 qty.
    """

    median_spread_bps: float
    p90_spread_bps: float
    median_l1_qty: float
    n_snapshots: int
    quality_ok: bool

    def engine_spread_bps(self) -> float:
        """Повний спред у bps — аргумент `run_backtest(spread_bps=…)`."""
        return float(self.median_spread_bps)


def calibrate_queue_from_depth(depth: pd.DataFrame) -> QueueCalibration:
    """Калібрування черги зі снапшотів depth5 (без lookahead: лише статистика ≤ t).

    MM/OBI не відроджуються тут: роадмап 5.4 чекає quality_ok покриття + окремий аудит.
    """
    from scalper_hft.data.validate import validate_depth
    from scalper_hft.features.depth_features import depth_spread_bps

    if depth is None or depth.empty:
        return QueueCalibration(2.0, 2.0, 0.0, 0, False)
    q = validate_depth(depth)
    spread = depth_spread_bps(depth).dropna()
    spread = spread[spread > 0]
    if spread.empty:
        med, p90 = 2.0, 2.0
    else:
        med = float(spread.median())
        p90 = float(spread.quantile(0.9))
    l1 = 0.0
    if "bid1_qty" in depth.columns and "ask1_qty" in depth.columns:
        tot = pd.to_numeric(depth["bid1_qty"], errors="coerce") + pd.to_numeric(depth["ask1_qty"], errors="coerce")
        tot = tot[tot > 0]
        if not tot.empty:
            l1 = float(tot.median())
    return QueueCalibration(
        median_spread_bps=med,
        p90_spread_bps=p90,
        median_l1_qty=l1,
        n_snapshots=int(len(depth)),
        quality_ok=bool(q.ok),
    )
