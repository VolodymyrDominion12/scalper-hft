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
