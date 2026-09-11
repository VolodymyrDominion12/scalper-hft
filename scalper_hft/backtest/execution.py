"""Модель транзакційних витрат (книга, гл. 5).

Три компоненти:
    1. Комісії (maker/taker) — фіксовані частки ноціоналу;
    2. Slippage — погіршення ціни виконання (bps);
    3. Market impact — для retail-розмірів на Binance ф'ючерсах нехтовно малий
       (глибокий стакан), тому за замовчуванням 0, але параметризується.

Важливо: для скальпінгу комісії + slippage часто перевищують очікуваний
прибуток — це головний фільтр життєздатності стратегії (див. RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Вартість одного round-trip = (fee + slippage) × 2 (вхід і вихід).

    Розширення (Спринт 2, Narang гл. 5):
        - vol_aware_slippage: slippage зростає з волатильністю
          (slippage = base·(σ_now/σ_ref)^exp);
        - sqrt_law_impact: market impact за Square-Root Law
          I = k·σ·√(Q/ADV) (Almgren–Chriss-стиль), калібровка k — з барів;
        - breakeven_move_pct як поріг входу (apply_breakeven_gate).
    """

    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_frac: float = 0.0002  # 2 bps
    impact_frac: float = 0.0
    # ── Спринт 2 ──
    impact_k: float = 0.1  # константа Square-Root Law (калібрується)
    vol_ref: float = 0.0  # базова волатильність (частка ціни); 0 = без масштабування
    vol_exp: float = 1.0  # показник масштабування slippage волатильністю

    @classmethod
    def from_settings(cls, settings=None, df: pd.DataFrame | None = None) -> "CostModel":
        """Авто-калібровка vol_ref з Parkinson-vol якщо не задано."""
        from scalper_hft.config import get_settings
        s = settings or get_settings()
        vol_ref = s.vol_aware_slippage_ref
        if vol_ref <= 0 and df is not None:
            from scalper_hft.features.microstructure import parkinson_vol
            pv = parkinson_vol(df["high"], df["low"], window=20)
            vol_ref = float(pv.dropna().median()) if not pv.dropna().empty else 0.0
        return cls(
            maker_fee=s.maker_fee, 
            taker_fee=s.taker_fee,
            slippage_frac=s.slippage_bps / 10_000.0, 
            vol_ref=vol_ref,
            vol_exp=s.vol_aware_slippage_exp,
            impact_k=s.impact_k
        )

    def taker_cost_per_side(self) -> float:
        return self.taker_fee + self.slippage_frac + self.impact_frac

    def maker_cost_per_side(self) -> float:
        return self.maker_fee + self.impact_frac  # лімітні ордери без slippage

    def round_trip_taker(self) -> float:
        return 2 * self.taker_cost_per_side()

    def round_trip_maker(self) -> float:
        return 2 * self.maker_cost_per_side()

    def taker_roundtrip_cost(self) -> float:
        """Аліас для round_trip_taker."""
        return self.round_trip_taker()

    def maker_roundtrip_cost(self) -> float:
        """Аліас для round_trip_maker."""
        return self.round_trip_maker()

    def cost(self, side_is_maker: bool) -> float:
        return self.maker_cost_per_side() if side_is_maker else self.taker_cost_per_side()

    @property
    def breakeven_move_pct(self) -> float:
        """Мінімальний рух ціни (%), що покриває round-trip taker — бар'єр для скальпера."""
        return self.round_trip_taker() * 100

    # ── Спринт 2: волатильність-залежні витрати ─────────────────────────────

    def vol_aware_slippage(self, vol_frac: float | pd.Series | None = None) -> float | pd.Series:
        """Slippage, масштабований поточною волатильністю.

        vol_frac: поточна волатильність як частка ціни (напр., ATR/close).
        Якщо vol_ref ≤ 0 або vol_frac None — повертає базовий slippage_frac.
        """
        if vol_frac is None or self.vol_ref <= 0:
            return self.slippage_frac
        scale = (vol_frac / self.vol_ref) ** self.vol_exp
        return self.slippage_frac * scale

    def sqrt_law_impact(
        self,
        qty_notional: float,
        adv_notional: float,
        sigma_frac: float,
    ) -> float:
        """Market impact за Square-Root Law: I = k·σ·√(Q/ADV).

        qty_notional: розмір ордера (ноціонал); adv_notional: середньоденний
        об'єм (ноціонал); sigma_frac: волатильність (частка ціни, на горизонті
        виконання). Q/ADV > 1 обмежується 1 (повний об'єм ринку).
        """
        if adv_notional <= 0 or qty_notional <= 0:
            return 0.0
        share = min(qty_notional / adv_notional, 1.0)
        return self.impact_k * sigma_frac * float(np.sqrt(share))

    def total_cost_per_side(
        self,
        is_maker: bool,
        vol_frac: float | None = None,
        qty_notional: float | None = None,
        adv_notional: float | None = None,
        sigma_frac: float | None = None,
    ) -> float:
        """Повна вартість однієї сторони: комісія + slippage(σ) + impact(Q)."""
        fee = self.maker_fee if is_maker else self.taker_fee
        slip = self.vol_aware_slippage(vol_frac)
        impact = 0.0
        if qty_notional is not None and adv_notional is not None and sigma_frac is not None:
            impact = self.sqrt_law_impact(qty_notional, adv_notional, sigma_frac)
        return float(fee + slip + impact)

    def with_is_slippage(self, records: list[object], kind: str = "all") -> CostModel:
        from dataclasses import replace

        return replace(self, slippage_frac=calibrate_from_is(records, kind=kind))


def calibrate_from_is(
    records: list[object],
    *,
    kind: str = "all",
    fallback: float = 0.0002,
) -> float:
    """Median IS (bps) → slippage_frac. kind: all | maker | chase."""
    rows = list(records)
    if kind == "maker":
        rows = [r for r in rows if bool(getattr(r, "is_maker", True))]
    elif kind == "chase":
        rows = [r for r in rows if not bool(getattr(r, "is_maker", True))]
    if not rows:
        return fallback
    vals = sorted(float(getattr(r, "shortfall_bps", 0.0)) for r in rows)
    median_bps = vals[len(vals) // 2]
    return max(median_bps / 10_000.0, 0.0)


def estimate_impact_k_from_bars(df: pd.DataFrame, sigma_col: str | None = None) -> float:
    """Калібровка константи Square-Root Law k з OHLCV-барів (проксі).

    Для кожного бару: q/adv = volume / rolling_mean(volume); за моделлю
    |ret| ≈ k·σ·√(q/adv) → k̂ = медіана(|ret| / (σ̂·√(q/adv))).
    σ̂ береться з sigma_col (напр. 'atr_14'/'close') або Parkinson-волі.
    """
    if df is None or df.empty or len(df) < 60:
        return 0.1
    close = df["close"]
    if sigma_col and sigma_col in df.columns:
        sigma = df[sigma_col].astype(float)
    else:
        hl = np.log(df["high"] / df["low"])
        sigma = np.sqrt(hl.rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))).fillna(0.0)
    adv = df["volume"].rolling(100, min_periods=50).mean().replace(0, np.nan)
    share = (df["volume"] / adv).clip(upper=1.0).fillna(0.0)
    ret = close.pct_change().abs()
    denom = (sigma * np.sqrt(share)).replace(0, np.nan)
    k = (ret / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if len(k) < 20:
        return 0.1
    return float(np.median(k.values))


def estimate_spread_from_bookticker(bt: pd.DataFrame) -> float:
    """Середній відносний спред з bookTicker (best bid/ask).

    bt: DataFrame з колонками bid/ask. Повертає
    медіану (ask−bid)/mid — реалістичний slippage для maker-моделі.
    """
    if bt is None or bt.empty:
        return 0.0
    if "bid" not in bt.columns or "ask" not in bt.columns:
        return 0.0
    bid = bt["bid"].astype(float)
    ask = bt["ask"].astype(float)
    mid = (bid + ask) / 2.0
    spread = ((ask - bid) / mid.replace(0, np.nan)).dropna()
    return float(spread.median()) if len(spread) else 0.0


def estimate_spread_from_depth(depth: pd.DataFrame) -> float:
    """Медіанний відносний спред з depth5 (ask1−bid1)/mid — як bookTicker."""
    if depth is None or depth.empty:
        return 0.0
    bid_col = "bid1" if "bid1" in depth.columns else ("bid" if "bid" in depth.columns else None)
    ask_col = "ask1" if "ask1" in depth.columns else ("ask" if "ask" in depth.columns else None)
    if bid_col is None or ask_col is None:
        return 0.0
    bid = depth[bid_col].astype(float)
    ask = depth[ask_col].astype(float)
    mid = (bid + ask) / 2.0
    spread = ((ask - bid) / mid.replace(0, np.nan)).dropna()
    return float(spread.median()) if len(spread) else 0.0


def _atr_from_ohlc(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR-фолбек з OHLC, якщо у df немає готової колонки atr."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def apply_breakeven_gate(
    signals: pd.Series,
    df: pd.DataFrame,
    cost: CostModel,
    is_maker: bool = False,
    move_col: str = "atr_14",
) -> pd.Series:
    """Вимикає сигнали, де очікуваний рух менший за round-trip витрати.

    Ідея (Narang, гл. 5; RESEARCH.md): якщо очікуваний прибуток менший за
    транзакційні витрати — стратегія мертва незалежно від сигналу. Гейт
    порівнює міру очікуваного руху (за замовчуванням ATR) з
    `cost.breakeven_move_pct` і обнуляє сигнали, які не покривають витрати.

    Args:
        signals: Series позицій у [-1, 1], індексована як df.index.
        df: OHLCV DataFrame (опційно з колонкою `move_col`).
        cost: CostModel — звідки береться breakeven_move_pct.
        is_maker: якщо True — поріг рахується за maker round-trip.
        move_col: колонка очікуваного руху у df (за замовч. 'atr_14').

    Returns:
        signals з обнуленими позиціями там, де рух < round-trip витрат.
    """
    threshold = cost.round_trip_taker() if not is_maker else cost.round_trip_maker()
    if move_col in df.columns:
        move = df[move_col]
    else:
        move = _atr_from_ohlc(df)
    move_frac = move / df["close"].replace(0, np.nan)
    gate = move_frac >= threshold
    return signals.where(gate.reindex(signals.index, fill_value=False), other=0.0)


class ImplementationShortfallTracker:
    """Трекер Implementation Shortfall (IS) та онлайн-калібровка CostModel (Narang Ch. 5, 7).

    IS = Side * (Fill_Price - Decision_Price) / Decision_Price
    Додатний IS означає погіршення ціни (slippage cost).
    Забезпечує зворотний зв'язок між реальними виконаннями та CostModel.
    """

    def __init__(self, alpha_decay: float = 0.05) -> None:
        self.alpha_decay = alpha_decay
        self.records: list[dict[str, float]] = []
        self._ewma_is_bps: float = 0.0
        self._ewma_vol: float = 0.0
        self._count: int = 0

    def record_execution(
        self,
        decision_price: float,
        fill_price: float,
        side: int,
        qty_notional: float = 0.0,
        vol_frac: float = 0.0,
        is_maker: bool = False,
    ) -> float:
        """Фіксація філа та розрахунок IS в bps.

        side: +1 (Buy), -1 (Sell).
        Returns: IS у базисних пунктах (bps).
        """
        if decision_price <= 0:
            return 0.0
        shortfall = side * (fill_price - decision_price) / decision_price
        is_bps = shortfall * 10_000.0

        self.records.append(
            {
                "decision_price": decision_price,
                "fill_price": fill_price,
                "side": float(side),
                "qty_notional": qty_notional,
                "vol_frac": vol_frac,
                "is_maker": 1.0 if is_maker else 0.0,
                "is_bps": is_bps,
                "shortfall": shortfall,
            }
        )

        # Оновлюємо EWMA показники
        if self._count == 0:
            self._ewma_is_bps = is_bps
            self._ewma_vol = vol_frac
        else:
            self._ewma_is_bps = (1.0 - self.alpha_decay) * self._ewma_is_bps + self.alpha_decay * is_bps
            if vol_frac > 0:
                self._ewma_vol = (1.0 - self.alpha_decay) * self._ewma_vol + self.alpha_decay * vol_frac
        self._count += 1

        return is_bps

    @property
    def mean_slippage_bps(self) -> float:
        """Середній реалізований slippage (bps)."""
        if not self.records:
            return 0.0
        return float(np.mean([r["is_bps"] for r in self.records]))

    @property
    def current_slippage_frac(self) -> float:
        """Поточний згладжений slippage як частка ціни."""
        return max(self._ewma_is_bps / 10_000.0, 0.0)

    def calibrate_cost_model(self, base_cost: CostModel | None = None) -> CostModel:
        """Створює оновлену CostModel з каліброваним значенням slippage_frac."""
        base = base_cost or CostModel()
        new_slip = max(self.current_slippage_frac, 0.0001) if self._count >= 5 else base.slippage_frac
        return CostModel(
            maker_fee=base.maker_fee,
            taker_fee=base.taker_fee,
            slippage_frac=new_slip,
            impact_frac=base.impact_frac,
            impact_k=base.impact_k,
            vol_ref=self._ewma_vol if self._ewma_vol > 0 else base.vol_ref,
            vol_exp=base.vol_exp,
        )
