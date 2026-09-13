"""Статистичний арбітраж пар (perps BTC/ETH/SOL).

Ідея: висококорельовані пари (BTC/ETH, BTC/SOL, ETH/SOL) мають середньо-
реверсійний log-ratio. Коли спред відхиляється від норми — шортуємо
"переоцінену" ногу і лонгуємо "недооцінену" (delta-neutral пара на перпах).

Переваги проти funding/basis:
    - обидві ноги на USDT-M перпах (без spot-short обмеження);
    - спред-рухи більші за basis (можуть покрити 2-leg комісії);
    - збір фандінгу обох ніг частково компенсує витрати.

⚠ Ризики: спред може не ревертнутись (тренд у відносній силі — напр. ETH
випереджає BTC у ралі); обидві ноги мають фандінг — потрібен чистий облік.

Нові параметри (покращення результативності):
    breakeven_gate  : bool = False — відключає сигнали де ATR < round-trip витрат.
                      Потребує передачі cost_model ззовні або дефолтного CostModel.
    hmm_vol_gate    : bool = False — відключає входи у HMM-стані max-волатильності
                      (state 2 при n_states=3). М'якше за hmm_reversion: лише
                      найгірший стан блокується.
    regime_scale    : bool = True — масштабує експозицію входу за режимом leg2
                      (high-vol/trend → ×factor). Дефолт увімкнено з версії 1.3:
                      iter6/6b (CSCV PBO=0.000, Calmar пік при factor=0.25,
                      maxDD ≈ вдвічі менший на всіх парах, див.
                      docs/reports/iter6_regime_scale.md) і VALIDATED_PAIRS
                      у live/pairs_runner.py використовують саме цю конфігурацію.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy


class PairsArb(Strategy):
    name = "pairs_arb"
    family = "relative_value"
    preferred_regimes = frozenset()
    requires = frozenset({"multi_symbol"})

    param_space = {
        "entry_z": (1.5, 4.0, 0.25),
        "exit_z": (0.0, 1.0, 0.1),
        "lookback": (60.0, 1440.0, 60.0),
    }

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        lookback: int = 480,
        breakeven_gate: bool = False,
        hmm_vol_gate: bool = False,
        use_kalman: bool = False,
        kalman_q: float = 1e-5,
        kalman_r: float = 1e-3,
        dynamic_half_life: bool = False,
        regime_scale: bool = True,
        regime_scale_factor: float = 0.25,
        coint_gate: bool = False,
        coint_window: int = 1440,
        flow_toxicity_gate: bool = False,
        vpin_threshold: float = 0.9,
        hawkes_imbalance_threshold: float = 0.7,
        time_stop: bool = False,
        time_stop_mult: float = 2.0,
        time_stop_lookback: int = 480,
    ) -> None:
        super().__init__(
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=int(lookback),
            breakeven_gate=bool(breakeven_gate),
            hmm_vol_gate=bool(hmm_vol_gate),
            use_kalman=bool(use_kalman),
            kalman_q=float(kalman_q),
            kalman_r=float(kalman_r),
            dynamic_half_life=bool(dynamic_half_life),
            regime_scale=bool(regime_scale),
            regime_scale_factor=float(regime_scale_factor),
            coint_gate=bool(coint_gate),
            coint_window=int(coint_window),
            flow_toxicity_gate=bool(flow_toxicity_gate),
            vpin_threshold=float(vpin_threshold),
            hawkes_imbalance_threshold=float(hawkes_imbalance_threshold),
            time_stop=bool(time_stop),
            time_stop_mult=float(time_stop_mult),
            time_stop_lookback=int(time_stop_lookback),
        )
        self.betas: pd.Series | None = None

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Сигнал пари: +1 = шорт leg1/лонг leg2 (спред високий), −1 = дзеркально.

        df: DataFrame з колонками leg1/leg2 (ціни закриття перпів).
        Сигнал на закритті t → виконання t+1 (рушій робить shift).
        """
        # Скидаємо betas на початку: інакше при early-return сюди потрапляє
        # застаріле значення з ПОПЕРЕДНЬОГО виклику (stale-bleed у pairs.py).
        self.betas = None
        if "leg1" not in df.columns or "leg2" not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        use_kalman = bool(self.get("use_kalman", False))
        dynamic_half_life = bool(self.get("dynamic_half_life", False))

        if use_kalman:
            from scalper_hft.features.signal_processing import dynamic_hedge_ratio

            kq = float(self.get("kalman_q", 1e-5))
            kr = float(self.get("kalman_r", 1e-3))
            kf_res = dynamic_hedge_ratio(np.log(df["leg1"]), np.log(df["leg2"]), q_beta=kq, q_alpha=kq, r=kr)
            spread_series = kf_res["spread"]
            self.betas = kf_res["beta"]
        else:
            spread_series = np.log(df["leg1"] / df["leg2"])  # type: ignore[assignment]
            self.betas = pd.Series(1.0, index=df.index)

        lookback = int(self.get("lookback", 480))
        if dynamic_half_life:
            from scalper_hft.features.signal_processing import estimate_half_life

            hl = estimate_half_life(spread_series.tail(min(lookback, len(spread_series))))
            if np.isfinite(hl) and hl > 5:
                lookback = int(np.clip(hl * 2.0, 30.0, 1440.0))

        entry_z = float(self.get("entry_z", 2.0))
        exit_z = float(self.get("exit_z", 0.3))
        use_breakeven_gate = bool(self.get("breakeven_gate", False))
        use_hmm_vol_gate = bool(self.get("hmm_vol_gate", False))

        mean = spread_series.rolling(lookback, min_periods=lookback // 2).mean()
        std = spread_series.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
        z = ((spread_series - mean) / std).fillna(0.0)

        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[z > entry_z] = 1.0
        sig[z < -entry_z] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        exit_any = (prev_pos != 0.0) & (z.abs() < exit_z)
        sig[exit_any] = 0.0

        sig = sig.ffill().fillna(0.0).astype(int)  # type: ignore[assignment]

        # ── Breakeven gate: відключаємо сигнали де ATR спреду < 2-leg round-trip ──
        if use_breakeven_gate:
            sig = self._apply_breakeven_gate(sig, spread_series, df)

        # ── HMM-фільтр: відключаємо входи у стані максимальної волатильності ──
        if use_hmm_vol_gate:
            sig = self._apply_hmm_vol_gate(sig, df)

        # ── Regime-scale: масштабуємо силу сигналу за режимом leg2 ──
        if bool(self.get("regime_scale", True)):
            factor = float(self.get("regime_scale_factor", 0.25))
            sig = self._apply_regime_scale(sig, df, factor=factor)

        # ── Cointegration Gate: блокуємо входи при втраті коінтеграції (ADF) ──
        if bool(self.get("coint_gate", False)):
            coint_window = int(self.get("coint_window", 1440))
            sig = self._apply_adf_coint_gate(sig, spread_series, window=coint_window)

        # ── Flow Toxicity Gate: блокуємо входи при токсичному потоці (VPIN/Hawkes) ──
        # Дослідження §1.1–1.2: високий VPIN + Hawkes-дисбаланс = інформований
        # потік, що пробиває support/resistance → mean-reversion небезпечна.
        if bool(self.get("flow_toxicity_gate", False)) and trades is not None:
            vpin_thr = float(self.get("vpin_threshold", 0.9))
            hawkes_thr = float(self.get("hawkes_imbalance_threshold", 0.7))
            sig = self._apply_flow_toxicity_gate(sig, trades, vpin_threshold=vpin_thr, hawkes_threshold=hawkes_thr)

        # ── Time Stop: примусово закриває позицію після time_stop_mult × half-life ──
        # Дослідження §3.3: якщо позиція утримується довше 2 періодів напіврозпаду,
        # коінтеграція зламана — примусово ліквідувати.
        if bool(self.get("time_stop", False)):
            ts_mult = float(self.get("time_stop_mult", 2.0))
            ts_lookback = int(self.get("time_stop_lookback", 480))
            sig = self._apply_time_stop(sig, spread_series, mult=ts_mult, lookback=ts_lookback)

        return sig

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_breakeven_gate(sig: pd.Series, ratio: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Блокує сигнали де ATR спреду (ratio) < maker round-trip витрат.

        ratio = log(leg1/leg2) — вже безрозмірна величина (частка ціни),
        тому ATR спреду напряму порівнюється з round-trip costs у %.
        Для 2-leg позиції: threshold = 2 × maker round-trip ≈ 0.08%.
        """
        from scalper_hft.backtest.execution import CostModel

        cost = CostModel()
        # ATR спреду: середнє |Δratio| за 14 барів
        ratio_atr = ratio.diff().abs().rolling(14, min_periods=7).mean()
        # ratio вже у fractional units (log-scale ≈ fractional) — не ділимо на ціну
        threshold = cost.round_trip_maker() * 2.0
        gate = ratio_atr >= threshold
        # зберігаємо позиції що вже відкриті (не перекриваємо виходи)
        holding = sig.shift(1).fillna(0.0) != 0.0
        return sig.where(gate.reindex(sig.index, fill_value=False) | holding, other=0)

    @staticmethod
    def _apply_hmm_vol_gate(sig: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Блокує нові входи у HMM-стані з максимальною волатильністю.

        Відмінно від hmm_reversion: блокує лише state 2 (найгірший стан
        при n_states=3), тому ≈60-70% сигналів зберігаються.
        causal=True — без lookahead.
        """
        try:
            from scalper_hft.features.hmm_regime import hmm_regime_features

            close = df.get("leg1", df.get("close", None))
            if close is None or len(close) < 100:
                return sig
            hmm = hmm_regime_features(close, n_states=3, causal=True)
            # state 2 = найвища волатильність (сортування за середнім ret → state 2 ≠ завжди vol!)
            # використовуємо hmm_p2 (posterior P(state 2)) як міру небезпеки
            p_high_vol = hmm.get("hmm_p2", pd.Series(0.0, index=hmm.index))
            p_high_vol_aligned = p_high_vol.reindex(sig.index, method="ffill").fillna(0.0)
            # блокуємо лише якщо P(state 2) > 0.6 (тверда впевненість у небезпечному стані)
            block = p_high_vol_aligned > 0.6
            # блокуємо лише нові входи; виходи/утримання позиції — не блокуємо
            holding = sig.shift(1).fillna(0.0) != 0.0
            return sig.where(~block | holding, other=0)
        except Exception:  # noqa: BLE001
            return sig

    @staticmethod
    def _apply_regime_scale(sig: pd.Series, df: pd.DataFrame, factor: float = 0.5) -> pd.Series:
        """Масштабувати силу сигналу за режимом leg2 (ринковий годинник).

        leg2 — зазвичай BTC (валюта котирування пари). У спокійному range/normal
        mean-reversion спреду працює краще; у high-vol або тренді BTC відносна
        сила може трендити (break co-integration) → зменшуємо експозицію до factor.

        Каузально: regime обчислюється на close ≤ t (named_market_state —
        EMA/vol-percentile без lookahead). Розмір фіксується на моменті входу
        (ffill entry-scale під час угоди) — без resize-churn між барами.
        Виходи (sig==0) та утримання напрямку зберігають повний розмір, як і
        інші гейти (hmm_vol_gate/breakeven_gate): лише нові входи масштабуються.
        """
        from scalper_hft.features.regimes import named_market_state

        leg2 = df.get("leg2")
        if leg2 is None or len(leg2) < 100:
            return sig.astype(float)
        state = named_market_state(leg2)
        vol = state["vol"].reindex(sig.index, method="ffill").fillna("normal")
        structure = state["structure"].reindex(sig.index, method="ffill").fillna("range")
        # scale: 1.0 у сприятливому (range/normal/low), factor у несприятливому
        scale = pd.Series(1.0, index=sig.index)
        scale = scale.where(~(vol == "high"), factor)
        scale = scale.where(~structure.isin(["trend_up", "trend_down"]), factor)

        sig_f = sig.astype(float)
        # новий вхід: sig != 0 після sig == 0
        new_entry = (sig_f != 0.0) & (sig_f.shift(1).fillna(0.0) == 0.0)
        entry_scale = pd.Series(np.nan, index=sig.index, dtype=float)
        entry_scale.loc[new_entry.values] = scale.loc[new_entry.values].values  # type: ignore[index]
        # lock entry-time scale на час угоди (ffill); поза позицією — байдуже (sig==0)
        entry_scale = entry_scale.ffill().fillna(1.0)
        return (sig_f * entry_scale).clip(-1.0, 1.0)

    @staticmethod
    def _apply_adf_coint_gate(sig: pd.Series, spread: pd.Series, window: int = 1440, step: int = 60) -> pd.Series:
        """Перевіряє коінтеграцію спреду через ADF тест (p-value).

        Оскільки ADF тест повільний, він обчислюється лише кожні `step` барів (напр. кожні 60 хв).
        Якщо p-value > 0.05, коінтеграція вважається розірваною -> блокуємо НОВІ входи.
        """
        import warnings

        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return sig

        if len(spread) < window + step:
            return sig

        p_values = pd.Series(index=spread.index, dtype=float)

        # Sparse calculation to avoid huge backtest overhead
        calc_indices = np.arange(window, len(spread), step)
        if len(calc_indices) == 0 or calc_indices[-1] != len(spread) - 1:
            calc_indices = np.append(calc_indices, len(spread) - 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in calc_indices:
                slice_data = spread.iloc[i - window : i].dropna()
                if len(slice_data) > window // 2:
                    try:
                        res = adfuller(slice_data.values, maxlag=1, autolag=None)
                        p_values.iloc[i] = res[1]
                    except Exception:
                        p_values.iloc[i] = 1.0

        p_values = p_values.ffill().fillna(0.0)
        block = p_values > 0.05

        holding = sig.shift(1).fillna(0.0) != 0.0
        return sig.where(~block | holding, other=0)

    @staticmethod
    def _apply_flow_toxicity_gate(
        sig: pd.Series,
        trades: pd.DataFrame,
        vpin_threshold: float = 0.9,
        hawkes_threshold: float = 0.7,
    ) -> pd.Series:
        """Блокує нові входи при токсичному потоці ордерів (VPIN + Hawkes).

        Дослідження §1.1–1.2: VPIN (Volume-Synchronized PIN) міряє дисбаланс
        агресивних покупців/продавців у об'ємних барах; Hawkes-дисбаланс —
        само-збудження потоку. Високі значення обох = інформований потік,
        що пробиває support/resistance → mean-reversion спреду небезпечна
        (коінтеграція може зламатись). Тому блокуємо лише нові входи; виходи
        та утримання позиції — не блокуємо (консистентно з іншими гейтами).

        Каузально: VPIN/Hawkes обчислюються лише з trades ≤ t (ffill на
        kline-індекс). Без trades — no-op (гейт вимкнений фактично).
        """
        if trades is None or trades.empty or "side" not in trades.columns:
            return sig

        try:
            from scalper_hft.features.hawkes import order_flow_toxicity_hawkes
            from scalper_hft.features.microstructure import vpin
        except ImportError:  # noqa: BLE001
            return sig

        block = pd.Series(False, index=sig.index)

        # VPIN: об'ємні бари ≈ 10% середнього об'єму свічки (AFML Ch.19.5.2).
        try:
            vpin_series = vpin(trades, bar_volume=1000.0, n=50)
            if not vpin_series.empty:
                # VPIN індексується часом закриття об'ємних барів; при токсичному
                # потоці декілька об'ємних барів закриваються у межах одного kline —
                # дублікати. Беремо останнє значення на кожен timestamp, потім ffill.
                if vpin_series.index.has_duplicates:
                    vpin_series = vpin_series.groupby(level=0).last()
                vpin_aligned = vpin_series.reindex(sig.index, method="ffill").fillna(0.0)
                block = block | (vpin_aligned > vpin_threshold)
        except Exception:  # noqa: BLE001
            pass

        # Hawkes-дисбаланс: |imbalance| > threshold = спрямований токсичний потік.
        try:
            hawkes_df = order_flow_toxicity_hawkes(trades, alpha=0.1, beta=0.5)
            if not hawkes_df.empty and "hawkes_imbalance" in hawkes_df.columns:
                imb = hawkes_df["hawkes_imbalance"]
                if imb.index.has_duplicates:
                    imb = imb.groupby(level=0).last()
                imb_aligned = imb.reindex(sig.index, method="ffill").fillna(0.0)
                block = block | (imb_aligned.abs() > hawkes_threshold)
        except Exception:  # noqa: BLE001
            pass

        # Блокуємо лише нові входи; позиції що вже відкриті — не перекриваємо.
        holding = sig.shift(1).fillna(0.0) != 0.0
        return sig.where(~block | holding, other=0)

    @staticmethod
    def _apply_time_stop(
        sig: pd.Series,
        spread: pd.Series,
        mult: float = 2.0,
        lookback: int = 480,
        step: int = 60,
    ) -> pd.Series:
        """Примусово закриває позицію після `mult × half-life` барів утримання.

        Дослідження §3.3: якщо позиція утримується довше 2 періодів напіврозпаду,
        коінтеграція зламана — алгоритм примусово ліквідує її. Half-life оцінюється
        каузально (на спреді ≤ t) з вікна `lookback`, обчислюється кожні `step`
        барів (як ADF-гейт) для швидкості.

        На відміну від інших гейтів, time stop форсує ВИХІД (sig=0), а не лише
        блокує входи. Після time-stop виходу можливий повторний вхід, якщо z-score
        все ще екстремальний (наступний бар).
        """
        from scalper_hft.features.signal_processing import estimate_half_life

        n = len(spread)
        if n < lookback + step:
            return sig

        # Sparse rolling half-life (кожні `step` барів) — каузально, без lookahead.
        hl = pd.Series(np.nan, index=spread.index)
        calc_idx = np.arange(lookback, n, step)
        if len(calc_idx) == 0 or calc_idx[-1] != n - 1:
            calc_idx = np.append(calc_idx, n - 1)
        for i in calc_idx:
            slice_data = spread.iloc[i - lookback : i].dropna()
            if len(slice_data) > lookback // 2:
                hl.iloc[i] = estimate_half_life(slice_data, min_obs=20)
        hl = hl.ffill().fillna(float("inf"))

        # Тривалість утримання: кількість послідовних non-zero барів до поточного.
        # run_id змінюється при кожному sig==0 → cumsum дає унікальний id для кожного run.
        run_id = (sig == 0).cumsum()
        holding_bars = sig.ne(0).astype(int).groupby(run_id).cumsum()

        # Time stop: holding_bars > mult × half_life → форсуємо вихід (sig=0).
        # inf half-life (нема mean-reversion) → time stop ніколи не спрацьовує.
        max_hold = mult * hl
        max_hold = max_hold.replace([float("inf")], np.inf)
        stop = holding_bars > max_hold
        stop = stop.fillna(False)  # NaN у holding_bars (де sig==0) → не стопаємо
        return sig.where(~stop, other=0)
