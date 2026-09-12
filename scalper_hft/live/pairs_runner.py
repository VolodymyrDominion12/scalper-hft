"""Paper pairs: дві ноги, maker post-only, all-or-none філл, ризик портфеля.

Сигнал PairsArb: +1 = шорт leg1 / лонг leg2; −1 = дзеркально; 0 = флет.
Сигнал на закритті бару t → лімітки по close t → філл на барі t+1, якщо
обидві ноги торкнулись рівня. Інакше unfilled (чекаємо wait_bars, потім скасовуємо).

Позиції ключаться як `{pair}:{symbol}`, щоб BTC у кількох парах не злипався
(як у бектест-портфелі: ноги незалежні).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.bar_clock import daemon_sleep_sec
from scalper_hft.live.control import DEFAULT_CONTROL_PATH, ControlState, load_control, save_control
from scalper_hft.live.intent_store import IntentStore
from scalper_hft.live.pairs_engine import (
    PairsEngine,
    PairsPaperResult,
    PendingOrder,
    align_ohlc,
    legs_for_want,
    pair_id,
    pair_size_pct,
    pos_key,
)
from scalper_hft.live.reconcile import KillSwitch, reconcile_exchange_state
from scalper_hft.live.store import PaperStore
from scalper_hft.live.sync_engine import SyncEngine
from scalper_hft.live.trader import closed_klines
from scalper_hft.live.ws_user_stream import OrderTradeEvent
from scalper_hft.portfolio.sizing import resolve_vol_target_ann
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.pairs_arb import PairsArb

__all__ = [
    "PendingOrder",
    "PairsEngine",
    "PairsLiveRunner",
    "PairsPaperResult",
    "PairsPaperRunner",
    "PairsPortfolioRunner",
    "VALIDATED_PAIRS",
    "align_ohlc",
    "legs_for_want",
    "pair_id",
    "pair_size_pct",
    "pos_key",
    "replay_pairs",
]

logger = logging.getLogger(__name__)


def _sync_kill_switch(reason: str) -> None:
    """Callback SyncEngine: зупинити торгівлю при drift позицій."""
    logger.critical("SyncEngine KillSwitch: %s", reason)
    save_control(pause=True, flatten=True)


def _make_sync_engine(
    store: PaperStore | None,
    *,
    account: PaperAccount,
    scope: set[str],
    dry_run: bool,
    mode: str,
) -> SyncEngine | None:
    if store is None:
        return None
    return SyncEngine(
        store=store,
        exchange_id="binance",
        mode=mode,
        account=account,
        scope=scope,
        dry_run=dry_run,
        on_kill_switch=_sync_kill_switch,
    )


# Пари за замовчуванням для paper-портфеля — перевалідація 2026-09-07
# (поточний код, 3y 2023-09..2026-09, 1h maker, 49 WF-вікон; див.
# docs/reports/iter5_pairs_revalidate / STRATEGY_STATUS.md):
#   LINK/BTC  з=2.0/lb=120: 3y +86.5% (PF 1.74), WF OOS>0 у 71% вікон, 400д +0.4%  ✅ ядро
#   LINK/BTC  з=2.0/lb=240: 3y +54.9% (PF 1.95), WF pos 65%                        альтернатива
# XRP/BTC, BTC/ETH, LINK/ETH на 3y від'ємні (XRP/BTC — лише моніторинг);
# раніше «валідовані» цифри (2026-08-30) отримані до фіксів моделі виконання.
#
# regime_scale overlay (iter6, 2026-09-08): CSCV PBO=0.000 (PASS), Calmar пік
# при factor=0.25 (3.12 vs 2.65 при 0.5). maxDD зменшено вдвічі на всіх парах.
# Див. docs/reports/iter6_regime_scale.md.
VALIDATED_PAIRS: tuple[dict, ...] = (
    {
        "leg1": "LINKUSDT",
        "leg2": "BTCUSDT",
        "entry_z": 2.0,
        "exit_z": 0.3,
        "lookback": 120,
        "regime_scale": True,
        "regime_scale_factor": 0.25,
    },
)

_RECENT_BARS = 800


def replay_pairs(
    leg1: str,
    leg2: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    strategy: Strategy | None = None,
    account: PaperAccount | None = None,
    store: PaperStore | None = None,
    funding1: pd.DataFrame | None = None,
    funding2: pd.DataFrame | None = None,
    n_pairs: int = 1,
    wait_bars: int | None = None,
    is_maker: bool = True,
    now: pd.Timestamp | None = None,
    interval: str = "1h",
) -> PairsPaperResult:
    """Історичний прогін з моделлю maker-філлів (сигнал t → філл t+1)."""
    settings = get_settings()
    c1 = closed_klines(df1, interval, now=now)
    c2 = closed_klines(df2, interval, now=now)
    common = align_ohlc(c1, c2)
    if len(common) < 50:
        raise ValueError("Замало спільних барів для пари")
    strategy = strategy or PairsArb()
    account = account or PaperAccount(
        initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
    )
    engine = PairsEngine(
        leg1, leg2, strategy, account, store=store, n_pairs=n_pairs, wait_bars=wait_bars, is_maker=is_maker
    )
    sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
    signals = strategy.generate_signals(sig_df)

    equity_pts: list[tuple[pd.Timestamp, float]] = []
    actions: list[str] = []
    prev: pd.Timestamp | None = None
    for ts_raw, row in common.iterrows():
        ts = pd.Timestamp(str(ts_raw))
        action = engine.on_bar(
            ts,
            float(row["l1_high"]),
            float(row["l1_low"]),
            float(row["l1_close"]),
            float(row["l2_high"]),
            float(row["l2_low"]),
            float(row["l2_close"]),
            float(signals.loc[ts]) if ts in signals.index else 0.0,
            funding1=_funding_between(funding1, prev, ts),
            funding2=_funding_between(funding2, prev, ts),
        )
        actions.append(action)
        equity_pts.append((ts, account.equity_at(engine._marks(float(row["l1_close"]), float(row["l2_close"])))))
        prev = ts

    eq = pd.Series({t: v for t, v in equity_pts}).sort_index()
    return PairsPaperResult(
        equity=eq,
        actions=actions,
        n_filled=engine.n_filled,
        n_unfilled=engine.n_unfilled,
        pair=engine.pid,
        account=account,
    )


def _funding_between(funding: pd.DataFrame | None, prev: pd.Timestamp | None, ts: pd.Timestamp) -> float | None:
    if funding is None or funding.empty or "fundingRate" not in funding.columns:
        return None
    idx = funding.index
    mask = idx <= ts if prev is None else (idx > prev) & (idx <= ts)
    rates = funding.loc[mask, "fundingRate"]
    if rates.empty:
        return None
    return float(rates.sum())


def _fetch_ohlcv(symbol: str, interval: str, limit: int = _RECENT_BARS) -> pd.DataFrame:
    from scalper_hft.data.client import recent_since_ms

    client = ExchangeClient()
    # `since_ms` — реальний час, НЕ 0: з нулем Binance віддає найстаріші свічки
    # лістингу (див. recent_since_ms), через що pairs-демон назавжди застрягав
    # на `hold:same_bar`.
    batch = client.fetch_klines(symbol, interval, since_ms=recent_since_ms(interval, limit), limit=limit)
    if not batch:
        raise RuntimeError(f"Немає даних для {symbol}")
    df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts").sort_index()


def should_persist_action(action: str) -> bool:
    """Не писати знімок на pause / той самий бар / помилку."""
    if action.startswith("error:") or action.startswith("hold:paused"):
        return False
    stripped = action.replace("halt:portfolio_loss ", "")
    parts = [p.strip() for p in stripped.split("||")]
    if parts and all("hold:same_bar" in p for p in parts):
        return False
    return True


def _install_stop_signals(stop: threading.Event) -> None:
    def _handle(signum: int, _frame: object) -> None:
        logger.info("сигнал %s — зупиняю paper-демон", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _paper_loop(
    step: Callable[[], str],
    save: Callable[[], None] | None,
    interval: str,
    account: PaperAccount,
    pair: str,
    n_filled: Callable[[], int],
    n_unfilled: Callable[[], int],
    *,
    daemon: bool,
    iterations: int,
    sleep_sec: int,
    stop: threading.Event | None = None,
    install_signals: bool = True,
    on_stop: Callable[[], Any] | None = None,
    control_path: Path | str | None = None,
) -> PairsPaperResult:
    halt = stop or threading.Event()
    if daemon and install_signals:
        _install_stop_signals(halt)
    actions: list[str] = []
    pts: list[tuple[pd.Timestamp, float]] = []
    i = 0
    try:
        while not halt.is_set():
            if not daemon and i >= iterations:
                break
            try:
                action = step()
            except KillSwitch as exc:
                logger.critical("KillSwitch у кроці %d: %s — pause+flatten, зупиняю цикл", i, exc)
                save_control(pause=True, flatten=True, path=control_path)
                action = f"killed:{exc}"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Крок %d: %s", i, exc)
                action = f"error:{exc}"
            actions.append(action)
            pts.append((pd.Timestamp.now(tz="UTC").tz_convert(None), account.equity))
            i += 1
            if action.startswith("killed:"):
                halt.set()
                break
            if daemon:
                if halt.wait(timeout=daemon_sleep_sec(interval, action)):
                    break
            elif i < iterations:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        logger.info("Paper loop перервано користувачем (Ctrl+C)")
    finally:
        if on_stop is not None:
            try:
                on_stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Помилка on_stop cleanup: %s", exc)
        if save is not None:
            save()
    eq = pd.Series({t: v for t, v in pts}).sort_index() if pts else pd.Series(dtype=float)
    return PairsPaperResult(
        equity=eq,
        actions=actions,
        n_filled=n_filled(),
        n_unfilled=n_unfilled(),
        pair=pair,
        account=account,
    )


def merge_runtime_payload(
    existing: dict[str, Any] | None,
    account: PaperAccount,
    runners: dict[str, dict[str, Any]],
    portfolio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(existing or {})
    payload["version"] = 1
    payload["account"] = account.to_snapshot()
    merged = dict(payload.get("runners") or {})
    merged.update(runners)
    payload["runners"] = merged
    if portfolio is not None:
        payload["portfolio"] = portfolio
    return payload


class PairsPaperRunner:
    """Цикл paper на одній парі (REST → закритий бар → engine).

    ⚠ Paper-only: рушій СИМУЛЮЄ maker-філи на локальному PaperAccount і НЕ
    ставить реальні ордери. Тому DRY_RUN=false тут заборонено — звірка з
    реальною біржею була б безглуздою (локальні ноги ніколи не співпадуть з
    реальними позиціями) і закінчувалась KillSwitch. Live-pairs потребує
    окремого адаптера з реальними ордерами ніг (див. M3 у code review).
    """

    def __init__(
        self,
        leg1: str,
        leg2: str,
        interval: str = "1h",
        strategy: Strategy | None = None,
        account: PaperAccount | None = None,
        store: PaperStore | None = None,
        n_pairs: int = 1,
        is_maker: bool = True,
        client: object | None = None,
        restore: bool = True,
        control_path: Path | str | None = None,
        require_audit: bool = True,
        audit_path: Path | None = None,
        vol_target_ann: float | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError(
                "PairsPaperRunner — paper-only: реальні ордери ніг не реалізовані. "
                "Використовуйте DRY_RUN=true (paper); live-pairs потребує адаптера "
                "з реальними ордерами."
            )
        self._dry_run = settings.dry_run
        self.leg1 = leg1
        self.leg2 = leg2
        self.interval = interval or "1h"
        self.strategy = strategy or PairsArb()
        # Paper pairs: завжди fail-closed на комірку пари (не двох ніг).
        # Юніт-тести передають require_audit=False.
        if require_audit:
            from scalper_hft.live.audit_gate import require_pair_audit_pass

            require_pair_audit_pass(
                self.strategy.name,
                self.leg1,
                self.leg2,
                self.interval,
                max_age_days=settings.audit_max_age_days,
                path=audit_path,
            )
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
        self.control_path = Path(control_path) if control_path is not None else DEFAULT_CONTROL_PATH
        payload: dict[str, Any] | None = None
        if restore and store is not None and account is None:
            payload = store.load_runtime()
            if payload and "account" in payload:
                account = PaperAccount.from_snapshot(payload["account"])
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        from scalper_hft.live.trader_bars import interval_seconds

        bars_per_year = 365.0 * 86400.0 / interval_seconds(self.interval)
        # Phase 5.2: ENABLE_VOL_TARGET з settings, якщо caller не передав явну ціль.
        self.enable_vol_target = bool(settings.enable_vol_target)
        vol_target_ann = resolve_vol_target_ann(
            vol_target_ann,
            enabled=self.enable_vol_target,
            target=float(settings.vol_target_ann),
        )
        self.engine = PairsEngine(
            leg1,
            leg2,
            self.strategy,
            self.account,
            store=store,
            n_pairs=n_pairs,
            is_maker=is_maker,
            vol_target_ann=vol_target_ann,
            bars_per_year=bars_per_year,
        )
        mode = "paper" if self._dry_run else "live"
        self.sync_engine = _make_sync_engine(
            store,
            account=self.account,
            scope={self.leg1, self.leg2},
            dry_run=self._dry_run,
            mode=mode,
        )
        self._last_ts: pd.Timestamp | None = None
        if restore and store is not None:
            payload = payload or store.load_runtime()
            state = (payload or {}).get("runners", {}).get(self.engine.pid)
            if state:
                self.engine.apply_snapshot(state)
                self._last_ts = self.engine.last_bar_ts

    def save_runtime(self) -> None:
        if self.store is None:
            return
        existing = self.store.load_runtime()
        payload = merge_runtime_payload(existing, self.account, {self.engine.pid: self.engine.to_snapshot()})
        self.store.save_runtime(payload)

    def _persist_if_needed(self, action: str) -> None:
        if self.store is not None and should_persist_action(action):
            self.save_runtime()

    def step(
        self,
        now: pd.Timestamp | None = None,
        *,
        reconcile: bool = True,
        control: ControlState | None = None,
        persist: bool = True,
    ) -> str:
        ctrl = control if control is not None else load_control(self.control_path)
        if ctrl.pause:
            return "hold:paused"
        if reconcile:
            reconcile_exchange_state(
                self.account,
                self.client,
                dry_run=self._dry_run,
                scope={self.leg1, self.leg2},
            )
        self.engine.control_block_entries = ctrl.no_new_entries
        df1 = closed_klines(_fetch_ohlcv(self.leg1, self.interval), self.interval, now=now)
        df2 = closed_klines(_fetch_ohlcv(self.leg2, self.interval), self.interval, now=now)
        common = align_ohlc(df1, df2)
        if len(common) < 50:
            return "hold:мало барів"
        self.engine.seed_spread_from_ohlc(common)
        ts = common.index[-1]
        if self._last_ts is not None and ts == self._last_ts:
            return "hold:same_bar"
        sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
        signal = 0.0 if ctrl.flatten else float(self.strategy.generate_signals(sig_df).iloc[-1])
        row = common.iloc[-1]
        action = self.engine.on_bar(
            ts,
            float(row["l1_high"]),
            float(row["l1_low"]),
            float(row["l1_close"]),
            float(row["l2_high"]),
            float(row["l2_low"]),
            float(row["l2_close"]),
            signal,
        )
        self._last_ts = ts
        logger.info("%s %s | equity=%.2f", self.engine.pid, action, self.account.equity)
        if persist:
            self._persist_if_needed(action)
        return action

    def on_ws_order_trade(self, event: OrderTradeEvent, now: pd.Timestamp | None = None) -> str:
        """Передати подію WebSocket стріму до PairsEngine."""
        res = self.engine.on_ws_order_trade(event, now=now)
        if res not in ("no_pending", "symbol_mismatch", "ignored"):
            self._persist_if_needed(res)
        return res

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 300,
        *,
        daemon: bool = False,
        stop: threading.Event | None = None,
        install_signals: bool = True,
    ) -> PairsPaperResult:
        if self.sync_engine:
            self.sync_engine.start()

        def _on_stop():
            if self.sync_engine:
                self.sync_engine.stop()
            self.engine.cancel_pending(reason="shutdown")

        return _paper_loop(
            self.step,
            self.save_runtime if self.store is not None else None,
            self.interval,
            self.account,
            self.engine.pid,
            lambda: self.engine.n_filled,
            lambda: self.engine.n_unfilled,
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            stop=stop,
            install_signals=install_signals,
            on_stop=_on_stop,
            control_path=self.control_path,
        )


class PairsPortfolioRunner:
    """Кілька валідованих пар на спільному рахунку.

    ⚠ Paper-only: як і PairsPaperRunner, не ставить реальні ордери →
    DRY_RUN=false заборонено (див. M3 у code review).
    """

    def __init__(
        self,
        configs: list[dict] | None = None,
        interval: str = "1h",
        account: PaperAccount | None = None,
        store: PaperStore | None = None,
        is_maker: bool = True,
        client: object | None = None,
        restore: bool = True,
        control_path: Path | str | None = None,
        require_audit: bool = True,
        audit_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError(
                "PairsPortfolioRunner — paper-only: реальні ордери ніг не реалізовані. "
                "Використовуйте DRY_RUN=true (paper); live-pairs потребує адаптера "
                "з реальними ордерами."
            )
        self._dry_run = settings.dry_run
        self.configs = configs or [dict(p) for p in VALIDATED_PAIRS]
        self.interval = interval
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
        self.control_path = Path(control_path) if control_path is not None else DEFAULT_CONTROL_PATH
        payload: dict[str, Any] | None = None
        if restore and store is not None and account is None:
            payload = store.load_runtime()
            if payload and "account" in payload:
                account = PaperAccount.from_snapshot(payload["account"])
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        n = len(self.configs)
        # Phase 5.2: vol-target sizing у PairsEngine
        # (масштаб ноціоналу входу = clip(target/realized σ спреду, 0, 1)).
        self.enable_vol_target = bool(settings.enable_vol_target)
        vol_target_ann = resolve_vol_target_ann(
            None,
            enabled=self.enable_vol_target,
            target=float(settings.vol_target_ann),
        )
        self.runners: list[PairsPaperRunner] = []
        for cfg in self.configs:
            strat = PairsArb(
                entry_z=float(cfg.get("entry_z", 2.0)),
                exit_z=float(cfg.get("exit_z", 0.3)),
                lookback=int(cfg.get("lookback", 240)),
                regime_scale=bool(cfg.get("regime_scale", False)),
                regime_scale_factor=float(cfg.get("regime_scale_factor", 0.5)),
            )
            self.runners.append(
                PairsPaperRunner(
                    cfg["leg1"],
                    cfg["leg2"],
                    interval=interval,
                    strategy=strat,
                    account=self.account,
                    store=store,
                    n_pairs=n,
                    is_maker=is_maker,
                    client=self.client,
                    restore=False,
                    control_path=self.control_path,
                    require_audit=require_audit,
                    audit_path=audit_path,
                    vol_target_ann=vol_target_ann,
                )
            )
        self.daily_loss_limit = settings.daily_loss_limit
        self.weekly_loss_limit = settings.weekly_loss_limit
        self.portfolio_var_limit = settings.portfolio_var_limit
        self.week_start_equity = self.account.equity
        self._last_week: tuple[int, int] | None = None
        # Ковзне вікно equity для hist-VaR: зберігаємо 100 останніх значень
        # (при 1h-інтервалі — ~4 дні; достатньо для VaR-сигналу корельованого
        # стресу, але не надто мало для сплайнів вибірки).
        self._equity_window: deque[float] = deque(maxlen=100)
        mode = "paper" if self._dry_run else "live"
        all_legs = {cfg["leg1"] for cfg in self.configs} | {cfg["leg2"] for cfg in self.configs}
        self.sync_engine = _make_sync_engine(
            store,
            account=self.account,
            scope=all_legs,
            dry_run=self._dry_run,
            mode=mode,
        )

        if payload:
            for r in self.runners:
                state = (payload.get("runners") or {}).get(r.engine.pid)
                if state:
                    r.engine.apply_snapshot(state)
                    r._last_ts = r.engine.last_bar_ts
            port = payload.get("portfolio") or {}
            if port.get("week_start_equity") is not None:
                self.week_start_equity = float(port["week_start_equity"])
            lw = port.get("last_week")
            if lw and len(lw) == 2:
                self._last_week = (int(lw[0]), int(lw[1]))

    def _roll_week(self, now: pd.Timestamp | None) -> None:
        ts = now if now is not None else pd.Timestamp.now(tz="UTC").tz_convert(None)
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))
        if self._last_week is not None and week != self._last_week:
            self.week_start_equity = self.account.equity
        self._last_week = week

    def _update_equity_window(self) -> None:
        """Зафіксувати поточне значення equity у ковзному вікні."""
        self._equity_window.append(self.account.equity)

    def _portfolio_var_ok(self) -> bool:
        """True якщо hist-VaR(95%) портфеля в межах ліміту або ліміт вимкнений.

        Повертає True (no-op) якщо:
        - portfolio_var_limit == 0.0 (дефолт, backward-compatible)
        - недостатньо даних у вікні (< 10 точок → чекаємо накопичення)
        """
        if self.portfolio_var_limit <= 0.0:
            return True
        win = list(self._equity_window)
        if len(win) < 10:
            return True  # недостатньо даних — не блокуємо
        arr = np.array(win, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        # Hist-VaR(95%): 5-й перцентиль доходності → втрата (додатнє число)
        var_95 = float(-np.quantile(returns, 0.05))
        if var_95 > self.portfolio_var_limit:
            logger.warning(
                "Portfolio hist-VaR(95%%) = %.4f > limit %.4f — halting new entries",
                var_95,
                self.portfolio_var_limit,
            )
            return False
        return True

    def _should_halt_entries(self) -> bool:
        eq = self.account.equity
        if eq <= self.account.day_start_equity * (1.0 - self.daily_loss_limit):
            return True
        if eq <= self.week_start_equity * (1.0 - self.weekly_loss_limit):
            return True
        if not self._portfolio_var_ok():
            return True
        return False

    def save_runtime(self) -> None:
        if self.store is None:
            return
        runners = {r.engine.pid: r.engine.to_snapshot() for r in self.runners}
        last_week = list(self._last_week) if self._last_week is not None else None
        payload = merge_runtime_payload(
            self.store.load_runtime(),
            self.account,
            runners,
            portfolio={"week_start_equity": self.week_start_equity, "last_week": last_week},
        )
        self.store.save_runtime(payload)

    def _persist_if_needed(self, action: str) -> None:
        if self.store is not None and should_persist_action(action):
            self.save_runtime()

    def step(self, now: pd.Timestamp | None = None) -> str:
        ctrl = load_control(self.control_path)
        if ctrl.pause:
            return "hold:paused"
        self._roll_week(now)
        self._update_equity_window()
        scope = {sym for r in self.runners for sym in (r.leg1, r.leg2)}
        reconcile_exchange_state(self.account, self.client, dry_run=self._dry_run, scope=scope)
        halt = self._should_halt_entries()
        var_halt = not self._portfolio_var_ok() if not halt else False
        for r in self.runners:
            r.engine.portfolio_block_entries = halt
            r.engine.control_block_entries = ctrl.no_new_entries
        if var_halt:
            prefix = "halt:portfolio_var "
        elif halt:
            prefix = "halt:portfolio_loss "
        else:
            prefix = ""
        action = prefix + " || ".join(
            r.step(now=now, reconcile=False, control=ctrl, persist=False) for r in self.runners
        )
        self._persist_if_needed(action)
        return action

    def cancel_all_pending(self, reason: str = "shutdown") -> int:
        count = 0
        for r in self.runners:
            if r.engine.cancel_pending(reason=reason):
                count += 1
        return count

    def on_ws_order_trade(self, event: OrderTradeEvent, now: pd.Timestamp | None = None) -> list[str]:
        """Диспетчеризація WS подій до відповідного парного раннера."""
        results: list[str] = []
        for r in self.runners:
            res = r.on_ws_order_trade(event, now=now)
            if res not in ("no_pending", "symbol_mismatch", "ignored"):
                results.append(f"{r.engine.pid}:{res}")
        return results

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 300,
        *,
        daemon: bool = False,
        stop: threading.Event | None = None,
        install_signals: bool = True,
    ) -> PairsPaperResult:
        if self.sync_engine:
            self.sync_engine.start()

        def _on_stop():
            if self.sync_engine:
                self.sync_engine.stop()
            self.cancel_all_pending(reason="shutdown")

        return _paper_loop(
            self.step,
            self.save_runtime if self.store is not None else None,
            self.interval,
            self.account,
            "portfolio",
            lambda: sum(r.engine.n_filled for r in self.runners),
            lambda: sum(r.engine.n_unfilled for r in self.runners),
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            stop=stop,
            install_signals=install_signals,
            on_stop=_on_stop,
            control_path=self.control_path,
        )


class PairsLiveRunner(PairsPaperRunner):
    """Live-раннер однієї пари: реальні ордери ніг через PairsLiveAdapter.

    Фаза 3 (DEPLOY_PLAN). На відміну від PairsPaperRunner:
      - вимагає DRY_RUN=false + live-ключі (require_live_credentials);
      - рушій = PairsLiveAdapter (реальні post_only ліміти, reconcile, KillSwitch);
      - на старті: adapter.start_live() (звірка + гідратація);
      - після кожного бару: adapter.reconcile_runtime() (drift → KillSwitch).

    ⚠ Не стартує без явного DRY_RUN=false та ключів. Немає тихого шляху в live.
    """

    def __init__(
        self,
        leg1: str,
        leg2: str,
        interval: str = "1h",
        strategy: Strategy | None = None,
        account: PaperAccount | None = None,
        store: PaperStore | None = None,
        n_pairs: int = 1,
        client: object | None = None,
        restore: bool = True,
        control_path: Path | str | None = None,
        require_audit: bool = True,
        audit_path: Path | None = None,
    ) -> None:
        from scalper_hft.config import require_live_credentials
        from scalper_hft.live.pairs_live import PairsLiveAdapter

        settings = get_settings()
        require_live_credentials(settings)  # fail-closed: без ключів не стартує
        if settings.dry_run:
            raise RuntimeError(
                "PairsLiveRunner — live: DRY_RUN=true заборонено. Для paper використовуйте PairsPaperRunner."
            )
        self._dry_run = False
        self.leg1 = leg1
        self.leg2 = leg2
        self.interval = interval or "1h"
        self.strategy = strategy or PairsArb()
        self.store = store
        self.client = client  # обов'язковий для live
        if self.client is None:
            raise RuntimeError("PairsLiveRunner потребує ExchangeClient (client)")
        if require_audit:
            from scalper_hft.live.audit_gate import require_live_audit_if_not_dry_run, require_pair_audit_pass

            require_pair_audit_pass(
                self.strategy.name,
                self.leg1,
                self.leg2,
                self.interval,
                max_age_days=settings.audit_max_age_days,
                path=audit_path,
            )
            require_live_audit_if_not_dry_run(
                settings,
                pair=(self.strategy.name, self.leg1, self.leg2, self.interval),
                path=audit_path,
            )
        self.control_path = Path(control_path) if control_path is not None else DEFAULT_CONTROL_PATH
        payload: dict[str, Any] | None = None
        if restore and store is not None and account is None:
            payload = store.load_runtime()
            if payload and "account" in payload:
                account = PaperAccount.from_snapshot(payload["account"])
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        from scalper_hft.live.trader_bars import interval_seconds

        bars_per_year = 365.0 * 86400.0 / interval_seconds(self.interval)
        self.enable_vol_target = bool(settings.enable_vol_target)
        vol_target_ann = resolve_vol_target_ann(
            None,
            enabled=self.enable_vol_target,
            target=float(settings.vol_target_ann),
        )
        self.engine = PairsLiveAdapter(
            leg1,
            leg2,
            self.strategy,
            self.account,
            client=self.client,
            store=store,
            n_pairs=n_pairs,
            is_maker=True,
            legging_mode="chase",
            intent_store=IntentStore(),
            vol_target_ann=vol_target_ann,
            bars_per_year=bars_per_year,
        )
        self.sync_engine = _make_sync_engine(
            store,
            account=self.account,
            scope={leg1, leg2},
            dry_run=False,
            mode="live",
        )
        self._last_ts: pd.Timestamp | None = None
        if restore and store is not None:
            payload = payload or store.load_runtime()
            state = (payload or {}).get("runners", {}).get(self.engine.pid)
            if state:
                self.engine.apply_snapshot(state)
                self._last_ts = self.engine.last_bar_ts

    def step(
        self,
        now: pd.Timestamp | None = None,
        *,
        reconcile: bool = True,
        control: ControlState | None = None,
        persist: bool = True,
    ) -> str:
        ctrl = control if control is not None else load_control(self.control_path)
        if ctrl.pause:
            return "hold:paused"
        # live: reconcile з біржею (drift → KillSwitch) замість paper no-op.
        # Fail-closed: KillSwitch → control plane pause+flatten, без краху циклу.
        try:
            self.engine.reconcile_runtime()
        except KillSwitch as exc:
            logger.critical("KillSwitch у live-step: %s — pause+flatten", exc)
            save_control(pause=True, flatten=True, path=self.control_path)
            return "killed:killswitch"
        self.engine.control_block_entries = ctrl.no_new_entries
        df1 = closed_klines(_fetch_ohlcv(self.leg1, self.interval), self.interval, now=now)
        df2 = closed_klines(_fetch_ohlcv(self.leg2, self.interval), self.interval, now=now)
        common = align_ohlc(df1, df2)
        if len(common) < 50:
            return "hold:мало барів"
        self.engine.seed_spread_from_ohlc(common)
        ts = common.index[-1]
        if self._last_ts is not None and ts == self._last_ts:
            return "hold:same_bar"
        sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
        signal = 0.0 if ctrl.flatten else float(self.strategy.generate_signals(sig_df).iloc[-1])
        row = common.iloc[-1]
        try:
            action = self.engine.on_bar(
                ts,
                float(row["l1_high"]),
                float(row["l1_low"]),
                float(row["l1_close"]),
                float(row["l2_high"]),
                float(row["l2_low"]),
                float(row["l2_close"]),
                signal,
            )
        except KillSwitch as exc:
            logger.critical("KillSwitch у live on_bar: %s — pause+flatten", exc)
            save_control(pause=True, flatten=True, path=self.control_path)
            return f"killed:{exc}"
        self._last_ts = ts
        logger.info("%s %s | equity=%.2f", self.engine.pid, action, self.account.equity)
        if persist:
            self._persist_if_needed(action)
        return action

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 300,
        *,
        daemon: bool = False,
        stop: threading.Event | None = None,
        install_signals: bool = True,
    ) -> PairsPaperResult:
        # live: на старті обов'язкова звірка + гідратація з біржі
        self.engine.start_live()
        if self.sync_engine:
            self.sync_engine.start()

        def _on_stop():
            if self.sync_engine:
                self.sync_engine.stop()
            self.engine.cancel_pending(reason="shutdown")

        return _paper_loop(
            self.step,
            self.save_runtime if self.store is not None else None,
            self.interval,
            self.account,
            self.engine.pid,
            lambda: self.engine.n_filled,
            lambda: self.engine.n_unfilled,
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            stop=stop,
            install_signals=install_signals,
            on_stop=_on_stop,
            control_path=self.control_path,
        )
