"""Оптимізація параметрів через Optuna з CV-цільовою функцією.

Критично (книга, гл. 9–11): оптимізація на всьому наборі = перенавчання.
Тому:
    1. цільова функція = середній OOS Sharpe по purged K-fold;
    2. фінальна оцінка — на повному OOS holdout (поза оптимізацією);
    3. deflated Sharpe коригує результат на кількість спроб.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
from scalper_hft.strategies.base import Strategy
from scalper_hft.validation.cv import purged_kfold_indices

logger = logging.getLogger(__name__)

try:  # optuna — опційна залежність (pyproject: [optim])
    import optuna  # type: ignore

    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    _HAS_OPTUNA = False


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    cv_scores: list[float]
    n_trials: int
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        hold = self.details.get("holdout_sharpe")
        hold_line = f"\nHoldout Sharpe: {hold:.3f}" if isinstance(hold, (int, float)) and np.isfinite(hold) else ""
        return (
            f"Оптимізація: {self.n_trials} спроб; кращі параметри: {self.best_params}\n"
            f"CV OOS Sharpe: {self.cv_scores} (avg {np.mean(self.cv_scores):.3f})"
            f"{hold_line}"
        )


def _effective_purge(space: dict, purge: int | None) -> int:
    """Purge для CV: явний аргумент або з найдовшого горизонту в param_space.

    Параметри виду holding_bars/lookback/period задають горизонт, протягом
    якого лейбл/позиція може заходити у сусіднє вікно.
    """
    if purge is not None:
        return int(purge)
    horizon_hi = 0
    for pname, (lo, hi, _step) in space.items():
        if any(k in pname for k in ("holding", "lookback", "period", "window", "horizon")):
            horizon_hi = max(horizon_hi, int(hi))
    return max(50, horizon_hi)


def _default_objective(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    params: dict,
    cost: CostModel,
    trades: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    n_splits: int,
    embargo: int,
    position_pct: float,
    purge: int = 50,
) -> float:
    """Середній Sharpe по purged K-fold на OOS-зрізах.

    trades/funding ріжуться за ЧАСОМ test-вікна (їхній індекс — час подій,
    не позиції барів); purge — з максимального горизонту лейблів/утримання.
    """
    oos_sharpes: list[float] = []
    for _train_idx, test_idx in purged_kfold_indices(len(df), n_splits, purge=purge, embargo=embargo):
        te = df.iloc[test_idx]
        trades_te = _slice_funding(trades, te.index[0], te.index[-1]) if trades is not None else None
        funding_te = _slice_funding(funding, te.index[0], te.index[-1]) if funding is not None else None
        try:
            res = run_backtest(
                te, strategy_cls(**params), cost=cost, trades=trades_te, funding=funding_te, position_pct=position_pct
            )
            sharpe = res.metrics.sharpe
        except Exception:  # noqa: BLE001
            sharpe = -1.0
        if not np.isfinite(sharpe):
            sharpe = -1.0
        oos_sharpes.append(sharpe)
    return float(np.mean(oos_sharpes)) if oos_sharpes else -1.0


def _slice_funding(funding: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Часовий зріз [t0, t1] для рядів з власним індексом (funding, aggTrades)."""
    mask = (funding.index >= t0) & (funding.index <= t1)
    return funding[mask]


def optimize_params(
    df: pd.DataFrame,
    strategy: Strategy,
    n_trials: int = 60,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    n_splits: int = 4,
    embargo: int = 30,
    position_pct: float = 0.01,
    sampler: str = "tpe",
    purge: int | None = None,
) -> OptimizationResult:
    """Оптимізація параметрів з param_space стратегії.

    sampler: 'tpe' (байєсівський) або 'random'.
    purge: барів purge на межі train/test у CV. None → max(50, верхня межа
        holding_bars/lookback-параметрів з param_space) — purge має покривати
        найдовший горизонт лейбла/утримання серед усіх trial-ів.
    """
    if not _HAS_OPTUNA:
        raise ImportError("Встановіть optuna: uv add --optional optim optuna")

    cost = cost or CostModel()
    strategy_cls = type(strategy)
    space = strategy.param_space
    purge_eff = _effective_purge(space, purge)

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        for pname, (lo, hi, step) in space.items():
            is_int = float(step) == int(step) and float(lo) == int(lo) and float(hi) == int(hi)
            if is_int:
                params[pname] = trial.suggest_int(pname, int(lo), int(hi))
            else:
                params[pname] = trial.suggest_float(pname, float(lo), float(hi))
        return _default_objective(
            df, strategy_cls, params, cost, trades, funding, n_splits, embargo, position_pct, purge=purge_eff
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # CV-оцінки найкращого варіанта (перезапуск objective з фіксованими параметрами)
    cv_scores: list[float] = []
    if study.best_params:
        for train_idx, test_idx in purged_kfold_indices(len(df), n_splits, purge=purge_eff, embargo=embargo):
            te = df.iloc[test_idx]
            # часові зрізи: індекс trades/funding ≠ позиції барів
            trades_te = _slice_funding(trades, te.index[0], te.index[-1]) if trades is not None else None
            funding_te = _slice_funding(funding, te.index[0], te.index[-1]) if funding is not None else None
            try:
                res = run_backtest(
                    te,
                    strategy_cls(**study.best_params),
                    cost=cost,
                    trades=trades_te,
                    funding=funding_te,
                    position_pct=position_pct,
                )
                score = res.metrics.sharpe
            except Exception:  # noqa: BLE001
                score = -1.0
            cv_scores.append(score if np.isfinite(score) else -1.0)

    return OptimizationResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        cv_scores=cv_scores,
        n_trials=n_trials,
        details={"direction": "maximize", "sampler": sampler},
    )


# ── ML-специфічна оптимізація (AFML Ch.7) ────────────────────────────────────


def optimize_ml_params(
    df: pd.DataFrame,
    n_trials: int = 40,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    scoring: str = "neg_log_loss",
    decay: float = 0.9,
    frac_d: float = 0.4,
    add_frac_diff: bool = True,
    trades: pd.DataFrame | None = None,
    sampler: str = "tpe",
) -> OptimizationResult:
    """Оптимізація pt/sl/holding_bars через PurgedKFold + LightGBM CV.

    На відміну від optimize_params() (який оптимізує по Sharpe бектесту),
    ця функція оптимізує ПАРАМЕТРИ ЛЕЙБЛІНГУ (pt, sl, holding_bars) через
    PurgedKFold.cross_val_score на labeled dataset — без lookahead по t1.

    Це правильний AFML підхід: оцінюємо якість ML-моделі, а не PnL.

    Args:
        df: OHLCV DataFrame.
        n_trials: кількість Optuna спроб.
        n_splits: кількість фолдів PurgedKFold.
        embargo_pct: embargo для PurgedKFold.
        scoring: 'neg_log_loss' або 'accuracy'.
        decay: time-decay для sample weights.
        frac_d: ступінь fractional differencing.
        add_frac_diff: включити FFD фічі.
        trades: aggTrades для CVD фіч.
        sampler: 'tpe' або 'random'.

    Returns:
        OptimizationResult з best_params = {pt, sl, holding_bars}.
    """
    if not _HAS_OPTUNA:
        raise ImportError("Встановіть optuna: uv add --optional optim optuna")

    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except ImportError as e:
        raise ImportError("Встановіть lightgbm: uv add --optional ml lightgbm") from e

    from scalper_hft.ml.features import build_labeled_dataset
    from scalper_hft.validation.cv import PurgedKFold

    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    def objective(trial: optuna.Trial) -> float:
        pt = trial.suggest_float("pt", 0.5, 2.5, step=0.25)
        sl = trial.suggest_float("sl", 0.5, 2.5, step=0.25)
        holding_bars = trial.suggest_int("holding_bars", 5, 30, step=5)

        try:
            X, y, w, t1 = build_labeled_dataset(
                df=df,
                trades=trades,
                mode="triple_barrier",
                pt=pt,
                sl=sl,
                holding_bars=holding_bars,
                decay=decay,
                frac_d=frac_d,
                add_frac_diff=add_frac_diff,
                return_t1=True,
            )
        except ValueError:
            return -1.0 if scoring == "accuracy" else -10.0

        if len(X) < n_splits * 30:
            return -1.0 if scoring == "accuracy" else -10.0

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            class_weight="balanced",
            verbosity=-1,
        )

        try:
            scores = pkf.cross_val_score(
                estimator=model,
                X=X,
                y=y,
                t1=t1,
                sample_weight=w,
                scoring=scoring,
            )
            return float(np.mean(scores)) if len(scores) > 0 else -1.0
        except Exception as e:
            logger.debug("Trial failed: %s", e)
            return -1.0 if scoring == "accuracy" else -10.0

    direction = "maximize"
    sampler_obj = optuna.samplers.TPESampler(seed=42) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=sampler_obj)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # CV-оцінки фінальної моделі з кращими параметрами
    cv_scores: list[float] = []
    if study.best_params:
        try:
            X_best, y_best, w_best, t1_best = build_labeled_dataset(
                df=df,
                trades=trades,
                mode="triple_barrier",
                pt=study.best_params["pt"],
                sl=study.best_params["sl"],
                holding_bars=study.best_params["holding_bars"],
                decay=decay,
                frac_d=frac_d,
                add_frac_diff=add_frac_diff,
                return_t1=True,
            )

            model_final = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=30,
                class_weight="balanced",
                verbosity=-1,
            )
            cv_scores = list(
                pkf.cross_val_score(
                    model_final,
                    X_best,
                    y_best,
                    t1=t1_best,
                    sample_weight=w_best,
                    scoring=scoring,
                )
            )
        except Exception as e:
            logger.warning("Final CV failed: %s", e)

    return OptimizationResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        cv_scores=cv_scores,
        n_trials=n_trials,
        details={"scoring": scoring, "n_splits": n_splits, "method": "PurgedKFold"},
    )


def evaluate_holdout(
    holdout: pd.DataFrame,
    strategy_cls: type[Strategy],
    params: dict,
    *,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
) -> float:
    """Сліпий Sharpe на holdout після Optuna (не входить у objective)."""
    if holdout is None or holdout.empty or len(holdout) < 30:
        return float("nan")
    trades_ho = _slice_funding(trades, holdout.index[0], holdout.index[-1]) if trades is not None else None
    funding_ho = _slice_funding(funding, holdout.index[0], holdout.index[-1]) if funding is not None else None
    try:
        res = run_backtest(
            holdout,
            strategy_cls(**params),
            cost=cost or CostModel(),
            trades=trades_ho,
            funding=funding_ho,
            position_pct=position_pct,
        )
        sharpe = float(res.metrics.sharpe)
    except Exception:  # noqa: BLE001
        return float("nan")
    return sharpe if np.isfinite(sharpe) else float("nan")
