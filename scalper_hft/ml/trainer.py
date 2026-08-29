"""Тренування ML-класифікатора з walk-forward (без lookahead і без leakage).

AFML-інтеграція:
  - `model.fit(X_train, y_train, sample_weight=w_train)` — ваги uniqueness+decay
  - frac_diff фічі з features.py (збереження пам'яті ряду)
  - triple-barrier labels замість простого horizon-таргету

Схема walk-forward:
    для кожного ковзного вікна:
        events_train = triple_barrier(train_slice)
        model.fit(X_tr, y_tr, sample_weight=w_tr)  ← AFML ваги
        preds = model.predict_proba(X_test)          ← OOS прогноз
    → конкатенація OOS-прогнозів = чесна оцінка генералізації.

Backward compatibility:
    train_walk_forward() приймає (X, y) або повний df → автоматично вибирає режим.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.ml.features import build_labeled_dataset

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
    _HAS_LGBM = False


# ── Результат ─────────────────────────────────────────────────────────────────

@dataclass
class MlResult:
    oos_accuracy: float
    oos_logloss: float
    oos_sharpe: float       # Sharpe симуляції: позиція = знак прогнозу
    predictions: pd.Series
    n_train: int
    n_test: int
    n_windows: int
    feature_importance: pd.Series | None = None
    probabilities: pd.Series | None = None  # P(клас +1) на OOS, вирівняна з predictions
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ML walk-forward: {self.n_windows} вікон; train ~{self.n_train}, test {self.n_test}\n"
            f"OOS accuracy: {self.oos_accuracy:.3f} | logloss: {self.oos_logloss:.4f} | OOS Sharpe: {self.oos_sharpe:.3f}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simulate_sharpe(preds: pd.Series, close: pd.Series) -> float:
    """Sharpe стратегії 'позиція = знак прогнозу' на OOS-даних."""
    pos = np.sign(preds).astype(float)
    ret = close.pct_change().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret
    if strat.std() == 0 or len(strat) < 2:
        return 0.0
    return float(strat.mean() / strat.std(ddof=0) * np.sqrt(len(strat)))


def _logloss(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    """Binary log-loss: -mean(log p(true class))."""
    eps = 1e-9
    p_true = np.where(y_true == 1, p_pos, 1.0 - p_pos)
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0 - eps))))


def _default_lgbm_params() -> dict:
    return {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",   # компенсація незбалансованих класів
        "verbosity": -1,
    }


# ── Walk-Forward з AFML ──────────────────────────────────────────────────────

def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int = 2000,
    test_size: int = 500,
    params: dict | None = None,
    close: pd.Series | None = None,
    sample_weights: pd.Series | None = None,
) -> MlResult:
    """Walk-forward навчання LightGBM з AFML sample weights.

    Args:
        X: матриця фіч (з build_labeled_dataset).
        y: лейбли {-1, +1}.
        train_size: кількість зразків у навчальному вікні.
        test_size: крок / розмір тестового вікна.
        params: гіперпараметри LightGBM (None = дефолт).
        close: ряд цін для симуляції Sharpe.
        sample_weights: ваги зразків (з compute_sample_weights або build_labeled_dataset).
                        None = рівні ваги.

    Returns:
        MlResult з OOS метриками.
    """
    if not _HAS_LGBM:
        raise ImportError("Встановіть lightgbm: uv add --optional ml lightgbm scikit-learn")

    params = params or _default_lgbm_params()
    oos_preds: list[pd.Series] = []
    oos_proba: list[np.ndarray] = []
    feat_imp_list: list[np.ndarray] = []
    windows = 0
    start = 0

    while start + train_size + test_size <= len(X):
        idx_tr = slice(start, start + train_size)
        idx_te = slice(start + train_size, start + train_size + test_size)

        X_tr = X.iloc[idx_tr]
        y_tr = y.iloc[idx_tr]
        X_te = X.iloc[idx_te]

        # sample weights для навчального вікна
        if sample_weights is not None:
            w_tr = sample_weights.reindex(X_tr.index).fillna(0.0).values
            # нормалізуємо до суми=1 у кожному вікні
            s = w_tr.sum()
            w_tr = w_tr / s if s > 0 else np.ones(len(w_tr)) / len(w_tr)
        else:
            w_tr = None

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        proba = model.predict_proba(X_te)
        classes = list(model.classes_)

        # ймовірність класу +1
        if 1 in classes:
            p_pos = proba[:, classes.index(1)]
        else:
            p_pos = np.full(len(X_te), 0.5)

        # Універсальне мапування: {−1,+1} (primary) або {0,1} (мета-лейблінг)
        other = classes[0]
        pred = pd.Series(np.where(p_pos >= 0.5, 1, other), index=X_te.index)
        oos_preds.append(pred)
        oos_proba.append(p_pos.astype(float))

        # feature importance (gain)
        if hasattr(model, "feature_importances_"):
            feat_imp_list.append(model.feature_importances_)

        windows += 1
        start += test_size

    if not oos_preds:
        raise ValueError(
            f"Не вийшло жодного вікна — збільшіть train_size/test_size або датасет. "
            f"Доступно {len(X)} зразків, потрібно мінімум {train_size + test_size}."
        )

    preds = pd.concat(oos_preds).sort_index()
    y_oos = y.reindex(preds.index)
    p_pos_all = np.concatenate(oos_proba)
    proba_series = pd.Series(p_pos_all, index=preds.index).sort_index()

    acc = float((preds == y_oos).mean())
    ll = _logloss(y_oos.values.astype(int), p_pos_all)

    close_aligned = close.reindex(preds.index) if close is not None else None
    sharpe = _simulate_sharpe(preds, close_aligned) if close_aligned is not None else 0.0

    # середня feature importance по всіх вікнах
    feat_importance = None
    if feat_imp_list and len(feat_imp_list[0]) == len(X.columns):
        avg_imp = np.mean(feat_imp_list, axis=0)
        feat_importance = pd.Series(avg_imp, index=X.columns).sort_values(ascending=False)

    return MlResult(
        oos_accuracy=acc,
        oos_logloss=ll,
        oos_sharpe=sharpe,
        predictions=preds,
        n_train=train_size,
        n_test=len(preds),
        n_windows=windows,
        feature_importance=feat_importance,
        probabilities=proba_series,
    )


# ── Зручний end-to-end helper ─────────────────────────────────────────────────

def train_from_ohlcv(
    df: pd.DataFrame,
    train_size: int = 2000,
    test_size: int = 500,
    mode: str = "triple_barrier",
    pt: float = 1.0,
    sl: float = 1.0,
    holding_bars: int = 10,
    decay: float = 0.9,
    frac_d: float = 0.4,
    add_frac_diff: bool = True,
    params: dict | None = None,
    trades: pd.DataFrame | None = None,
    add_micro: bool = True,
    add_hmm: bool = False,
    add_garch: bool = False,
    hmm_states: int = 3,
) -> MlResult:
    """End-to-end: OHLCV → triple-barrier labels → walk-forward LightGBM.

    Один виклик замість ручного поєднання build_labeled_dataset + train_walk_forward.

    Args:
        df: OHLCV DataFrame з DatetimeIndex.
        train_size: кількість labeled-зразків у train-вікні.
        test_size: крок walk-forward.
        mode: 'triple_barrier' (AFML) або 'horizon'.
        pt/sl: бар'єри (× ATR).
        holding_bars: вертикальний бар'єр (барів).
        decay: time-decay для sample weights.
        frac_d: ступінь fractional differencing.
        add_frac_diff: включити FFD фічі.
        params: LightGBM параметри.
        trades: aggTrades для CVD/micro-фіч.
        add_micro/add_hmm/add_garch: фічі Спринту 2–3 (micro — з trades;
            HMM/GARCH — каузальні, без lookahead).
        hmm_states: кількість HMM-станів.

    Returns:
        MlResult з усіма метриками.
    """
    X, y, w = build_labeled_dataset(
        df=df,
        trades=trades,
        mode=mode,
        pt=pt,
        sl=sl,
        holding_bars=holding_bars,
        decay=decay,
        frac_d=frac_d,
        add_frac_diff=add_frac_diff,
        add_micro=add_micro,
        add_hmm=add_hmm,
        add_garch=add_garch,
        hmm_states=hmm_states,
    )
    logger.info(
        "Dataset: %d зразків | labels: %s | frac_diff: %s",
        len(X),
        dict(y.value_counts().to_dict()),
        add_frac_diff,
    )
    return train_walk_forward(
        X=X,
        y=y,
        train_size=train_size,
        test_size=test_size,
        params=params,
        close=df["close"],
        sample_weights=w,
    )


# ── Walk-forward мета-лейблінг (AFML Ch.3.6–3.7) ─────────────────────────────

def _oof_primary_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray | None,
    params: dict,
    n_splits: int = 3,
    gap: int = 10,
) -> np.ndarray:
    """Чесні (OOF) прогнози primary на train-вікні через time-series split.

    Потрібно, щоб мета-мітки y_meta = (primary вгадав) не були зашумлені
    IS-оптимізмом: primary на власних train-даних майже завжди «правий»,
    що робить y_meta однокласовим і мета-модель дегенеративною (p≈0.5).
    """
    preds = np.zeros(len(X), dtype=int)
    from scalper_hft.validation.cv import time_series_split

    folds = list(time_series_split(len(X), n_splits=n_splits, gap=gap))
    for tr, te in folds:
        if len(te) == 0 or len(tr) < 2:
            continue
        m = LGBMClassifier(**params)
        w_tr = w[tr] if w is not None else None
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w_tr)
        preds[te] = _predict_binary(m, X.iloc[te])
    return preds


def train_walk_forward_meta(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int = 2000,
    test_size: int = 500,
    sample_weights: pd.Series | None = None,
    params: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Walk-forward мета-лейблінг: primary задає сторону, мета — «торгувати чи ні».

    У кожному вікні:
      1. primary-модель навчається на (X_tr, y_tr);
      2. чесні OOF-прогнози primary на X_tr (time-series split, без IS-оптимізму)
         → мета-мітки y_meta = (primary_pred_oof == y) ∈ {0, 1};
      3. мета-модель навчається на (X_tr + 'primary_pred' як фіча, y_meta);
      4. на OOS: side = primary-прогноз, p_meta = P(meta=1).

    Без lookahead: і primary, і мета навчаються лише на train-вікні.

    Returns:
        (side, p_meta): Series, вирівняні на OOS-індексі X.
            side ∈ {−1, +1}; p_meta ∈ [0, 1] — імовірність, що угода прибуткова.
    """
    if not _HAS_LGBM:
        raise ImportError("Встановіть lightgbm: uv add --optional ml lightgbm scikit-learn")

    params = params or _default_lgbm_params()
    oos_side: list[pd.Series] = []
    oos_pmeta: list[np.ndarray] = []
    start = 0

    while start + train_size + test_size <= len(X):
        idx_tr = slice(start, start + train_size)
        idx_te = slice(start + train_size, start + train_size + test_size)
        X_tr = X.iloc[idx_tr]
        y_tr = y.iloc[idx_tr]
        X_te = X.iloc[idx_te]

        w_tr = None
        if sample_weights is not None:
            w_tr = sample_weights.reindex(X_tr.index).fillna(0.0).values
            s = w_tr.sum()
            w_tr = w_tr / s if s > 0 else np.ones(len(w_tr)) / len(w_tr)

        # 1) primary: fit на train
        primary = LGBMClassifier(**params)
        primary.fit(X_tr, y_tr, sample_weight=w_tr)
        side_te = _predict_binary(primary, X_te)  # OOS side (сигнал)

        # 2) чесні OOF-прогнози primary на train → мета-мітки
        side_tr_oof = _oof_primary_predictions(X_tr, y_tr, w_tr, params)
        y_meta_tr = (side_tr_oof == y_tr.values).astype(int)

        # 3) мета: P(primary правий | фічі + side)
        meta = None
        if np.unique(y_meta_tr).size < 2:
            p_meta_te = np.full(len(X_te), 0.5)
        else:
            X_meta_tr = X_tr.copy()
            X_meta_tr["primary_pred"] = side_tr_oof
            X_meta_te = X_te.copy()
            X_meta_te["primary_pred"] = side_te
            meta = LGBMClassifier(**params)
            meta.fit(X_meta_tr, y_meta_tr, sample_weight=w_tr)
            proba = meta.predict_proba(X_meta_te)
            classes = list(meta.classes_)
            p_meta_te = proba[:, classes.index(1)] if 1 in classes else np.full(len(X_te), 0.5)

        oos_side.append(pd.Series(side_te, index=X_te.index))
        oos_pmeta.append(np.asarray(p_meta_te, dtype=float))
        start += test_size

    if not oos_side:
        raise ValueError("Не вийшло жодного вікна для мета-лейблінгу")

    side = pd.concat(oos_side).sort_index()
    p_meta = pd.Series(np.concatenate(oos_pmeta), index=side.index).sort_index()
    return side, p_meta


def _predict_binary(model: object, X: pd.DataFrame) -> np.ndarray:
    """Прогноз класів {−1, +1} або {0, 1} відповідно до model.classes_."""
    proba = model.predict_proba(X)  # type: ignore[attr-defined]
    classes = list(model.classes_)  # type: ignore[attr-defined]
    p_pos = proba[:, classes.index(1)] if 1 in classes else np.full(len(X), 0.5)
    other = classes[0]
    return np.where(p_pos >= 0.5, 1, other)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    """Прогноз напрямку (+1/-1) для нових даних."""
    return _predict_binary(model, X)
