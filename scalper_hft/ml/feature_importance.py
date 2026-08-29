"""Feature importance (AFML Ch.8): MDI / MDA / SFI + PCA-перевірка.

«Backtesting is not a research tool. Feature importance is» (Перший закон LdP):
замість ітерацій бектесту для відбору фіч — три рівні важливості:

    MDI (IS, impurity): сумарне зменшення нечистоти дерев; щоб уникнути
        masking — малий feature_fraction у LGBM; нулі → NaN перед усередненням;
    MDA (OOS, permutation): для кожної фічі переставляємо колонку в тесті →
        втрата скора (neg log-loss); CV purged+embargoed; важливість = відносне
        покращення;
    SFI (OOS, single-feature): скоринг кожної фічі окремо — без substitution
        effects, але губить joint effects.

PCA-перевірка: weighted Kendall τ між MDI та оберненим PCA-рангом > 0.8 —
статистичне підтвердження, що патерн не оверфіт (у книзі corr 0.85).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mdi(model: object, X: pd.DataFrame) -> pd.Series:
    """Mean Decrease Impurity: важливість з дерев (IS).

    Для LGBM використовує gain-важливість (impurity); нулі замінюються NaN
    перед поверненням (AFML: нулі — це masking, не «нульова важливість»).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("model має мати feature_importances_ (LGBM/sklearn tree)")
    imp = np.asarray(model.feature_importances_, dtype=float)
    if len(imp) != X.shape[1]:
        raise ValueError("довжина feature_importances_ не збігається з X")
    out = pd.Series(imp, index=list(X.columns))
    out[out == 0] = np.nan
    return out


def _iter_folds(cv, X: pd.DataFrame, t1: pd.Series | None):
    """Генератор (train_idx, test_idx) з cv.

    PurgedKFold потребує t1; без t1 — time-series split (expanding).
    cv може бути: об'єктом з .split, або готовою послідовністю фолдів.
    """
    if hasattr(cv, "split"):
        if t1 is not None:
            yield from cv.split(X, t1=t1)
        else:
            from scalper_hft.validation.cv import time_series_split

            n = len(X)
            embargo = int(n * getattr(cv, "embargo_pct", 0.01))
            n_splits = getattr(cv, "n_splits", 5)
            yield from time_series_split(n, n_splits=n_splits, embargo=embargo)
    else:
        yield from cv


def _fold_score(
    clf_factory,
    X_tr: pd.DataFrame, y_tr: pd.Series, w_tr,
    X_te: pd.DataFrame, y_te: pd.Series,
    score: str,
) -> float:
    """neg-log-loss або accuracy на тестовому зрізі."""
    clf = clf_factory()
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    proba = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    if 1 in classes:
        p_pos = proba[:, classes.index(1)]
    else:
        p_pos = np.full(len(X_te), 0.5)
    if score == "accuracy":
        pred = np.where(p_pos >= 0.5, 1, classes[0])
        return float((pred == y_te.values).mean())
    # neg log loss (вище = краще; значення ≤ 0)
    p_true = np.where(y_te.values == 1, np.clip(p_pos, 1e-9, 1 - 1e-9),
                      1 - np.clip(p_pos, 1e-9, 1 - 1e-9))
    return float(np.mean(np.log(p_true)))


def mda(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    cv,
    sample_weights: pd.Series | None = None,
    score: str = "neg_log_loss",
    n_repeats: int = 1,
    t1: pd.Series | None = None,
) -> pd.Series:
    """Mean Decrease Accuracy: permutation importance на OOS з purged CV.

    cv: генератор розбиттів (PurgedKFold або подібний), split(X) → (tr, te).
    clf_factory: нуль-аргументна функція, що повертає новий класифікатор.
    Returns: важливість = відносне падіння скора при перестановці фічі.
    """
    base_scores: list[float] = []
    perm_scores: dict[str, list[float]] = {c: [] for c in X.columns}
    for train_idx, test_idx in _iter_folds(cv, X, t1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = sample_weights.reindex(X_tr.index).fillna(0.0).values if sample_weights is not None else None
        base = _fold_score(clf_factory, X_tr, y_tr, w_tr, X_te, y_te, score)
        base_scores.append(base)
        for col in X.columns:
            X_perm = X_te.copy()
            drops = []
            for _ in range(n_repeats):
                X_perm[col] = np.random.permutation(X_perm[col].values)
                drops.append(base - _fold_score(clf_factory, X_tr, y_tr, w_tr, X_perm, y_te, score))
            perm_scores[col].append(float(np.mean(drops)))
    base_mean = float(np.mean(base_scores)) if base_scores else 0.0
    if abs(base_mean) < 1e-12:
        return pd.Series(0.0, index=X.columns)
    return pd.Series({col: float(np.mean(v)) / abs(base_mean) for col, v in perm_scores.items()})


def sfi(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    cv,
    sample_weights: pd.Series | None = None,
    score: str = "neg_log_loss",
    t1: pd.Series | None = None,
) -> pd.Series:
    """Single-Feature Importance: скоринг кожної фічі окремо (OOS, purged CV)."""
    out: dict[str, list[float]] = {c: [] for c in X.columns}
    for col in X.columns:
        X1 = X[[col]]
        for train_idx, test_idx in _iter_folds(cv, X1, t1):
            X_tr, X_te = X1.iloc[train_idx], X1.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            w_tr = sample_weights.reindex(X_tr.index).fillna(0.0).values if sample_weights is not None else None
            out[col].append(_fold_score(clf_factory, X_tr, y_tr, w_tr, X_te, y_te, score))
    return pd.Series({c: float(np.mean(v)) for c, v in out.items()})


def pca_importance_corr(mdi_values: pd.Series, X: pd.DataFrame, method: str = "kendall") -> float:
    """Weighted Kendall τ між MDI-рангом та оберненим PCA-рангом (AFML Ch.8.6).

    Висока кореляція (>0.8 у книзі) — патерн важливості не випадковий
    (фічі з більшою дисперсією мають вищу важливість — узгоджено з PCA).
    """
    cols = [c for c in X.columns if c in mdi_values.index]
    if len(cols) < 3:
        return 0.0
    Z = (X[cols] - X[cols].mean()) / X[cols].std().replace(0, np.nan)
    Z = Z.fillna(0.0).values
    try:
        from scipy.linalg import eigh

        _, eigvecs = eigh(np.cov(Z, rowvar=False))
    except Exception:  # noqa: BLE001
        return 0.0
    # важливість кожної фічі = внесок у перші головні компоненти (вага |навантаження|)
    pca_imp = pd.Series(np.abs(eigvecs[:, : min(3, eigvecs.shape[1])]).sum(axis=1), index=cols)
    mdi_rank = mdi_values[cols].rank(ascending=True)
    pca_rank = pca_imp.rank(ascending=False)  # обернений PCA-ранг
    if method == "spearman":
        return float(mdi_rank.corr(pca_rank, method="spearman"))
    try:
        from scipy.stats import weightedtau

        tau, _ = weightedtau(mdi_rank.values, pca_rank.values)
        return float(tau)
    except Exception:  # noqa: BLE001
        return float(mdi_rank.corr(pca_rank, method="spearman"))


def feature_importance_report(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    cv,
    sample_weights: pd.Series | None = None,
    score: str = "neg_log_loss",
    t1: pd.Series | None = None,
) -> pd.DataFrame:
    """Зведений звіт: MDI-проксі (перша модель) + MDA + SFI + PCA-τ.

    Returns:
        DataFrame з колонками mda, sfi, pca_tau, відсортований за MDA.
    """
    mda_vals = mda(X, y, clf_factory, cv, sample_weights=sample_weights, score=score, t1=t1)
    sfi_vals = sfi(X, y, clf_factory, cv, sample_weights=sample_weights, score=score, t1=t1)
    # MDI-проксі: середній gain з однієї моделі на повних даних
    clf = clf_factory()
    clf.fit(X, y, sample_weight=sample_weights.reindex(X.index).fillna(0.0).values if sample_weights is not None else None)
    mdi_vals = mdi(clf, X)
    tau = pca_importance_corr(mdi_vals.dropna(), X)
    out = pd.DataFrame({"mda": mda_vals, "sfi": sfi_vals, "mdi": mdi_vals})
    out["pca_tau"] = tau
    return out.sort_values("mda", ascending=False)


__all__ = ["mdi", "mda", "sfi", "pca_importance_corr", "feature_importance_report"]
