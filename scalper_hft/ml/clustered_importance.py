"""Clustered Feature Importance (CFI) — AFML Ch. 8.5 (López de Prado).

Кластерна важливість ознак усуває дві фундаментальні вади стандартних MDI/MDA:
    1. Заміщення ознак (Substitution Effect): якщо дві фічі сильно корельовані,
       перестановка однієї не знижує скор (MDA≈0), оскільки друга компенсує її.
    2. Розмивання важливості (Dilution): дерева ділять спліти між дублями ознак,
       штучно занижуючи їх MDI.

CFI групує ознаки в ієрархічні кластери та переставляє цілі кластери одночасно.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from scalper_hft.ml.feature_importance import _fold_score, _iter_folds


@dataclass
class CfiResult:
    clustered_mda: pd.Series  # важливість кластерів (cluster_id -> importance)
    feature_clusters: pd.Series  # прив'язка фіч до кластерів (feature_name -> cluster_id)
    feature_mda: pd.Series  # скоригована важливість окремих фіч
    clusters_dict: dict[int, list[str]]  # cluster_id -> список фіч


def cluster_features(
    X: pd.DataFrame,
    max_clusters: int | None = None,
    threshold: float = 0.5,
) -> tuple[pd.Series, dict[int, list[str]]]:
    """Ієрархічна кластеризація ознак на основі кореляційної відстані.

    Відстань d_ij = sqrt(0.5 * (1 - rho_ij)).
    """
    corr = X.corr(method="spearman").fillna(0.0).values
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # Condense distance matrix for scipy linkage
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="ward")

    if max_clusters is not None and max_clusters > 0:
        clusters = fcluster(link, t=min(max_clusters, X.shape[1]), criterion="maxclust")
    else:
        clusters = fcluster(link, t=threshold, criterion="distance")

    feat_clusters = pd.Series(clusters, index=X.columns, name="cluster_id")
    clusters_dict: dict[int, list[str]] = {}
    for feat, cl_id in feat_clusters.items():
        clusters_dict.setdefault(int(cl_id), []).append(str(feat))

    return feat_clusters, clusters_dict


def clustered_mda(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    cv,
    sample_weights: pd.Series | None = None,
    score: str = "neg_log_loss",
    max_clusters: int | None = None,
    threshold: float = 0.5,
    n_repeats: int = 1,
    t1: pd.Series | None = None,
) -> CfiResult:
    """Clustered Mean Decrease Accuracy (Clustered MDA, AFML Ch. 8.5).

    Одночасно переставляє всі ознаки в межах кожного кластера для запобігання
    substitution effects на OOS фолдах (Purged CV).
    """
    feat_clusters, clusters_dict = cluster_features(X, max_clusters=max_clusters, threshold=threshold)
    unique_clusters = sorted(clusters_dict.keys())

    base_scores: list[float] = []
    cluster_drops: dict[int, list[float]] = {cl_id: [] for cl_id in unique_clusters}

    for train_idx, test_idx in _iter_folds(cv, X, t1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = sample_weights.reindex(X_tr.index).fillna(0.0).values if sample_weights is not None else None

        base = _fold_score(clf_factory, X_tr, y_tr, w_tr, X_te, y_te, score)
        base_scores.append(base)

        for cl_id, feats in clusters_dict.items():
            drops = []
            for _ in range(n_repeats):
                X_perm = X_te.copy()
                for f in feats:
                    X_perm[f] = np.random.permutation(X_perm[f].values)
                drops.append(base - _fold_score(clf_factory, X_tr, y_tr, w_tr, X_perm, y_te, score))
            cluster_drops[cl_id].append(float(np.mean(drops)))

    base_mean = float(np.mean(base_scores)) if base_scores else 0.0
    denom = abs(base_mean) if abs(base_mean) > 1e-12 else 1.0

    clustered_mda_series = pd.Series(
        {cl_id: float(np.mean(v)) / denom for cl_id, v in cluster_drops.items()},
        name="clustered_mda",
    )

    # Розподіляємо важливість кластера порівну між його ознаками
    feat_mda_dict: dict[str, float] = {}
    for cl_id, feats in clusters_dict.items():
        cl_imp = clustered_mda_series.get(cl_id, 0.0)
        for f in feats:
            feat_mda_dict[f] = cl_imp / len(feats)

    feature_mda_series = pd.Series(feat_mda_dict, index=X.columns, name="feature_mda")

    return CfiResult(
        clustered_mda=clustered_mda_series,
        feature_clusters=feat_clusters,
        feature_mda=feature_mda_series,
        clusters_dict=clusters_dict,
    )
