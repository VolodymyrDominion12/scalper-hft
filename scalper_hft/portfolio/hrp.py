"""Hierarchical Risk Parity (HRP) портфельна алокація капіталу.

Реалізація за методологією Маркоса Лопеса де Прадо:
"Building Diversified Portfolios that Outperform Out-of-Sample" (Journal of Portfolio Management, 2016).

Переваги над Markowitz / MVO / ERC:
1. НЕ вимагає інверсії коваріаційної матриці (Σ⁻¹), тому стійкий до сингулярних або високоскорельованих активів.
2. Працює у 3 кроки:
   - Tree Clustering (кластеризація активів за кореляційною відстанню).
   - Quasi-Diagonalization (впорядкування активів уздовж діагоналі).
   - Recursive Bisection (рекурсивний поділ ваг між кластерами за оберненою дисперсією).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Розрахунок кореляційної матриці відстаней: d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))."""
    clean_corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - clean_corr))
    np.fill_diagonal(dist, 0.0)
    return dist


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Квазі-діагоналізація: отримання відсортованого списку індексів активів з дендрограми."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # загальна кількість елементів

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # розріджуємо індекси
        df0 = sort_ix[sort_ix >= num_items]  # кластери, що потребують розкриття
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # ліва дитина
        df1 = pd.Series(link[j, 1], index=i + 1)  # права дитина
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return [int(x) for x in sort_ix.tolist()]


def get_cluster_variance(cov: np.ndarray, cluster_indices: list[int]) -> float:
    """Розрахунок дисперсії кластера активів за inverse-variance вагами."""
    sub_cov = cov[np.ix_(cluster_indices, cluster_indices)]
    inv_diag = 1.0 / np.diag(sub_cov)
    inv_diag[np.isinf(inv_diag)] = 0.0
    sum_inv = np.sum(inv_diag)
    if sum_inv <= 0:
        w = np.full(len(cluster_indices), 1.0 / len(cluster_indices))
    else:
        w = inv_diag / sum_inv
    cluster_var = float(np.dot(w, np.dot(sub_cov, w)))
    return max(1e-12, cluster_var)


def get_rec_bisection(cov: np.ndarray, sorted_indices: list[int]) -> pd.Series:
    """Рекурсивна бісекція: розподіл ваг між кластерами за оберненою дисперсією кластерів."""
    w = pd.Series(1.0, index=sorted_indices)
    cluster_list = [sorted_indices]

    while cluster_list:
        new_cluster_list = []
        for cluster in cluster_list:
            if len(cluster) > 1:
                # Розбиваємо кластер навпіл
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = get_cluster_variance(cov, left)
                var_right = get_cluster_variance(cov, right)

                # Частка ваги лівого кластера
                alpha = 1.0 - var_left / (var_left + var_right)
                w[left] *= alpha
                w[right] *= 1.0 - alpha

                new_cluster_list.append(left)
                new_cluster_list.append(right)
        cluster_list = new_cluster_list

    return w


def hrp_weights(
    returns: np.ndarray | pd.DataFrame,
    method: str = "single",
) -> np.ndarray:
    """Hierarchical Risk Parity ваги капіталу.

    Args:
        returns: Матриця дохідностей (T x N), де N >= 2.
        method: Метод зв'язку для кластеризації ('single', 'ward', 'complete', 'average').

    Returns:
        np.ndarray ваг довжиною N, що сумуються до 1.0.
    """
    if isinstance(returns, pd.DataFrame):
        r = returns.to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float)

    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns має бути 2D матрицею (T x N) з кількістю активів N >= 2")

    n_assets = r.shape[1]

    # Перевірка на активи без варіації (dead assets)
    stds = np.std(r, axis=0)
    tol = max(float(stds.max()) * 1e-10, 1e-12)
    dead = stds <= tol
    if dead.all():
        return np.full(n_assets, 1.0 / n_assets)
    if dead.any():
        live = ~dead
        if live.sum() == 1:
            w = np.zeros(n_assets)
            w[live] = 1.0
            return w
        w_live = hrp_weights(r[:, live], method=method)
        w = np.zeros(n_assets)
        w[live] = w_live
        return w

    # Розрахунок коваріації та кореляції
    cov = np.cov(r, rowvar=False)
    # Забезпечуємо симетричність та мінімальну діагональну регуляризацію
    cov = 0.5 * (cov + cov.T) + np.eye(n_assets) * 1e-10

    d_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = np.clip(np.dot(d_inv, np.dot(cov, d_inv)), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # 1. Tree Clustering
    dist = correlation_distance(corr)
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method=method)

    # 2. Quasi-Diagonalization
    sorted_ix = get_quasi_diag(link)

    # 3. Recursive Bisection
    hrp_w_series = get_rec_bisection(cov, sorted_ix)

    # Відновлюємо початковий порядок активів (0..N-1)
    hrp_w = hrp_w_series.sort_index().to_numpy(dtype=float)

    # Нормалізація до 1.0
    sum_w = np.sum(hrp_w)
    if sum_w > 0:
        hrp_w = hrp_w / sum_w
    else:
        hrp_w = np.full(n_assets, 1.0 / n_assets)

    return hrp_w
