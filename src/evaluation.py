"""
Clustering evaluation metrics.

Computes Silhouette Score, Davies‑Bouldin Index, and Calinski‑Harabasz
Index for one or many *k* values.

Note: Silhouette Score uses a subsample when the dataset exceeds
``SILHOUETTE_SAMPLE_SIZE`` rows to avoid O(n²) memory issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Silhouette computes pairwise distances → O(n²) memory.
# Use manual subsampling for all metrics to keep memory reasonable and fast.
EVALUATION_SAMPLE_SIZE = 10_000


def calculate_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    sample_size: int = EVALUATION_SAMPLE_SIZE,
    random_state: int = 42,
) -> dict:
    """Compute internal clustering metrics for a single set of labels.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix.
    labels : np.ndarray
        Cluster labels.
    sample_size : int
        Max rows used for evaluation (subsampled when data is larger).
    random_state : int
        RNG seed for reproducible subsampling.

    Returns
    -------
    dict
        Keys: ``silhouette``, ``davies_bouldin``, ``calinski_harabasz``.
    """
    n = len(data)
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        data = data[idx]
        labels = labels[idx]

    sil = silhouette_score(data, labels)

    return {
        "silhouette": sil,
        "davies_bouldin": davies_bouldin_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
    }


def evaluate_multiple_k(
    data: np.ndarray,
    models: dict[int, KMeans],
) -> pd.DataFrame:
    """Evaluate all candidate models and return a comparison table.

    Parameters
    ----------
    data : np.ndarray
        The **same** scaled data used for training.
    models : dict[int, KMeans]
        ``{k: fitted_model}`` mapping.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``silhouette``, ``davies_bouldin``,
        ``calinski_harabasz``, ``inertia``.
    """
    rows = []
    for k in sorted(models):
        m = models[k]
        metrics = calculate_metrics(data, m.labels_)
        metrics["k"] = k
        metrics["inertia"] = m.inertia_
        rows.append(metrics)

    df = pd.DataFrame(rows)
    col_order = ["k", "silhouette", "davies_bouldin", "calinski_harabasz", "inertia"]
    return df[col_order]


def get_best_k(eval_df: pd.DataFrame) -> int:
    """Return the *k* with the highest Silhouette Score."""
    return int(eval_df.loc[eval_df["silhouette"].idxmax(), "k"])
