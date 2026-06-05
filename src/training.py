"""
K‑Means training utilities.

Trains models for a range of *k* values and provides persistence
helpers using joblib.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans

from src.config import K_RANGE, MODELS_DIR, RANDOM_STATE


def train_kmeans(
    data: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    """Train a single K‑Means model.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    k : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    KMeans
        Fitted model.
    """
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(data)
    return model


def train_multiple_k(
    data: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = RANDOM_STATE,
    progress_callback=None,
) -> dict[int, KMeans]:
    """Train K‑Means for several values of *k*.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix.
    k_range : range or list[int] or None
        Values of *k* to try.  Defaults to ``K_RANGE``.
    random_state : int
        Random seed.
    progress_callback : callable or None
        Called with ``(current_index, total)`` after each fit.

    Returns
    -------
    dict[int, KMeans]
        Mapping ``{k: fitted_model}``.
    """
    if k_range is None:
        k_range = K_RANGE
    k_list = list(k_range)
    models: dict[int, KMeans] = {}
    total = len(k_list)
    for idx, k in enumerate(k_list):
        models[k] = train_kmeans(data, k, random_state)
        if progress_callback:
            progress_callback(idx + 1, total)
    return models


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────
def save_model(
    model: KMeans,
    scaler,
    metadata: dict,
    path: Path | None = None,
) -> Path:
    """Save model, scaler and metadata to *path*.

    Creates ``kmeans_model.joblib``, ``scaler.joblib``, and
    ``metadata.json`` inside *path*.
    """
    path = path or MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path / "kmeans_model.joblib")
    joblib.dump(scaler, path / "scaler.joblib")

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return path


def load_model(
    path: Path | None = None,
) -> tuple[KMeans, object, dict]:
    """Load a previously saved model bundle.

    Returns
    -------
    tuple[KMeans, scaler, metadata_dict]
    """
    path = path or MODELS_DIR
    model = joblib.load(path / "kmeans_model.joblib")
    scaler = joblib.load(path / "scaler.joblib")
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata
