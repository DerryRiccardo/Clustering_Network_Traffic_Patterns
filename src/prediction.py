"""Utilities for assigning new traffic flows to trained K-Means clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import transform_new_data


def predict_flow_cluster(
    input_df: pd.DataFrame,
    model: KMeans,
    scaler: StandardScaler,
    feature_names: list[str],
    skewed_features: list[str],
    log_shifts: dict[str, float],
) -> dict:
    """Predict the nearest cluster for one or more user-provided flows."""
    scaled = transform_new_data(
        input_df,
        scaler=scaler,
        feature_names=feature_names,
        skewed_features=skewed_features,
        log_shifts=log_shifts,
    )
    labels = model.predict(scaled)
    distances = model.transform(scaled)

    return {
        "scaled_data": scaled,
        "cluster_labels": labels,
        "distances": distances,
        "nearest_distances": distances[np.arange(len(labels)), labels],
    }
