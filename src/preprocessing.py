"""
Preprocessing pipeline: feature selection, cleaning, transformation, scaling.

Every function is stateless and reusable so the Streamlit pages can
show before / after comparisons without re‑running the full pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.config import SELECTED_FEATURES, SKEW_THRESHOLD


# ──────────────────────────────────────────────
# Feature Selection
# ──────────────────────────────────────────────
def select_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Select the specified feature columns from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (may contain extra columns).
    features : list[str] or None
        Column names to keep.  Defaults to ``SELECTED_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Dataframe with only the selected columns.
    """
    features = features or SELECTED_FEATURES
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[features].copy()


# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
def clean_data(
    df: pd.DataFrame,
    labels: pd.Series | None = None,
    missing_strategy: str = "Drop Rows",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Handle NaN, infinity values, and duplicates, keeping labels in sync.

    Steps
    -----
    1. Replace ±inf with NaN.
    2. Drop rows containing any NaN.
    3. Remove duplicate rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        Cleaned dataframe and corresponding cleaned labels.
    """
    out = df.copy()
    if labels is not None:
        out["__label__"] = labels.copy()

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if missing_strategy == "Drop Rows":
        out.dropna(inplace=True)
    elif missing_strategy == "Fill with Mean":
        out.fillna(out.mean(numeric_only=True), inplace=True)
    elif missing_strategy == "Fill with Median":
        out.fillna(out.median(numeric_only=True), inplace=True)
        
    out.drop_duplicates(inplace=True)

    if labels is not None:
        out_labels = out.pop("__label__")
    else:
        out_labels = None

    out.reset_index(drop=True, inplace=True)
    if out_labels is not None:
        out_labels.reset_index(drop=True, inplace=True)

    return out, out_labels


# ──────────────────────────────────────────────
# Skewness helpers
# ──────────────────────────────────────────────
def get_skewed_features(
    df: pd.DataFrame,
    threshold: float = SKEW_THRESHOLD,
) -> list[str]:
    """Return feature names whose absolute skewness exceeds *threshold*."""
    skew = df.skew(numeric_only=True)
    return skew[skew.abs() > threshold].index.tolist()


# ──────────────────────────────────────────────
# Log Transform
# ──────────────────────────────────────────────
def apply_log_transform(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    shifts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply ``log1p`` to highly skewed columns.

    If *columns* is ``None``, columns are detected automatically via
    :func:`get_skewed_features`.
    """
    out = df.copy()
    if columns is None:
        columns = get_skewed_features(out)
    shifts = shifts or {}
    for col in columns:
        if col in out.columns:
            shift = shifts.get(col)
            if shift is None:
                min_val = out[col].min()
                shift = -float(min_val) if min_val < 0 else 0.0
            if shift:
                out[col] = out[col] + shift
            out[col] = np.log1p(out[col])
    return out


def get_log_shifts(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Return per-column shifts used before log1p transformation."""
    shifts: dict[str, float] = {}
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            shifts[col] = -float(min_val) if min_val < 0 else 0.0
    return shifts


def transform_new_data(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: list[str],
    skewed_features: list[str] | None = None,
    log_shifts: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply the fitted preprocessing steps to new user-provided flows."""
    selected = select_features(df, feature_names)
    selected = selected.replace([np.inf, -np.inf], np.nan)
    if selected.isnull().any().any():
        missing_cols = selected.columns[selected.isnull().any()].tolist()
        raise ValueError(f"Input contains missing or infinite values in: {missing_cols}")

    transformed = apply_log_transform(
        selected,
        columns=skewed_features or [],
        shifts=log_shifts or {},
    )
    if not np.isfinite(transformed.to_numpy()).all():
        raise ValueError(
            "Input produces non-finite values after preprocessing. "
            "Please check negative or extremely large feature values."
        )
    return scaler.transform(transformed)


# ──────────────────────────────────────────────
# Scaling
# ──────────────────────────────────────────────
def scale_data(
    df: pd.DataFrame,
    scaler_type: str = "StandardScaler",
) -> tuple[np.ndarray | pd.DataFrame, object | None, list[str]]:
    """Scale features using the specified scaler.

    Returns
    -------
    tuple
        (scaled_array_or_df, fitted_scaler, feature_names)
    """
    feature_names = df.columns.tolist()
    
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df)
    else:  # "Tanpa Scaling"
        scaler = None
        scaled = df.values
        
    return scaled, scaler, feature_names


# ──────────────────────────────────────────────
# Full pipeline helper
# ──────────────────────────────────────────────
def run_preprocessing_pipeline(
    df: pd.DataFrame,
    labels: pd.Series | None = None,
    features: list[str] | None = None,
    apply_log: bool = True,
    missing_strategy: str = "Drop Rows",
    scaler_type: str = "StandardScaler",
) -> dict:
    """Run the complete preprocessing pipeline and return all artefacts.

    Returns a dict with keys:
        selected, cleaned, labels, log_transformed, scaled_data, scaler,
        feature_names, rows_before, rows_after, skewed_features
    """
    selected = select_features(df, features)
    rows_before = len(selected)

    cleaned, cleaned_labels = clean_data(selected, labels, missing_strategy)
    rows_after = len(cleaned)

    skewed = get_skewed_features(cleaned)

    log_shifts = get_log_shifts(cleaned, skewed)

    if apply_log and skewed:
        transformed = apply_log_transform(cleaned, skewed, shifts=log_shifts)
    else:
        transformed = cleaned.copy()

    scaled_data, scaler, feature_names = scale_data(transformed, scaler_type)

    return {
        "selected": selected,
        "cleaned": cleaned,
        "labels": cleaned_labels,
        "log_transformed": transformed,
        "scaled_data": scaled_data,
        "scaler": scaler,
        "feature_names": feature_names,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "skewed_features": skewed,
        "log_shifts": log_shifts,
    }
